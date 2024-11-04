import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import evaluate  # 평가 라이브러리 사용

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)

# 커스텀 데이터셋 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        targets = self.tokenizer(item['tran'], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return inputs['input_ids'].squeeze(), targets['input_ids'].squeeze()

# JSONL 파일 로드 함수
def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({'text': item['text'], 'tran': item['tran']})
    return data

# 모델과 데이터를 준비하는 함수
def prepare_data_and_model(data_path, batch_size=16, train_ratio=0.8, valid_ratio=0.1):
    data = load_data(data_path)
    random.shuffle(data)

    tokenizer = BartTokenizer.from_pretrained("gogamza/kobart-base-v2")
    dataset = TranslationDataset(data, tokenizer)
    
    # 데이터셋 분할
    train_size = int(len(dataset) * train_ratio)
    valid_size = int(len(dataset) * valid_ratio)
    test_size = len(dataset) - train_size - valid_size
    
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    # KoBART 모델 로드
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
    
    return train_loader, valid_loader, test_loader, model, tokenizer

# 체크포인트 저장 함수
def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

# 체크포인트 로드 함수
def load_checkpoint(model, optimizer, scheduler, path="checkpoint.pt"):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded, resuming from epoch {epoch+1}")
        return epoch + 1  # 다음 에포크부터 시작
    else:
        print("No checkpoint found, starting from scratch.")
        return 0  # 처음부터 시작

# 학습 함수 정의
def train_model(data_path, output_dir="output", epochs=10, batch_size=16, lr=5e-5, weight_decay=1e-4, patience=3, warmup_steps=500):
    os.makedirs(output_dir, exist_ok=True)  # output 디렉터리가 없으면 생성합니다.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, model, tokenizer = prepare_data_and_model(data_path, batch_size)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    early_stopping = EarlyStopping(patience=patience)
    
    # evaluate 라이브러리를 사용하여 ROUGE 메트릭 로드
    rouge = evaluate.load("rouge")
    
    # 이전 체크포인트 로드
    start_epoch = load_checkpoint(model, optimizer, scheduler, path=os.path.join(output_dir, "checkpoint.pt"))

    for epoch in range(start_epoch, epochs):  # start_epoch 부터 시작
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(input_ids=inputs, labels=targets)
                loss = outputs.loss
                val_loss += loss.item()
                
                predictions = outputs.logits.argmax(dim=-1)
                accuracy = (predictions == targets).float().mean().item()
                val_accuracy += accuracy
                
                decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(targets, skip_special_tokens=True)
                rouge.add_batch(predictions=decoded_preds, references=decoded_labels)
        
        val_loss /= len(valid_loader)
        val_accuracy /= len(valid_loader)
        rouge_score = rouge.compute()
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Rouge-1: {rouge_score['rouge1']:.4f}, "
              f"Rouge-2: {rouge_score['rouge2']:.4f}, Rouge-L: {rouge_score['rougeL']:.4f}")
        
        early_stopping(val_loss, model, path=os.path.join(output_dir, "checkpoint.pt"))
        
        # 체크포인트 저장
        save_checkpoint(model, optimizer, scheduler, epoch, path=os.path.join(output_dir, "checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # 테스트
    model.load_state_dict(torch.load(os.path.join(output_dir, "checkpoint.pt")))
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            test_loss += loss.item()
            
            predictions = outputs.logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            test_accuracy += accuracy
    
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 모델 학습 실행
train_model(data_path="data.jsonl")