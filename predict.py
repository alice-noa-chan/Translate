import torch
import warnings  # 경고 제어를 위한 모듈
from transformers import BartForConditionalGeneration, BartTokenizer

# 경고 출력 설정
IgnoreWarning = True  # True이면 경고를 무시하고, False이면 경고 출력

if IgnoreWarning:
    warnings.filterwarnings("ignore", category=FutureWarning)  # FutureWarning 무시
else:
    warnings.filterwarnings("default")  # 기본 경고 출력

# 모델과 토크나이저를 로드하는 함수
def load_model(model_path="output/checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA가 가능하면 GPU 사용
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")  # KoBART 모델 로드
    tokenizer = BartTokenizer.from_pretrained("gogamza/kobart-base-v2")  # KoBART 전용 토크나이저 로드
    
    # 학습된 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # 모델을 평가 모드로 전환
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

# 입력된 텍스트를 번역하는 함수
def generate_text(model, tokenizer, device, input_text, max_length=128):
    # 입력 텍스트를 토크나이즈하고, 디바이스에 맞게 이동
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    
    # 모델 예측 수행
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    
    # 결과 디코딩
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# 메인 실행 함수
def main():
    model_path = "output/checkpoint.pt"  # 모델 파일 경로 변수화
    model, tokenizer, device = load_model(model_path)
    
    print("Enter text to translate (type 'quit' to exit):")
    
    # 사용자에게 무한 입력을 받아 번역
    while True:
        input_text = input(">> ")  # 사용자 입력 받기
        if input_text.lower() == 'quit':  # 'quit' 입력 시 종료
            print("Exiting translation...")
            break
        output_text = generate_text(model, tokenizer, device, input_text)
        print("Generated Output:", output_text)

if __name__ == "__main__":
    main()
