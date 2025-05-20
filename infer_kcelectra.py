import torch
import json
import pickle
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 전처리 함수
def clean_text(text):
    text = re.sub(r"[^\w가-힣\s]", "", text)
    text = re.sub(r"\bP\b|ML|ml|G|g|g\b", "", text, flags=re.IGNORECASE)
    return text.strip()

# 모델, 토크나이저, 라벨 인코더 로드
model = AutoModelForSequenceClassification.from_pretrained("kc_model_merge")
tokenizer = AutoTokenizer.from_pretrained("kc_model_merge")
with open("kc_model_merge/kc_label_encoder_merge.pkl", "rb") as f:
    le = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 입력 로드
with open("gpt_receipt_result.json", "r", encoding="utf-8") as f:
    gpt_items = json.load(f)

# 입력 정제 및 필터링
texts = []
valid_items = []
for entry in gpt_items:
    item = entry["item"]
    cleaned = clean_text(item)
    if cleaned and len(cleaned) >= 2:
        texts.append(cleaned)
        valid_items.append(entry)

inputs = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 예측
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1)

# 라벨 변환
labels = le.inverse_transform(preds.cpu().numpy())
for i, entry in enumerate(valid_items):
    entry["category"] = labels[i]
    print(f'{entry["item"]} → {entry["category"]}')

# 저장
with open("gpt_receipt_with_categories.json", "w", encoding="utf-8") as f:
    json.dump(gpt_items, f, ensure_ascii=False, indent=2)
