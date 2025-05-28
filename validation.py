import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from matplotlib import font_manager, rc 

# 한글출력위한 폰트
font_path = "malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

# 1. 데이터 로드 + 셔플
df = pd.read_csv("val_split.csv").dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 라벨 인코더 불러오기
with open("kc_model_merge/kc_label_encoder_merge.pkl", "rb") as f:
    le = pickle.load(f)
df["label"] = le.transform(df["target"])

# val만 분리 (전체 20%)
val_size = int(len(df))
val_df = df[:val_size]

# 토크나이저 + 모델 로드
tokenizer = AutoTokenizer.from_pretrained("kc_model_merge")
model = AutoModelForSequenceClassification.from_pretrained("kc_model_merge")

# Dataset 정의
class ProductDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(self.encodings[k][idx]) for k in self.encodings}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

val_dataset = ProductDataset(val_df["input"].tolist(), val_df["label"].tolist())
val_loader = DataLoader(val_dataset, batch_size=8)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 평가
y_true, y_pred = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 리포트 출력
print("\n📊 Classification Report:")
print(classification_report(
    y_true, y_pred,
    labels=list(range(len(le.classes_))),
    target_names=le.classes_,
    zero_division=0
))

# Confusion Matrix 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_val.png")
plt.show()

# 오분류 인덱스 찾기
mismatched_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

# 오분류 데이터 추출
misclassified_df = val_df.iloc[mismatched_indices].copy()
misclassified_df["예측"] = [le.inverse_transform([p])[0] for p in [y_pred[i] for i in mismatched_indices]]
misclassified_df["실제"] = [le.inverse_transform([t])[0] for t in [y_true[i] for i in mismatched_indices]]

# 필요한 열만 포함하여 JSON 변환
misclassified_json = misclassified_df[["input", "cleaned_input", "실제", "예측"]].to_dict(orient="records")

import json
with open("misclassified_samples.json", "w", encoding="utf-8") as f:
    json.dump(misclassified_json, f, indent=2, ensure_ascii=False)
