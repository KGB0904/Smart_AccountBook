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
from matplotlib import font_manager as fm, rc
from sklearn.model_selection import train_test_split
import os
from torch.optim import AdamW
from tqdm import tqdm


# 데이터 로드 및 전처리
df = pd.read_csv("total_product_sorted_cleaned_merge.csv").dropna()

def clean_text(text):
    return text

df["cleaned_input"] = df["input"].apply(clean_text)

# 라벨 인코딩
le = LabelEncoder()
df["label"] = le.fit_transform(df["target"])

#  train/val 분할
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#  split 파일 저장
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

# 저장 경로
os.makedirs("kc_model_merge", exist_ok=True)
with open("kc_model_merge/kc_label_encoder_merge.pkl", "wb") as f:
    pickle.dump(le, f)

# 토크나이저 및 데이터셋 정의
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

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

# 학습에 train_df만 사용해서 변환
train_dataset = ProductDataset(train_df["cleaned_input"].tolist(), train_df["label"].tolist())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 모델 초기화 및 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=len(le.classes_)).to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

model.train()
epochs = 8
for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    total_loss = 0
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f" Epoch {epoch+1} 평균 손실: {total_loss / len(train_loader):.4f}")

# 모델 및 토크나이저 저장
model.save_pretrained("kc_model_merge")
tokenizer.save_pretrained("kc_model_merge")
