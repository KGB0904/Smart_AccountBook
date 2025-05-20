import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re
import pickle
import os

# 데이터 로드 및 전처리
df = pd.read_csv("total_product_sorted_merge.csv").dropna()

def clean_text(text):
    text = re.sub(r"[^\w가-힣\s]", "", text)
    text = re.sub(r"\bP\b|ML|ml|G|g|g\b", "", text, flags=re.IGNORECASE)
    return text.strip()

df["cleaned_input"] = df["input"].apply(clean_text)

# 라벨 인코딩
le = LabelEncoder()
df["label"] = le.fit_transform(df["target"])

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

dataset = ProductDataset(df["cleaned_input"].tolist(), df["label"].tolist())
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델 초기화 및 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=len(le.classes_)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
epochs = 10
for epoch in range(epochs):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
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

    print(f" Epoch {epoch+1} 평균 손실: {total_loss / len(loader):.4f}")

# 모델 및 토크나이저 저장
model.save_pretrained("kc_model_merge")
tokenizer.save_pretrained("kc_model_merge")
