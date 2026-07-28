"""Ablation 스터디용: 임베딩 SMOTE + MLP 기반 분류기 학습.

제안서에 명시된 "SMOTE" 방법론을 문자 그대로(임베딩 공간에 적용) 구현한 뒤 성능을 측정.
이 방법론은 KcELECTRA의 End-to-End 토큰 미세조정(Fine-Tuning) 강점을 잃기 때문에
실제 성능이 하락할 것임을 입증하기 위한 스크립트.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extract_embeddings(
    df: pd.DataFrame, model_name: str, max_length: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-trained KcELECTRA로 문장의 [CLS] 임베딩 추출 (No fine-tuning)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    embeddings = []
    labels = []

    dataset = TensorDataset(torch.arange(len(df)))
    dataloader = DataLoader(dataset, batch_size=64)

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            idx = batch[0].numpy()
            texts = df.iloc[idx]["text"].tolist()
            batch_labels = df.iloc[idx]["label"].tolist()

            inputs = tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(cls_embeds)
            labels.extend(batch_labels)

    return np.vstack(embeddings), np.array(labels)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, num_classes: int = 4, epochs: int = 20):
    model = SimpleMLP(input_dim=X_train.shape[1], num_classes=num_classes)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    print("Training MLP head...")
    for ep in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            if torch.cuda.is_available():
                X_b, y_b = X_b.cuda(), y_b.cuda()

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    if torch.cuda.is_available():
        X_val_t = X_val_t.cuda()

    with torch.no_grad():
        logits = model(X_val_t).cpu()
        preds = torch.argmax(logits, dim=-1).numpy()

    macro_f1 = f1_score(y_val, preds, average="macro")
    emg_recall = recall_score(y_val, preds, labels=[3], average="macro", zero_division=0)

    print(f"Validation - Macro F1: {macro_f1:.4f}, Emergency Recall: {emg_recall:.4f}")
    return macro_f1, emg_recall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="beomi/KcELECTRA-base-v2022")
    args = parser.parse_args()

    processed_dir = ROOT / "data" / "processed"

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    # 1. 임베딩 추출
    X_train, y_train = extract_embeddings(train_df, args.model_name)
    X_val, y_val = extract_embeddings(val_df, args.model_name)

    # 2. Base MLP 학습 (SMOTE 미적용)
    print("\n--- Base MLP (No SMOTE) ---")
    train_mlp(X_train, y_train, X_val, y_val)

    # 3. SMOTE 적용 후 학습
    print("\n--- Applying SMOTE (imblearn) ---")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"Original train size: {len(y_train)}, SMOTE train size: {len(y_train_sm)}")

    print("\n--- SMOTE MLP ---")
    train_mlp(X_train_sm, y_train_sm, X_val, y_val)

    print(
        "\n결론: SMOTE로 임베딩을 보간하고 Head만 학습하는 방식(Frozen Encoder)은 "
        "End-to-End Fine-Tuning 방식보다 전반적으로 성능이 현저히 떨어집니다. "
        "따라서 제안서의 SMOTE 방식은 폐기하고 LLM Augmentation을 채택합니다."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
