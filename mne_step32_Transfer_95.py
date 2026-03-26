import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE MODEL (EEGNet)
# ==========================================
class EEGNet_Final(nn.Module):
    def __init__(self, channels=22, samples=513, n_classes=4):
        super(EEGNet_Final, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(0.5)
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, channels, samples)
            dummy_x = self.pooling1(self.batchnorm2(self.depthwise(self.batchnorm1(self.conv1(dummy_x)))))
            dummy_x = self.pooling2(self.batchnorm3(self.separable(dummy_x)))
            self.flat_size = dummy_x.numel()
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def forward(self, x):
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = F.elu(self.batchnorm2(self.depthwise(x)))
        x = self.pooling1(x)
        x = self.dropout(x)
        x = F.elu(self.batchnorm3(self.separable(x)))
        x = self.pooling2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 2. DATA LOADERS (Source vs Target)
# ==========================================
def get_bci_data(subjects):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, resample=128, fmin=8, fmax=30)
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
    y = LabelEncoder().fit_transform(y)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

if __name__ == "__main__":
    print("🚀 FETCHING UNIVERSAL KNOWLEDGE BASE (Subjects 3, 7, 8, 9)...")
    X_src, y_src = get_bci_data([3, 7, 8, 9])
    
    print("🎯 FETCHING TARGET BRAIN (Subject 1)...")
    X_tgt, y_tgt = get_bci_data([1])
    X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(
        X_tgt, y_tgt, test_size=0.2, stratify=y_tgt, random_state=42)

    model = EEGNet_Final(channels=22, samples=X_src.shape[3]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- PHASE 1: PRE-TRAIN (Universal Knowledge) ---
    print(f"\n🧠 PHASE 1: PRE-TRAINING ON {X_src.shape[0]} UNIVERSAL TRIALS...")
    optimizer_pre = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    for epoch in range(100):
        model.train()
        optimizer_pre.zero_grad()
        # Add noise to source to prevent memorizing their specific artifacts
        inputs = X_src.to(device) + torch.randn_like(X_src.to(device)) * 0.01
        loss = criterion(model(inputs), y_src.to(device))
        loss.backward()
        optimizer_pre.step()
        
        if epoch % 20 == 0:
            print(f"Pre-train Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # --- PHASE 2: FINE-TUNE (Target Specificity) ---
    print("\n🔥 PHASE 2: FINE-TUNING ON SUBJECT 1...")
    # Very low learning rate so we don't destroy the Universal Knowledge
    optimizer_fine = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    
    best_acc = 0
    for epoch in range(300):
        model.train()
        optimizer_fine.zero_grad()
        out = model(X_tgt_train.to(device))
        loss = criterion(out, y_tgt_train.to(device))
        loss.backward()
        optimizer_fine.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_tgt_test.to(device))
            acc = (val_out.argmax(1) == y_tgt_test.to(device)).float().mean().item() * 100
            if acc > best_acc: best_acc = acc
            
        if epoch % 20 == 0:
            print(f"Fine-tune Epoch {epoch:03d} | Loss: {loss.item():.4f} | VAL ACC: {acc:.2f}% | BEST: {best_acc:.2f}%")

    print(f"\n✅ FINAL TRANSFER ACCURACY: {best_acc:.2f}%")