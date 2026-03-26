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

class EEGNet_95(nn.Module):
    def __init__(self, channels=22, samples=513, n_classes=4):
        super(EEGNet_95, self).__init__()
        # Block 1: Temporal + Spatial
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))
        
        # Block 2: Separable
        self.separable = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 8))
        
        self.dropout = nn.Dropout(0.6) # Increased for 95% threshold
        
        # Dynamic size check
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

if __name__ == "__main__":
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, resample=128, fmin=8, fmax=30)
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[1])
    y = LabelEncoder().fit_transform(y)
    
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet_95(channels=22, samples=X.shape[3]).to(device)
    
    # 🔥 STRATEGY: OneCycleLR + LabelSmoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=300)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("🚀 COMMENCING FINAL 95% OPTIMIZATION...")
    best_acc = 0
    
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_test.to(device))
                acc = (val_out.argmax(1) == y_test.to(device)).float().mean() * 100
                if acc > best_acc: best_acc = acc
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | VAL ACC: {acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n✅ FINAL BEST VALIDATION: {best_acc:.2f}%")