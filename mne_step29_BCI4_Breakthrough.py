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

# ==========================================
# 1. ROBUST EEGNET ARCHITECTURE
# ==========================================
class EEGNet(nn.Module):
    def __init__(self, channels=22, samples=513, n_classes=4):
        super(EEGNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pooling1 = nn.AvgPool2d((1, 4))
        
        # Block 2
        self.separable = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.pooling2 = nn.AvgPool2d((1, 8))
        
        # Dynamic shape inference for the linear layer
        self.n_outputs = self._get_flatten_size(channels, samples)
        self.classifier = nn.Linear(self.n_outputs, n_classes)

    def _get_flatten_size(self, channels, samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, channels, samples)
            x = self.pooling1(self.batchnorm2(self.depthwise(self.batchnorm1(self.conv1(x)))))
            x = self.pooling2(self.batchnorm3(self.separable(x)))
            return x.numel()

    def forward(self, x):
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = F.elu(self.batchnorm2(self.depthwise(x)))
        x = self.pooling1(x)
        x = F.elu(self.batchnorm3(self.separable(x)))
        x = self.pooling2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 2. MOABB EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 FETCHING BCI COMPETITION IV 2A (SUBJECT 1)...")
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, resample=128, fmin=8, fmax=30)
    
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[1])
    y = LabelEncoder().fit_transform(y)
    
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(channels=22, samples=X.shape[3]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"🔥 NEURONS READY | SAMPLES: {X.shape[3]} | FLAT SIZE: {model.n_outputs}")
    
    for epoch in range(201):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_test.to(device))
                acc = (val_out.argmax(1) == y_test.to(device)).float().mean()
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | VAL ACC: {acc*100:.2f}%")

    print(f"\n✅ FINAL VALIDATION: {acc*100:.2f}%")