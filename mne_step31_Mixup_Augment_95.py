import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. MIXUP AUGMENTATION LOGIC
# ==========================================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 2. THE MODEL (EEGNet with 95% Tuning)
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
        self.dropout = nn.Dropout(0.6)
        
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
# 3. TRAINING LOOP WITH AUDIT TRACKING
# ==========================================
if __name__ == "__main__":
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, resample=128, fmin=8, fmax=30)
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[1])
    y = LabelEncoder().fit_transform(y)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    total_epochs = 401
    model = EEGNet_Final(channels=22, samples=X.shape[3]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=1, epochs=total_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Audit Arrays
    history = {'loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0

    print(f"🔥 STARTING AUGMENTED AUDIT | EPOCHS: {total_epochs} | DEVICE: {device}")
    
    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Capture current LR for audit
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Apply Mixup + Noise
        inputs, targets_a, targets_b, lam = mixup_data(X_train.to(device), y_train.to(device))
        inputs = inputs + torch.randn_like(inputs) * 0.01
        
        out = model(inputs)
        loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        history['loss'].append(loss.item())

        # Validation Audit
        model.eval()
        with torch.no_grad():
            val_out = model(X_test.to(device))
            acc = (val_out.argmax(1) == y_test.to(device)).float().mean().item() * 100
            history['val_acc'].append(acc)
            if acc > best_acc: best_acc = acc

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | VAL ACC: {acc:.2f}% | BEST: {best_acc:.2f}%")

    # ==========================================
    # 4. THE BREAKTHROUGH VISUALIZER
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mixup Loss / Accuracy', color='black')
    ax1.plot(history['loss'], color='red', label='Augmented Loss', alpha=0.5)
    ax1.plot(history['val_acc'], color='green', label='Validation Acc', linewidth=2)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate (OneCycle)', color='blue')
    ax2.plot(history['lr'], color='blue', linestyle=':', label='LR Schedule', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title(f"BCI IV 2a - Mixup Audit | Peak Accuracy: {best_acc:.2f}%")
    plt.tight_layout()
    plt.savefig("step31_mixup_audit.png")
    print("\n✅ Audit Plot saved as 'step31_mixup_audit.png'")
    plt.show()