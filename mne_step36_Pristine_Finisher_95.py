import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyriemann.estimation import Covariances
from scipy.linalg import fractional_matrix_power
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def temporal_shift(x, max_shift=10):
    batch_size = x.size(0)
    shifted_x = torch.zeros_like(x)
    shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,))
    for i in range(batch_size):
        shift = shifts[i].item()
        if shift > 0:
            shifted_x[i, :, :, shift:] = x[i, :, :, :-shift]
            shifted_x[i, :, :, :shift] = 0
        elif shift < 0:
            shifted_x[i, :, :, :shift] = x[i, :, :, -shift:]
            shifted_x[i, :, :, shift:] = 0
        else:
            shifted_x[i] = x[i]
    return shifted_x

def mixup_data(x, y, alpha=0.4):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_aligned_bci_data(subjects):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, resample=128, fmin=8, fmax=30)
    X_all, y_all = [], []
    for sub in subjects:
        X_sub, y_sub, _ = paradigm.get_data(dataset=dataset, subjects=[sub])
        covs = Covariances(estimator='oas').fit_transform(X_sub)
        R_mean = np.mean(covs, axis=0)
        R_inv_sqrt = fractional_matrix_power(R_mean, -0.5)
        X_aligned = np.zeros_like(X_sub)
        for i in range(X_sub.shape[0]):
            X_aligned[i] = R_inv_sqrt @ X_sub[i]
        X_all.append(X_aligned)
        y_all.append(y_sub)
        
    y = LabelEncoder().fit_transform(np.concatenate(y_all, axis=0))
    X = torch.tensor(np.concatenate(X_all, axis=0), dtype=torch.float32).unsqueeze(1)
    return X, torch.tensor(y, dtype=torch.long)

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

if __name__ == "__main__":
    print("🚀 FETCHING ALL 8 BACKGROUND BRAINS (Maximal Universal Base)...")
    X_src, y_src = get_aligned_bci_data([2, 3, 4, 5, 6, 7, 8, 9])
    
    print("🎯 FETCHING TARGET BRAIN (Sub 1)...")
    X_tgt, y_tgt = get_aligned_bci_data([1])
    X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(
        X_tgt, y_tgt, test_size=0.2, stratify=y_tgt, random_state=42)

    model = EEGNet_Final(channels=22, samples=X_src.shape[3]).to(device)
    
    criterion_pre = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_fine = nn.CrossEntropyLoss()

    # --- PHASE 1: PRE-TRAIN (ALL SUBJECTS) ---
    optimizer_pre = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler_pre = optim.lr_scheduler.OneCycleLR(optimizer_pre, max_lr=0.005, steps_per_epoch=1, epochs=200)
    
    print("\n🧠 PHASE 1: PRE-TRAINING (200 Epochs)...")
    for epoch in range(200):
        model.train()
        optimizer_pre.zero_grad()
        shifted_x = temporal_shift(X_src)
        inputs, targets_a, targets_b, lam = mixup_data(shifted_x.to(device), y_src.to(device))
        loss = mixup_criterion(criterion_pre, model(inputs), targets_a, targets_b, lam)
        loss.backward()
        optimizer_pre.step()
        scheduler_pre.step()
        
        # Fixed: You will now see progress here
        if epoch % 50 == 0:
            print(f"Pre-train Epoch {epoch:03d} | Mixup Loss: {loss.item():.4f}")

    # --- PHASE 2: LIMIT BREAK + PRISTINE FINISHER ---
    fine_epochs = 300
    swa_start = 220
    
    optimizer_fine = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01) 
    scheduler_fine = optim.lr_scheduler.OneCycleLR(optimizer_fine, max_lr=0.002, steps_per_epoch=1, epochs=fine_epochs)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer_fine, swa_lr=0.0001) 

    history = {'loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0

    print(f"\n🔥 PHASE 2: FINE-TUNING (Pristine Cooldown at Epoch {swa_start})...")
    
    for epoch in range(fine_epochs):
        model.train()
        optimizer_fine.zero_grad()
        history['lr'].append(optimizer_fine.param_groups[0]['lr'])
        
        # ---------------------------------------------------------
        # THE PRISTINE FINISHER LOGIC
        # ---------------------------------------------------------
        if epoch < swa_start:
            # Phase 2A: Heavy Augmentation (Exploration)
            shifted_x = temporal_shift(X_tgt_train)
            inputs, targets_a, targets_b, lam = mixup_data(shifted_x.to(device), y_tgt_train.to(device))
            inputs = inputs + torch.randn_like(inputs) * 0.01
            out = model(inputs)
            loss = mixup_criterion(criterion_fine, out, targets_a, targets_b, lam) 
        else:
            # Phase 2B: Pure Data SWA (Exploitation / Cooldown)
            inputs = X_tgt_train.to(device)
            targets = y_tgt_train.to(device)
            out = model(inputs)
            loss = criterion_fine(out, targets) # Standard confident loss
        # ---------------------------------------------------------

        loss.backward()
        optimizer_fine.step()
        
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler_fine.step()

        history['loss'].append(loss.item())

        model.eval()
        with torch.no_grad():
            val_out = model(X_tgt_test.to(device))
            acc = (val_out.argmax(1) == y_tgt_test.to(device)).float().mean().item() * 100
            history['val_acc'].append(acc)
            if acc > best_acc: best_acc = acc
            
        if epoch % 30 == 0 or epoch == swa_start:
            if epoch == swa_start:
                print(f"\n🛑 EPOCH {swa_start}: INITIATING PRISTINE DATA COOLDOWN...")
            print(f"Fine-tune Epoch {epoch:03d} | Loss: {loss.item():.4f} | VAL ACC: {acc:.2f}% | BEST: {best_acc:.2f}%")

    # --- PHASE 3: SWA RESOLUTION ---
    print("\n🎯 CALCULATING FINAL 95% SWA CENTER...")
    train_dataset = TensorDataset(X_tgt_train.to(device), y_tgt_train.to(device))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    
    swa_model.eval()
    with torch.no_grad():
        final_out = swa_model(X_tgt_test.to(device))
        final_swa_acc = (final_out.argmax(1) == y_tgt_test.to(device)).float().mean().item() * 100

    print(f"✅ SWA AVERAGED ACCURACY: {final_swa_acc:.2f}%")
    print(f"✅ BEST SINGLE EPOCH: {best_acc:.2f}%")

    # ==========================================
    # 5. THE AUDIT PLOT
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Fine-Tuning Epoch')
    ax1.set_ylabel('Loss / Accuracy', color='black')
    ax1.plot(history['loss'], color='red', label='Training Loss', alpha=0.5)
    ax1.plot(history['val_acc'], color='green', label='Validation Acc', linewidth=2)
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='blue')
    ax2.plot(history['lr'], color='blue', linestyle=':', label='LR Schedule', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.axvline(x=swa_start, color='purple', linestyle='--', linewidth=2, label='SWA / Cooldown Start')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f"BCI IV 2a Pristine Finisher | Peak Single: {best_acc:.2f}% | Final SWA: {final_swa_acc:.2f}%")
    plt.tight_layout()
    plt.savefig("step36_pristine_finisher.png")
    print("\n✅ Audit Plot saved as 'step36_pristine_finisher.png'")
    plt.show()