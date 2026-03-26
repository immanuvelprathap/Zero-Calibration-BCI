import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

# Suppression for clean output
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE ARCHITECTURE (Anatomically Masked)
# ==========================================
class BreakthroughEEGNet(nn.Module):
    def __init__(self, channels=64):
        super(BreakthroughEEGNet, self).__init__()
        # Temporal: Narrower kernel (64) to capture sharper Mu-rhythm transients
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 🎯 THE BIOLOGICAL PRIORITY MASK
        self.spatial_mask = nn.Parameter(torch.ones(channels))
        with torch.no_grad():
            self.spatial_mask.fill_(0.05)    # Punish noise/EOG
            self.spatial_mask[[8, 10, 12]] = 5.0 # Aggressive boost for C3, Cz, C4
        
        self.conv2 = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        
        self.pool = nn.AvgPool2d((1, 8)) 
        self.dropout = nn.Dropout(0.6) 
        
        self.fc = nn.Linear(32 * 60, 2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = x * self.spatial_mask.view(1, 1, -1, 1)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.elu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)

# ==========================================
# 2. DATA AGGREGATION
# ==========================================
def load_and_process_data(num_subjects=109):
    X_list, y_list = [], []
    print(f"🚀 AGGREGATING {num_subjects} SUBJECTS...")
    
    for sub in range(1, num_subjects + 1):
        try:
            runs = [4, 8, 12] 
            raw_fnames = mne.datasets.eegbci.load_data(sub, runs, update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames])
            
            if raw.info['sfreq'] != 160:
                raw.resample(160, npad="auto")
            
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0, tmax=3.0, 
                                baseline=None, preload=True, verbose=False)
            
            data = epochs.get_data(copy=True)
            if data.shape[0] == 0: continue
            
            if data.shape[2] != 481:
                if data.shape[2] > 481:
                    data = data[:, :, :481]
                else:
                    continue
            
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            
            X_list.append(data.astype('float32'))
            y_list.append((epochs.events[:, -1] - 2).astype('int64'))
            
            if sub % 10 == 0: print(f"✅ Synced through Subject {sub}...")
        except Exception:
            continue
        
    return X_list, y_list

# ==========================================
# 3. REFINED TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    X_all, y_all = load_and_process_data(109)

    X_train = np.concatenate(X_all[:100], axis=0)
    y_train = np.concatenate(y_all[:100], axis=0)
    X_val_list = X_all[100:]
    y_val_list = y_all[100:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), 
                                          torch.from_numpy(y_train).to(device)), 
                              batch_size=32, shuffle=True)

    model = BreakthroughEEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Removed 'verbose' argument to fix TypeError
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n🔥 STARTING REFINED TRAINING PHASE | Samples: {len(y_train)}")
    last_lr = 0.001
    for epoch in range(100):
        model.train()
        train_loss = 0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            output = model(b_x)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_accs = []
        with torch.no_grad():
            for xt, yt in zip(X_val_list, y_val_list):
                out = model(torch.from_numpy(xt).to(device))
                pred = torch.max(out, 1)[1]
                acc = (pred == torch.from_numpy(yt).to(device)).sum().item() / len(yt)
                val_accs.append(acc)
        
        avg_val = np.mean(val_accs) * 100
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Avg Val Acc: {avg_val:.2f}% | LR: {current_lr}")
        
        if current_lr < last_lr:
            print(f"📉 Learning rate reduced to {current_lr}")
            last_lr = current_lr

        scheduler.step(avg_val)

        if avg_val >= 95.0:
            print("🎯 TARGET REACHED. BREAKTHROUGH COMPLETE.")
            torch.save(model.state_dict(), "breakthrough_95_model.pth")
            break