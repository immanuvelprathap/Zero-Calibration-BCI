import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import inv, sqrtm
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE ARCHITECTURE (Remains Inception)
# ==========================================
class InceptionEEGNet(nn.Module):
    def __init__(self, channels=64):
        super(InceptionEEGNet, self).__init__()
        self.t1 = nn.Conv2d(1, 8, (1, 32), padding=(0, 16), bias=False)
        self.t2 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.t3 = nn.Conv2d(1, 8, (1, 128), padding=(0, 64), bias=False)
        self.bn_in = nn.BatchNorm2d(24)
        self.spatial = nn.Conv2d(24, 48, (channels, 1), groups=24, bias=False)
        self.bn_out = nn.BatchNorm2d(48)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(48 * 60, 2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = torch.cat([self.t1(x), self.t2(x), self.t3(x)], dim=1)
        x = self.elu(self.bn_out(self.spatial(self.bn_in(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))

# ==========================================
# 2. THE SECRET SAUCE: EUCLIDEAN ALIGNMENT
# ==========================================
def euclidean_alignment(X):
    # X shape: (trials, channels, samples)
    R = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        R += np.dot(X[i], X[i].T)
    R /= X.shape[0] # Subject-wise average covariance
    
    # Calculate R^-1/2
    R_inv_sqrt = inv(sqrtm(R))
    
    # Transform trials: X_aligned = R^-1/2 * X
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = np.dot(R_inv_sqrt, X[i])
    return X_aligned

def load_and_process_data_EA(num_subjects=109):
    X_list, y_list = [], []
    print(f"🚀 AGGREGATING {num_subjects} SUBJECTS WITH EUCLIDEAN ALIGNMENT...")
    
    for sub in range(1, num_subjects + 1):
        try:
            runs = [4, 8, 12]
            fnames = mne.datasets.eegbci.load_data(sub, runs, update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames])
            mne.datasets.eegbci.standardize(raw)
            if raw.info['sfreq'] != 160: raw.resample(160)
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0, tmax=3.0, baseline=None, preload=True, verbose=False)
            data = epochs.get_data(copy=True)
            if data.shape[0] == 0: continue
            data = data[:, :, :481] if data.shape[2] >= 481 else None
            if data is None: continue
            
            # 🔥 APPLY EA BEFORE NORMALIZATION
            data_aligned = euclidean_alignment(data)
            
            X_list.append(data_aligned.astype('float32'))
            y_list.append((epochs.events[:, -1] - 2).astype('int64'))
            if sub % 10 == 0: print(f"✅ Aligned Subject {sub}...")
        except Exception: continue
    return X_list, y_list

# ==========================================
# 3. TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    X_all, y_all = load_and_process_data_EA(109)
    X_train = np.concatenate(X_all[:100], axis=0)
    y_train = np.concatenate(y_all[:100], axis=0)
    X_val_list, y_val_list = X_all[100:], y_all[100:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), 
                                          torch.from_numpy(y_train).to(device)), 
                              batch_size=32, shuffle=True)

    model = InceptionEEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n🔥 STARTING ALIGNED PHASE | Samples: {len(y_train)}")
    for epoch in range(100):
        model.train()
        t_loss = 0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval(); val_accs = []
        with torch.no_grad():
            for xt, yt in zip(X_val_list, y_val_list):
                out = model(torch.from_numpy(xt).to(device))
                acc = (torch.max(out, 1)[1] == torch.from_numpy(yt).to(device)).sum().item() / len(yt)
                val_accs.append(acc)
        
        avg_val = np.mean(val_accs) * 100
        print(f"Epoch {epoch+1:02d} | Loss: {t_loss/len(train_loader):.4f} | Avg Val Acc: {avg_val:.2f}% | LR: {optimizer.param_groups[0]['lr']}")
        scheduler.step(avg_val)
        if avg_val >= 95.0:
            print("🎯 TARGET REACHED."); torch.save(model.state_dict(), "EA_inception_95.pth"); break