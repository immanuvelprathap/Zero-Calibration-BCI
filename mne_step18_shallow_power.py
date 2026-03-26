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
# 1. THE SHALLOW-POWER ARCHITECTURE (Fixed Dim)
# ==========================================
class ShallowPowerNet(nn.Module):
    def __init__(self, channels=64):
        super(ShallowPowerNet, self).__init__()
        # Temporal Conv: Captures the phase (Mu/Beta)
        self.temp_conv = nn.Conv2d(1, 40, (1, 25), bias=False)
        # Spatial Conv: The learned spatial filter
        self.spat_conv = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        # AvgPool: Smooths the power estimate
        self.avg_pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        
        # 🔥 DIMENSION FIX: 40 filters * 26 temporal windows = 1040
        self.fc = nn.Linear(40 * 26, 2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        
        # Linear transformations
        x = self.temp_conv(x)
        x = self.spat_conv(x)
        x = self.bn(x)
        
        # 🔥 BIOLOGICAL TRANSFORMATION: Square -> Pool -> Log
        # This converts the signal to Log-Band Power (ERD/ERS)
        x = x ** 2 
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        
        x = self.dropout(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)

# ==========================================
# 2. EUCLIDEAN ALIGNMENT
# ==========================================
def euclidean_alignment(X):
    R = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        R += np.dot(X[i], X[i].T)
    R /= X.shape[0]
    R_inv_sqrt = inv(sqrtm(R))
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = np.dot(R_inv_sqrt, X[i])
    return X_aligned

def load_and_process_data(num_subjects=109):
    X_list, y_list = [], []
    print(f"🚀 AGGREGATING {num_subjects} SUBJECTS FOR SHALLOW-POWER...")
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
            data = data[:, :, :481]
            
            # Align subject space
            data = euclidean_alignment(data)
            
            X_list.append(data.astype('float32'))
            y_list.append((epochs.events[:, -1] - 2).astype('int64'))
            if sub % 10 == 0: print(f"✅ Sync Subject {sub}...")
        except Exception: continue
    return X_list, y_list

# ==========================================
# 3. TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    X_all, y_all = load_and_process_data(109)
    X_train = np.concatenate(X_all[:100], axis=0)
    y_train = np.concatenate(y_all[:100], axis=0)
    X_val_list, y_val_list = X_all[100:], y_all[100:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), 
                                          torch.from_numpy(y_train).to(device)), 
                              batch_size=64, shuffle=True)

    model = ShallowPowerNet().to(device)
    # Stiffer weight decay (1e-2) to prevent over-specialization
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔥 STARTING POWER-EXTRACTION PHASE | Samples: {len(y_train)}")
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
            print("🎯 TARGET REACHED."); break