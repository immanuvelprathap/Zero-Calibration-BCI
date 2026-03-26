import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reset to 0.60 just to prove the model is actually ALIVE
CONF_THRESHOLD = 0.60  

print(f"🚀 STEP 11: THE FOUNDATION REBUILD")

# ==========================================
# 1. FIXED ARCHITECTURE (Static Initialization)
# ==========================================
class FixedEEGNet(nn.Module):
    def __init__(self, channels=64, samples=801):
        super(FixedEEGNet, self).__init__()
        # BIO: Capture frequency components
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # BIO: Capture spatial motor imagery patterns
        self.conv2 = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        
        # MATH: Massive pooling to ensure we don't look at tiny noise jitters
        self.pool = nn.AvgPool2d((1, 8)) 
        self.dropout = nn.Dropout(0.5)
        
        # MATH: Hard-coded linear layer to ensure optimizer sees it
        # Based on (samples // 8) -> 801 // 8 = 100
        self.fc = nn.Linear(32 * 100, 2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.elu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1) # Use reshape for safety
        return self.fc(x)

# ==========================================
# 2. ROBUST DATA LOADING
# ==========================================
all_X, all_y = [], []

for sub in range(1, 21):
    print(f"Sub {sub:02d}...", end="\r")
    raw_fnames = mne.datasets.eegbci.load_data(sub, [4, 8, 12])
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    raw.filter(7.0, 30.0, fir_design='firwin', verbose=False)
    
    events, _ = mne.events_from_annotations(raw)
    # Using a 4-second window (801 samples)
    epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0, tmax=4.0, 
                        baseline=None, preload=True, verbose=False)
    
    X = epochs.get_data(copy=True)
    # MATH: Unit Variance Scaling
    X = (X - np.mean(X)) / (np.std(X) + 1e-6)
    
    all_X.append(X.astype('float32'))
    all_y.append((epochs.events[:, -1] - 2).astype('int64'))

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
for test_idx in range(20):
    X_train = np.concatenate([all_X[i] for i in range(20) if i != test_idx])
    y_train = np.concatenate([all_y[i] for i in range(20) if i != test_idx])
    
    # Re-initialize for every subject to prevent leakage
    model = FixedEEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), 
                                          torch.from_numpy(y_train).to(device)), 
                              batch_size=16, shuffle=True)
    
    model.train()
    for epoch in range(30):
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_x = torch.from_numpy(all_X[test_idx]).to(device)
        test_y = torch.from_numpy(all_y[test_idx]).to(device)
        
        logits = model(test_x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        
        mask = conf >= CONF_THRESHOLD
        exec_rate = (mask.sum().item() / len(test_y)) * 100
        acc = (pred[mask] == test_y[mask]).sum().item() / mask.sum().item() * 100 if mask.sum() > 0 else 0.0
        
    print(f"Sub {test_idx+1:02d} | Exec: {exec_rate:04.1f}% | Acc: {acc:04.1f}%")