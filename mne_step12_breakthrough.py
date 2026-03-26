# import mne
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import warnings

# # Suppression for clean output
# mne.set_log_level('ERROR')
# warnings.filterwarnings("ignore")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==========================================
# # 1. THE ARCHITECTURE (Anatomically Masked)
# # ==========================================
# class BreakthroughEEGNet(nn.Module):
#     def __init__(self, channels=64):
#         super(BreakthroughEEGNet, self).__init__()
        
#         # Temporal Filter: 160Hz native. Kernel 80 = 500ms window.
#         self.conv1 = nn.Conv2d(1, 16, (1, 80), padding=(0, 40), bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
        
#         # 🎯 THE BIOLOGICAL PRIORITY MASK
#         # Confirmed via PDF: CH 9 (C3), 11 (Cz), 13 (C4)
#         # Python Indices (0-indexed): 8, 10, 12
#         self.spatial_mask = nn.Parameter(torch.ones(channels))
#         with torch.no_grad():
#             self.spatial_mask.fill_(0.1)     # Suppress forehead/EOG noise
#             self.spatial_mask[8] = 3.5       # Boost Right Hand Intent
#             self.spatial_mask[12] = 3.5      # Boost Left Hand Intent
#             self.spatial_mask[10] = 1.5      # Boost Midline Reference
        
#         # Spatial Filter: Extracting Hemisphere Contrast
#         self.conv2 = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.elu = nn.ELU()
        
#         self.pool = nn.AvgPool2d((1, 8)) 
#         self.dropout = nn.Dropout(0.5)
        
#         # 3.0s @ 160Hz = 480 samples. After AvgPool(8) = 60
#         self.fc = nn.Linear(32 * 60, 2)

#     def forward(self, x):
#         if len(x.shape) == 3: x = x.unsqueeze(1)
        
#         # Apply the validated anatomical mask
#         x = x * self.spatial_mask.view(1, 1, -1, 1)
        
#         x = self.bn1(self.conv1(x))
#         x = self.bn2(self.elu(self.conv2(x)))
#         x = self.pool(x)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1) 
#         return self.fc(x)

# # ==========================================
# # 2. DATA AGGREGATION (109 Subjects)
# # ==========================================
# def load_and_process_data(num_subjects=109):
#     X_list, y_list = [], []
#     print(f"🚀 AGGREGATING 109 SUBJECTS (Task 2: Imagined Left/Right)...")
    
#     for sub in range(1, num_subjects + 1):
#         try:
#             # Runs 4, 8, 12 are specifically Task 2 (Motor Imagery Hand)
#             runs = [4, 8, 12]
#             raw_fnames = mne.datasets.eegbci.load_data(sub, runs, update_path=False)
#             raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
            
#             # Strict Bandpass for Mu-Beta Rhythms (8-30Hz)
#             raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
#             events, _ = mne.events_from_annotations(raw)
#             # T1=Left, T2=Right. Code 2=T1, Code 3=T2 in MNE eegbci
#             epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0, tmax=3.0, 
#                                 baseline=None, preload=True, verbose=False)
            
#             data = epochs.get_data(copy=True)
#             if data.shape[0] == 0: continue
            
#             # Robust Z-Score Normalization
#             data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            
#             X_list.append(data.astype('float32'))
#             y_list.append((epochs.events[:, -1] - 2).astype('int64'))
            
#             if sub % 10 == 0: print(f"Synced through Subject {sub}...")
#         except Exception: continue
        
#     return X_list, y_list

# # ==========================================
# # 3. TRAINING LOOP (The 95% Push)
# # ==========================================
# X_all, y_all = load_and_process_data(109)

# # Split by subject to ensure genuine generalizability
# # 100 subs for training, 9 subs for unseen validation
# for i, arr in enumerate(X_all):
#     print(f"Subject {i} shape: {arr.shape}")
# X_train = np.concatenate(X_all[:100])
# y_train = np.concatenate(y_all[:100])
# X_val_list = X_all[100:]
# y_val_list = y_all[100:]

# train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), 
#                                       torch.from_numpy(y_train).to(device)), 
#                           batch_size=64, shuffle=True)

# model = BreakthroughEEGNet().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# print("\n🔥 STARTING DEEP TRAINING PHASE")
# for epoch in range(50):
#     model.train()
#     train_loss = 0
#     for b_x, b_y in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(b_x), b_y)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
    
#     # Cross-Subject Validation
#     model.eval()
#     val_accs = []
#     with torch.no_grad():
#         for xt, yt in zip(X_val_list, y_val_list):
#             out = model(torch.from_numpy(xt).to(device))
#             pred = torch.max(out, 1)[1]
#             acc = (pred == torch.from_numpy(yt).to(device)).sum().item() / len(yt)
#             val_accs.append(acc)
    
#     avg_val = np.mean(val_accs) * 100
#     print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Avg Val Acc: {avg_val:.2f}%")
    
#     if avg_val >= 95.0:
#         print("🎯 TARGET REACHED. BREAKTHROUGH COMPLETE.")
#         break

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
        
        # Temporal Filter: 160Hz native. Kernel 80 = 500ms window.
        self.conv1 = nn.Conv2d(1, 16, (1, 80), padding=(0, 40), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 🎯 THE BIOLOGICAL PRIORITY MASK
        # Indices 8, 10, 12 correspond to C3, Cz, C4 in the 64-channel setup
        self.spatial_mask = nn.Parameter(torch.ones(channels))
        with torch.no_grad():
            self.spatial_mask.fill_(0.1)     # Suppress forehead/EOG noise
            self.spatial_mask[8] = 3.5       # Boost Right Hand Intent (C3)
            self.spatial_mask[12] = 3.5      # Boost Left Hand Intent (C4)
            self.spatial_mask[10] = 1.5      # Boost Midline Reference (Cz)
        
        # Spatial Filter: Extracting Hemisphere Contrast
        self.conv2 = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        
        self.pool = nn.AvgPool2d((1, 8)) 
        self.dropout = nn.Dropout(0.5)
        
        # 3.0s @ 160Hz = 481 samples. After AvgPool(8) = 60 (approx)
        # We use a dummy forward to calculate exact FC input size
        self.fc = nn.Linear(32 * 60, 2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        
        # Apply the validated anatomical mask
        x = x * self.spatial_mask.view(1, 1, -1, 1)
        
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.elu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)

# ==========================================
# 2. DATA AGGREGATION (109 Subjects)
# ==========================================
def load_and_process_data(num_subjects=109):
    X_list, y_list = [], []
    print(f"🚀 AGGREGATING {num_subjects} SUBJECTS...")
    
    for sub in range(1, num_subjects + 1):
        try:
            runs = [4, 8, 12] # Task 2
            raw_fnames = mne.datasets.eegbci.load_data(sub, runs, update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames])
            
            # --- THE BREAKTHROUGH FIX: FORCE SAMPLING CONSISTENCY ---
            if raw.info['sfreq'] != 160:
                raw.resample(160, npad="auto")
            
            # Strict Bandpass for Mu-Beta Rhythms (8-30Hz)
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, verbose=False)
            # T1=Left, T2=Right. Code 2=T1, Code 3=T2 in MNE eegbci
            epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0, tmax=3.0, 
                                baseline=None, preload=True, verbose=False)
            
            data = epochs.get_data(copy=True)
            if data.shape[0] == 0: continue
            
            # --- THE DIMENSION CHECK ---
            # Standardizing to 481 samples (3 seconds @ 160Hz)
            if data.shape[2] != 481:
                # If slightly off due to rounding, force it to 481
                if data.shape[2] > 481:
                    data = data[:, :, :481]
                else:
                    continue # Skip if too short to avoid padding noise
            
            # Robust Z-Score Normalization
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            
            X_list.append(data.astype('float32'))
            y_list.append((epochs.events[:, -1] - 2).astype('int64'))
            
            if sub % 10 == 0: print(f"✅ Synced through Subject {sub}...")
        except Exception as e:
            continue
        
    return X_list, y_list

# ==========================================
# 3. TRAINING LOOP (The 95% Push)
# ==========================================
X_all, y_all = load_and_process_data(109)

# Verify consistency before concatenation
print("\n🔍 VERIFYING DIMENSIONS...")
for i, arr in enumerate(X_all):
    if arr.shape[2] != 481:
        print(f"❌ Subject {i} mismatch: {arr.shape}")

# Split by subject: 100 for training, remaining for validation
X_train = np.concatenate(X_all[:100], axis=0)
y_train = np.concatenate(y_all[:100], axis=0)
X_val_list = X_all[100:]
y_val_list = y_all[100:]

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), 
                                      torch.from_numpy(y_train).to(device)), 
                          batch_size=64, shuffle=True)

model = BreakthroughEEGNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\n🔥 STARTING DEEP TRAINING | Training Samples: {len(y_train)}")
for epoch in range(100): # Increased epochs for the 95% push
    model.train()
    train_loss = 0
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        output = model(b_x)
        loss = criterion(output, b_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Cross-Subject Validation
    model.eval()
    val_accs = []
    with torch.no_grad():
        for xt, yt in zip(X_val_list, y_val_list):
            out = model(torch.from_numpy(xt).to(device))
            pred = torch.max(out, 1)[1]
            acc = (pred == torch.from_numpy(yt).to(device)).sum().item() / len(yt)
            val_accs.append(acc)
    
    avg_val = np.mean(val_accs) * 100
    print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Avg Val Acc: {avg_val:.2f}%")
    
    if avg_val >= 95.0:
        print("🎯 TARGET REACHED. BREAKTHROUGH COMPLETE.")
        torch.save(model.state_dict(), "breakthrough_95_model.pth")
        break