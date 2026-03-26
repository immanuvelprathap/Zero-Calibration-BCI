import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm
from scipy.linalg import fractional_matrix_power
import warnings

# --- CLINICAL RIGOR SETTINGS ---
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REJECTION THRESHOLDS FOR 95% TARGET
CONFIDENCE_THRESHOLD = 0.85     
ADAPT_LR = 0.0001               
ADAPT_EPOCHS = 20               

print(f"🔬 STEP 9 (FIXED): RIEMANNIAN PROCRUSTES + DYNAMIC EEGNET")

# ==========================================
# 1. NEURAL NETWORK (Robust Dynamic EEGNet)
# ==========================================
class FilterBankEEGNet(nn.Module):
    def __init__(self, n_classes=2, channels=64):
        super(FilterBankEEGNet, self).__init__()
        # BIO: Temporal filters focus on Mu/Beta ERD signatures.
        self.temp_conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temp_conv2 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # BIO: Spatial filters localize the motor imagery source.
        # MATH: Depthwise convolution learns subject-specific spatial patterns.
        self.depth_conv = nn.Conv2d(32, 64, (channels, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(0.35)
        
        self.fc = None # Calculated dynamically below
        self.n_classes = n_classes

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = torch.cat((self.temp_conv1(x), self.temp_conv2(x)), dim=1)
        x = self.bn1(x)
        x = self.dropout(self.avg_pool1(self.bn2(self.elu(self.depth_conv(x)))))
        x = x.view(x.size(0), -1)
        
        # MATH: Dynamic Linear Layer Initialization
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.n_classes).to(x.device)
            
        return self.fc(x)

# ==========================================
# 2. DATA EXTRACTION & MANIFOLD ALIGNMENT
# ==========================================
all_X, all_y = [], []
cov_estimator = Covariances(estimator='oas')

print("Phase 1: Riemannian Manifold Mapping (Procrustes Stage 1)...")

for subject in range(1, 21):
    runs = [4, 8, 12]
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    mne.datasets.eegbci.standardize(raw)
    raw.filter(3.0, 35.0, fir_design='firwin', verbose=False)
    events, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, {'T1': event_dict['T1'], 'T2': event_dict['T2']}, 
                        tmin=-1.0, tmax=4.0, baseline=(None, 0), preload=True, verbose=False)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - event_dict['T1']
    
    # MATH: Centering on the SPD Manifold
    # BIO: This ensures the 'Rest' state of every subject is mathematically identical.
    covs = cov_estimator.transform(X)
    C_ref = mean_covariance(covs, metric='riemann')
    C_inv_sq = invsqrtm(C_ref)
    
    X_aligned = np.array([C_inv_sq @ trial for trial in X])
    all_X.append(X_aligned)
    all_y.append(y)
    print(f"Subject {subject:02d} Aligned.", end="\r")

# ==========================================
# 3. ADAPTIVE LOSO 
# ==========================================
print("\nPhase 2: Adaptive Decoding...")
subject_metrics = []

for test_idx in range(20):
    X_test_t = torch.tensor(all_X[test_idx], dtype=torch.float32).to(device)
    y_test_t = torch.tensor(all_y[test_idx], dtype=torch.long).to(device)
    X_train = np.concatenate([all_X[i] for i in range(20) if i != test_idx])
    y_train = np.concatenate([all_y[i] for i in range(20) if i != test_idx])
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), 
                                          torch.tensor(y_train, dtype=torch.long).to(device)), 
                              batch_size=32, shuffle=True)
    
    model = FilterBankEEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    for _ in range(30):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    # --- REFINED TTA (UDA) ---
    model.eval()
    with torch.no_grad():
        out = model(X_test_t)
        probs = torch.softmax(out, dim=1)
        conf, pseudo = torch.max(probs, 1)

    mask = conf >= 0.95 
    if mask.sum() > 3:
        model.train()
        opt_tta = optim.Adam(model.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_EPOCHS):
            opt_tta.zero_grad()
            loss = criterion(model(X_test_t[mask]), pseudo[mask])
            loss.backward()
            opt_tta.step()
        status = f"RPA+TTA ({mask.sum().item()} trials)"
    else:
        status = "Baseline (Low SNR)"

    model.eval()
    with torch.no_grad():
        final_out = model(X_test_t)
        final_probs = torch.softmax(final_out, dim=1)
        f_conf, f_pred = torch.max(final_probs, 1)
        
        c_mask = f_conf >= CONFIDENCE_THRESHOLD
        exec_rate = (c_mask.sum().item() / len(y_test_t)) * 100
        acc = (f_pred[c_mask] == y_test_t[c_mask]).sum().item() / c_mask.sum().item() * 100 if c_mask.sum() > 0 else 0.0
        
    print(f"Sub {test_idx+1:02d} | {status} | Exec: {exec_rate:04.1f}% | Func Acc: {acc:04.1f}%")
    subject_metrics.append((acc, exec_rate))

print("\n" + "="*55)
print(f"STEP 9 SUMMARY: RIEMANNIAN + ADAPTIVE")
print(f"FUNCTIONAL ACCURACY : {np.mean([m[0] for m in subject_metrics]):.2f}%")
print(f"EXECUTION RATE      : {np.mean([m[1] for m in subject_metrics]):.2f}%")
print("="*55)