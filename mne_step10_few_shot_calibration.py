import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm
import warnings

# --- CLINICAL TARGET: 95% ACCURACY ---
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calibration Hyperparameters
CALIB_TRIALS = 10      # BIO: First 10 trials used for "Warm-up" (Supervised)
CONF_THRESHOLD = 0.85  # MATH: Only execute if Softmax probability > 0.85
CALIB_LR = 0.0005      # MATH: Learning rate for subject-specific nudge
CALIB_EPOCHS = 20      # MATH: Iterations to align filters to the new brain

print(f"🔬 STEP 10: FEW-SHOT SUPERVISED CALIBRATION")
print(f"Goal: Breaking the 90% Barrier | Device: {device}")

# ==========================================
# 1. DYNAMIC NEURAL NETWORK (Filter Bank EEGNet)
# ==========================================
class AdaptiveEEGNet(nn.Module):
    def __init__(self, n_classes=2, channels=64):
        super(AdaptiveEEGNet, self).__init__()
        # BIO: Temporal filters for Mu (8-12Hz) and Beta (13-30Hz) ERD.
        self.temp_conv = nn.Conv2d(1, 32, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # BIO: Spatial filters to localize motor imagery on the cortex.
        # MATH: Depthwise convolution reduces parameters to prevent overfitting.
        self.depth_conv = nn.Conv2d(32, 64, (channels, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(0.35)
        
        self.fc = None # Initialized dynamically on first forward pass
        self.n_classes = n_classes

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = self.bn1(self.temp_conv(x))
        x = self.dropout(self.pool(self.bn2(self.elu(self.depth_conv(x)))))
        x = x.view(x.size(0), -1)
        
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.n_classes).to(x.device)
        return self.fc(x)

# ==========================================
# 2. DATA LOADING & RIEMANNIAN ALIGNMENT
# ==========================================
all_X, all_y = [], []
cov_est = Covariances(estimator='oas')

for sub in range(1, 21):
    print(f"Riemannian Centering Sub {sub:02d}...", end="\r")
    raw_fnames = mne.datasets.eegbci.load_data(sub, [4, 8, 12])
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    mne.datasets.eegbci.standardize(raw)
    raw.filter(7.0, 30.0, fir_design='firwin', verbose=False) # Tightened to Mu/Beta
    
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=-1.0, tmax=4.0, 
                        baseline=(None, 0), preload=True, verbose=False)
    
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - 2
    
    # MATH: Riemannian Centering (Stage 1 Procrustes)
    # BIO: Aligns the "Resting State" of the brain across all subjects.
    C_ref = mean_covariance(cov_est.transform(X), metric='riemann')
    C_inv_sq = invsqrtm(C_ref)
    all_X.append(np.array([C_inv_sq @ trial for trial in X]))
    all_y.append(y)

# ==========================================
# 3. FEW-SHOT LOSO EVALUATION
# ==========================================
print("\nPhase 2: Supervised Calibration & Deployment...")
subject_metrics = []

for test_idx in range(20):
    # Split into Train (19 subjects) and Test (1 subject)
    X_train = np.concatenate([all_X[i] for i in range(20) if i != test_idx])
    y_train = np.concatenate([all_y[i] for i in range(20) if i != test_idx])
    
    # Split Test Subject into Calibration (10 trials) and Evaluation (Rest)
    X_sub = all_X[test_idx]
    y_sub = all_y[test_idx]
    
    X_calib_t = torch.tensor(X_sub[:CALIB_TRIALS], dtype=torch.float32).to(device)
    y_calib_t = torch.tensor(y_sub[:CALIB_TRIALS], dtype=torch.long).to(device)
    X_eval_t = torch.tensor(X_sub[CALIB_TRIALS:], dtype=torch.float32).to(device)
    y_eval_t = torch.tensor(y_sub[CALIB_TRIALS:], dtype=torch.long).to(device)
    
    # --- 1. GLOBAL PRE-TRAINING ---
    model = AdaptiveEEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), 
                                          torch.tensor(y_train, dtype=torch.long).to(device)), 
                              batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(30):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(); criterion(model(batch_X), batch_y).backward(); optimizer.step()

    # --- 2. SUPERVISED FEW-SHOT CALIBRATION ---
    # BIO: Anchoring the model to the subject's specific "Left" vs "Right" clusters.
    # MATH: Fine-tuning weights on 10 known samples to solve the Subject 04/05 shift.
    model.train()
    opt_calib = optim.Adam(model.parameters(), lr=CALIB_LR)
    for _ in range(CALIB_EPOCHS):
        opt_calib.zero_grad()
        loss = criterion(model(X_calib_t), y_calib_t)
        loss.backward(); opt_calib.step()

    # --- 3. CLINICAL DEPLOYMENT (Evaluation) ---
    model.eval()
    with torch.no_grad():
        out = model(X_eval_t)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        
        # Confidence Gating
        mask = conf >= CONF_THRESHOLD
        exec_rate = (mask.sum().item() / len(y_eval_t)) * 100
        acc = (pred[mask] == y_eval_t[mask]).sum().item() / mask.sum().item() * 100 if mask.sum() > 0 else 0.0
        
    print(f"Sub {test_idx+1:02d} | Calibrated | Exec: {exec_rate:04.1f}% | Func Acc: {acc:04.1f}%")
    subject_metrics.append((acc, exec_rate))

# ==========================================
# 4. FINAL PERFORMANCE SUMMARY
# ==========================================
print("\n" + "="*55)
print(f"STEP 10 SUMMARY: FEW-SHOT CLINICAL CALIBRATION")
print(f"FUNCTIONAL ACCURACY : {np.mean([m[0] for m in subject_metrics]):.2f}%")
print(f"EXECUTION RATE      : {np.mean([m[1] for m in subject_metrics]):.2f}%")
print("="*55)