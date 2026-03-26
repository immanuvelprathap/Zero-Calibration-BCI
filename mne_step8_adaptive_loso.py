import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pyriemann.estimation import Covariances
from scipy.linalg import fractional_matrix_power
import warnings

# --- PROJECT RIGOR SETTINGS ---
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS FOR THE 95% THRESHOLD
CONFIDENCE_THRESHOLD = 0.85     
PSEUDO_LABEL_THRESHOLD = 0.90   
ADAPT_LR = 0.0001               
ADAPT_EPOCHS = 15               

print(f"🚀 INITIATING STEP 8: ADAPTIVE UDA/TTA PIPELINE")

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class FilterBankEEGNet(nn.Module):
    def __init__(self, n_classes=2, channels=64, samples=801):
        super(FilterBankEEGNet, self).__init__()
        # BIO: Captures specific oscillations (Mu/Beta) across different scales.
        # MATH: Temporal convolutions acting as band-pass filters.
        self.temp_conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temp_conv2 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # BIO: Spatial filters to locate motor cortex activity.
        # MATH: Depthwise convolution to learn subject-specific spatial patterns.
        self.depth_conv = nn.Conv2d(32, 64, (channels, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.35)
        
        self.sep_conv = nn.Conv2d(64, 64, (1, 16), padding=(0, 8), groups=64, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.35)
        
        self.feature_size = 64 * (samples // 4 // 8) 
        self.fc = nn.Linear(self.feature_size, n_classes)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = torch.cat((self.temp_conv1(x), self.temp_conv2(x)), dim=1)
        x = self.bn1(x)
        x = self.dropout1(self.avg_pool1(self.bn2(self.elu(self.depth_conv(x)))))
        x = self.dropout2(self.avg_pool2(self.bn3(self.elu(self.sep_conv(x)))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 2. DATA EXTRACTION & EA ALIGNMENT
# ==========================================
all_X, all_y = [], []
cov_estimator = Covariances(estimator='oas')

for subject in range(1, 21):
    print(f"Aligning Subject {subject:02d}...", end="\r")
    runs = [4, 8, 12]
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    mne.datasets.eegbci.standardize(raw)
    raw.filter(3.0, 35.0, fir_design='firwin', verbose=False)
    events, event_dict = mne.events_from_annotations(raw)
    
    # BIO: Baseline correction removes DC offset and voltage drift from the hardware.
    epochs = mne.Epochs(raw, events, {'T1': event_dict['T1'], 'T2': event_dict['T2']}, 
                        tmin=-1.0, tmax=4.0, baseline=(None, 0), preload=True, verbose=False)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - event_dict['T1']
    
    # EUCLIDEAN ALIGNMENT (EA)
    # BIO: Compensates for electrode impedance and skull thickness differences.
    # MATH: Centers the subject's mean covariance matrix at the Identity Matrix (I).
    R_bar = np.mean(cov_estimator.transform(X), axis=0)
    R_bar_inv_half = fractional_matrix_power(R_bar, -0.5).real
    X_aligned = np.array([np.dot(R_bar_inv_half, trial) for trial in X])
    
    all_X.append(X_aligned)
    all_y.append(y)

# ==========================================
# 3. ADAPTIVE LOSO EVALUATION
# ==========================================
subject_metrics = []

for test_idx in range(20):
    # Split
    X_test_t = torch.tensor(all_X[test_idx], dtype=torch.float32).to(device)
    y_test_t = torch.tensor(all_y[test_idx], dtype=torch.long).to(device)
    X_train = np.concatenate([all_X[i] for i in range(20) if i != test_idx])
    y_train = np.concatenate([all_y[i] for i in range(20) if i != test_idx])
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), 
                                          torch.tensor(y_train, dtype=torch.long).to(device)), 
                              batch_size=32, shuffle=True)
    
    # PRE-TRAINING (Source Domain Knowledge)
    model = FilterBankEEGNet(samples=X_test_t.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    for _ in range(30): 
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    # --- TEST-TIME ADAPTATION (TTA) / UNSUPERVISED DOMAIN ADAPTATION (UDA) ---
    # BIO: Identifying trials where the subject's brain signature is most distinct.
    # MATH: Pseudo-labeling high-confidence trials to use as anchors for adaptation.
    
    model.eval()
    with torch.no_grad():
        initial_out = model(X_test_t)
        probs = torch.softmax(initial_out, dim=1)
        conf, pseudo = torch.max(probs, 1)

    mask = conf >= PSEUDO_LABEL_THRESHOLD
    if mask.sum() > 5:
        # BIO: Nudging the model to understand this specific subject's unique Motor Imagery manifold.
        # MATH: Minimizing loss on pseudo-labels to refine spatial filters for the target domain.
        model.train()
        opt_tta = optim.Adam(model.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_EPOCHS):
            opt_tta.zero_grad()
            loss = criterion(model(X_test_t[mask]), pseudo[mask])
            loss.backward()
            opt_tta.step()
        status = f"TTA-Active ({mask.sum().item()} trials)"
    else:
        # BIO: If no trials are clear, the subject's SNR is too low; we refuse to adapt to noise.
        status = "TTA-Skipped (Low SNR)"

    # --- FINAL FUNCTIONAL EVALUATION ---
    # BIO: Decisions are only made if the signal-to-noise ratio is sufficient for a safe command.
    # MATH: Applying Shannon Entropy/Softmax gating to ensure high functional accuracy.
    model.eval()
    with torch.no_grad():
        final_out = model(X_test_t)
        final_probs = torch.softmax(final_out, dim=1)
        f_conf, f_pred = torch.max(final_probs, 1)
        
        c_mask = f_conf >= CONFIDENCE_THRESHOLD
        exec_rate = (c_mask.sum().item() / len(y_test_t)) * 100
        acc = (f_pred[c_mask] == y_test_t[c_mask]).sum().item() / c_mask.sum().item() * 100 if c_mask.sum() > 0 else 0.0
        
    print(f"Sub {test_idx+1:02d} | {status} | Exec: {exec_rate:05.1f}% | Func Acc: {acc:04.1f}%")
    subject_metrics.append((acc, exec_rate))

# Final Metrics
print("\n" + "="*55)
print(f"BREAKTHROUGH CLINICAL METRICS (UDA/TTA ENABLED)")
print(f"FUNCTIONAL ACCURACY : {np.mean([m[0] for m in subject_metrics]):.2f}%")
print(f"EXECUTION RATE      : {np.mean([m[1] for m in subject_metrics]):.2f}%")
print("="*55)