import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pyriemann.estimation import Covariances
from scipy.linalg import fractional_matrix_power
import warnings

# Suppress MNE verbosity for a clean terminal output
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

print("🚀 INITIALIZING: MNE-Python + PyRiemann + PyTorch LOSO Pipeline")

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE (Filter Bank)
# ==========================================
class FilterBankEEGNet(nn.Module):
    def __init__(self, n_classes=2, channels=64, samples=801):
        super(FilterBankEEGNet, self).__init__()
        # Dual-scale temporal filters (Alpha and Beta focus)
        self.temp_conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temp_conv2 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
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
# 2. AUTOMATED DATA PIPELINE (MNE + PyRiemann)
# ==========================================
all_X = []
all_y = []
cov_estimator = Covariances(estimator='oas')

print("\n--- PHASE 1: Data Extraction & Riemannian Alignment ---")
for subject in range(1, 21):
    print(f"Processing Subject {subject:02d}...", end=" ")
    runs = [4, 8, 12] # Left/Right Fist
    
    # Load and concatenate
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
    
    # Isolate Motor Imagery band with +/- 5Hz tolerance (Alex's recommendation)
    raw.filter(3.0, 35.0, fir_design='firwin')
    
    # Extract Triggers
    events, event_dict = mne.events_from_annotations(raw)
    event_id = dict(Left_Hand=event_dict['T1'], Right_Hand=event_dict['T2'])
    
    # # Cut Epochs & Reject massive noise spikes automatically (>100 microvolts)
    # reject_criteria = dict(eeg=100e-6) 
    # epochs = mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0, 
    #                     baseline=None, preload=True, reject=reject_criteria)

    # Cut Epochs & Apply Baseline Correction (Zero-centering the signal)
    # We remove the strict amplitude rejection and let Euclidean Alignment handle the noise
    epochs = mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0, 
                        baseline=(None, 0), preload=True)
    
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - 2 # Shift labels to 0 and 1
    
    # EUCLIDEAN ALIGNMENT (The Math)
    R = cov_estimator.transform(X)
    R_bar = np.mean(R, axis=0)
    R_bar_inv_half = fractional_matrix_power(R_bar, -0.5)
    
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = np.dot(R_bar_inv_half, X[i])
        
    all_X.append(X_aligned)
    all_y.append(y)
    print(f"Done! ({len(X)} clean trials)")

# ==========================================
# 3. LEAVE-ONE-SUBJECT-OUT (LOSO) TRAINING
# ==========================================
print("\n--- PHASE 2: Deep Learning LOSO Cross-Validation ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {device.type.upper()} Detected\n")

subject_accuracies = []



for test_idx in range(20):
    # Split Data: 1 Subject for Test, 19 for Train
    X_test = all_X[test_idx]
    y_test = all_y[test_idx]
    
    X_train = np.concatenate([all_X[i] for i in range(20) if i != test_idx])
    y_train = np.concatenate([all_y[i] for i in range(20) if i != test_idx])
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t  = torch.tensor(y_test, dtype=torch.long).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    # Initialize Model
    model = FilterBankEEGNet(samples=X_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Fast Training Loop (50 Epochs)
    model.train()
    for epoch in range(50):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         test_outputs = model(X_test_t)
#         _, predicted = torch.max(test_outputs.data, 1)
#         correct = (predicted == y_test_t).sum().item()
#         accuracy = (correct / len(y_test_t)) * 100.0
        
#     subject_accuracies.append(accuracy)
#     print(f"Subject {test_idx + 1:02d} | Unseen Test Accuracy: {accuracy:.2f}%")

# # Final Results
# final_average = np.mean(subject_accuracies)
# print("\n=========================================")
# print(f"FINAL LOSO AVERAGE ACCURACY: {final_average:.2f}%")
# print("=========================================")# Evaluation with Confidence Gating
    model.eval()
    CONFIDENCE_THRESHOLD = 0.85  # 85% certainty required to execute command
    
    with torch.no_grad():
        test_outputs = model(X_test_t)
        
        # Convert raw logits to probabilities
        probabilities = torch.softmax(test_outputs.data, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        # Create a boolean mask of trials that pass the confidence threshold
        confident_mask = confidences >= CONFIDENCE_THRESHOLD
        
        total_trials = len(y_test_t)
        executed_trials = confident_mask.sum().item()
        rejected_trials = total_trials - executed_trials
        
        if executed_trials > 0:
            # Calculate accuracy ONLY on the trials the model chose to execute
            confident_predictions = predicted[confident_mask]
            confident_labels = y_test_t[confident_mask]
            correct = (confident_predictions == confident_labels).sum().item()
            functional_accuracy = (correct / executed_trials) * 100.0
        else:
            functional_accuracy = 0.0 # Model rejected everything (Safety trigger)
            
        execution_rate = (executed_trials / total_trials) * 100.0
        
    subject_accuracies.append((functional_accuracy, execution_rate))
    
    print(f"Subject {test_idx + 1:02d} | Executed: {execution_rate:05.2f}% of trials | Functional Accuracy: {functional_accuracy:05.2f}%")

# ==========================================
# 4. FINAL RIGOROUS METRICS
# ==========================================
valid_functional_accs = [acc for acc, rate in subject_accuracies if rate > 0]
average_functional_acc = np.mean(valid_functional_accs) if valid_functional_accs else 0
average_execution_rate = np.mean([rate for acc, rate in subject_accuracies])

print("\n=======================================================")
print(f"CLINICAL METRICS (Threshold: {CONFIDENCE_THRESHOLD*100}%)")
print(f"System Execution Rate  : {average_execution_rate:.2f}% (Trials passed to hardware)")
print(f"FUNCTIONAL ACCURACY    : {average_functional_acc:.2f}% (Accuracy of executed commands)")
print("=======================================================")