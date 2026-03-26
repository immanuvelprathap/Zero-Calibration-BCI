import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function
from scipy.linalg import inv, sqrtm
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE GRADIENT REVERSAL LAYER (GRL)
# ==========================================
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # This is the "Adversarial" magic: flip the gradient sign
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==========================================
# 2. ADVERSARIAL SANN ARCHITECTURE
# ==========================================
class SANN_PowerNet(nn.Module):
    def __init__(self, channels=64, num_subjects=100):
        super(SANN_PowerNet, self).__init__()
        # Shared Feature Extractor
        self.temp_conv = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.spat_conv = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.avg_pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        
        # Head 1: Task Classifier (Left vs Right)
        self.label_classifier = nn.Sequential(
            nn.Linear(40 * 26, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
        # Head 2: Subject Classifier (The Adversary)
        self.subject_classifier = nn.Sequential(
            nn.Linear(40 * 26, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_subjects)
        )

    def forward(self, x, alpha=1.0):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        
        # Shared Path
        features = self.temp_conv(x)
        features = self.spat_conv(features)
        features = self.bn(features)
        features = features ** 2 
        features = self.avg_pool(features)
        features = torch.log(torch.clamp(features, min=1e-6))
        features = features.view(features.size(0), -1)
        
        # Motor Imagery Prediction
        label_output = self.label_classifier(features)
        
        # Subject Identification (flipped gradient)
        reverse_features = ReverseLayerF.apply(features, alpha)
        subject_output = self.subject_classifier(reverse_features)
        
        return label_output, subject_output

# ==========================================
# 3. RESTRUCTURED DATA LOADING (Tracking Subject ID)
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

def load_adversarial_data(num_subjects=109):
    X_list, y_list, s_list = [], [], []
    print(f"🚀 INDEXING {num_subjects} SUBJECTS FOR ADVERSARIAL SANN...")
    for sub_idx in range(num_subjects):
        try:
            sub_id = sub_idx + 1
            runs = [4, 8, 12]
            fnames = mne.datasets.eegbci.load_data(sub_id, runs, update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames])
            mne.datasets.eegbci.standardize(raw)
            if raw.info['sfreq'] != 160: raw.resample(160)
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            events, _ = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0, tmax=3.0, baseline=None, preload=True, verbose=False)
            data = epochs.get_data(copy=True)[:, :, :481]
            
            data = euclidean_alignment(data)
            
            X_list.append(data.astype('float32'))
            y_list.append((epochs.events[:, -1] - 2).astype('int64'))
            # NEW: Tag every trial with its Subject ID
            s_list.append(np.full(len(data), sub_idx, dtype='int64'))
            
            if sub_id % 10 == 0: print(f"✅ Synced Subject {sub_id}...")
        except Exception: continue
    return X_list, y_list, s_list

# ==========================================
# 4. TRAINING WITH ADVERSARIAL PENALTY
# ==========================================
if __name__ == "__main__":
    X_all, y_all, s_all = load_adversarial_data(109)
    
    # 100 subjects for training, 9 for unseen validation
    X_train = np.concatenate(X_all[:100], axis=0)
    y_train = np.concatenate(y_all[:100], axis=0)
    s_train = np.concatenate(s_all[:100], axis=0)
    
    X_val_list, y_val_list = X_all[100:], y_all[100:]

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(y_train), 
        torch.from_numpy(s_train)), 
        batch_size=64, shuffle=True)

    model = SANN_PowerNet(num_subjects=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔥 STARTING ADVERSARIAL SANN PHASE | Samples: {len(y_train)}")
    for epoch in range(100):
        model.train()
        # Schedule Alpha to increase gradually (0.0 -> 1.0)
        alpha = min(1.0, (epoch + 1) / 40.0)
        
        t_label_loss, t_subj_loss = 0, 0
        for b_x, b_y, b_s in train_loader:
            b_x, b_y, b_s = b_x.to(device), b_y.to(device), b_s.to(device)
            
            optimizer.zero_grad()
            l_pred, s_pred = model(b_x, alpha=alpha)
            
            loss_label = criterion(l_pred, b_y)
            loss_subject = criterion(s_pred, b_s)
            
            # Combine losses: Task + Adversarial
            total_loss = loss_label + loss_subject
            total_loss.backward()
            optimizer.step()
            
            t_label_loss += loss_label.item()
            t_subj_loss += loss_subject.item()

        model.eval(); val_accs = []
        with torch.no_grad():
            for xt, yt in zip(X_val_list, y_val_list):
                l_pred, _ = model(torch.from_numpy(xt).to(device), alpha=0)
                acc = (torch.max(l_pred, 1)[1] == torch.from_numpy(yt).to(device)).sum().item() / len(yt)
                val_accs.append(acc)
        
        avg_val = np.mean(val_accs) * 100
        print(f"Epoch {epoch+1:02d} | Task Loss: {t_label_loss/len(train_loader):.4f} | Subj Loss: {t_subj_loss/len(train_loader):.4f} | Val Acc: {avg_val:.2f}% | Alpha: {alpha:.2f}")
        
        if avg_val >= 95.0:
            print("🎯 TARGET ACHIEVED. CROSS-SUBJECT BOUNDARY BROKEN."); break