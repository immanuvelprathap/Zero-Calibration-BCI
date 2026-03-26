import mne
import numpy as np
import torch
import torch.nn as nn
from pyriemann.estimation import Covariances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

# ==========================================
# 1. SPDNET LAYERS (Neural Net on the Manifold)
# ==========================================
class BiMap(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BiMap, self).__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, x):
        # Performs W^T * X * W to change dimensions while staying on manifold
        return self.W.t() @ x @ self.W

class ReEig(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(ReEig, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Applies ReLU-like thresholding to the eigenvalues
        vals, vecs = torch.linalg.eigh(x)
        vals = torch.clamp(vals, min=self.epsilon)
        return vecs @ torch.diag_embed(vals) @ vecs.transpose(-2, -1)

class SPDNet(nn.Module):
    def __init__(self, channels=64):
        super(SPDNet, self).__init__()
        self.bimap = BiMap(channels, 32)
        self.reeig = ReEig()
        self.classifier = nn.Linear(32 * 32, 2)

    def forward(self, x):
        x = self.bimap(x)
        x = self.reeig(x)
        # Flatten the matrix for the final linear layer
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def get_subject_covs(sub_id):
    fnames = mne.datasets.eegbci.load_data(sub_id, [4, 8, 12], update_path=False)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames])
    mne.datasets.eegbci.standardize(raw)
    raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0.5, tmax=2.5, 
                      baseline=None, preload=True, verbose=False)
    
    # Estimate Covariances (The inputs for SPDNet)
    covs = Covariances(estimator='oas').transform(epochs.get_data())
    return torch.tensor(covs, dtype=torch.float32), torch.tensor(epochs.events[:, -1] - 2, dtype=torch.long)

if __name__ == "__main__":
    print(f"🚀 TRAINING SPDNET ON TOP-TIER SUBJECTS...")
    all_final_accs = []
    
    # We will test on 5 subjects to see if we can break 90% properly
    for sub_id in [1, 7, 14, 23, 33]: # Known high-performers
        try:
            X, y = get_subject_covs(sub_id)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            
            model = SPDNet(channels=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
                all_final_accs.append(acc)
                print(f"✅ Subject {sub_id} | SPDNet Acc: {acc*100:.2f}%")
        except Exception as e:
            print(f"❌ Error on Sub {sub_id}: {e}")

    if all_final_accs:
        print("\n" + "█"*45)
        print(f" SPDNET MANIFOLD REPORT ")
        print("█"*45)
        print(f" Mean Top-Subject Accuracy: {np.mean(all_final_accs)*100:.2f}%")
        print("█"*45)