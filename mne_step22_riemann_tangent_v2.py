import mne
import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

# Force clean output
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

# ==========================================
# 1. THE RIGOROUS DATA LOADER
# ==========================================
def load_riemann_data_strict(num_subjects=109):
    X_list, y_list = [], []
    # Target: 2.0s window at 160Hz = 321 samples
    EXPECTED_SAMPLES = 321 
    
    print(f"🚀 MAPPING {num_subjects} SUBJECTS TO SPD MANIFOLD...")
    print("⚠️ STRICT MODE: Subjects with mismatched temporal lengths will be force-aligned or dropped.")
    
    for sub_id in range(1, num_subjects + 1):
        try:
            # Runs 4, 8, 12: Motor Imagery (Left vs Right Hand)
            runs = [4, 8, 12]
            fnames = mne.datasets.eegbci.load_data(sub_id, runs, update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames])
            
            mne.datasets.eegbci.standardize(raw)
            # Motor imagery bandpass (Mu and Beta rhythms)
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, verbose=False)
            # Crop to the most stable part of imagery (0.5s to 2.5s)
            epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0.5, tmax=2.5, 
                              baseline=None, preload=True, verbose=False)
            
            data = epochs.get_data(copy=True)
            
            # --- THE DIMENSIONALITY FIX ---
            # If the subject has more than 321 samples, crop it. 
            # If they have fewer, they are statistically invalid for this batch; drop them.
            if data.shape[2] > EXPECTED_SAMPLES:
                data = data[:, :, :EXPECTED_SAMPLES]
            elif data.shape[2] < EXPECTED_SAMPLES:
                continue 
            
            X_list.append(data.astype('float32'))
            y_list.append(epochs.events[:, -1] - 2)
            
            if sub_id % 10 == 0:
                print(f"✅ Subject {sub_id} Validated & Manifold-Mapped.")
                
        except Exception:
            continue
            
    return X_list, y_list

# ==========================================
# 2. THE RIEMANNIAN GEOMETRY ENGINE
# ==========================================
if __name__ == "__main__":
    X_all, y_all = load_riemann_data_strict(109)
    
    # Validation check to ensure we didn't drop too many subjects
    if len(X_all) < 100:
        print(f"⚠️ Warning: Only {len(X_all)} subjects passed strict validation.")
    
    # Split: 100 training subjects, remaining for unseen validation
    train_idx = min(100, len(X_all) - 1)
    
    X_train = np.concatenate(X_all[:train_idx], axis=0)
    y_train = np.concatenate(y_all[:train_idx], axis=0)
    X_test = np.concatenate(X_all[train_idx:], axis=0)
    y_test = np.concatenate(y_all[train_idx:], axis=0)

    # THE RIEMANNIAN PIPELINE:
    # 1. Covariances: Converts (N, Channels, Time) -> (N, Channels, Channels) SPD matrices
    # 2. TangentSpace: Maps SPD matrices from the curved manifold to a flat tangent plane
    # 3. StandardScaler: Normalizes the tangent vectors
    # 4. LogisticRegression: Final classification
    pipe = make_pipeline(
        Covariances(estimator='oas'), 
        TangentSpace(metric='riemann'), 
        StandardScaler(),
        LogisticRegression(C=0.8, solver='lbfgs', max_iter=2000, penalty='l2')
    )

    print(f"\n🧠 Dataset Size: {len(y_train)} Training Trials | {len(y_test)} Unseen Trials")
    print("🔥 FITTING RIEMANNIAN MANIFOLD...")
    
    pipe.fit(X_train, y_train)
    
    train_acc = pipe.score(X_train, y_train) * 100
    test_acc = pipe.score(X_test, y_test) * 100

    print("\n" + "█"*40)
    print(f" RIEMANNIAN PERFORMANCE REPORT ")
    print("█"*40)
    print(f" Training (Seen Brains):   {train_acc:.2f}%")
    print(f" Validation (Unseen Brains): {test_acc:.2f}%")
    print("█"*40)

    if test_acc >= 80.0:
        print("\n🎯 BREAKTHROUGH: You have reached the 'Elite BCI' tier.")
    elif test_acc >= 70.0:
        print("\n📉 PLATEAU: The geometry is correct, but we need subject-specific frequency optimization.")
    else:
        print("\n❌ FAILED: The signal is too noisy. We may need to revisit Euclidean Alignment (EA) combined with Riemann.")