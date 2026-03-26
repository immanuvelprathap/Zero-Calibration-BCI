import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

# ==========================================
# 1. MULTI-BAND DATA LOADER
# ==========================================
def load_fbr_data(num_subjects=109):
    X_list, y_list = [], []
    # Narrow bands: 8-12Hz (Mu), 12-16Hz, 16-20Hz, 20-24Hz, 24-30Hz (Beta)
    BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    EXPECTED_SAMPLES = 321
    
    print(f"🚀 EXTRACTING MULTI-BAND MANIFOLDS FOR {num_subjects} SUBJECTS...")
    
    for sub_id in range(1, num_subjects + 1):
        try:
            fnames = mne.datasets.eegbci.load_data(sub_id, [4, 8, 12], update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames])
            mne.datasets.eegbci.standardize(raw)
            
            events, _ = mne.events_from_annotations(raw, verbose=False)
            
            # Temporary storage for this subject's filtered versions
            subject_bands = []
            
            for low, high in BANDS:
                # Filter specific band
                raw_filt = raw.copy().filter(low, high, fir_design='firwin', verbose=False)
                epochs = mne.Epochs(raw_filt, events, {'T1': 2, 'T2': 3}, tmin=0.5, tmax=2.5, 
                                  baseline=None, preload=True, verbose=False)
                
                d = epochs.get_data(copy=True)
                # Force shape alignment
                if d.shape[2] > EXPECTED_SAMPLES:
                    d = d[:, :, :EXPECTED_SAMPLES]
                elif d.shape[2] < EXPECTED_SAMPLES:
                    raise ValueError("Samples too short")
                    
                subject_bands.append(d)
                
            # Pack as (Bands, Trials, Channels, Time)
            X_list.append(np.stack(subject_bands)) 
            y_list.append(epochs.events[:, -1] - 2)
            
            if sub_id % 10 == 0: print(f"✅ Filter Bank Indexed: Subject {sub_id}")
        except:
            continue
            
    return X_list, y_list

# ==========================================
# 2. ENSEMBLE TRAINING & VOTING
# ==========================================
if __name__ == "__main__":
    X_all, y_all = load_fbr_data(109)
    
    NUM_BANDS = 5
    # Use 100 subjects for training, the rest for unseen validation
    train_limit = 100 
    
    # Store a trained model for each frequency band
    band_experts = []
    
    print(f"\n🔥 TRAINING {NUM_BANDS} BAND-SPECIFIC EXPERTS...")
    
    for b in range(NUM_BANDS):
        # Extract the b-th band for the first 100 subjects
        X_train_band = np.concatenate([x[b] for x in X_all[:train_limit]], axis=0)
        y_train = np.concatenate(y_all[:train_limit], axis=0)
        
        # Expert Pipeline: Manifold Mapping -> Regularized Classifier
        expert = make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            StandardScaler(),
            # C=0.01 is EXTREME regularization to fight that 99% overfitting
            LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=1000)
        )
        
        expert.fit(X_train_band, y_train)
        band_experts.append(expert)
        print(f"   Expert {b+1} ({8+b*4}-{12+b*4}Hz) Trained.")

    # ==========================================
    # 3. UNSEEN BRAIN EVALUATION (SOFT VOTING)
    # ==========================================
    print("\n🧠 CROSS-SUBJECT EVALUATION (ENSEMBLE VOTING)...")
    
    final_accs = []
    for sub_idx in range(train_limit, len(X_all)):
        # Collect probability predictions from all 5 experts
        all_probs = []
        for b in range(NUM_BANDS):
            # Predict probabilities for this subject's b-th band data
            probs = band_experts[b].predict_proba(X_all[sub_idx][b])
            all_probs.append(probs)
            
        # Average the probabilities across experts (Soft Voting)
        mean_probs = np.mean(all_probs, axis=0)
        final_preds = np.argmax(mean_probs, axis=1)
        
        # Calculate accuracy for this specific unseen subject
        subject_acc = np.mean(final_preds == y_all[sub_idx])
        final_accs.append(subject_acc)

    print("\n" + "█"*45)
    print(f" FILTER BANK RIEMANNIAN ENSEMBLE REPORT ")
    print("█"*45)
    print(f" Mean Val Accuracy (Unseen Subjects): {np.mean(final_accs)*100:.2f}%")
    print(f" Best Subject Acc: {np.max(final_accs)*100:.2f}%")
    print(f" Worst Subject Acc: {np.min(final_accs)*100:.2f}%")
    print("█"*45)

    if np.mean(final_accs) > 0.70:
        print("\n🎯 STABILITY REACHED: Multiple bands are filtering out subject noise.")
    else:
        print("\n⚠️ FAILED: 51% suggests the label-to-signal mapping is still broken.")