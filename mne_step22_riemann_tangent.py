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
# 1. MANIFOLD DATA PREPARATION
# ==========================================
def load_riemann_data(num_subjects=109):
    X_list, y_list = [], []
    print(f"🚀 MAPPING {num_subjects} SUBJECTS TO SPD MANIFOLD...")
    
    for sub_id in range(1, num_subjects + 1):
        try:
            runs = [4, 8, 12] # Motor Imagery: Left vs Right Hand
            fnames = mne.datasets.eegbci.load_data(sub_id, runs, update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames])
            
            mne.datasets.eegbci.standardize(raw)
            # Motor imagery lives in the Mu (8-13Hz) and Beta (13-30Hz) bands
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, verbose=False)
            # We use a 2-second window where imagery is most stable
            epochs = mne.Epochs(raw, events, {'T1': 2, 'T2': 3}, tmin=0.5, tmax=2.5, 
                              baseline=None, preload=True, verbose=False)
            
            X_list.append(epochs.get_data(copy=True))
            y_list.append(epochs.events[:, -1] - 2)
            
            if sub_id % 10 == 0: print(f"✅ Extracted Manifold for Subject {sub_id}...")
        except Exception as e:
            continue
            
    return X_list, y_list

# ==========================================
# 2. THE RIEMANNIAN GEOMETRY PIPELINE
# ==========================================
if __name__ == "__main__":
    X_all, y_all = load_riemann_data(109)
    
    # Pure Cross-Subject Split
    # Training on first 100 subjects
    X_train = np.concatenate(X_all[:100], axis=0)
    y_train = np.concatenate(y_all[:100], axis=0)
    
    # Testing on the remaining 9 (completely unseen brains)
    X_test = np.concatenate(X_all[100:], axis=0)
    y_test = np.concatenate(y_all[100:], axis=0)

    # THE PIPELINE:
    # 1. Covariances: Compute SPD matrices (OAS is a robust shrinkage estimator)
    # 2. TangentSpace: Project onto the tangent plane at the Riemannian Mean
    # 3. LogisticRegression: Linear classification in the flattened space
    pipe = make_pipeline(
        Covariances(estimator='oas'), 
        TangentSpace(metric='riemann'), 
        StandardScaler(),
        LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, penalty='l2')
    )

    print(f"\n🔥 TOTAL SAMPLES: {len(y_train)}")
    print("🧠 TRAINING RIEMANNIAN TANGENT SPACE CLASSIFIER...")
    
    pipe.fit(X_train, y_train)
    
    train_acc = pipe.score(X_train, y_train) * 100
    test_acc = pipe.score(X_test, y_test) * 100

    print("\n" + "="*30)
    print(f"RIEMANNIAN RESULTS")
    print("-" * 30)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"VAL ACCURACY (UNSEEN SUBJECTS): {test_acc:.2f}%")
    print("="*30)

    if test_acc > 75:
        print("\n🚀 BREAKTHROUGH: Geometrical features outperform neural networks.")
        print("This confirms the signal is in the COVARIANCE, not the raw wave.")
    else:
        print("\n⚠️ STUCK: If accuracy is still <75%, the inter-subject noise is in the spectral domain.")