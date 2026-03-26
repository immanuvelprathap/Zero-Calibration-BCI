import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

def get_spectral_features(raw, events):
    # We split into 10 narrow bands (2Hz wide)
    BANDS = [(f, f+2) for f in range(8, 28, 2)]
    features = []
    
    for low, high in BANDS:
        r_f = raw.copy().filter(low, high, method='iir', verbose=False)
        epochs = mne.Epochs(r_f, events, {'T1':2, 'T2':3}, tmin=0.5, tmax=2.5, 
                          baseline=None, preload=True, verbose=False)
        # Riemannian Tangent Space vectorization for this band
        covs = Covariances(estimator='oas').transform(epochs.get_data())
        ts = TangentSpace(metric='riemann').fit_transform(covs)
        features.append(ts)
    
    # Stack all bands together: (Trials, Features * Bands)
    return np.concatenate(features, axis=1), epochs.events[:, -1] - 2

def run_spectral_hunter(num_subjects=109):
    print(f"🚀 HUNTING FOR SUBJECT-SPECIFIC FREQUENCY PEAKS...")
    all_accs = []
    
    # We process each subject INDIVIDUALLY to find their personal 95%
    for sub_id in range(1, num_subjects + 1):
        try:
            fnames = mne.datasets.eegbci.load_data(sub_id, [4, 8, 12])
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in fnames])
            mne.datasets.eegbci.standardize(raw)
            events, _ = mne.events_from_annotations(raw, verbose=False)
            
            X, y = get_spectral_features(raw, events)
            
            # Split into training and testing for THIS subject
            split = int(len(y) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # 🔥 THE HUNTER: Logistic Regression with L1 (Lasso) 
            # It will 'kill' the frequency bands that are just noise.
            clf = LogisticRegressionCV(Cs=10, cv=3, penalty='l1', solver='liblinear', max_iter=1000)
            clf.fit(X_train, y_train)
            
            acc = clf.score(X_test, y_test)
            all_accs.append(acc)
            
            if sub_id % 10 == 0:
                print(f"✅ Sub {sub_id} | Best Acc so far: {np.max(all_accs)*100:.2f}%")
        except: continue
        
    return all_accs

if __name__ == "__main__":
    results = run_spectral_hunter(109)
    print("\n" + "█"*45)
    print(f" SPECTRAL HUNTER FINAL REPORT ")
    print("█"*45)
    print(f" Subject-Specific Mean: {np.mean(results)*100:.2f}%")
    print(f" TOP PERFORMER ACCURACY: {np.max(results)*100:.2f}%")
    print("█"*45)