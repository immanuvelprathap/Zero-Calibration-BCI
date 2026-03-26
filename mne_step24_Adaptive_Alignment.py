import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

def load_and_center_data(num_subjects=109):
    X_list, y_list = [], []
    print(f"🚀 ALIGNING MANIFOLDS VIA CENTROID RELOCATION...")
    
    for sub_id in range(1, num_subjects + 1):
        try:
            fnames = mne.datasets.eegbci.load_data(sub_id, [4, 8, 12])
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in fnames])
            mne.datasets.eegbci.standardize(raw)
            raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
            
            epochs = mne.Epochs(raw, mne.events_from_annotations(raw)[0], {'T1':2, 'T2':3}, 
                              tmin=0.5, tmax=2.5, baseline=None, preload=True)
            
            # 1. Estimate Covariances
            covs = Covariances(estimator='oas').transform(epochs.get_data())
            
            # 2. 🔥 ADAPTIVE STEP: Calculate this subject's Riemannian Mean
            C_mean = mean_covariance(covs, metric='riemann')
            
            # 3. 🔥 ALIGNMENT: Whiten the data using the subject's own mean
            # This centers the subject's 'cloud' at the Identity matrix (the origin)
            from scipy.linalg import inv, sqrtm
            C_inv_sq = inv(sqrtm(C_mean))
            aligned_covs = np.stack([C_inv_sq @ C @ C_inv_sq for C in covs])
            
            X_list.append(aligned_covs)
            y_list.append(epochs.events[:, -1] - 2)
            if sub_id % 10 == 0: print(f"✅ Aligned Subject {sub_id}")
        except: continue
    return X_list, y_list

if __name__ == "__main__":
    X_all, y_all = load_and_center_data(109)
    
    X_train = np.concatenate(X_all[:100], axis=0)
    y_train = np.concatenate(y_all[:100], axis=0)
    
    # Classification in Tangent Space of the ALIGNED data
    pipe = make_pipeline(
        TangentSpace(metric='riemann'),
        StandardScaler(),
        LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    )

    pipe.fit(X_train, y_train)
    
    # Test on unseen, but ALSO ALIGNED subjects
    test_accs = [pipe.score(X_all[i], y_all[i]) for i in range(100, len(X_all))]
    
    print("\n" + "█"*45)
    print(f" ADAPTIVE CENTROID ALIGNMENT (ACA) REPORT ")
    print("█"*45)
    print(f" Mean Val Accuracy: {np.mean(test_accs)*100:.2f}%")
    print("█"*45)