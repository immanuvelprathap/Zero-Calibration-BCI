import mne
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

print("--- STEP 5: BCI Classification (Traditional Benchmark) ---")

# 1. Load, Filter, ICA, and Epoch (Silent execution of Steps 1-3)
subject = 1
runs = [4, 8, 12] 
raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
mne.datasets.eegbci.standardize(raw) 
raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)

from mne.preprocessing import ICA
ica = ICA(n_components=15, random_state=97, max_iter='auto')
ica.fit(raw, verbose=False)
ica.exclude = [0]
raw_clean = ica.apply(raw.copy(), verbose=False)

events, event_dict = mne.events_from_annotations(raw_clean, verbose=False)
event_id = dict(Left_Hand=event_dict['T1'], Right_Hand=event_dict['T2'])
epochs = mne.Epochs(raw_clean, events, event_id, tmin=-1.0, tmax=4.0, baseline=None, preload=True, verbose=False)

# 2. Extract Data and Labels
X = epochs.get_data(copy=True)
y = epochs.events[:, -1] # The labels (2 for Left, 3 for Right)

# 3. Create the PyRiemann Benchmark Pipeline
# This takes the Epochs -> Computes Covariance -> Maps to Tangent Space -> Classifies
print("\nBuilding PyRiemann Pipeline (Covariance -> Tangent Space -> Logistic Regression)...")
clf_pipeline = make_pipeline(
    Covariances(estimator='oas'), 
    TangentSpace(metric='riemann'), 
    LogisticRegression(max_iter=1000)
)

# 4. Evaluate using 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf_pipeline, X, y, cv=cv)

print("\nRESULTS:")
print(f"Individual Fold Accuracies: {np.round(scores * 100, 2)}%")
print(f"Mean Baseline Accuracy: {np.mean(scores) * 100:.2f}%")
print("---------------------------------------------------------")
print("If your EEGNet beats this baseline on the LOSO loop, you have a solid paper!")