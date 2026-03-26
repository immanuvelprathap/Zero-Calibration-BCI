import mne
import numpy as np
from pyriemann.estimation import Covariances
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

# 1. Re-create the clean epochs silently (combining Steps 1-3)
print("Loading and cleaning data (Steps 1-3)...")
subject = 1
runs = [4, 8, 12] 
raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
mne.datasets.eegbci.standardize(raw) 
raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)

# Assuming ICA000 was the blink (from Step 2)
from mne.preprocessing import ICA
ica = ICA(n_components=15, random_state=97, max_iter='auto')
ica.fit(raw, verbose=False)
ica.exclude = [0]
raw_clean = ica.apply(raw.copy(), verbose=False)

events, event_dict = mne.events_from_annotations(raw_clean, verbose=False)
event_id = dict(Left_Hand=event_dict['T1'], Right_Hand=event_dict['T2'])
epochs = mne.Epochs(raw_clean, events, event_id, tmin=-1.0, tmax=4.0, baseline=None, preload=True, verbose=False)

# 2. Extract Data for Riemannian Math
# X shape: (Trials, Channels, Time)
X = epochs.get_data(copy=True)
print(f"\nExtracted Epochs Shape: {X.shape}")

# 3. Compute Spatial Covariances using PyRiemann
# R shape: (Trials, Channels, Channels)
cov_estimator = Covariances(estimator='oas')
R = cov_estimator.transform(X)
print(f"Computed Covariance Matrices Shape: {R.shape}")

# 4. Euclidean Alignment (EA)
print("\nApplying Euclidean Alignment (EA)...")
# Calculate the subject's mean covariance matrix (R_bar)
R_bar = np.mean(R, axis=0)

# Calculate the alignment operator: R_bar^(-1/2)
R_bar_inv_half = fractional_matrix_power(R_bar, -0.5)

# Apply the alignment operator to the raw trials: X_aligned = R_bar^(-1/2) * X
X_aligned = np.zeros_like(X)
for i in range(X.shape[0]):
    X_aligned[i] = np.dot(R_bar_inv_half, X[i])

# 5. Verify Alignment
# The new mean covariance of X_aligned should be the Identity Matrix (I)
R_aligned = cov_estimator.transform(X_aligned)
R_aligned_mean = np.mean(R_aligned, axis=0)

print("\nVerification:")
print("If EA was successful, the aligned mean covariance should be the Identity Matrix.")
print("Diagonal elements (should be ~1.0):")
print(np.diag(R_aligned_mean)[:5], "...") # Print first 5 diagonal elements
print("Off-diagonal elements (should be ~0.0):")
print(R_aligned_mean[0, 1:6], "...") # Print first 5 off-diagonal elements

# Plotting the Covariance Matrices to visually prove it to Alex
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im1 = axes[0].imshow(R_bar, cmap='RdBu_r', vmin=-np.max(np.abs(R_bar)), vmax=np.max(np.abs(R_bar)))
axes[0].set_title("Before EA: Original Mean Covariance")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(R_aligned_mean, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title("After EA: Aligned Mean Covariance (Identity Matrix)")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()