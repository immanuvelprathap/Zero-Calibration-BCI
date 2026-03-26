import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# 1. Load & Filter (Same as before)
subject = 2
runs = [4] # Left/Right fist motor imagery
raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)

eegbci_standard = mne.channels.make_standard_montage('standard_1005')
mne.datasets.eegbci.standardize(raw) 
raw.set_montage(eegbci_standard)

print("Applying broad filter (1-40 Hz)...")
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

# 2. Fit ICA and REMOVE the Artifact
print("Fitting ICA and removing artifacts...")
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(raw)

# We identify ICA000 as the eye blink and tell MNE to exclude it
ica.exclude = [0] 
raw_clean = raw.copy()
ica.apply(raw_clean)
print("Eye blinks successfully removed!")

# 3. Extracting Events (The Triggers)
# PhysioNet uses T1 (Left Hand) and T2 (Right Hand)
events, event_dict = mne.events_from_annotations(raw_clean)
print(f"Found events: {event_dict}")

# We map T1 and T2 to their integer event codes (usually 2 and 3)
event_id = dict(Left_Hand=event_dict['T1'], Right_Hand=event_dict['T2'])

# 4. Epoching the Data
# We cut from -1 seconds (before the cue) to +4 seconds (during the imagery)
tmin, tmax = -1.0, 4.0

print("Cutting the continuous data into Epochs...")
epochs = mne.Epochs(raw_clean, events, event_id, tmin, tmax, 
                    proj=True, baseline=None, preload=True)

# 5. Visualize the Clean Epochs
# This will show you the individual, clean 5-second trials
epochs.plot(n_epochs=3, scalings={'eeg': 75e-6})
plt.show()