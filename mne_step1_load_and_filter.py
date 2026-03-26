import mne
import matplotlib.pyplot as plt

# 1. Fetch PhysioNet Data natively through MNE
# Subject 1, Runs 4, 8, 12 are typically Left/Right Fist Motor Imagery
subject = 1
runs = [4, 8, 12] 
print(f"Fetching data for Subject {subject}...")
raw_fnames = mne.datasets.eegbci.load_data(subject, runs)

# 2. Load the EDF files into an MNE 'Raw' object
# We concatenate the runs into one continuous recording
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)

# 3. Standardize the Electrode Montage (The "Spatial Map")
# This tells MNE exactly where each electrode sits on the 3D skull
eegbci_standard = mne.channels.make_standard_montage('standard_1005')
# PhysioNet uses slightly older naming conventions, so we map them
mne.datasets.eegbci.standardize(raw) 
raw.set_montage(eegbci_standard)

# 4. The "Broad" Filter (Alex's Recommendation)
# Before ICA, we use a broad bandpass filter (e.g., 1 Hz to 40 Hz)
# 1 Hz removes slow sweat/cable movement drifts. 
# 40 Hz removes 50Hz electrical wall noise.
print("Applying broad filter (1-40 Hz)...")
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

# 5. Visualize the Continuous Data
# This opens an interactive window where you can scroll through the brainwaves
print("Opening visualization window...")
# We scale the display so the waves fit nicely on screen
fig = raw.plot(duration=5, n_channels=20, scalings={'eeg': 75e-6})
plt.show()