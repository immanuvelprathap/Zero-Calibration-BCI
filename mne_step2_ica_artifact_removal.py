import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# 1. Load Data and Standardize (Same as Step 1)
subject = 1
runs = [4, 8, 12] # runs --> Per Session ************************************************************** split 
print(f"Fetching data for Subject {subject}...")
raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws) # NO

eegbci_standard = mne.channels.make_standard_montage('standard_1005')
mne.datasets.eegbci.standardize(raw) 
raw.set_montage(eegbci_standard)

# Filter (1-40 Hz)
print("Applying broad filter (1-40 Hz)...")
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

# 2. Setup and Fit ICA
print("Fitting ICA...")
# We use 15 components, which is plenty to isolate eye blinks/heartbeats
ica = ICA(n_components=15, max_iter='auto', random_state=97) # 32 Componets Why Halve --> Look into it!
ica.fit(raw)

# 3. Visualize ICA Components
print("Plotting ICA spatial topographies...")
# This shows the "heatmaps" of where each component is coming from
ica.plot_components()
raw.plot()

print("Plotting ICA time sources...")
# This shows the actual waveform of each component
ica.plot_sources(raw, show_scrollbars=False)


plt.show()