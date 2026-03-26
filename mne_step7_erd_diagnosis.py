import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

print("🔬 RUNNING ERD DIAGNOSTIC: Subject 17 (Best) vs Subject 16 (Worst)")

def plot_motor_erd(subject_id):
    print(f"Loading and processing Subject {subject_id:02d}...")
    
    # 1. Load Data
    runs = [4, 8, 12] # Left/Right hand imagery
    raw_fnames = mne.datasets.eegbci.load_data(subject_id, runs)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    
    # Standardize electrode names to 10-05 system
    mne.datasets.eegbci.standardize(raw) 
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
    
    # 2. Filter & Epoch
    raw.filter(3.0, 35.0, fir_design='firwin')
    events, event_dict = mne.events_from_annotations(raw)
    event_id = dict(Left_Hand=event_dict['T1'], Right_Hand=event_dict['T2'])
    
    # Epoching from -1s (rest) to 4s (imagined movement)
    epochs = mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0, 
                        baseline=(None, 0), preload=True)
    
    # Isolate Motor Cortex Electrodes
    epochs.pick(['C3', 'C4'])
    
    # 3. Time-Frequency Analysis (Morlet Wavelets)
    # We analyze frequencies from 8Hz to 30Hz
    freqs = np.arange(8, 31, 1)
    n_cycles = freqs / 2.0  
    
    print(f"Calculating Morlet Wavelets for Subject {subject_id:02d}...")
    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, 
                       return_itc=False, decim=3, n_jobs=1)
    
    # 4. Plotting
    # We plot the power drop relative to the baseline (-1.0 to 0.0 seconds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Subject {subject_id:02d} - ERD/ERS Motor Cortex (C3/C4)", fontsize=14, fontweight='bold')
    
    # mode='percent' shows the percentage drop in power (ERD is blue)
    power.plot([0], baseline=(-1.0, 0), mode='percent', axes=axes[0], show=False, colorbar=False)
    axes[0].set_title('C3 Electrode (Left Hemisphere)')
    axes[0].axvline(0, color='k', linestyle='--', linewidth=2)
    
    power.plot([1], baseline=(-1.0, 0), mode='percent', axes=axes[1], show=False, colorbar=True)
    axes[1].set_title('C4 Electrode (Right Hemisphere)')
    axes[1].axvline(0, color='k', linestyle='--', linewidth=2)
    
    plt.tight_layout()

# Run the diagnostic on our top and bottom performers
plot_motor_erd(17) # The 82.22% Subject
plot_motor_erd(16) # The 46.67% Subject

print("Opening diagnostic plots...")
plt.show()