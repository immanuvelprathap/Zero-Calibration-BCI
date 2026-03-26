import mne
from mne.preprocessing import ICA
from moabb.datasets import BNCI2014_001
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def generate_pipeline_comparison_plot():
    print("🚀 FETCHING DATA FOR PIPELINE COMPARISON...")
    dataset = BNCI2014_001()
    
    # 1. Dynamically load the run to avoid KeyError across MOABB versions
    session_data = dataset.get_data(subjects=[1])[1]['0train']
    run_key = list(session_data.keys())[-1] # Grabs the last available run dynamically
    print(f"   -> Successfully loaded run: {run_key}...")
    raw = session_data[run_key].copy()
    
    # We will look specifically at the Fz electrode (Front of the head, catches blinks)
    channel_to_plot = 'Fz' 
    
    # ---------------------------------------------------------
    # PIPELINE 1: THE MOABB BASELINE (Dirty Data)
    # ---------------------------------------------------------
    raw_baseline = raw.copy()
    raw_baseline.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
    
    # Find the events to create epochs
    events = mne.find_events(raw_baseline, stim_channel='stim')
    
    # Epoch the dirty baseline data
    epochs_baseline = mne.Epochs(raw_baseline, events, tmin=-0.5, tmax=4.0, baseline=None, preload=True)
    
    # ---------------------------------------------------------
    # PIPELINE 2: YOUR CUSTOM ICA PIPELINE (Clean Data)
    # ---------------------------------------------------------
    raw_custom = raw.copy()
    # ICA needs 1-40Hz to fit properly without drifting
    raw_custom.filter(1., 40., fir_design='firwin')
    
    print("🧠 FITTING PER-SESSION ICA...")
    ica = ICA(n_components=11, random_state=42, max_iter='auto')
    ica.fit(raw_custom, picks='eeg')
    
    # Automatically find and remove the eye blink component
    eog_indices, _ = ica.find_bads_eog(raw_custom, ch_name=['EOG1', 'EOG2', 'EOG3'])
    ica.exclude = eog_indices
    raw_custom_clean = ica.apply(raw_custom)
    
    # Now bandpass it to the 8-30Hz motor imagery band (matching the baseline)
    raw_custom_clean.filter(8., 30., fir_design='firwin')
    
    # Epoch the clean custom data
    epochs_custom = mne.Epochs(raw_custom_clean, events, tmin=-0.5, tmax=4.0, baseline=None, preload=True)


###################################################THE VISUAL PROOF (Plotting a specific artifact epoch)########################################################################
#     # ---------------------------------------------------------
#     # THE VISUAL PROOF (Plotting a specific artifact epoch)
#     # ---------------------------------------------------------
#     # Let's grab Epoch #5 (often contains early artifacts as the user settles in)
#     epoch_idx = 5 
#     ch_idx = raw.ch_names.index(channel_to_plot)
    
#     baseline_signal = epochs_baseline.get_data()[epoch_idx, ch_idx, :] * 1e6 # Convert to microvolts
#     custom_signal = epochs_custom.get_data()[epoch_idx, ch_idx, :] * 1e6
#     times = epochs_baseline.times
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(times, baseline_signal, color='red', alpha=0.7, linewidth=2, label='Baseline Model Input (Dirty - Contains Blinks)')
#     plt.plot(times, custom_signal, color='green', linewidth=2, label='Your Model Input (Clean - Post ICA)')
    
#     plt.title(f"Subject 1 | Epoch #{epoch_idx} | Channel {channel_to_plot} (Frontal Lobe)", fontsize=14, fontweight='bold')
#     plt.xlabel("Time (seconds)", fontsize=12)
#     plt.ylabel("Amplitude (µV)", fontsize=12)
#     plt.axvline(x=0, color='black', linestyle='--', label='Motor Imagery Cue Starts')
#     plt.legend(loc='upper right', fontsize=11)
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig("pipeline_epoch_comparison.png", dpi=300)
#     print("✅ SAVED: pipeline_epoch_comparison.png")
#     plt.show()

# if __name__ == "__main__":
#     generate_pipeline_comparison_plot()



##################################################THE VISUAL PROOF (Plotting multiple epochs for Alex)#########################################################################
#     # ---------------------------------------------------------
#     # THE VISUAL PROOF (Plotting multiple epochs for Alex)
#     # ---------------------------------------------------------
#     # Put any epoch numbers you want to inspect in this list. 
#     # Let's look at an early, middle, and late epoch.
#     epochs_to_plot = [5, 15, 30] 
#     ch_idx = raw.ch_names.index(channel_to_plot)
#     times = epochs_baseline.times
    
#     # Create a stacked graphic based on how many epochs are in the list
#     fig, axes = plt.subplots(len(epochs_to_plot), 1, figsize=(12, 4 * len(epochs_to_plot)), sharex=True)
    
#     # Safety check in case you only put 1 epoch in the list
#     if len(epochs_to_plot) == 1:
#         axes = [axes]
        
#     for i, epoch_idx in enumerate(epochs_to_plot):
#         baseline_signal = epochs_baseline.get_data()[epoch_idx, ch_idx, :] * 1e6
#         custom_signal = epochs_custom.get_data()[epoch_idx, ch_idx, :] * 1e6
        
#         axes[i].plot(times, baseline_signal, color='red', alpha=0.7, linewidth=2, label='Baseline (Dirty)')
#         axes[i].plot(times, custom_signal, color='green', linewidth=2, label='Custom (Clean)')
        
#         axes[i].set_title(f"Subject 1 | Epoch #{epoch_idx} | Channel {channel_to_plot}", fontsize=12, fontweight='bold')
#         axes[i].set_ylabel("Amplitude (µV)", fontsize=10)
#         axes[i].axvline(x=0, color='black', linestyle='--', label='Cue Starts')
#         axes[i].grid(True, alpha=0.3)
        
#         # Only put the legend on the top graph so it doesn't get cluttered
#         if i == 0:
#             axes[i].legend(loc='upper right', fontsize=10)

#     axes[-1].set_xlabel("Time (seconds)", fontsize=12)
#     plt.tight_layout()
    
#     file_name = "pipeline_multi_epoch_comparison.png"
#     plt.savefig(file_name, dpi=300)
#     print(f"✅ SAVED: {file_name}")
#     plt.show()

# if __name__ == "__main__":
#     generate_pipeline_comparison_plot()



#####################################################THE VISUAL PROOF (The Spatial Gradient / Volume Conduction)######################################################################
#     # ---------------------------------------------------------
#     # THE VISUAL PROOF (The Spatial Gradient / Volume Conduction)
#     # ---------------------------------------------------------
#     # Lock onto the epoch with the known blink
#     epoch_idx = 5 
    
#     # We will plot Front (Eyes), Middle (Motor), and Back (Visual)
#     channels_to_plot = ['Fz', 'Cz', 'Pz'] 
#     times = epochs_baseline.times
    
#     fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 10), sharex=True)
    
#     for i, ch_name in enumerate(channels_to_plot):
#         ch_idx = raw.ch_names.index(ch_name)
        
#         baseline_signal = epochs_baseline.get_data()[epoch_idx, ch_idx, :] * 1e6
#         custom_signal = epochs_custom.get_data()[epoch_idx, ch_idx, :] * 1e6
        
#         axes[i].plot(times, baseline_signal, color='red', alpha=0.7, linewidth=2, label='Baseline (Dirty)')
#         axes[i].plot(times, custom_signal, color='green', linewidth=2, label='Custom (Clean - Post ICA)')
        
#         # Add labels indicating WHERE on the head this is
#         region = "Frontal" if ch_name == 'Fz' else "Central (Motor)" if ch_name == 'Cz' else "Parietal (Back)"
#         axes[i].set_title(f"Subject 1 | Epoch #{epoch_idx} | Channel {ch_name} [{region}]", fontsize=12, fontweight='bold')
#         axes[i].set_ylabel("Amplitude (µV)", fontsize=10)
#         axes[i].axvline(x=0, color='black', linestyle='--', label='Cue Starts')
#         axes[i].grid(True, alpha=0.3)
        
#         # Keep the Y-axis scale identical across all plots to prove the variance drops off
#         axes[i].set_ylim([-20, 20]) 
        
#         if i == 0:
#             axes[i].legend(loc='upper right', fontsize=10)

#     axes[-1].set_xlabel("Time (seconds)", fontsize=12)
#     plt.tight_layout()
    
#     file_name = "pipeline_spatial_gradient.png"
#     plt.savefig(file_name, dpi=300)
#     print(f"✅ SAVED: {file_name}")
#     plt.show()

# if __name__ == "__main__":
#     generate_pipeline_comparison_plot()

#####################################################THE VISUAL PROOF (The Midline Spatial Gradient)######################################################################

    # ---------------------------------------------------------
    # THE VISUAL PROOF (The Midline Spatial Gradient)
    # ---------------------------------------------------------
    # Lock onto the epoch with the known blink
    epoch_idx = 5 
    
    # The Sagittal Midline Slice (Front to Back)
    channels_to_plot = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz'] 
    
    # A dictionary to automatically label the physical brain regions for the presentation
    region_labels = {
        'Fz': 'Frontal (Closest to Eyes)',
        'FCz': 'Frontal-Central',
        'Cz': 'Central (Motor Cortex)',
        'CPz': 'Centro-Parietal',
        'Pz': 'Parietal',
        'POz': 'Parieto-Occipital (Back of Head)'
    }
    
    times = epochs_baseline.times
    
    # Make the figure taller to comfortably fit 6 plots
    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 14), sharex=True)
    
    for i, ch_name in enumerate(channels_to_plot):
        ch_idx = raw.ch_names.index(ch_name)
        
        baseline_signal = epochs_baseline.get_data()[epoch_idx, ch_idx, :] * 1e6
        custom_signal = epochs_custom.get_data()[epoch_idx, ch_idx, :] * 1e6
        
        axes[i].plot(times, baseline_signal, color='red', alpha=0.7, linewidth=2, label='Baseline (Dirty)')
        axes[i].plot(times, custom_signal, color='green', linewidth=2, label='Custom (Clean - Post ICA)')
        
        region = region_labels[ch_name]
        axes[i].set_title(f"Subject 1 | Epoch #{epoch_idx} | Channel {ch_name} [{region}]", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Amplitude (µV)", fontsize=10)
        axes[i].axvline(x=0, color='black', linestyle='--', label='Cue Starts')
        axes[i].grid(True, alpha=0.3)
        
        # Lock the Y-axis scale to mathematically prove the variance drops off
        axes[i].set_ylim([-20, 20]) 
        
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=10)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()
    
    file_name = "pipeline_midline_gradient.png"
    plt.savefig(file_name, dpi=300)
    print(f"✅ SAVED: {file_name}")
    plt.show()

if __name__ == "__main__":
    generate_pipeline_comparison_plot()