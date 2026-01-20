import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

# Simple EDA decomposition: lowpass for tonic, subtract for phasic
wesad_dir = Path("data/wesad")
subject_dirs = sorted([d for d in wesad_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
subject_file = None
for d in subject_dirs:
    pkl_files = list(d.glob("*.pkl"))
    if pkl_files:
        subject_file = pkl_files[0]
        break

if subject_file is None:
    raise FileNotFoundError("No WESAD subject .pkl file found.")

with open(subject_file, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

eda = data['signal']['wrist']['EDA'].flatten()
eda_sr = 4

# Use a 5-minute window for clarity
window = slice(1000, 1000 + 5*60*eda_sr)
eda_win = eda[window]

# Butterworth lowpass filter for tonic (cutoff 0.05 Hz)
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

eda_tonic = lowpass_filter(eda_win, cutoff=0.05, fs=eda_sr)
eda_phasic = eda_win - eda_tonic

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(eda_win, label='Raw EDA', color='#219ebc', lw=2)
ax.plot(eda_tonic, label='Tonic (Lowpass)', color='#8ecae6', lw=2)
ax.plot(eda_phasic, label='Phasic (Raw - Tonic)', color='#fb8500', lw=2)
ax.set_title('EDA Decomposition: Tonic and Phasic Components', fontsize=15, weight='bold')
ax.set_xlabel('Sample Index (5 min window)')
ax.set_ylabel('EDA (μS)')
ax.legend()
plt.tight_layout()
plt.savefig('results/advanced_figures/eda_decomposition.png', dpi=300)
plt.close()
print('EDA decomposition figure saved to results/advanced_figures/eda_decomposition.png')
