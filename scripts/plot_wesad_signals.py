import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use first available WESAD subject
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

labels = data['label'].flatten()

# Find a baseline and a stress segment
baseline_idx = np.where(labels == 1)[0]
stress_idx = np.where(labels == 2)[0]

# Use first 2 minutes of each (sampling rate ~700Hz for chest, 64Hz for wrist)
window_sec = 120
bvp = data['signal']['wrist']['BVP'].flatten()
eda = data['signal']['wrist']['EDA'].flatten()
temp = data['signal']['wrist']['TEMP'].flatten()
acc = data['signal']['wrist']['ACC']

bvp_sr = 64
eda_sr = 4
temp_sr = 4
acc_sr = 32

# Get baseline and stress windows (start at first index of each)
def get_window(arr, idx, sr, sec=120):
    if len(idx) == 0:
        return np.array([])
    start = idx[0]
    return arr[start:start+sr*sec]

bvp_base = get_window(bvp, baseline_idx, bvp_sr)
bvp_stress = get_window(bvp, stress_idx, bvp_sr)
eda_base = get_window(eda, baseline_idx, eda_sr)
eda_stress = get_window(eda, stress_idx, eda_sr)
temp_base = get_window(temp, baseline_idx, temp_sr)
temp_stress = get_window(temp, stress_idx, temp_sr)
acc_base = get_window(acc[:,0], baseline_idx, acc_sr), get_window(acc[:,1], baseline_idx, acc_sr), get_window(acc[:,2], baseline_idx, acc_sr)
acc_stress = get_window(acc[:,0], stress_idx, acc_sr), get_window(acc[:,1], stress_idx, acc_sr), get_window(acc[:,2], stress_idx, acc_sr)

fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=False)

# BVP/HR
axs[0,0].plot(bvp_base, color='tab:blue')
axs[0,0].set_title('Baseline: PPG/BVP')
axs[0,1].plot(bvp_stress, color='tab:red')
axs[0,1].set_title('Stress: PPG/BVP')

# EDA
axs[1,0].plot(eda_base, color='tab:blue')
axs[1,0].set_title('Baseline: EDA')
axs[1,1].plot(eda_stress, color='tab:red')
axs[1,1].set_title('Stress: EDA')

# Temp
axs[2,0].plot(temp_base, color='tab:blue')
axs[2,0].set_title('Baseline: Temp')
axs[2,1].plot(temp_stress, color='tab:red')
axs[2,1].set_title('Stress: Temp')

# ACC
axs[3,0].plot(acc_base[0], label='X', alpha=0.7)
axs[3,0].plot(acc_base[1], label='Y', alpha=0.7)
axs[3,0].plot(acc_base[2], label='Z', alpha=0.7)
axs[3,0].set_title('Baseline: ACC')
axs[3,0].legend()
axs[3,1].plot(acc_stress[0], label='X', alpha=0.7)
axs[3,1].plot(acc_stress[1], label='Y', alpha=0.7)
axs[3,1].plot(acc_stress[2], label='Z', alpha=0.7)
axs[3,1].set_title('Stress: ACC')
axs[3,1].legend()

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"Physiological Signals (Subject: {subject_file.parent.name})", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/advanced_figures/wesad_signals_baseline_vs_stress.png', dpi=300)
plt.close()
print('Physiological signals figure saved to results/advanced_figures/wesad_signals_baseline_vs_stress.png')
