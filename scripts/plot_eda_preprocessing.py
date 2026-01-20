import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

eda = data['signal']['wrist']['EDA'].flatten()
eda_sr = 4

# Simulate artifacts: randomly set 10% of points in a window to NaN
def simulate_artifact(signal, frac=0.1, window=400):
    np.random.seed(42)
    start = 1000
    end = start + window
    artifact_signal = signal.copy()
    idx = np.arange(start, end)
    n_art = int(frac * window)
    artifact_idx = np.random.choice(idx, n_art, replace=False)
    artifact_signal[artifact_idx] = np.nan
    return artifact_signal, artifact_idx

eda_art, artifact_idx = simulate_artifact(eda)

# Interpolate to fill gaps
def interpolate_signal(signal):
    nans = np.isnan(signal)
    x = np.arange(len(signal))
    signal_interp = signal.copy()
    if np.any(nans):
        signal_interp[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return signal_interp

eda_interp = interpolate_signal(eda_art)

# Plot
fig, axs = plt.subplots(1, 1, figsize=(10, 4))
window = slice(900, 1500)
axs.plot(eda[window], label='Original EDA', color='tab:blue')
axs.plot(eda_art[window], label='With Artifacts', color='tab:red', linestyle='--', alpha=0.7)
axs.plot(eda_interp[window], label='Interpolated', color='tab:green', linestyle=':')
axs.scatter(artifact_idx - 900, eda_interp[artifact_idx], color='black', s=20, label='Artifact Points', zorder=5)
axs.set_title('EDA Signal Preprocessing: Artifact Removal via Interpolation')
axs.set_xlabel('Sample Index (windowed)')
axs.set_ylabel('EDA (μS)')
axs.legend()
plt.tight_layout()
plt.savefig('results/advanced_figures/eda_preprocessing_artifact_interpolation.png', dpi=300)
plt.close()
print('EDA preprocessing figure saved to results/advanced_figures/eda_preprocessing_artifact_interpolation.png')
