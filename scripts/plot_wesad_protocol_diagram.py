import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 2))

# Protocol segments (start, duration, label, color)
segments = [
    (0, 5, 'Baseline', '#8ecae6'),
    (5, 10, 'Stressor 1', '#ffb703'),
    (15, 5, 'Recovery 1', '#b7e4c7'),
    (20, 10, 'Stressor 2', '#fb8500'),
    (30, 5, 'Recovery 2', '#b7e4c7'),
    (35, 5, 'Amusement', '#219ebc'),
    (40, 5, 'Recovery 3', '#b7e4c7'),
]

for start, duration, label, color in segments:
    ax.barh(0, duration, left=start, height=0.5, color=color, edgecolor='k')
    ax.text(start + duration/2, 0, label, ha='center', va='center', fontsize=12, color='black')

# Timeline
ax.set_xlim(0, 45)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_title('WESAD Experimental Protocol', fontsize=14, weight='bold')

# Legend
legend_patches = [
    mpatches.Patch(color='#8ecae6', label='Baseline'),
    mpatches.Patch(color='#ffb703', label='Stressor'),
    mpatches.Patch(color='#fb8500', label='Stressor'),
    mpatches.Patch(color='#b7e4c7', label='Recovery'),
    mpatches.Patch(color='#219ebc', label='Amusement'),
]
ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig('results/advanced_figures/wesad_protocol_diagram.png', dpi=300)
plt.close()
print('WESAD protocol diagram saved to results/advanced_figures/wesad_protocol_diagram.png')
