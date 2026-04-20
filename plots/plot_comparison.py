import matplotlib.pyplot as plt
import numpy as np
import os

# Data derived from results.md Summary Table
models = [
    {"name": "Three-Class", "acc": 89.44, "group": "Ablation"},
    {"name": "Base Model", "acc": 90.08, "group": "Proposed"},
    {"name": "SSL FT (20%)", "acc": 91.96, "group": "Proposed"},
    {"name": "SSL FT (100%)", "acc": 93.03, "group": "Proposed"},
    {"name": "Nested CV", "acc": 93.09, "group": "Proposed"},
    {"name": "TimesFM+LoRA", "acc": 93.26, "group": "Foundation Model"},
    {"name": "SSL Linear Probe", "acc": 94.01, "group": "Proposed"},
    {"name": "TimesFM Zero-Shot", "acc": 86.40, "group": "Foundation Model"}
]

# Sort by accuracy
models = sorted(models, key=lambda x: x["acc"])

names = [m["name"] for m in models]
accs = [m["acc"] for m in models]
groups = [m["group"] for m in models]

plt.figure(figsize=(11, 6), facecolor="#fdfdfd")
ax = plt.gca()
ax.set_facecolor("#fdfdfd")

# Draw line
x = np.arange(len(names))
slope, intercept = np.polyfit(x, accs, 1)
trend_line = slope * x + intercept
plt.plot(x, trend_line, linestyle='--', color='#aeb6bf', zorder=1, alpha=0.8)

# Fill between actual line and trend line similar to user's plot
plt.fill_between(x, [accs[0]]*len(x), trend_line, color='#f2f4f4', alpha=0.5, zorder=0)

# Colors matching user's image
group_colors = {
    "Ablation": "#e74c3c",       
    "Proposed": "#2980b9",       
    "Foundation Model": "#27ae60" 
}

# Scatter
for i, m in enumerate(models):
    c = group_colors[m["group"]]
    plt.scatter(i, m["acc"], color=c, s=120, zorder=2, edgecolors='w', linewidth=1.5)
    
    # Adjust text position
    offset = -0.5
    va = 'top'
    plt.text(i, m["acc"] + offset, f'{m["acc"]:.1f}%', ha='center', va=va, fontsize=8, color='#424949', fontweight='medium')

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='#aeb6bf', linestyle='--', label=f'Trend (slope={slope:.2f})')]
for g in ["Ablation", "Proposed", "Foundation Model"]:
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors[g], markersize=10, label=g))

plt.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True, facecolor='white', edgecolor='#d5dbdb')

plt.xticks(x, names, rotation=40, ha='right', fontsize=9)
plt.ylabel('Combined Accuracy (%)', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#bdc3c7')
ax.spines['bottom'].set_color('#bdc3c7')
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.ylim(84, 96)
plt.tight_layout()

os.makedirs('d:/Documents/parkinsons/plots', exist_ok=True)
out_path = 'd:/Documents/parkinsons/plots/model_comparison_filtered.png'
plt.savefig(out_path, dpi=300)
print(f"Plot saved to: {out_path}")
