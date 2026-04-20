import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────
# Model data  — sourced from results.md Summary Table
# ──────────────────────────────────────────────────────────────────

models = [
    # name                          hcpd_acc   pddd_acc   group
    ("BaseModel (no BandPass)",     0.8598,    0.8539,    "Ablation"),
    ("Three-Class",                 None,      None,      "Ablation"),   # 3-class; skip per-task
    ("TimesFM Zero-shot (KNN)",     0.8034,    0.7663,    "Foundation Model"),
    ("TimesFM Zero-shot (LogReg)",  0.8739,    0.8542,    "Foundation Model"),
    ("Base Model + BandPass",       0.9312,    0.8704,    "Proposed"),
    ("SSL FT (20%)",                0.9226,    0.9167,    "Proposed"),
    ("SSL FT (100%)",               0.9356,    0.9250,    "Proposed"),
    ("Supervised Baseline (100%)",  None,      None,      "Proposed"),   # combined only
    ("SSL Linear Probe (100%)",     0.9412,    0.9390,    "Proposed"),
    ("Nested CV Transformer",       0.9361,    0.9257,    "Proposed"),
    ("TimesFM+LoRA",                0.9362,    0.9289,    "Foundation Model"),
    ("CNN Task-Wise (avg)",         0.8721,    0.8282,    "Ablation"),
    ("LSTM Task-Wise (avg)",        0.9280,    0.8917,    "Proposed"),
]

# Filter to only models that have per-task scores
models = [(n, h, p, g) for (n, h, p, g) in models if h is not None and p is not None]

# ──────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────
group_colors = {
    "Ablation":         "#e74c3c",
    "Proposed":         "#2980b9",
    "Foundation Model": "#27ae60",
}

# ──────────────────────────────────────────────────────────────────
# Shared helper — draw one panel
# ──────────────────────────────────────────────────────────────────
def draw_panel(ax, models_sorted, acc_key, ylabel, title):
    """
    models_sorted : list of (name, hcpd, pddd, group) already sorted by acc_key
    acc_key       : 'hcpd' or 'pddd'
    """
    names = [m[0] for m in models_sorted]
    accs  = [m[1] if acc_key == "hcpd" else m[2] for m in models_sorted]
    groups = [m[3] for m in models_sorted]

    x = np.arange(len(names))

    # Trend line
    slope, intercept = np.polyfit(x, accs, 1)
    trend = slope * x + intercept
    ax.plot(x, trend, linestyle="--", color="#aeb6bf", zorder=1, alpha=0.85, linewidth=1.4)

    # Subtle fill between baseline and trend
    ax.fill_between(x, [min(accs)] * len(x), trend,
                    color="#f2f4f4", alpha=0.55, zorder=0)

    # Scatter points
    for i, m in enumerate(models_sorted):
        acc = accs[i]
        c = group_colors[m[3]]
        ax.scatter(i, acc, color=c, s=130, zorder=3,
                   edgecolors="white", linewidth=1.6)
        # Label: place below the point
        ax.text(i, acc - 0.004, f"{acc*100:.2f}%",
                ha="center", va="top", fontsize=7.8,
                color="#2c3e50", fontweight="medium")

    # Legend (inside each panel)
    legend_els = [
        Line2D([0], [0], color="#aeb6bf", linestyle="--", linewidth=1.4,
               label=f"Trend (slope={slope*100:.2f} pp/step)")
    ]
    for g in ["Ablation", "Proposed", "Foundation Model"]:
        legend_els.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=group_colors[g],
                   markersize=9, label=g)
        )
    ax.legend(handles=legend_els, loc="upper left",
              fontsize=8.5, frameon=True,
              facecolor="white", edgecolor="#d5dbdb",
              framealpha=0.92)

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)

    y_lo = max(0.75, min(accs) - 0.025)
    y_hi = max(accs) + 0.025
    ax.set_ylim(y_lo, y_hi)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdc3c7")
    ax.spines["bottom"].set_color("#bdc3c7")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_facecolor("#fdfdfd")


# ──────────────────────────────────────────────────────────────────
# Build figure
# ──────────────────────────────────────────────────────────────────
fig, (ax_hcpd, ax_pddd) = plt.subplots(1, 2, figsize=(22, 5.2),
                                        facecolor="#fdfdfd",
                                        gridspec_kw={"wspace": 0.28})
fig.patch.set_facecolor("#fdfdfd")

# Sort each panel independently by its own accuracy
sorted_hcpd = sorted(models, key=lambda m: m[1])
sorted_pddd  = sorted(models, key=lambda m: m[2])

draw_panel(ax_hcpd, sorted_hcpd,
           acc_key="hcpd",
           ylabel="HC vs PD Accuracy",
           title="HC vs PD — Model Comparison")

draw_panel(ax_pddd, sorted_pddd,
           acc_key="pddd",
           ylabel="PD vs DD Accuracy",
           title="PD vs DD — Model Comparison")

plt.suptitle("Per-Task Model Comparison",
             fontsize=15, fontweight="bold", y=1.01, color="#1a252f")

plt.tight_layout(rect=[0, 0, 1, 0.98])

# ──────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────
os.makedirs("d:/Documents/parkinsons/plots", exist_ok=True)
out_png = "d:/Documents/parkinsons/plots/model_comparison_dual.png"
out_pdf = "d:/Documents/parkinsons/plots/model_comparison_dual.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
print("Saved: " + out_png)
print("Saved: " + out_pdf)
