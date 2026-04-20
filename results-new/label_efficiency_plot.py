import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ──────────────────────────────────────────────
# Data Arrays
# ──────────────────────────────────────────────
pct = np.array([5, 10, 20, 50, 70, 100])

# SSL Hardcoded Arrays
ssl_hc_pd_mean = np.array([0.7939, 0.8703, 0.9226, 0.9363, 0.9344, 0.9356])
ssl_hc_pd_std  = np.array([0.0287, 0.0110, 0.0057, 0.0070, 0.0025, 0.0004])

ssl_pd_dd_mean = np.array([0.7878, 0.8591, 0.9167, 0.9238, 0.9213, 0.9250])
ssl_pd_dd_std  = np.array([0.0235, 0.0129, 0.0014, 0.0031, 0.0035, 0.0001])

# Supervised Hardcoded Arrays
sup_hc_pd_mean = np.array([0.8412, 0.8748, 0.9103, 0.9352, 0.9360, 0.9299])
sup_hc_pd_std  = np.array([0.0280, 0.0239, 0.0262, 0.0010, 0.0012, 0.0120])

sup_pd_dd_mean = np.array([0.8470, 0.8730, 0.9013, 0.9255, 0.9256, 0.9209])
sup_pd_dd_std  = np.array([0.0131, 0.0213, 0.0192, 0.0007, 0.0013, 0.0086])


# ──────────────────────────────────────────────
# Plot (1x2 Subplots)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

SSL_COLOR = "#2563EB"   # blue
SUP_COLOR = "#DC2626"   # red

def plot_ax(ax, title, ssl_m, ssl_s, sup_m, sup_s):
    # SSL curve
    ax.plot(pct, ssl_m, color=SSL_COLOR, linewidth=2.3, marker="o",
            markersize=7, label="SSL fine-tune", zorder=4)
    ax.fill_between(pct, ssl_m - ssl_s, ssl_m + ssl_s,
                    alpha=0.13, color=SSL_COLOR, zorder=2)
    
    # Supervised curve
    ax.plot(pct, sup_m, color=SUP_COLOR, linewidth=2.3, marker="s",
            markersize=7, label="Supervised baseline", zorder=4)
    ax.fill_between(pct, sup_m - sup_s, sup_m + sup_s,
                    alpha=0.13, color=SUP_COLOR, zorder=2)

    # Annotate all points cleanly
    for i in range(len(pct)):
        x = pct[i]
        
        # We adjust horizontal alignment based on whether the curve is on top or bottom
        # to avoid overlapping labels on the line itself.
        ax.annotate(f"{ssl_m[i]:.3f}",
                    xy=(x, ssl_m[i]),
                    xytext=(x - 1, ssl_m[i] + 0.010),
                    fontsize=8.5, color=SSL_COLOR, fontweight="bold", ha="center")
        ax.annotate(f"{sup_m[i]:.3f}",
                    xy=(x, sup_m[i]),
                    xytext=(x + 1, sup_m[i] - 0.014),
                    fontsize=8.5, color=SUP_COLOR, fontweight="bold", ha="center")

    ax.set_xlabel("Percentage of Labelled Training Data (%)", fontsize=11)
    ax.set_ylabel("Accuracy (mean over folds)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    
    # Set explicit ticks to override default matplotlib ticks that might show '0'
    ax.set_xticks(pct)
    ax.set_xticklabels([f"{p}%" for p in pct], fontsize=10)
    
    # Use explicit xlim so 0 isn't added by default as a major tick
    ax.set_xlim(0, 105)
    
    min_val = min(np.min(ssl_m - ssl_s), np.min(sup_m - sup_s))
    ax.set_ylim(max(0.60, min_val - 0.02), 0.98)
    
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=10.5, framealpha=0.92, loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)

# -- Plot HC vs PD --
plot_ax(axes[0], "HC vs PD - Label Efficiency", 
        ssl_hc_pd_mean, ssl_hc_pd_std, 
        sup_hc_pd_mean, sup_hc_pd_std)

# -- Plot PD vs DD --
plot_ax(axes[1], "PD vs DD - Label Efficiency", 
        ssl_pd_dd_mean, ssl_pd_dd_std, 
        sup_pd_dd_mean, sup_pd_dd_std)

plt.tight_layout(w_pad=2.5)

base_path = Path(__file__).parent
out = base_path / "label_efficiency_curve.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved -> {out}")

out_png = out.with_suffix(".png")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved -> {out_png}")
