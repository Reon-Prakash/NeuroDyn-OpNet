"""
visualize_results.py
--------------------
Publication-Quality LOSO Cross-Validation Bar Chart
NeuroDyn-OpNet EEG Research Results

Output : loso_performance.png  (300 DPI)

Author : NeuroDyn-OpNet Visualization Pipeline
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.max_open_warning"] = 0

# ──────────────────────────────────────────────────────────────────────────────
# 0.  WINDOWS CONSOLE ENCODING FIX
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

ROOT_PATH = r"C:\Users\KIIT0001\.vscode\Codes\NeuroDyn-OpNet"
os.chdir(ROOT_PATH)
print(f"[INFO] Working directory : {os.getcwd()}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA
# ──────────────────────────────────────────────────────────────────────────────

subjects   = ["S1",  "S2",  "S3",  "S4",  "S5",
              "S6",  "S7",  "S8",  "S9",  "S10", "S11"]

accuracies = [70.05, 77.86, 70.47, 71.43, 90.13,
              93.94, 59.41, 79.85, 72.20, 73.83, 67.11]

mean_acc   = 75.12   # pre-computed mean
std_acc    =  9.49   # pre-computed std

n_subjects = len(subjects)
x_pos      = np.arange(n_subjects)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.edgecolor"    : "#2d2d2d",
    "axes.linewidth"    : 1.2,
    "grid.color"        : "#e0e0e0",
    "grid.linewidth"    : 0.8,
    "grid.linestyle"    : "--",
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "xtick.major.size"  : 5,
    "ytick.major.size"  : 5,
    "xtick.color"       : "#2d2d2d",
    "ytick.color"       : "#2d2d2d",
})

# ──────────────────────────────────────────────────────────────────────────────
# 4.  COLOR PALETTE  (viridis — maps accuracy to color intensity)
# ──────────────────────────────────────────────────────────────────────────────

# Normalize accuracies to [0, 1] for colormap mapping
norm_vals  = (np.array(accuracies) - min(accuracies)) / \
             (max(accuracies) - min(accuracies))
cmap       = plt.cm.get_cmap("viridis", n_subjects)
bar_colors = [cmap(v) for v in norm_vals]

# ──────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE SETUP
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  BAR CHART
# ──────────────────────────────────────────────────────────────────────────────

bars = ax.bar(
    x_pos,
    accuracies,
    width     = 0.58,
    color     = bar_colors,
    edgecolor = "#2d2d2d",
    linewidth = 0.9,
    zorder    = 3,
)

# ──────────────────────────────────────────────────────────────────────────────
# 7.  MEAN ACCURACY REFERENCE LINE
# ──────────────────────────────────────────────────────────────────────────────

ax.axhline(
    y          = mean_acc,
    color      = "#e63946",
    linestyle  = "--",
    linewidth  = 2.0,
    zorder     = 4,
    label      = f"Mean Accuracy = {mean_acc:.2f}%",
)

# Shade ± 1 std band around mean
ax.axhspan(
    mean_acc - std_acc,
    mean_acc + std_acc,
    alpha     = 0.08,
    color     = "#e63946",
    zorder    = 2,
    label     = f"Mean +/- Std ({mean_acc-std_acc:.2f}% - "
                f"{mean_acc+std_acc:.2f}%)",
)

# ──────────────────────────────────────────────────────────────────────────────
# 8.  BAR ANNOTATIONS  (value on top of each bar)
# ──────────────────────────────────────────────────────────────────────────────

for bar, acc in zip(bars, accuracies):
    bar_x = bar.get_x() + bar.get_width() / 2
    bar_h = bar.get_height()

    # Choose text color based on bar darkness
    norm_v     = (acc - min(accuracies)) / (max(accuracies) - min(accuracies))
    text_color = "#ffffff" if norm_v > 0.65 else "#1a1a1a"

    # Value inside bar (bottom area)
    ax.text(
        bar_x,
        bar_h - 3.5,
        f"{acc:.2f}%",
        ha          = "center",
        va          = "top",
        fontsize    = 9.5,
        fontweight  = "bold",
        color       = text_color,
        zorder      = 5,
    )

    # Small upward tick line above bar
    ax.annotate(
        "",
        xy        = (bar_x, bar_h + 0.4),
        xytext    = (bar_x, bar_h),
        arrowprops= dict(arrowstyle="-", color="#555555", lw=0.8),
        zorder    = 5,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 9.  STATISTICS TEXT BOX
# ──────────────────────────────────────────────────────────────────────────────

stats_text = (
    f"  Statistics (LOSO)  \n"
    f"  Mean    :  {mean_acc:.2f}%  \n"
    f"  Std Dev :  {std_acc:.2f}%  \n"
    f"  Min     :  {min(accuracies):.2f}%  \n"
    f"  Max     :  {max(accuracies):.2f}%  \n"
    f"  N Folds :  {n_subjects}"
)

props = dict(
    boxstyle = "round,pad=0.6",
    facecolor = "#ffffff",
    edgecolor = "#aaaaaa",
    alpha     = 0.88,
    linewidth = 1.2
)

ax.text(
    0.985, 0.975,
    stats_text,
    transform           = ax.transAxes,
    fontsize            = 9.8,
    verticalalignment   = "top",
    horizontalalignment = "right",
    bbox                = props,
    family              = "monospace",
    color               = "#2d2d2d",
    zorder              = 6,
)

# ──────────────────────────────────────────────────────────────────────────────
# 10.  COLORBAR  (performance scale legend)
# ──────────────────────────────────────────────────────────────────────────────

sm   = plt.cm.ScalarMappable(
    cmap  = plt.cm.viridis,
    norm  = plt.Normalize(vmin=min(accuracies), vmax=max(accuracies))
)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax           = ax,
    orientation  = "vertical",
    fraction     = 0.028,
    pad          = 0.02,
    aspect       = 28,
)
cbar.set_label(
    "Test Accuracy (%)",
    fontsize   = 10,
    labelpad   = 10,
    color      = "#2d2d2d"
)
cbar.ax.tick_params(labelsize=9, colors="#2d2d2d")
cbar.outline.set_edgecolor("#aaaaaa")

# ──────────────────────────────────────────────────────────────────────────────
# 11.  AXES FORMATTING
# ──────────────────────────────────────────────────────────────────────────────

ax.set_xticks(x_pos)
ax.set_xticklabels(
    subjects,
    fontsize   = 12,
    fontweight = "semibold",
    color      = "#2d2d2d"
)

ax.set_ylim(50, 101)
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.tick_params(axis="y", which="minor", length=3, color="#cccccc")
ax.set_yticklabels(
    [f"{int(t)}%" for t in ax.get_yticks()],
    fontsize = 11,
    color    = "#2d2d2d"
)

ax.set_xlabel(
    "Subject ID",
    fontsize   = 13,
    fontweight = "bold",
    labelpad   = 10,
    color      = "#2d2d2d"
)
ax.set_ylabel(
    "Test Accuracy (%)",
    fontsize   = 13,
    fontweight = "bold",
    labelpad   = 10,
    color      = "#2d2d2d"
)

ax.set_title(
    "NeuroDyn-OpNet: LOSO Cross-Validation Results\n"
    "SADT Dataset  |  11-Subject Leave-One-Out Evaluation",
    fontsize   = 15,
    fontweight = "bold",
    color      = "#1a1a1a",
    pad        = 18,
)

# ──────────────────────────────────────────────────────────────────────────────
# 12.  LEGEND
# ──────────────────────────────────────────────────────────────────────────────

legend_handles = [
    mpatches.Patch(
        facecolor  = "#e63946",
        alpha      = 0.9,
        linestyle  = "--",
        edgecolor  = "#e63946",
        label      = f"Mean Accuracy = {mean_acc:.2f}%"
    ),
    mpatches.Patch(
        facecolor  = "#e63946",
        alpha      = 0.15,
        edgecolor  = "#e63946",
        label      = f"Mean +/- 1 Std  "
                     f"({mean_acc - std_acc:.2f}% - "
                     f"{mean_acc + std_acc:.2f}%)"
    ),
]

ax.legend(
    handles    = legend_handles,
    loc        = "upper left",
    fontsize   = 9.5,
    framealpha = 0.88,
    edgecolor  = "#aaaaaa",
    facecolor  = "#ffffff",
    borderpad  = 0.8,
)

# ──────────────────────────────────────────────────────────────────────────────
# 13.  BEST / WORST SUBJECT CALLOUTS
# ──────────────────────────────────────────────────────────────────────────────

best_idx  = int(np.argmax(accuracies))
worst_idx = int(np.argmin(accuracies))

# Best subject arrow annotation
ax.annotate(
    f"Best\n{subjects[best_idx]}: {accuracies[best_idx]:.2f}%",
    xy           = (x_pos[best_idx], accuracies[best_idx]),
    xytext       = (x_pos[best_idx] - 1.5, accuracies[best_idx] + 3),
    fontsize     = 9,
    fontweight   = "bold",
    color        = "#2b9348",
    arrowprops   = dict(
        arrowstyle  = "->",
        color       = "#2b9348",
        lw          = 1.4,
        connectionstyle = "arc3,rad=-0.2"
    ),
    bbox = dict(
        boxstyle    = "round,pad=0.3",
        facecolor   = "#d8f3dc",
        edgecolor   = "#2b9348",
        alpha       = 0.85
    ),
    zorder = 7,
)

# Worst subject arrow annotation
ax.annotate(
    f"Worst\n{subjects[worst_idx]}: {accuracies[worst_idx]:.2f}%",
    xy           = (x_pos[worst_idx], accuracies[worst_idx]),
    xytext       = (x_pos[worst_idx] + 1.0, accuracies[worst_idx] - 5),
    fontsize     = 9,
    fontweight   = "bold",
    color        = "#c1121f",
    arrowprops   = dict(
        arrowstyle  = "->",
        color       = "#c1121f",
        lw          = 1.4,
        connectionstyle = "arc3,rad=0.2"
    ),
    bbox = dict(
        boxstyle    = "round,pad=0.3",
        facecolor   = "#ffe5e5",
        edgecolor   = "#c1121f",
        alpha       = 0.85
    ),
    zorder = 7,
)

# ──────────────────────────────────────────────────────────────────────────────
# 14.  ABOVE / BELOW MEAN INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    bar_x = bar.get_x() + bar.get_width() / 2
    bar_h = bar.get_height()

    if acc >= mean_acc:
        marker_color = "#2b9348"
        marker       = "^"
        offset       = bar_h + 1.2
    else:
        marker_color = "#c1121f"
        marker       = "v"
        offset       = bar_h + 1.2

    ax.plot(
        bar_x, offset,
        marker     = marker,
        color      = marker_color,
        markersize = 6,
        zorder     = 5,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 15.  LAYOUT & SAVE
# ──────────────────────────────────────────────────────────────────────────────

plt.tight_layout(pad=2.0)

out_path = os.path.join(ROOT_PATH, "loso_performance.png")
fig.savefig(
    out_path,
    dpi         = 300,
    bbox_inches = "tight",
    facecolor   = fig.get_facecolor(),
    edgecolor   = "none",
    format      = "png"
)

print(f"[SAVE] Plot saved -> {out_path}")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 16.  CONSOLE SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 52)
print("  LOSO Results Summary")
print("=" * 52)
print(f"  {'Subject':<10}  {'Accuracy':>10}  {'vs Mean':>10}")
print("-" * 52)
for subj, acc in zip(subjects, accuracies):
    delta = acc - mean_acc
    flag  = "(+)" if delta >= 0 else "(-)"
    print(f"  {subj:<10}  {acc:>9.2f}%  {delta:>+9.2f}%  {flag}")
print("-" * 52)
print(f"  {'Mean':<10}  {mean_acc:>9.2f}%")
print(f"  {'Std Dev':<10}  {std_acc:>9.2f}%")
print(f"  {'Min':<10}  {min(accuracies):>9.2f}%  ({subjects[int(np.argmin(accuracies))]})")
print(f"  {'Max':<10}  {max(accuracies):>9.2f}%  ({subjects[int(np.argmax(accuracies))]})")
print("=" * 52)