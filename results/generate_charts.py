"""Generate the worklog's benchmark charts from measured H100 data.

Palette is the dataviz skill's validated categorical set (colorblind-safe,
worst-adjacent CVD ΔE 47). Every bar carries a direct value label (satisfies the
contrast-relief rule), grid is recessive, spines trimmed.

    python results/generate_charts.py   # writes results/*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OURS, TORCH, VLLM = "#2a78d6", "#eda100", "#1baf7a"   # blue / yellow / aqua
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=12, loc="left")
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def label_bars(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)


# ── Chart 1: memory-bound kernels, % of peak HBM, ours vs PyTorch vs vLLM ──
fig, ax = plt.subplots(figsize=(8, 4.4))
kernels = ["RMSNorm", "LayerNorm"]
ours   = [82, 80]
torch_ = [20, 65]
vllm   = [82, None]
x = range(len(kernels)); w = 0.26
b1 = ax.bar([i - w for i in x], ours, w, label="inferneo (ours)", color=OURS, zorder=3)
b2 = ax.bar([i for i in x], torch_, w, label="PyTorch", color=TORCH, zorder=3)
b3 = ax.bar([i + w for i in x if vllm[i] is not None], [v for v in vllm if v is not None],
            w, label="vLLM (fused)", color=VLLM, zorder=3)
ax.axhline(100, color=MUTED, linestyle="--", linewidth=1, zorder=2)
ax.text(len(kernels) - 0.5, 101, "peak HBM (100%)", ha="right", va="bottom",
        fontsize=9, color=MUTED)
label_bars(ax, b1, "{:.0f}%"); label_bars(ax, b2, "{:.0f}%"); label_bars(ax, b3, "{:.0f}%")
ax.set_xticks(list(x)); ax.set_xticklabels(kernels)
ax.set_ylim(0, 112); ax.yaxis.set_major_formatter(PercentFormatter())
style(ax, "Memory-bound kernels reach the HBM ceiling", "% of peak HBM bandwidth")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig("results/memory_bound.png", dpi=150, bbox_inches="tight")

# ── Chart 2: SiLU optimization progression ──
fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(["v1\nscalar", "v2\nfloat4"], [52, 89], 0.5,
              color=["#9ec3ee", OURS], zorder=3)   # light→full blue = progression
ax.axhline(100, color=MUTED, linestyle="--", linewidth=1, zorder=2)
label_bars(ax, bars, "{:.0f}%")
ax.set_ylim(0, 112); ax.yaxis.set_major_formatter(PercentFormatter())
style(ax, "SiLU: float4 vectorization → +1.7×", "% of peak HBM bandwidth")
fig.tight_layout()
fig.savefig("results/silu.png", dpi=150, bbox_inches="tight")

# ── Chart 3: FlashAttention TFLOP/s ──
fig, ax = plt.subplots(figsize=(7, 4.4))
labels = ["v1\nper-query", "v2\ntiled-smem", "PyTorch SDPA\n(FlashAttn-2)"]
vals = [2.8, 7.9, 19.6]
colors = ["#9ec3ee", OURS, TORCH]
bars = ax.bar(labels, vals, 0.55, color=colors, zorder=3)
label_bars(ax, bars, "{:.1f}")
ax.set_ylim(0, 22)
style(ax, "FlashAttention: shared-memory tiling → 2.8×", "TFLOP/s (higher is better)")
ax.text(0.0, -0.28, "ours: fp32, no tensor cores · both exact to 4.9e-8 · 0 GB for scores (vs 0.5 GB naive)",
        transform=ax.transAxes, fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig("results/flash_attention.png", dpi=150, bbox_inches="tight")

print("wrote results/memory_bound.png, results/silu.png, results/flash_attention.png")
