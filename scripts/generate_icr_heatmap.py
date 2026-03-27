"""
Interest rate sensitivity heatmap (Seaborn + Matplotlib).
SOFR scenarios x leverage multiples -> illustrative ICR grid.
Run from repo root: python scripts/generate_icr_heatmap.py
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT = os.path.join("outputs", "graphprompts", "icr_sensitivity_heatmap.png")


def main() -> None:
    sofr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    lev = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])

    # Synthetic ICR: falls with leverage and with higher base rates
    grid = np.zeros((len(lev), len(sofr)))
    for i, L in enumerate(lev):
        for j, r in enumerate(sofr):
            grid[i, j] = 4.2 - 0.35 * (L - 3) - 0.22 * (r - 3) + np.sin(i + j * 0.4) * 0.08

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    cmap = sns.blend_palette(["#f5f0e6", "#9ca3af", "#7f1d1d"], as_cmap=True)

    sns.heatmap(
        grid,
        xticklabels=[f"{x:.1f}%" for x in sofr],
        yticklabels=[f"{x:.1f}x" for x in lev],
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.35,
        linecolor="#e5e5e5",
        cbar_kws={"label": "Interest coverage (x)"},
        ax=ax,
        annot_kws={"size": 7, "family": "monospace"},
    )

    ax.set_xlabel("SOFR scenario", fontsize=11, labelpad=8)
    ax.set_ylabel("Borrower leverage (EBITDA multiple)", fontsize=11, labelpad=8)
    ax.set_title(
        "PORTFOLIO INTEREST COVERAGE — RATE × LEVERAGE",
        loc="left",
        fontsize=14,
        fontweight=600,
        pad=16,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
