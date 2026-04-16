"""
FPGA_Fidelity — ECoG Grid Spatial-Fidelity Verification
========================================================

Purpose
-------
Verify that the FPGA ECoG grid remains centred over the sagittal sinus
across the 21-day corticosterone paradigm.

Approach
--------
The sagittal sinus bisects the brain midline.  Electrodes directly over
the sinus are farthest from cortical tissue, so high-frequency (30–100 Hz)
neural activity is maximally attenuated there — the sinus appears as a
*trough* in the high-gamma power profile across the electrode columns.

For each sub-grid (rostral & caudal) we sample the **first row** of 16
electrodes and compare the high-gamma depth profile between Day 0 (BL)
and Day 21 (MDD).  The figure overlays both days:

    • Blue shaded band  = expected midline (centre columns 7–8)
    • Orange shaded band = detected midline (column with minimum high-γ)

If the trough has not shifted columns, the grid placement is stable.

Grid layout
-----------
Row-major, 16 × 16 per sub-grid, 200 µm inter-electrode pitch:
    • Caudal  sub-grid: channels   0–255  (RSP / VIS territory)
    • Rostral sub-grid: channels 256–511  (MO / ACA territory)
First row of each = the 16 channels with the lowest row index (row 0).

Usage
-----
    python FPGA_Fidelity.py
    >> Which animal do you want to verify spatial fidelity from? RB15

Output
------
Composite PNG saved in Results/ subfolder.

Author
------
Generated for Raz's DMN electrophysiology PhD project (JoVE reviewer response).
"""

from __future__ import annotations

import os
import re
import glob
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import mne
from scipy import signal as sig
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """All tuneable parameters for FPGA Fidelity verification."""

    # --- Paths ---
    data_dir: str = ""
    output_dir: str = ""

    # --- Recording ---
    sample_rate: float = 250.0
    t_start_s: float = 0.0
    t_end_s: float = 120.0

    # --- Grid geometry ---
    n_cols: int = 16          # electrodes per row
    n_rows: int = 16          # rows per sub-grid
    inter_electrode_um: int = 200  # µm pitch

    # First-row channel mapping (col 0 = leftmost software column):
    #   Rostral sub-grid row 0:  ch 256, 257, … 271  (ascending)
    #   Caudal sub-grid row 0:   ch 0, 1, … 15       (ascending)
    #
    # Note: the caudal sub-grid's internal wiring maps the physical midline
    # to a different software column than the rostral grid.  This is fine —
    # the tool's purpose is to verify STABILITY (same trough position on
    # Day 0 vs Day 21), not absolute centering.  Empirically, ch 0-15 gives
    # cross-day r ≈ 0.99 on well-centred mice — the correct first row.
    rostral_row0_start: int = 256
    rostral_row0_step: int = 1      # +1 = ascending
    caudal_row0_start: int = 0
    caudal_row0_step: int = 1       # +1 = ascending

    # --- Expected midline (0-indexed centre columns) ---
    expected_midline_cols: Tuple[int, int] = (7, 8)

    # --- Midline search window (inner columns only, avoids edge artefacts) ---
    # The craniotomy is cut to the grid dimensions, so the sinus is always
    # within a few columns of centre — never at the periphery.
    midline_search_cols: Tuple[int, int] = (3, 12)  # inclusive bounds

    # --- High-gamma band for sinus detection ---
    hg_lo: float = 30.0
    hg_hi: float = 100.0

    # --- Welch PSD ---
    welch_nperseg_s: float = 2.0
    welch_overlap_frac: float = 0.5

    # --- Plotting ---
    dpi: int = 300
    palette_day0: str = "#2196F3"   # blue
    palette_day21: str = "#F44336"  # red


# ============================================================================
# File Parsing
# ============================================================================

def _parse_metadata(fpath: str) -> Optional[Tuple[str, str, str, str]]:
    """Extract animal_id, probe, group, timepoint from filename."""
    fname = os.path.basename(fpath).lower()

    match = re.search(r'_(rb[a-z0-9]+)_', fname)
    if not match:
        match = re.search(r'(rb[a-z0-9]+)', fname)
    if not match:
        return None
    animal_id = match.group(1).upper()

    if "probea" in fname:
        probe = "ProbeA"
    elif "probeb" in fname:
        probe = "ProbeB"
    else:
        probe = "FPGA"

    basename = os.path.splitext(os.path.basename(fpath))[0].lower()
    basename = basename.split(".")[0]

    group, timepoint = None, None
    patterns = [
        ("bl-ctrl",  "Control",    "Baseline"),
        ("bl ctrl",  "Control",    "Baseline"),
        ("mdd-ctrl", "Control",    "Post-Stress"),
        ("mdd ctrl", "Control",    "Post-Stress"),
        ("lsd-ctrl", "Control",    "Post-LSD"),
        ("lsd ctrl", "Control",    "Post-LSD"),
        ("bl",       "Experiment", "Baseline"),
        ("mdd",      "Experiment", "Post-Stress"),
        ("lsd",      "Experiment", "Post-LSD"),
    ]
    for pattern, g, t in patterns:
        if pattern in basename:
            group, timepoint = g, t
            break

    if group is None:
        return None
    return animal_id, probe, group, timepoint


def discover_animal_files(data_dir: str, animal_id: str) -> Dict[str, str]:
    """Find FPGA _raw.fif files for a specific animal."""
    pattern = os.path.join(data_dir, "*_raw.fif")
    files = glob.glob(pattern)
    result = {}

    for f in sorted(files):
        meta = _parse_metadata(f)
        if meta is None:
            continue
        aid, probe, group, timepoint = meta
        if aid != animal_id.upper() or probe != "FPGA":
            continue

        if timepoint == "Baseline":
            result["Day 0"] = f
            log.info(f"  Day 0 (Baseline): {os.path.basename(f)}  [{group}]")
        elif timepoint == "Post-Stress":
            result["Day 21"] = f
            log.info(f"  Day 21 (Post-Stress): {os.path.basename(f)}  [{group}]")

    return result


# ============================================================================
# Signal Processing
# ============================================================================

def compute_first_row_hg_profile(
    raw: mne.io.BaseRaw,
    row0_start: int,
    row0_step: int,
    cfg: Config,
) -> np.ndarray:
    """Compute high-gamma power for the first row (row 0) of a sub-grid.

    Row 0 channel list is built as:
        [row0_start, row0_start + row0_step, ..., row0_start + (n_cols-1)*row0_step]

    For ascending grids (rostral):  start=256, step=+1  → [256, 257, …, 271]
    For descending grids (caudal):  start=511, step=−1  → [511, 510, …, 496]

    Returns: ndarray of shape (n_cols,) — log10(high-gamma power) per electrode.
             Column 0 = leftmost physical electrode.
    """
    tmax = min(cfg.t_end_s, raw.times[-1])
    raw_crop = raw.copy().crop(tmin=cfg.t_start_s, tmax=tmax)
    ch_names = raw_crop.ch_names
    ch_to_idx = {ch: i for i, ch in enumerate(ch_names)}

    _integrate = getattr(np, "trapezoid", np.trapz)

    # Build first-row channel list (col 0 = leftmost physical electrode)
    first_row_chs = [row0_start + col * row0_step for col in range(cfg.n_cols)]
    indices = []
    valid_cols = []
    for col, ch_num in enumerate(first_row_chs):
        ch_name = f"Ch_{ch_num}"
        if ch_name in ch_to_idx:
            indices.append(ch_to_idx[ch_name])
            valid_cols.append(col)

    if not indices:
        log.warning(f"  No channels found for first row "
                    f"(start={row0_start}, step={row0_step})")
        return np.full(cfg.n_cols, np.nan)

    data = raw_crop.get_data(picks=[ch_names[i] for i in indices])

    nperseg = int(cfg.welch_nperseg_s * cfg.sample_rate)
    noverlap = int(nperseg * cfg.welch_overlap_frac)

    freqs, pxx = sig.welch(
        data, fs=cfg.sample_rate, window="hann",
        nperseg=nperseg, noverlap=noverlap, axis=-1,
    )

    hg_mask = (freqs >= cfg.hg_lo) & (freqs <= cfg.hg_hi)

    profile = np.full(cfg.n_cols, np.nan)
    for i, col in enumerate(valid_cols):
        hg_power = _integrate(pxx[i, hg_mask], freqs[hg_mask])
        profile[col] = np.log10(hg_power) if hg_power > 0 else np.nan

    return profile


def detect_midline_column(
    profile: np.ndarray,
    search_range: Tuple[int, int] = (3, 12),
) -> int:
    """Find the column with the lowest high-gamma power (sinus trough).

    Search is restricted to ``search_range`` (inclusive) to avoid edge
    artefacts at the grid periphery — channels at the border often have
    anomalous impedance/contact and would fool a naive global argmin.
    """
    lo, hi = search_range
    inner = profile[lo:hi + 1].copy()
    return int(lo + np.nanargmin(inner))


# ============================================================================
# Plotting
# ============================================================================

def _plot_subgrid_panel(
    ax: plt.Axes,
    profile_d0: np.ndarray,
    profile_d21: np.ndarray,
    cfg: Config,
    subgrid_name: str,
    panel_label: str,
):
    """Plot overlaid first-row high-gamma profiles for Day 0 vs Day 21."""

    cols = np.arange(cfg.n_cols)
    col_pos_um = cols * cfg.inter_electrode_um  # physical position in µm

    # --- Expected midline shading (blue) ---
    exp_left = cfg.expected_midline_cols[0] * cfg.inter_electrode_um - cfg.inter_electrode_um * 0.5
    exp_right = (cfg.expected_midline_cols[1] + 1) * cfg.inter_electrode_um - cfg.inter_electrode_um * 0.5
    ax.axvspan(exp_left, exp_right, color="#2196F3", alpha=0.15, zorder=0,
               label="Expected midline (cols 7–8)")
    exp_centre_um = (cfg.expected_midline_cols[0] + cfg.expected_midline_cols[1]) / 2 * cfg.inter_electrode_um
    ax.axvline(exp_centre_um, color="#2196F3", linewidth=1.5, linestyle="--",
               alpha=0.5, zorder=1)

    # --- Detected midlines (orange shading) ---
    det_d0 = detect_midline_column(profile_d0, cfg.midline_search_cols)
    det_d21 = detect_midline_column(profile_d21, cfg.midline_search_cols)

    det_d0_um = det_d0 * cfg.inter_electrode_um
    det_d21_um = det_d21 * cfg.inter_electrode_um

    # Show the union of Day 0 and Day 21 detected midlines as orange band
    det_left_um = min(det_d0_um, det_d21_um) - cfg.inter_electrode_um * 0.5
    det_right_um = max(det_d0_um, det_d21_um) + cfg.inter_electrode_um * 0.5
    ax.axvspan(det_left_um, det_right_um, color="#FF9800", alpha=0.18, zorder=0,
               label="Detected midline range")

    # Day 0 detected midline marker
    ax.axvline(det_d0_um, color=cfg.palette_day0, linewidth=1.8, linestyle="-",
               alpha=0.6, zorder=2)
    # Day 21 detected midline marker
    ax.axvline(det_d21_um, color=cfg.palette_day21, linewidth=1.8, linestyle="-",
               alpha=0.6, zorder=2)

    # --- Data traces ---
    ax.plot(col_pos_um, profile_d0, color=cfg.palette_day0, linewidth=2.0,
            marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8,
            label="Day 0 (Baseline)", zorder=4, alpha=0.9)
    ax.plot(col_pos_um, profile_d21, color=cfg.palette_day21, linewidth=2.0,
            marker="s", markersize=6, markeredgecolor="white", markeredgewidth=0.8,
            label="Day 21 (Post-Stress)", zorder=4, alpha=0.9)

    # --- Filled area showing the trough shape ---
    y_top = max(np.nanmax(profile_d0), np.nanmax(profile_d21))
    ax.fill_between(col_pos_um, profile_d0, y_top, color=cfg.palette_day0,
                    alpha=0.06, zorder=0)
    ax.fill_between(col_pos_um, profile_d21, y_top, color=cfg.palette_day21,
                    alpha=0.06, zorder=0)

    # --- Stability annotation (emphasis on inter-session consistency) ---
    inter_day_shift = (det_d21 - det_d0) * cfg.inter_electrode_um

    if abs(inter_day_shift) == 0:
        stability = "STABLE (no shift)"
        stab_color = "#4CAF50"
    elif abs(inter_day_shift) <= cfg.inter_electrode_um:
        stability = f"MINOR SHIFT ({inter_day_shift:+.0f} µm)"
        stab_color = "#FF9800"
    else:
        stability = f"DRIFT ({inter_day_shift:+.0f} µm)"
        stab_color = "#F44336"

    annot = (f"Day 0 trough: col {det_d0}\n"
             f"Day 21 trough: col {det_d21}\n"
             f"{stability}")

    ax.text(0.98, 0.97, annot, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=stab_color,
                      alpha=0.92, linewidth=1.5),
            zorder=6)

    # --- Correlation between Day 0 and Day 21 profiles ---
    valid = ~(np.isnan(profile_d0) | np.isnan(profile_d21))
    if valid.sum() > 3:
        r_val, _ = sp_stats.pearsonr(profile_d0[valid], profile_d21[valid])
        ax.text(0.02, 0.03, f"Profile r = {r_val:.3f}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9" if r_val > 0.9
                          else "#FFF3E0" if r_val > 0.7 else "#FFEBEE",
                          ec="none", alpha=0.9),
                zorder=6)

    # --- Axis formatting ---
    ax.set_xlabel("Column (L → R across midline)", fontsize=9)
    ax.set_ylabel("log₁₀(high-γ power)", fontsize=9)
    ax.set_xticks(col_pos_um)
    ax.set_xticklabels([str(c) for c in cols], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

    ax.set_title(f"{panel_label}   {subgrid_name} Sub-Grid — First Row (16 electrodes)",
                 fontsize=11, fontweight="bold", loc="left")

    # Tidy spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_barcode_strip(
    ax: plt.Axes,
    profile_d0: np.ndarray,
    profile_d21: np.ndarray,
    cfg: Config,
    subgrid_name: str,
    panel_label: str,
):
    """Plot a stacked colour barcode strip for intuitive visual comparison.

    Two rows: top = Day 0, bottom = Day 21.
    Colour = high-gamma power (dark = sinus trough, bright = cortex).
    """
    # Stack into 2-row array
    combined = np.vstack([profile_d0, profile_d21])
    vmin = np.nanmin(combined)
    vmax = np.nanmax(combined)

    im = ax.imshow(combined, aspect="auto", cmap="inferno",
                   vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=[-0.5, cfg.n_cols - 0.5, 1.5, -0.5])

    # Expected midline
    exp_left = cfg.expected_midline_cols[0] - 0.5
    exp_right = cfg.expected_midline_cols[1] + 0.5
    ax.axvspan(exp_left, exp_right, color="#2196F3", alpha=0.25, zorder=2)
    ax.axvline(np.mean(cfg.expected_midline_cols), color="#2196F3",
               linewidth=2, linestyle="--", alpha=0.6, zorder=3)

    # Detected midlines
    det_d0 = detect_midline_column(profile_d0, cfg.midline_search_cols)
    det_d21 = detect_midline_column(profile_d21, cfg.midline_search_cols)
    ax.plot(det_d0, 0, marker="v", color="#FF9800", markersize=10,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.plot(det_d21, 1, marker="v", color="#FF9800", markersize=10,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Day 0", "Day 21"], fontsize=8, fontweight="bold")
    ax.set_xticks(range(cfg.n_cols))
    ax.set_xticklabels([str(c) for c in range(cfg.n_cols)], fontsize=6)
    ax.set_xlabel("Column", fontsize=8)
    ax.set_title(f"{panel_label}   {subgrid_name} — Barcode",
                 fontsize=10, fontweight="bold", loc="left")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.04, orientation="vertical")
    cbar.set_label("log₁₀(high-γ)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def create_composite_figure(
    animal_id: str,
    rostral_d0: np.ndarray, rostral_d21: np.ndarray,
    caudal_d0: np.ndarray, caudal_d21: np.ndarray,
    day0_fname: str, day21_fname: str,
    cfg: Config,
) -> plt.Figure:
    """Build the composite figure: profile plots + barcode strips."""

    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(
        4, 1, height_ratios=[3, 1, 3, 1],
        hspace=0.40, top=0.92, bottom=0.06, left=0.08, right=0.92,
    )

    # --- Rostral profile ---
    ax1 = fig.add_subplot(gs[0])
    _plot_subgrid_panel(ax1, rostral_d0, rostral_d21, cfg,
                        "Rostral", "a")

    # --- Rostral barcode ---
    ax2 = fig.add_subplot(gs[1])
    _plot_barcode_strip(ax2, rostral_d0, rostral_d21, cfg,
                        "Rostral", "b")

    # --- Caudal profile ---
    ax3 = fig.add_subplot(gs[2])
    _plot_subgrid_panel(ax3, caudal_d0, caudal_d21, cfg,
                        "Caudal", "c")

    # --- Caudal barcode ---
    ax4 = fig.add_subplot(gs[3])
    _plot_barcode_strip(ax4, caudal_d0, caudal_d21, cfg,
                        "Caudal", "d")

    # --- Suptitle ---
    fig.suptitle(
        f"FPGA ECoG Grid Spatial Fidelity — {animal_id}\n"
        f"Day 0: {os.path.basename(day0_fname)}   |   "
        f"Day 21: {os.path.basename(day21_fname)}",
        fontsize=14, fontweight="bold",
    )

    # --- Footer ---
    fig.text(
        0.5, 0.01,
        "High-gamma (30–100 Hz) power via Welch PSD. Electrodes over the "
        "sagittal sinus are farthest from cortex → high-γ is maximally "
        "attenuated (trough). Blue band = expected midline (cols 7–8, "
        "rostral sub-grid wiring). Orange markers/band = detected trough. "
        "Stability criterion: trough at same column on Day 0 and Day 21. "
        "Profile r = Pearson correlation of Day 0 vs Day 21 first-row profiles.",
        ha="center", fontsize=7.5, fontstyle="italic", color="grey",
        wrap=True,
    )

    return fig


# ============================================================================
# Main
# ============================================================================

def _resolve_paths(cfg: Config) -> Config:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not cfg.data_dir:
        cfg.data_dir = script_dir
    if not cfg.output_dir:
        cfg.output_dir = os.path.join(script_dir, "Results")
    return cfg


def main():
    cfg = _resolve_paths(Config())

    print("=" * 60)
    print("  FPGA Fidelity — ECoG Grid Spatial Verification")
    print("=" * 60)
    animal_id = input("\n  Which animal do you want to verify spatial fidelity from? ").strip()
    if not animal_id:
        print("  No animal ID provided. Exiting.")
        return

    animal_id = animal_id.upper()
    if not animal_id.startswith("RB"):
        animal_id = "RB" + animal_id
    print(f"\n  Searching for {animal_id} FPGA recordings...")

    files = discover_animal_files(cfg.data_dir, animal_id)

    if "Day 0" not in files:
        print(f"\n  ERROR: No Day 0 (Baseline) FPGA file found for {animal_id}.")
        print(f"  Searched in: {cfg.data_dir}")
        return
    if "Day 21" not in files:
        print(f"\n  ERROR: No Day 21 (Post-Stress) FPGA file found for {animal_id}.")
        print(f"  Searched in: {cfg.data_dir}")
        return

    # --- Load Day 0 ---
    print(f"\n  Loading Day 0: {os.path.basename(files['Day 0'])}")
    raw_d0 = mne.io.read_raw_fif(files["Day 0"], preload=True, verbose=False)
    print(f"    {len(raw_d0.ch_names)} channels, {raw_d0.times[-1]:.1f} s")

    rostral_d0 = compute_first_row_hg_profile(raw_d0, cfg.rostral_row0_start, cfg.rostral_row0_step, cfg)
    caudal_d0 = compute_first_row_hg_profile(raw_d0, cfg.caudal_row0_start, cfg.caudal_row0_step, cfg)
    del raw_d0

    # --- Load Day 21 ---
    print(f"\n  Loading Day 21: {os.path.basename(files['Day 21'])}")
    raw_d21 = mne.io.read_raw_fif(files["Day 21"], preload=True, verbose=False)
    print(f"    {len(raw_d21.ch_names)} channels, {raw_d21.times[-1]:.1f} s")

    rostral_d21 = compute_first_row_hg_profile(raw_d21, cfg.rostral_row0_start, cfg.rostral_row0_step, cfg)
    caudal_d21 = compute_first_row_hg_profile(raw_d21, cfg.caudal_row0_start, cfg.caudal_row0_step, cfg)
    del raw_d21

    # --- Report ---
    for name, d0, d21, start, step in [
        ("Rostral", rostral_d0, rostral_d21,
         cfg.rostral_row0_start, cfg.rostral_row0_step),
        ("Caudal", caudal_d0, caudal_d21,
         cfg.caudal_row0_start, cfg.caudal_row0_step),
    ]:
        ch_first = start
        ch_last = start + (cfg.n_cols - 1) * step
        det_0 = detect_midline_column(d0, cfg.midline_search_cols)
        det_21 = detect_midline_column(d21, cfg.midline_search_cols)
        shift = (det_21 - det_0) * cfg.inter_electrode_um
        print(f"\n  {name} sub-grid row 0 "
              f"(ch {ch_first}→{ch_last}, step={step:+d}):")
        print(f"    Day 0  midline → col {det_0}")
        print(f"    Day 21 midline → col {det_21}")
        print(f"    Inter-session shift: {shift:+.0f} µm")

    # --- Build figure ---
    fig = create_composite_figure(
        animal_id,
        rostral_d0, rostral_d21,
        caudal_d0, caudal_d21,
        files["Day 0"], files["Day 21"],
        cfg,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"FPGA_Fidelity_{animal_id}.png")
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
