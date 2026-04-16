"""
NPX_Fidelity — Neuropixels ProbeB Spatial-Fidelity Verification
================================================================

Purpose
-------
Verify that the Neuropixels ProbeB laminar position is stable across the
21-day corticosterone paradigm by comparing Current Source Density (CSD)
depth profiles between Day 0 (Baseline) and Day 21 (Post-Stress) recordings
of the same animal.

Method
------
- CSD: 2nd spatial derivative of laminar LFP, Hamming smoothing (3-ch kernel)
- Side-by-side CSD depth-time heatmaps for a representative 2 s epoch
- Mean CSD power depth profiles overlaid for quantitative comparison
- Pearson correlation of mean CSD power profiles as a stability metric

Recording
---------
Neuropixels ProbeB only (_raw.fif files).

Usage
-----
    python NPX_Fidelity.py
    >> Which animal do you want to verify spatial fidelity from? RB15

Output
------
Composite PNG in Results/ subfolder.

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

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """All tuneable parameters for NPX Fidelity verification."""

    # --- Paths (auto-resolved from script location) ---
    data_dir: str = ""
    output_dir: str = ""

    # --- Recording ---
    sample_rate: float = 250.0
    t_start_s: float = 0.0
    t_end_s: float = 120.0

    # --- CSD ---
    csd_ch_start: int = 100   # ILA end of the column
    csd_ch_end: int = 384     # MOs end of the column
    csd_smooth_kernel: int = 3  # Hamming window width (channels)
    csd_sigma: float = 1.0      # Conductivity (arbitrary units)

    # --- Display ---
    csd_display_t: float = 2.0  # seconds of CSD to display (representative)

    # --- Welch PSD (for mean power profile) ---
    welch_nperseg_s: float = 2.0
    welch_overlap_frac: float = 0.5

    # --- Plotting ---
    dpi: int = 300

    # --- Layer boundaries (channel positions for annotation) ---
    layer_boundaries: Dict[str, int] = field(default_factory=lambda: {
        "OLF/ORBvl":  50,
        "ORBm":       100,
        "ILA L1":     120,
        "ILA L2/3":   140,
        "ILA L5":     160,
        "PL L5":      180,
        "PL L6a":     200,
        "ACA6a":      220,
        "MOs6a":      240,
        "MOs L5":     270,
        "MOs L2/3":   300,
        "MOs L1":     355,
    })

    # --- Region spans for coloured background ---
    probe_b_layers: List[Tuple[str, range]] = field(default_factory=lambda: [
        ("MOs",  range(240, 384)),
        ("ACA",  range(220, 240)),
        ("PL",   range(180, 220)),
        ("ILA",  range(100, 180)),
    ])

    layer_colors: Dict[str, str] = field(default_factory=lambda: {
        "MOs": "#27ae60", "ACA": "#c0392b",
        "PL": "#2980b9", "ILA": "#8e44ad", "ORBm": "#f39c12",
    })


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
    """Find ProbeB _raw.fif files for a specific animal.

    Returns dict mapping day label ('Day 0', 'Day 21') to filepath.
    Day 0 = Baseline (BL or BL-ctrl)
    Day 21 = Post-Stress (MDD or MDD-ctrl)
    """
    pattern = os.path.join(data_dir, "*_raw.fif")
    files = glob.glob(pattern)
    result = {}

    for f in sorted(files):
        meta = _parse_metadata(f)
        if meta is None:
            continue
        aid, probe, group, timepoint = meta
        if aid != animal_id.upper() or probe != "ProbeB":
            continue

        if timepoint == "Baseline":
            result["Day 0"] = f
            log.info(f"  Day 0 (Baseline): {os.path.basename(f)}  [{group}]")
        elif timepoint == "Post-Stress":
            result["Day 21"] = f
            log.info(f"  Day 21 (Post-Stress): {os.path.basename(f)}  [{group}]")

    return result


# ============================================================================
# CSD Computation (adapted from 4c_pipeline)
# ============================================================================

def compute_csd(
    raw: mne.io.BaseRaw,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Compute Current Source Density from the ProbeB laminar column.

    CSD(z,t) = -sigma * [V(z+dz) - 2V(z) + V(z-dz)] / dz^2

    Returns
    -------
    csd : ndarray, shape (n_csd_channels, n_times)
    times : ndarray, shape (n_times,)
    csd_channels : list of int, channel numbers corresponding to CSD rows
    """
    tmax = min(cfg.t_end_s, raw.times[-1])
    raw_crop = raw.copy().crop(tmin=cfg.t_start_s, tmax=tmax)
    ch_names = raw_crop.ch_names
    data = raw_crop.get_data()
    times = raw_crop.times

    ch_to_idx = {ch: i for i, ch in enumerate(ch_names)}
    csd_ch_nums = list(range(cfg.csd_ch_start, cfg.csd_ch_end + 1))
    csd_indices = []
    valid_ch_nums = []
    for ch_num in csd_ch_nums:
        ch_name = f"Ch_{ch_num}"
        if ch_name in ch_to_idx:
            csd_indices.append(ch_to_idx[ch_name])
            valid_ch_nums.append(ch_num)

    if len(csd_indices) < 3:
        log.warning("Not enough channels for CSD computation")
        return np.array([]), times, []

    laminar_lfp = data[csd_indices, :]

    # Spatial smoothing: Hamming window
    if cfg.csd_smooth_kernel > 1:
        kernel = np.hamming(cfg.csd_smooth_kernel)
        kernel /= kernel.sum()
        smoothed = np.zeros_like(laminar_lfp)
        for t_idx in range(laminar_lfp.shape[1]):
            smoothed[:, t_idx] = np.convolve(laminar_lfp[:, t_idx], kernel, mode="same")
        laminar_lfp = smoothed

    # 2nd spatial derivative
    n_ch = laminar_lfp.shape[0]
    csd = np.zeros((n_ch - 2, laminar_lfp.shape[1]))
    for i in range(1, n_ch - 1):
        csd[i - 1, :] = -cfg.csd_sigma * (
            laminar_lfp[i + 1, :] - 2 * laminar_lfp[i, :] + laminar_lfp[i - 1, :]
        )
    csd_channels = valid_ch_nums[1:-1]

    return csd, times, csd_channels


def compute_mean_csd_power_profile(
    csd: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Compute time-averaged CSD power (RMS) per channel for depth profile.

    Returns: ndarray of shape (n_csd_channels,)
    """
    return np.sqrt(np.mean(csd ** 2, axis=1))


# ============================================================================
# Plotting
# ============================================================================

def plot_csd_panel(
    csd: np.ndarray,
    times: np.ndarray,
    csd_channels: List[int],
    ax: plt.Axes,
    cfg: Config,
    title: str = "",
    vmax: Optional[float] = None,
    show_cbar: bool = True,
):
    """Plot a single CSD depth-time heatmap (representative segment)."""
    t_max = min(cfg.csd_display_t, times[-1])
    t_mask = times <= t_max
    csd_show = csd[:, t_mask]
    times_show = times[t_mask]

    if vmax is None:
        vmax = np.percentile(np.abs(csd_show), 95)

    ch_min, ch_max = csd_channels[0], csd_channels[-1]

    # Region background shading
    for layer_name, ch_range in cfg.probe_b_layers:
        ch_list = sorted(ch_range)
        span_lo = max(ch_list[0], ch_min)
        span_hi = min(ch_list[-1], ch_max)
        if span_lo > ch_max or span_hi < ch_min:
            continue
        lc = cfg.layer_colors.get(layer_name, "gray")
        ax.axhspan(span_lo, span_hi, color=lc, alpha=0.08, zorder=0)
        ax.axhline(span_lo, color=lc, linestyle="-", linewidth=1.2,
                   alpha=0.85, zorder=3)

    im = ax.imshow(
        csd_show, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="bilinear",
        extent=[times_show[0], times_show[-1], csd_channels[-1], csd_channels[0]],
        zorder=1,
    )

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.06)
        cbar.set_label("CSD (a.u.)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Channel (dorsal → ventral)", fontsize=9)
    ax.tick_params(labelsize=7)

    # Region labels
    for layer_name, ch_range in cfg.probe_b_layers:
        ch_list = sorted(ch_range)
        mid_ch = (max(ch_list[0], ch_min) + min(ch_list[-1], ch_max)) / 2
        if mid_ch < ch_min or mid_ch > ch_max:
            continue
        lc = cfg.layer_colors.get(layer_name, "gray")
        ax.text(times_show[0] + (times_show[-1] - times_show[0]) * 0.02,
                mid_ch, layer_name, fontsize=7, weight="bold",
                va="center", ha="left", color=lc, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=lc, alpha=0.85, linewidth=0.7))

    # Layer boundary tick marks
    for label, ch in cfg.layer_boundaries.items():
        if csd_channels[0] <= ch <= csd_channels[-1]:
            ax.axhline(ch, color="grey", linewidth=0.4, linestyle="--",
                       alpha=0.55, zorder=2)
            ax.text(times_show[-1] * 1.01, ch, label, fontsize=5.5,
                    va="center", clip_on=False)

    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

    return im, vmax


def plot_power_profile(
    power_d0: np.ndarray,
    csd_ch_d0: List[int],
    power_d21: np.ndarray,
    csd_ch_d21: List[int],
    ax: plt.Axes,
    cfg: Config,
    r_value: float,
):
    """Overlay mean CSD power depth profiles for Day 0 vs Day 21."""
    ax.plot(power_d0, csd_ch_d0, color="#2196F3", linewidth=1.5,
            label="Day 0 (Baseline)", alpha=0.85)
    ax.plot(power_d21, csd_ch_d21, color="#F44336", linewidth=1.5,
            label="Day 21 (Post-Stress)", alpha=0.85)

    # Region shading
    ch_min = min(csd_ch_d0[0], csd_ch_d21[0])
    ch_max = max(csd_ch_d0[-1], csd_ch_d21[-1])
    for layer_name, ch_range in cfg.probe_b_layers:
        ch_list = sorted(ch_range)
        span_lo = max(ch_list[0], ch_min)
        span_hi = min(ch_list[-1], ch_max)
        if span_lo > ch_max or span_hi < ch_min:
            continue
        lc = cfg.layer_colors.get(layer_name, "gray")
        ax.axhspan(span_lo, span_hi, color=lc, alpha=0.06, zorder=0)

    ax.invert_yaxis()
    ax.set_xlabel("Mean CSD Power (RMS, a.u.)", fontsize=9)
    ax.set_ylabel("Channel (dorsal → ventral)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(
        f"c   Laminar CSD Power Profile Overlay   (r = {r_value:.3f})",
        fontsize=11, fontweight="bold", loc="left",
    )


def create_composite_figure(
    animal_id: str,
    csd_d0: np.ndarray, times_d0: np.ndarray, ch_d0: List[int],
    csd_d21: np.ndarray, times_d21: np.ndarray, ch_d21: List[int],
    day0_fname: str, day21_fname: str,
    cfg: Config,
) -> plt.Figure:
    """Build the main composite figure."""

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.7], wspace=0.35)

    # Shared vmax for consistent colour scale
    vmax = max(
        np.percentile(np.abs(csd_d0[:, times_d0 <= cfg.csd_display_t]), 95),
        np.percentile(np.abs(csd_d21[:, times_d21 <= cfg.csd_display_t]), 95),
    )

    # --- Panel A: Day 0 CSD heatmap ---
    ax_a = fig.add_subplot(gs[0, 0])
    plot_csd_panel(csd_d0, times_d0, ch_d0, ax_a, cfg,
                   title=f"a   Day 0 — Baseline CSD", vmax=vmax, show_cbar=True)

    # --- Panel B: Day 21 CSD heatmap ---
    ax_b = fig.add_subplot(gs[0, 1])
    plot_csd_panel(csd_d21, times_d21, ch_d21, ax_b, cfg,
                   title=f"b   Day 21 — Post-Stress CSD", vmax=vmax, show_cbar=True)

    # --- Panel C: Mean CSD power depth profiles overlaid ---
    power_d0 = compute_mean_csd_power_profile(csd_d0, cfg)
    power_d21 = compute_mean_csd_power_profile(csd_d21, cfg)

    # Align channels for correlation: use intersection
    ch_set = set(ch_d0) & set(ch_d21)
    if len(ch_set) > 10:
        common = sorted(ch_set)
        idx_d0 = [ch_d0.index(c) for c in common]
        idx_d21 = [ch_d21.index(c) for c in common]
        r_val, _ = sp_stats.pearsonr(power_d0[idx_d0], power_d21[idx_d21])
    else:
        r_val = np.nan

    ax_c = fig.add_subplot(gs[0, 2])
    plot_power_profile(power_d0, ch_d0, power_d21, ch_d21, ax_c, cfg, r_val)

    # --- Suptitle ---
    fig.suptitle(
        f"Neuropixels ProbeB Spatial Fidelity — {animal_id}\n"
        f"Day 0: {os.path.basename(day0_fname)}   |   Day 21: {os.path.basename(day21_fname)}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    fig.text(
        0.5, -0.02,
        "CSD computed as 2nd spatial derivative of laminar LFP (Hamming-smoothed, "
        f"kernel={cfg.csd_smooth_kernel}). "
        "Representative 2 s epoch shown. Mean CSD power = RMS across full recording. "
        "Pearson r quantifies laminar profile stability between Day 0 and Day 21.",
        ha="center", fontsize=8, fontstyle="italic", color="grey",
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
    print("  NPX Fidelity — Neuropixels ProbeB Spatial Verification")
    print("=" * 60)
    animal_id = input("\n  Which animal do you want to verify spatial fidelity from? ").strip()
    if not animal_id:
        print("  No animal ID provided. Exiting.")
        return

    # Normalise to uppercase
    animal_id = animal_id.upper()
    if not animal_id.startswith("RB"):
        animal_id = "RB" + animal_id
    print(f"\n  Searching for {animal_id} ProbeB recordings...")

    files = discover_animal_files(cfg.data_dir, animal_id)

    if "Day 0" not in files:
        print(f"\n  ERROR: No Day 0 (Baseline) ProbeB file found for {animal_id}.")
        print(f"  Searched in: {cfg.data_dir}")
        return
    if "Day 21" not in files:
        print(f"\n  ERROR: No Day 21 (Post-Stress) ProbeB file found for {animal_id}.")
        print(f"  Searched in: {cfg.data_dir}")
        return

    print(f"\n  Loading Day 0: {os.path.basename(files['Day 0'])}")
    raw_d0 = mne.io.read_raw_fif(files["Day 0"], preload=True, verbose=False)
    csd_d0, times_d0, ch_d0 = compute_csd(raw_d0, cfg)
    del raw_d0

    print(f"  Loading Day 21: {os.path.basename(files['Day 21'])}")
    raw_d21 = mne.io.read_raw_fif(files["Day 21"], preload=True, verbose=False)
    csd_d21, times_d21, ch_d21 = compute_csd(raw_d21, cfg)
    del raw_d21

    if csd_d0.size == 0:
        print("  ERROR: CSD computation failed for Day 0.")
        return
    if csd_d21.size == 0:
        print("  ERROR: CSD computation failed for Day 21.")
        return

    print(f"\n  CSD Day 0:  {csd_d0.shape[0]} channels × {csd_d0.shape[1]} samples")
    print(f"  CSD Day 21: {csd_d21.shape[0]} channels × {csd_d21.shape[1]} samples")

    # Build figure
    fig = create_composite_figure(
        animal_id,
        csd_d0, times_d0, ch_d0,
        csd_d21, times_d21, ch_d21,
        files["Day 0"], files["Day 21"],
        cfg,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"NPX_Fidelity_{animal_id}.png")
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
