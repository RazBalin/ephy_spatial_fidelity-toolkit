# ephy_spatial_fidelity-toolkit

Quantitative verification of chronic electrophysiological implant stability across longitudinal recording sessions in freely moving mice.

## Motivation

Chronic electrophysiology experiments that span days to weeks require evidence that recording electrodes have not drifted relative to the brain. This is particularly critical for laminar probes (Neuropixels) and high-density surface grids (FPGA ECoG), where even sub-100 µm displacement can reassign channels to different cortical layers or shift grid coverage away from the target region.

This toolkit provides two standalone verification scripts, each exploiting a distinct biophysical signature to assess spatial fidelity without requiring additional imaging or histology:

| Script | Modality | Biophysical basis | Key metric |
|---|---|---|---|
| `NPX_Fidelity.py` | Neuropixels ProbeB (mPFC column) | Current Source Density laminar profile | Pearson *r* of depth-resolved CSD power between sessions |
| `FPGA_Fidelity.py` | FPGA ECoG surface grid (2 × 16 × 16) | High-gamma (30–100 Hz) attenuation over the sagittal sinus | Inter-session trough position shift (µm) |

Both scripts are designed for interactive use during data quality assessment and produce publication-ready composite figures suitable for supplementary material or reviewer response.

## Method

### Neuropixels ProbeB — Laminar CSD Profile Comparison

The Neuropixels probe records from a dorsal-to-ventral column through the medial prefrontal cortex (mPFC), spanning MOs → ACA → PL → ILA (channels 100–384). Current Source Density (CSD) is computed as the second spatial derivative of the laminar local field potential:

$$\text{CSD}(z,t) = -\sigma \, \frac{V(z + \Delta z) - 2\,V(z) + V(z - \Delta z)}{\Delta z^2}$$

with Hamming-window spatial smoothing (3-channel kernel) to suppress high-spatial-frequency noise. The script compares Day 0 (Baseline) and Day 21 (Post-Stress) recordings of the same animal by producing:

1. **Side-by-side CSD depth–time heatmaps** (representative 2 s epoch, shared colour scale) with anatomical layer boundaries annotated.
2. **Mean CSD power depth profiles** (RMS across the full 120 s recording) overlaid for both sessions.
3. **Pearson correlation** of the depth profiles as a scalar stability metric.

A high *r* (> 0.9) with layer boundaries at identical channel positions indicates the probe has not drifted. Note that amplitude differences are expected between Baseline and Post-Stress due to condition-dependent changes in neural activity (the corticosterone depression model modulates mPFC oscillatory power); the critical indicator of positional stability is spatial pattern preservation, not amplitude identity.

### FPGA ECoG Grid — Sagittal Sinus Trough Detection

The FPGA consists of two 16 × 16 electrode sub-grids (256 channels each, 200 µm pitch, 80 µm platinum sites) placed bilaterally over the dorsal cortex. The sagittal sinus — the major midline venous structure — runs beneath the grid centre. Electrodes directly over the sinus are farthest from cortical tissue, producing a characteristic **attenuation of high-frequency (30–100 Hz) neural activity**. This creates a reproducible trough in the high-gamma power profile across the electrode columns.

For each sub-grid, the script extracts the first row of 16 electrodes, computes per-channel high-gamma power via Welch PSD (2 s Hann windows, 50% overlap), and identifies the trough column. The composite figure overlays Day 0 and Day 21 profiles with:

- **Expected midline** (blue shading, centre columns) as an anatomical reference for the rostral sub-grid.
- **Detected trough** (orange markers) for each session.
- **Inter-session shift** in µm and **Pearson *r*** of the 16-electrode profile.

A stable grid shows the trough at the same column on both days (*r* > 0.95, shift = 0 µm). Because the two sub-grids have different internal wiring (the caudal sub-grid maps the physical midline to a different software column), the tool assesses **relative stability** (same position across days) rather than absolute centering.

## Installation

```bash
git clone https://github.com/RazBalin/ephy_spatial_fidelity-toolkit.git
cd ephy_spatial_fidelity-toolkit
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.9
- MNE-Python ≥ 1.0
- NumPy
- SciPy
- Matplotlib

## Usage

Both scripts use interactive CLI prompts. Place them in the same directory as your `_raw.fif` files (or edit the `data_dir` field in the `Config` dataclass).

### Neuropixels ProbeB

```
python NPX_Fidelity.py
============================================================
  NPX Fidelity — Neuropixels ProbeB Spatial Verification
============================================================
  Which animal do you want to verify spatial fidelity from? RB43

  Saved: Results/NPX_Fidelity_RB43.png
```

### FPGA ECoG Grid

```
python FPGA_Fidelity.py
============================================================
  FPGA Fidelity — ECoG Grid Spatial Verification
============================================================
  Which animal do you want to verify spatial fidelity from? RB43

  Saved: Results/FPGA_Fidelity_RB43.png
```

Output figures are saved at 300 DPI in a `Results/` subfolder.

## File naming convention

The scripts discover files automatically using the following naming pattern:

```
{date}_{time}_{Probe}_{AnimalID}_{Condition}_raw.fif
```

where `Probe` is one of `FPGA`, `ProbeA`, or `ProbeB`, `AnimalID` matches `RB[0-9]+`, and `Condition` is one of:

| Tag | Group | Timepoint | Day |
|---|---|---|---|
| `BL` | Experiment | Baseline | 0 |
| `BL-ctrl` | Control | Baseline | 0 |
| `MDD` | Experiment | Post-Stress | 21 |
| `MDD-ctrl` | Control | Post-Stress | 21 |

## Configuration

All tuneable parameters are exposed via a `@dataclass Config` at the top of each script. Key parameters:

### NPX_Fidelity.py

| Parameter | Default | Description |
|---|---|---|
| `csd_ch_start` | 100 | First channel for CSD computation (ILA) |
| `csd_ch_end` | 384 | Last channel for CSD computation (MOs) |
| `csd_smooth_kernel` | 3 | Hamming window width for spatial smoothing |
| `csd_display_t` | 2.0 s | Duration of the representative CSD epoch |
| `sample_rate` | 250.0 Hz | Sampling rate of the preprocessed FIF (originally 500Hz) |

### FPGA_Fidelity.py

| Parameter | Default | Description |
|---|---|---|
| `rostral_row0_start` | 256 | First channel of the rostral sub-grid row 0 |
| `caudal_row0_start` | 0 | First channel of the caudal sub-grid row 0 |
| `hg_lo` / `hg_hi` | 30 / 100 Hz | High-gamma band for sinus detection |
| `midline_search_cols` | (3, 12) | Inner-column window to exclude edge artefacts |
| `expected_midline_cols` | (7, 8) | Expected sinus columns (rostral sub-grid reference) |

## Experimental context

This toolkit was developed for the re-sealable chronic head-mount system described in:

> Bhatt, D.K. *et al.* (2025). A re-sealable chronic head mount for simultaneous Neuropixels and high-density ECoG recordings in freely moving mice. *bioRxiv*. DOI: [10.1101/2025.11.13.688186v2](https://doi.org/10.1101/2025.11.13.688186v2)

The experimental paradigm involves C57BL/6J mice (≥ 16 weeks) with simultaneous:

- **FPGA ECoG** — two 16 × 16 surface grids (E256, 200 µm pitch, 80 µm Pt sites) over the dorsal cortex, targeting four Default Mode Network nodes (R-ACA, L-ACA, R-RSP, L-RSP).
- **Neuropixels ProbeB** — laminar recording through the medial prefrontal cortex (MOs → ACA → PL → ILA → ORBm).
- **Neuropixels ProbeA** — hippocampal–thalamic–visual column (TH → DG → CA1 → VISam).

Recordings are acquired at three timepoints: Baseline (Day 0), Post-Stress (Day 21, after chronic corticosterone), and Post-LSD (Day 22). The fidelity scripts compare Day 0 and Day 21 to verify spatial stability across the 21-day stress paradigm.

## Interpretation guidelines

| Metric | Excellent | Acceptable | Concerning |
|---|---|---|---|
| CSD profile *r* (NPX) | > 0.90 | 0.75 – 0.90 | < 0.75 |
| High-γ profile *r* (FPGA) | > 0.95 | 0.85 – 0.95 | < 0.85 |
| FPGA trough shift | 0 µm | ≤ 200 µm (1 pitch) | > 200 µm |

Note: the CSD profile *r* is expected to be lower than the FPGA *r* because CSD power conflates probe position with neural activity level, which changes across experimental conditions. Spatial pattern preservation (peaks and troughs at the same channels) is the primary indicator of positional stability, not amplitude identity.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this toolkit, please cite:

```bibtex
@article{Kiiso2025resealable,
  title={A re-sealable chronic head mount for simultaneous Neuropixels and
         high-density ECoG recordings in freely moving mice},
  author={Kiiso, Balin and others},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.13.688186 }
}
```
