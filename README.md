# GCN–SINDy Lorenz Experiments

This repository contains experiments that combine graph convolutional networks (GCNs) with sparse identification of nonlinear dynamics (SINDy) on Lorenz-system time series. The scripts cover data upsampling, derivative estimation, sparse equation discovery, coefficient pruning, and loss-landscape analysis.

## Project contents

### Main experiments

| File | Purpose |
| --- | --- |
| `gcnsindy_losslandscape.py` | Jointly trains a GCN and SINDy coefficients, gradually prunes candidate terms, records training and pruning diagnostics, and plots the joint loss surface before and after a pruning event. This is the loss-landscape experiment. |
| `gcnSindy_loss.py` | Related joint GCN–SINDy pruning experiment. Logs loss and optimization diagnostics and summarizes coefficient/support recovery; does not include the loss-surface snapshots. |
| `GCNSINDy.py` | GCN-based interpolation followed by SINDy equation discovery and trajectory comparison. Uses the 2%-dropped Lorenz dataset. |
| `gcn_lorentz+rates5_0.001.py` | Dual-GCN experiment: one network estimates rates while another upscales the trajectory; exports a six-channel trajectory/rate CSV and creates verification plots. |

### GCN interpolation and derivative scripts

| File | Purpose |
| --- | --- |
| `gcn_lorentz5_0.001.py` | Trains a GCN on a Lorenz trajectory and interpolates the series to a 0.001-second grid. |
| `gcn_lorentz50_1.py` | GCN interpolation experiment for a longer Lorenz trajectory, with trajectory visualization. |
| `derivativeGCN.py` | Trains a coordinate upscaler and a derivative-prediction GCN, using finite-difference pseudo-labels; writes predicted derivatives. |

### SINDy equation discovery scripts

| File | Purpose |
| --- | --- |
| `sindy_lorentz.py` | Computes finite-difference derivatives, builds a polynomial feature library, fits sparse equations with sequential thresholded least squares (STLS), and compares the discovered model with the Lorenz system. |
| `sindy_lorentz_quinticdiff.py` | Similar SINDy workflow using derivatives from a quintic smoothing spline. |
| `sindy_lorentz_updated.py` | Discovers equations from the combined trajectory-and-rate data produced by the dual-GCN workflow. |

### Data preparation and saved outputs

| File | Contents |
| --- | --- |
| `Lorentz_attractor_5s_0.010_rate.csv` | Five-second Lorenz trajectory sampled at 0.01 seconds; columns are `Time_s`, `X`, `Y`, and `Z`. This is the input used by the joint pruning/loss-landscape scripts. |
| `Lorentz_attractor_5s_0.010_rate.csv 10-49-42-335.csv` | Additional copy/version of the 0.01-second trajectory data. |
| `Lorentz_attractor_dropped_2pct.csv`, `Lorentz_attractor_dropped_10pct.csv` | Datasets with randomly removed samples, used for missing-data/interpolation experiments. |
| `droppeddataset.py` | Creates a randomly subsampled dataset from `Lorentz_attractor_5s_0.010_rate.csv`. Edit `DROP_PERCENT` and `RANDOM_SEED` in the file to change the drop rate or reproducibility. |
| `Lorentz_attractor50_0_001.csv` | Long/high-resolution Lorenz trajectory data. |
| `Lorentz_derivatives_0_001.csv` | Derivative data on a 0.001-second grid. |
| `Lorentz_gcn1_gcn2_0_001.csv` | Combined GCN trajectory/rate data on a 0.001-second grid. |
| `lorenz_0.01_50.csv` | Lorenz trajectory data at 0.01-second sampling over a longer interval. |
| `gcn_sindy_losses.png` | Saved loss-curve figure from an experiment. |
| `results` | Text output containing example SINDy equations. |
| `python.code-workspace` | VS Code workspace configuration. |
| `tempCodeRunnerFile.py`, `tempCodeRunnerFile.python` | Small editor/Code Runner scratch files retained from the working folder. |

The CSV files are included as experiment inputs and outputs. Some scripts expect particular filenames and column names; check the `csv_path`, `INPUT_FILE`, and `output_file` settings near the top or in the relevant script section before running a different experiment.

## Setup

Use Python 3.10 or newer and install the packages used by the experiments:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib torch
```

PyTorch selects Apple Silicon MPS or CUDA when available in the scripts that support those backends; otherwise, they use the CPU. Install a PyTorch build appropriate for your operating system and accelerator if the default package does not provide the backend you need.

## Running

Run commands from the repository root so scripts that use relative input/output paths can find their CSV files. For example:

```bash
python gcnsindy_losslandscape.py
python gcnSindy_loss.py
python droppeddataset.py
```

The two joint-training scripts have experiment settings near the top, including training length, loss weights, pruning schedule, random seed, and plotting options. The loss-landscape script also exposes `PLOT_LANDSCAPE`, grid resolution, span, and the pruning event to snapshot. Training scripts may take substantial time, especially on CPU, and generally run the experiment immediately when launched.

Several earlier scripts have fixed paths or settings inside the source. Review those values before running them. Outputs may include generated CSV files, plots, pruning logs, coefficient arrays, and printed model equations. Output names and exact artifacts depend on the script configuration.

## Method overview

The Lorenz state is represented by coordinates `(x, y, z)`. The GCN models temporal relationships between samples and can fill in intermediate time points or estimate derivatives. SINDy forms a candidate feature matrix containing

```text
1, x, y, z, x², xy, xz, y², yz, z²
```

and fits coefficients that express each state derivative as a sparse combination of those features. The joint experiments optimize the GCN and SINDy coefficients together using data, derivative-consistency, and SINDy losses. Their pruning experiments remove small candidate coefficients over scheduled events and track how the fit responds. `gcnsindy_losslandscape.py` evaluates the combined loss along two parameter-space directions around selected pre-pruning and post-pruning snapshots.

## Notes

- Experiment parameters are set in the Python files; there is no separate command-line argument interface.
- The scripts are research/experiment workflows, not a packaged Python library.
- The local virtual environment and Python cache files are excluded by `.gitignore`.
