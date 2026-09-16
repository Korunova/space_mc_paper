[readme.md](https://github.com/user-attachments/files/30203006/readme.md)
Note: add dataset links!

# SPaCe-MC: Spatially Constrained Monte Carlo Analysis

## Overview

This repository contains the Python code accompanying the manuscript:

> **Spatially Constrained Monte Carlo Permutation Test Reveals Diffusion Changes Near Stress Granules**

The repository implements a complete workflow for:

* 3D stress granule segmentation from fluorescence microscopy images
* Extraction and spatial classification of single-particle tracking (SPT) trajectories
* Diffusion analysis using Brownian and anomalous diffusion models
* Spatially Constrained Monte Carlo (SPaCe-MC) permutation testing
* Simulation and validation using fractional Brownian motion (fBM)
* Visualization and statistical analysis of experimental and simulated datasets

The code was developed for the analysis of intracellular diffusion within stress granules but can be adapted to other spatially confined biological compartments.

---

# Repository Structure

```text
space_mc_paper/

├── Segmentation/
│   ├── 3D_segmentation_LoG.py
│   ├── 3D_segmentation_LoG_GEM.py
│   └── SG_segmentation_SPT.py
│
├── GlobalLocalDiffusion/
│   ├── space_mc.py
│   ├── space_mc_permutation_sampling.py
│   ├── space_mc_visualization.py
│   │
│   ├── space_mc_simulation_validation/
│   │   ├── fBm.py
│   │   └── space_mc_validation_module.py
│   │
│   └── utils/
│       ├── preprocessing.py
│       ├── postprocessing.py
│       ├── ABM_diffusion.py
│       └── space_mc_functions.py
```

---

# Segmentation

## 3D_segmentation_LoG.py

Performs automated three-dimensional segmentation of fluorescent particles and stress granules using a Laplacian-of-Gaussian (LoG)-based segmentation pipeline.

The script extracts:

* object morphology
* particle densities
* spatial relationships between particles and stress granules

Outputs include segmentation masks and quantitative measurements.

---

## 3D_segmentation_LoG_GEM.py

Extends the LoG segmentation workflow for multichannel fluorescence microscopy.

Additional analyses include:

* stress granule morphology
* intensity measurements
* Manders colocalization coefficients
* object-based colocalization
* cytofluorograms
* quantitative morphology tables

---

## SG_segmentation_SPT.py

Segments stress granules from live-cell microscopy images for downstream single-particle tracking analysis.

Generated masks are used for assigning trajectories to stress granules or cytoplasmic regions during SPaCe-MC analysis.

---

# GlobalLocalDiffusion

## space_mc_permutation_sampling.py

Generates randomized cytoplasmic control regions that preserve the spatial characteristics of stress granules.

The script:

* loads segmentation masks
* assigns SPT trajectories to stress granules or randomized controls
* generates Monte Carlo permutation datasets
* exports trajectories for downstream diffusion analysis

---

## space_mc.py

Main analysis script implementing the SPaCe-MC framework.

Capabilities include:

* Brownian diffusion analysis
* anomalous diffusion analysis
* Monte Carlo permutation testing
* trajectory statistics
* non-Gaussian displacement analysis
* analysis of simulated datasets

---

## space_mc_visualization.py

Generates publication-quality visualizations of SPaCe-MC results.

Includes:

* ridge (joy) plots
* null distributions
* confidence intervals
* simulation parameter distributions

---

# Simulation Validation

## fBm.py

Generates synthetic single-particle trajectories using fractional Brownian motion.

Simulation parameters include:

* Hurst exponent
* diffusion coefficient
* particle number
* trajectory duration

These datasets are used to benchmark the SPaCe-MC framework.

---

## space_mc_validation_module.py

Evaluates the statistical performance of SPaCe-MC using simulated datasets with known diffusion properties.

Calculates:

* empirical p-values
* false positive rate
* false negative rate
* specificity
* statistical power

---

# Utility Modules

The `utils` directory contains reusable functions shared across the analysis scripts.

| Module                  | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `preprocessing.py`      | Data loading, trajectory preprocessing, interpolation, segmentation utilities |
| `postprocessing.py`     | Statistical analysis and publication-quality plotting functions               |
| `ABM_diffusion.py`      | Brownian and anomalous diffusion model fitting                                |
| `space_mc_functions.py` | Core SPaCe-MC permutation testing algorithms                                  |

---

# System Requirements

The code was developed and run using Python 3 on a standard personal computer. No GPU or other non-standard hardware is required.

The main Python dependencies are:

* numpy
* scipy
* pandas
* matplotlib
* scikit-image
* tifffile
* napari
* fbm

Package versions were not formally fixed. The scripts may require minor adjustments depending on the local Python environment and package versions.

---

# Installation

Install Python 3, then install the required packages:

```bash
pip install numpy scipy pandas matplotlib scikit-image tifffile napari fbm
```

Installation typically takes a few minutes on a standard desktop or laptop computer.

---

# Demo Data for Reproducing the Analyses Presented in the Paper

The demonstration datasets are available through Figshare:

* [Single-particle tracking (SPT) data for GEMs](FIGSHARE_SPT_DATA_URL)
* [Stress-granule segmentation masks](FIGSHARE_SG_MASKS_URL)

The SPT dataset is approximately 13 GB; the segmentation-mask dataset is approximately 233 MB. Download and extract both datasets before running the demo.

# Input Data

The SPaCe-MC workflow requires:

* single-particle tracking CSV files;
* corresponding stress-granule segmentation masks;

Several scripts require users to specify local input and output directories near the beginning of the script.

# Demo: Running SPaCe-MC on the Figshare Data

1. Download and extract the SPT and segmentation-mask datasets from Figshare.

2. In `GlobalLocalDiffusion/space_mc_permutation_sampling.py`, update the input paths to the downloaded SPT CSV files and segmentation-mask files. Set an output directory for the generated permutation datasets.

3. Run:

   ```bash
   python GlobalLocalDiffusion/space_mc_permutation_sampling.py
   ```

   This step assigns trajectories to stress granules and randomized cytoplasmic control regions and exports the corresponding trajectory datasets.

4. In `GlobalLocalDiffusion/space_mc.py`, update the input paths to the output generated in the previous step and specify an output directory.

5. Run:

   ```bash
   python GlobalLocalDiffusion/space_mc.py
   ```

   This step performs Brownian and anomalous diffusion analysis and evaluates the observed stress-granule measurements against the SPaCe-MC permutation distribution.

6. To generate figures, update the input paths in `GlobalLocalDiffusion/space_mc_visualization.py` and run:

   ```bash
   python GlobalLocalDiffusion/space_mc_visualization.py
   ```

# Expected Demo Output

The demo produces output files in the directories specified in the scripts, including:

* trajectory datasets assigned to stress granules and randomized control regions;
* Monte Carlo permutation datasets;
* diffusion-analysis results for Brownian and anomalous diffusion models;
* summary tables and statistical results, including empirical permutation p-values;
* null-distribution plots, confidence intervals, and other publication-quality figures.

Runtime depends on the number of trajectories, image size, and number of Monte Carlo permutations. Analyses of the full Figshare SPT dataset may take substantially longer than a small test dataset.

# Instructions for Use with New Data

To analyze new data:

1. Prepare SPT trajectory files in CSV format and corresponding masks.
2. Update the local input and output paths near the beginning of each relevant script.
3. Run the analysis in the following order:

   1. Generate SPaCe-MC control regions (`GlobalLocalDiffusion/space_mc_permutation_sampling.py`).
   2. Perform diffusion analysis (`GlobalLocalDiffusion/space_mc.py`).
   3. Generate visualizations (`GlobalLocalDiffusion/space_mc_visualization.py`).
   4. Optionally evaluate the method using simulated trajectories (`GlobalLocalDiffusion/space_mc_simulation_validation/`).

# Reproduction of Manuscript Analyses

The Figshare datasets and scripts in this repository can be used to reproduce the analyses presented in the accompanying manuscript. Use the supplied data, set the input and output paths as described above, and run the scripts in the workflow order.

Because the scripts were developed for the original study, users may need to adjust local directory paths and analysis parameters to match their own system and data organization.

# Citation

If you use this code in your research, please cite:

*E. Korunova, V. Sikirzhytski,  J. L Twiss,  M. Shtutman,  P. Vasquez.* **Spatially Constrained Monte Carlo Permutation Test Reveals Diffusion Changes Near Stress Granules**

---

# License

This repository is released under the MIT License.

---

# Contact

For questions regarding the code or methodology, please contact the corresponding author of the manuscript.

