[readme.md](https://github.com/user-attachments/files/30203006/readme.md)
Note: line 9 and 227 to change before publicity of the script (authors and paper name)

# SPaCe-MC: Spatially Constrained Monte Carlo Analysis

## Overview

This repository contains the Python code accompanying the manuscript:

> **SPaCe-MC: Spatially Constrained Monte Carlo Analysis of Single-Particle Diffusion in Stress Granules**

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

# Requirements

The code requires Python 3 and the following major packages:

* numpy
* scipy
* pandas
* matplotlib
* scikit-image
* tifffile
* napari
* fbm

Additional package versions can be found in the corresponding Python scripts.

---

# Input Data

The workflow requires:

* fluorescence microscopy image stacks (OME-TIFF or TIFF)
* stress granule segmentation masks
* single-particle tracking CSV files

Several scripts expect users to specify local input and output directories near the beginning of each script.

---

# Running the Code

The scripts reproduce the analyses presented in the manuscript.

Because they were developed for the original study, input and output directories are specified as local file paths within each script. Before running the code, users should modify these paths to match the location of their own datasets and desired output directories.

The analysis workflow is:

1. Segment stress granules (`Segmentation/`)
2. Generate SPaCe-MC control regions (`space_mc_permutation_sampling.py`)
3. Perform diffusion analysis (`space_mc.py`)
4. Validate using simulations (`space_mc_simulation_validation/`)
5. Generate figures (`space_mc_visualization.py`)

---

# Citation

If you use this code in your research, please cite:

*Authors*. **SPaCe-MC: Spatially Constrained Monte Carlo Analysis of Single-Particle Diffusion in Stress Granules.**

---

# License

This repository is distributed for academic research purposes.

Please cite the accompanying manuscript when using this code.

---

# Contact

For questions regarding the code or methodology, please contact the corresponding author of the manuscript.

