# TandemFoilSet: Datasets for Flow Field Prediction of Tandem-Airfoil

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the dataset generation and extraction code for high-fidelity CFD simulations used in flow prediction benchmarks. It supports a wide range of aerodynamic configurations, from single and tandem airfoils to race cars.

## 📚 Citation

If you use this code or dataset in your research, please cite the following papers:

```bibtex
@article{lim2025accelerating,
  title={Accelerating fluid simulations with graph convolution network predicted flow fields},
  author={Lim, W. X. and Jessica, L. S. E. and Lv, Y. and Kong, A. W. K. and Chan, W. L.},
  journal={Aerospace Science and Technology},
  volume={164},
  pages={110414},
  year={2025},
  publisher={Elsevier}
}

@inproceedings{
lim2026tandemfoilset,
title={{TandemFoilSet}: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils},
author={Lim, W. X. and Jessica, L. S. E. and Li, Z. and Oo, T. Z. and Chan, W. L. and Kong, A. W. K.},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=4Z0P4Nbosn}
}
```

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Generation](#dataset-generation)
  - [Data Extraction](#data-extraction)
- [Dataset Access](#dataset-access)
- [Configuration Guide](#configuration-guide)
- [License](#license)

## 🎯 Overview

This repository provides tools for:

1. **Generating CFD datasets** using OpenFOAM for various aerodynamic configurations
2. **Extracting flow field data** from OpenFOAM simulations into PyTorch Geometric format

The codebase supports multiple geometry families:
- **Single airfoils** at various angles of attack and Reynolds numbers
- **Tandem airfoils** in cruise and takeoff configurations
- **Race car configurations** with ground effect
- **Three-airfoil configurations**

## 📁 Repository Structure

```
TandemFoilSet-ICLR/
├── GenData/                    # Dataset generation codes
│   ├── tandemAirfoil/         # Tandem airfoil generation
│   │   ├── autoSingle_woO_match/    # Single-airfoil datasets (AOA=0, 5)
│   │   ├── tandem_cruise/           # Tandem-airfoil cruise (AOA=0, 5)
│   │   └── tandem_takeoff/          # Tandem-airfoil takeoff (AOA=5, ground effect)
│   ├── randomFields/          # Random flow field generation
│   │   ├── run_random_single/       # Single-airfoil with random conditions
│   │   └── run_random_cruise/       # Tandem-airfoil with random conditions
│   └── raceCar/               # Race car configurations
│       ├── run_singleRaceCar/       # Inverted single-airfoil with ground effect
│       └── run_raceCar/             # Race car with ground effect
├── ExtractData/               # Data extraction codes
│   └── aa_extract_openfoam_data.py    # Extract OpenFOAM data to PyTorch format
└── README.md                  # This file
```

## 🔧 Prerequisites

### Required Software

- **OpenFOAM** (version 2112 or compatible)
  - For generating CFD datasets
  - Installation: [OpenFOAM Installation Guide](https://openfoam.org/download/)

- **Python** (3.7+)
  - Required packages:
    ```bash
    torch>=1.10.0
    torch-geometric>=2.0.0
    numpy>=1.20.0
    matplotlib>=3.3.0
    networkx>=2.6.0
    ```

### System Requirements

- **CPU**: Multi-core processor recommended for parallel simulations
- **RAM**: 16GB+ recommended for large datasets
- **GPU**: Optional but recommended for model training (CUDA-compatible)
- **Storage**: Sufficient space for OpenFOAM case files and extracted datasets

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/TandemFoilSet-ICLR.git
   cd TandemFoilSet-ICLR
   ```

2. **Install Python dependencies:**
   ```bash
   pip install torch torch-geometric numpy matplotlib networkx
   ```

3. **Set up OpenFOAM:**
   - Follow the [OpenFOAM installation guide](https://openfoam.org/download/)
   - Ensure OpenFOAM is properly sourced in your environment:
     ```bash
     source /path/to/OpenFOAM/OpenFOAM-2112/etc/bashrc
     ```

4. **Install additional dependencies:**
   - The code requires custom modules (`fun_LoadData`, `getDID`)
   - Ensure these modules are in your Python path or in the same directory

## 🚀 Usage

### Dataset Generation

The `GenData/` directory contains scripts for generating CFD datasets. Each subdirectory includes scripts to:
1. Create geometry
2. Generate mesh
3. Run OpenFOAM simulations

#### Example: Generating Single-Airfoil Dataset

```bash
cd GenData/tandemAirfoil/autoSingle_woO_match/
# Follow the instructions in the subdirectory
# Typically involves running Allrun scripts or PBS job scripts
```

#### Example: Generating Tandem-Airfoil Dataset

```bash
cd GenData/tandemAirfoil/tandem_cruise/
# Run the generation pipeline
./Allrun
```

**Note:** Generation scripts are typically configured for HPC clusters with PBS job schedulers. Modify the `run.pbs` files to match your system configuration.

### Data Extraction

The `ExtractData/aa_extract_openfoam_data.py` script extracts flow field data from OpenFOAM cases and converts them to PyTorch Geometric format.

#### Basic Usage

1. **Configure the script:**
   ```python
   # In aa_extract_openfoam_data.py, modify:
   root_dir = './'                    # Root directory
   dataname = 'your_dataset/'         # Dataset directory name
   pickle_name = 'output.pickle'      # Output filename
   ```

2. **Run the extraction:**
   ```bash
   cd ExtractData
   python aa_extract_openfoam_data.py
   ```

#### Configuration Options

- **`dataname`**: Directory containing OpenFOAM cases
- **`pickle_name`**: Output pickle filename
- **`fileList`**: Filter condition for selecting cases (e.g., `'naca' in file`)

#### Output Format

The script generates a pickle file containing a list of PyTorch Geometric `Data` objects. Each object includes:
- `pos`: Node positions (2D coordinates)
- `edge_index`: Graph edge connectivity
- `y`: Flow field variables (u, v, p velocities and pressure)
- `boundary`: Boundary condition labels
- `flowState`: Flow state parameters (Re, AoA, etc.)
- `af_pos`: Airfoil positions
- `NACA`: NACA airfoil codes
- `saf`: Signed airfoil field (geometry feature)
- `dsdf`: Distance to surface field (geometry feature)

## 📊 Dataset Access

### Pre-computed Datasets

- **Single-Airfoil Datasets** (pickle format)
  - [🔗 NTU Dataverse](https://doi.org/10.21979/N9/9OYSTD)

- **Tandem-Airfoil Datasets** (raw OpenFOAM + pickle format)
  - [🔗 NTU Dataverse](To be updated)

### Dataset Statistics

- **Single-Airfoil**: ~400 cases with varying NACA profiles, Re, and AoA
- **Tandem-Airfoil Cruise**: ~500 cases with varying configurations
- **Tandem-Airfoil Takeoff**: ~200 cases with ground effect
- **Race Car**: ~300 cases with ground effect and random conditions

## ⚙️ Configuration Guide

### Supported Path Patterns

The extraction script automatically detects configuration types based on path patterns:

- `'single'`: Single airfoil configurations
- `'takeoff'`: Tandem airfoil takeoff with ground effect
- `'wground_single_randomFields'`: Single airfoil with ground effect
- `'raceCar_randomFields'`: Race car configurations
- Other: Standard tandem airfoil cruise configurations

### Flow Conditions

- **Fixed Conditions**: Re=500, AoA=0 or 5
- **Random Conditions**: Re ∈ [10⁵, 5×10⁶], AoA ∈ [-5°, 6°]

### Geometry Features

The extraction script automatically computes:
- **SAF (Signed Airfoil Field)**: SV
- **DSDF (Distance to Surface)**: DID

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenFOAM community for the excellent CFD software
- PyTorch Geometric team for graph neural network tools
- NTU High Performance Computing Centre for computational resources

**Last Updated**: January 2026
