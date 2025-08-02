# GeoGAT-site: Face-Centered Geometric Graph Attention Network for Protein-Protein Interface Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)

Protein-protein interactions (PPIs) underpin the intricate machinery of cellular life, orchestrating processes from signal transduction to metabolic regulation, yet their precise interface prediction remains a cornerstone challenge in structural biology, demanding scalable computational paradigms that balance accuracy and efficiency. Herein, we present GeoGAT-site, a pioneering geometric graph attention network that leverages face-centered surface fingerprints extracted from three-dimensional protein architectures to forecast interaction sites. Diverging from conventional vertex-centric approaches, our face-centered methodology achieves a 3.64-fold acceleration in patch generation, mitigating computational bottlenecks while preserving granular surface descriptors. At its core, GeoGAT-site incorporates a bespoke attention mechanism that adaptively modulates inter-facial distances and normal vector orientations, synergistically integrating spatial geometries with physicochemical attributes for nuanced interface delineation. Harnessing a meticulously curated dataset of 150 million face-centered fingerprints from over 20,000 structurally diverse proteins, the model attains robust generalization across heterogeneous interaction motifs. Empirical validation on an independent cohort of 167 protein complexes yields a ROC AUC of 0.89, surpassing established benchmarks including MaSIF-site (0.845), SPPIDER (0.65), and PSIVER (0.63). By furnishing high-fidelity interface annotations, GeoGAT-site augments downstream structural modeling paradigms through targeted constraints, thereby providing a versatile scaffold for unraveling PPI dynamics with profound implications for therapeutic discovery, protein redesign, and molecular epistemology.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
GeoGAT-site predicts PPI interfaces by modeling protein surfaces as graphs with triangular mesh faces as nodes. It integrates geometric (distance, normal vector angles) and physicochemical (electrostatic potential, hydrophobicity, hydrogen bonds) features via a novel geometric attention mechanism. The pipeline supports drug design, protein engineering, and molecular recognition, offering high accuracy and efficiency.

Read the full paper: *GeoGAT-site: A Face-Centered Geometric Graph Attention Network for Protein-Protein Interface Prediction* (AAAI 2026).

## Features
- **Face-Centered Approach**: Uses triangular mesh faces, reducing patch generation time by 72.51% compared to vertex-centered methods.
- **Geometric Attention**: Weights inter-face distances and angles for precise predictions.
- **Physicochemical Features**:
  - Electrostatic potential (APBS).
  - Hydrophobicity (Kyte-Doolittle, extended for RNA).
  - Hydrogen bonds (HBPLUS).
  - Interface labels (FreeSASA, ΔSASA > 0.1 Å²).
- **Scalable Processing**: Handles 150M fingerprints from 20,000+ proteins using parallel processing and HDF5.
- **Visualization**: Colors PLY meshes (blue-to-red gradient) based on interface probabilities.
- **Performance**: ROC AUC of 0.89, with category-specific results (enzyme-substrate: 0.90, receptor-ligand: 0.88, antibody-antigen: 0.87).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/geogat-site.git
   cd geogat-site
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install external tools (see [Dependencies](#dependencies)).

## Dependencies
### Python Packages
```bash
pip install numpy torch torch-geometric h5py networkx scipy pandas biopython freesasa meshio scikit-learn tqdm
```

### External Tools
Install and add to PATH or specify via command-line arguments:
- **APBS**: Electrostatic potential.
  - `brew install apbs` or `conda install -c conda-forge apbs`
- **PDB2PQR**: PDB to PQR conversion.
  - `pip install pdb2pqr`
- **HBPLUS**: Hydrogen bond analysis.
  - Download: [EBI Thornton Group](http://www.ebi.ac.uk/thornton-srv/software/HBPLUS/)
- **MSMS**: Surface mesh generation.
  - Download: [MGLTools](http://mgltools.scripps.edu/downloads/tars/releases/MSMSRELEASE/REL2.6.1/)
  - Include `pdb_to_xyzr`
- **FreeSASA**: Solvent-accessible surface area.
  - Install: [FreeSASA GitHub](https://github.com/mittinatten/freesasa)

## Usage
Run the pipeline scripts in sequence or individually. Ensure external tools are installed.

### 1. Preprocess PDB and Compute Features
Generate PLY files with features:
```bash
python coculate_ply_interface.py --input_pdb path/to/input.pdb --output_dir output
```
- `--input_pdb`: PDB file (required).
- `--output_dir`: Output directory (default: `output`).
- `--apbs_path`, `--pdb2pqr_path`, `--msms_path`, `--pdb_to_xyzr_path`, `--hbplus_path`: Tool paths.
- `--sasa_threshold`: SASA threshold (default: 0.1).

### 2. Generate Training Patches
Create face-centered patches:
```bash
python dataprocess_ply.py --ply_dir path/to/ply_files --output_base_dir face_patches
```
- `--ply_dir`: Directory with PLY files (required).
- `--sample_size`: Number of faces to sample (default: all).
- `--num_workers`: Parallel processes (default: 4).
- `--max_radius`: Patch radius in Å (default: 9.0).
- `--output_base_dir`: Patch output directory (default: `face_patches`).
- `--batch_size`: Processing batch size (default: 3000).

### 3. Train GeoGAT Model
Train the model:
```bash
python train_geogat.py --h5_dir path/to/face_patches --model_path geogat_model.pth --device cuda
```
- `--h5_dir`: HDF5 patch directory (required).
- `--model_path`: Model save path (default: `geogat_model.pth`).
- `--device`: `cpu` or `cuda` (default: `cpu`).
- `--batch_size`: Training batch size (default: 64).
- `--max_h5_files`: Max HDF5 files (default: all).
- `--epoch`: Training epochs (default: 100).

### 4. Predict and Visualize
Predict interfaces and color PLY mesh:
```bash
python predict_and_color_ply.py --ply_file path/to/input.ply --h5_file path/to/patches.h5 --model_path geogat_model.pth --output_ply output_colored.ply
```
- `--ply_file`: PLY file with features (required).
- `--h5_file`: HDF5 patch file (required).
- `--model_path`: Trained model path (default: `geogat_model.pth`).
- `--output_ply`: Colored PLY output (default: `output_colored.ply`).
- `--batch_size`: Prediction batch size (default: 32).
- `--device`: `cpu` or `cuda` (default: `cpu`).

## Pipeline Steps
1. **PDB Preprocessing** (`coculate_ply_interface.py`):
   - Cleans PDB files, separates protein/RNA chains.
   - Computes electrostatic potential (APBS), hydrophobicity (Kyte-Doolittle), hydrogen bonds (HBPLUS), and interface labels (FreeSASA).
   - Generates PLY meshes with MSMS.
2. **Patch Generation** (`dataprocess_ply.py`):
   - Creates 9Å-radius face-centered patches.
   - Stores node (normals, charge, hydrophobicity, hydrogen bonds) and edge (distance, angle) features in HDF5.
3. **Model Training** (`train_geogat.py`):
   - Trains GeoGAT with weighted cross-entropy loss.
   - Evaluates ROC AUC and AUPR, saving best model.
4. **Prediction and Visualization** (`predict_and_color_ply.py`):
   - Predicts interface probabilities.
   - Maps probabilities to vertices (blue-to-red gradient).
   - Outputs PLY with vertex colors and face scores.

## File Structure
```
geogat-site/
├── coculate_ply_interface.py   # PDB preprocessing and feature computation
├── dataprocess_ply.py          # Face-centered patch generation
├── train_geogat.py             # GeoGAT model training
├── predict_and_color_ply.py    # Interface prediction and visualization
├── gat.py                      # GeoGAT model definition
├── requirements.txt            # Python dependencies
├── output/                     # Processed files
│   ├── pdb/                    # Cleaned PDB/PQR files
│   ├── pqr/                    # PQR and APBS inputs
│   ├── dx/                     # Electrostatic potential files
│   ├── hbplus/                 # Hydrogen bond files
│   ├── msms/                   # Surface mesh files
│   ├── ply/                    # Final PLY files
├── face_patches/               # Patch data
└── README.md                   # This file
```

## Dataset
- **Training Set**: 150M face-centered fingerprints from 20,000+ PDB protein structures (enzymes, receptors, antibodies).
- **Test Set**: ~1M fingerprints from 167 complexes, no overlap with training set (sequence identity <30%, TM score <0.5).
- **Features**:
  - **Node**: Normal vectors (nx, ny, nz), electrostatic potential, hydrophobicity, hydrogen bonds.
  - **Edge**: Euclidean distance, cosine angle between face centers.
  - **Labels**: Interface (1.0) or non-interface (0.0) based on ΔSASA > 0.1 Å².
- **Access**: [https://anonymous.4open.science/r/GeoGAT-site-1F7B](https://anonymous.4open.science/r/GeoGAT-site-1F7B).


## License
[MIT License](LICENSE)


## Acknowledgments
- **PyTorch Geometric**: Graph neural networks.
- **BioPython**, **FreeSASA**: Structure processing.
- **APBS**, **PDB2PQR**, **HBPLUS**, **MSMS**: Feature extraction.
- **NumPy**, **SciPy**, **Pandas**: Numerical processing.
- **Anonymous 4open.science**: Code and dataset hosting.