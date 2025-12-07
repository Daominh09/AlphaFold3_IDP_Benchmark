# AlphaFold3 and Intrinsically Disordered Proteins Prediction Benchmark

This repository contains the code and analysis scripts for our research paper "AlphaFold3 and Intrinsically Disordered Proteins: Reliable Monomer Prediction, Unpredictable Multimer Performance"

## Analysis Modules

### 1. DisProt Analysis
Analysis of disordered protein regions using the DisProt database.

- **analysis.py**: Main analysis script for disordered regions

### 2. Benchmark 90 Analysis
Evaluation pipeline for AF3 IDPs predictions using a multimer Benchmark 90 dataset.

- **evaluate_combined_score.py**: Computes combined score evaluation metrics
- **evaluate_dockq.py**: Evaluates protein-protein docking quality using DockQ
- **evaluate_interface.py**: Analyzes protein-protein interface characteristics
- **get_bsa.py**: Calculates buried surface area (BSA)
- **run_dockq.py**: Executes DockQ scoring
- **run_multi_seed_dockq.py**: Runs DockQ with multiple random seeds


### 3. MFIB Analysis
Multimer structure prediction and analysis on MFIB dataset.

- **af3_monomer_generator.py**: Generates monomer structure predictions
- **af3_multimer_generator.py**: Generates multimer structure predictions
- **morf_analysis.py**: Analyzes molecular recognition features (MoRFs)
- **rmsd_analysis.py**: Computes RMSD metrics for structure comparison

## Installation

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AF3_ANALYSIS

# Setup environment and install dependencies
conda create -n AF3xIDP python=3.11 -y
conda activate AF3xIDP
conda install -c conda-forge numpy pandas matplotlib seaborn scipy scikit-learn biopython requests -y
conda install -c conda-forge pymol-open-source -y
```

## Usage


### Running DisProt Analysis

```bash
cd disprot_analysis
python evaluate_monomers.py
```

### Running Multimer Benchmark 90 Analysis

```bash
cd benchmark_90_analysis

# Run DockQ evaluation
python evaluate_dockq.py

# Evaluate interface properties
python evaluate_interface.py

# Calculate combined scores
python evaluate_combined_score.py
```

### Running MFIB Analysis

```bash
cd mfib_analysis

# Analyze results
python rmsd_analysis.py
python morf_analysis.py
```

## Contact

For questions or issues, please contact tuanminhdao@usf.edu or open an issue on GitHub.
