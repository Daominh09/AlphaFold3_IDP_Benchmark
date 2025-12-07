import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
MULTI_SEED_DIR = "dataset/multi_seed"
NATIVE_FILES_DIR = "dataset/naive_files"  # Note: assuming this is the correct spelling
AF2_FILES_DIR = "dataset/alphafold2_files"
OUTPUT_DIR = "dockq_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_filename(filename):
    """
    Parse filename to extract protein name, fold number, and seed
    Example: fold_7ta3_1_model_0.cif -> protein=7ta3, fold=1, seed implied by fold
    """
    pattern = r'fold_(\w+)_(\d+)_model_0\.cif'
    match = re.match(pattern, filename)
    if match:
        protein = match.group(1)
        fold = int(match.group(2))
        return protein, fold
    return None, None

def find_native_structure(protein_name, native_dir):
    """
    Find the corresponding native structure file
    """
    native_path = Path(native_dir)
    protein_name = protein_name.upper()
    # Try common naming patterns
    for ext in ['.pdb', '.cif']:
        # Try exact match
        native_file = native_path / f"{protein_name}{ext}"
        if native_file.exists():
            return str(native_file)
        
        # Try uppercase
        native_file = native_path / f"{protein_name.upper()}{ext}"
        if native_file.exists():
            return str(native_file)
    
    # List all files and try to match
    for file in native_path.iterdir():
        if protein_name.lower() in file.name.lower():
            return str(file)
    
    return None

def find_af2_structure(protein_name, af2_dir):
    """
    Find the corresponding AlphaFold2 structure file
    """
    af2_path = Path(af2_dir)
    protein_name_lower = protein_name.lower()
    protein_name_upper = protein_name.upper()
    
    # Try common naming patterns
    for ext in ['.pdb', '.cif']:
        # Try exact match lowercase
        af2_file = af2_path / f"{protein_name_lower}{ext}"
        if af2_file.exists():
            return str(af2_file)
        
        # Try uppercase
        af2_file = af2_path / f"{protein_name_upper}{ext}"
        if af2_file.exists():
            return str(af2_file)
    
    # List all files and try to match
    for file in af2_path.iterdir():
        if protein_name.lower() in file.name.lower():
            return str(file)
    
    return None

def run_dockq(model_file, native_file):
    """
    Run DockQ to compare model with native structure
    Returns dictionary with DockQ score and other metrics or None if failed
    """
    try:
        # DockQ command - adjust based on your installation
        # Option 1: If DockQ is in PATH
        cmd = f"DockQ {model_file} {native_file}"
        
        # Option 2: If you need to specify python
        # cmd = f"python /path/to/DockQ/DockQ.py {model_file} {native_file}"
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse DockQ output
        output = result.stdout
        
        # Method 1: Look for "Total DockQ over X native interfaces: Y.YYY"
        for line in output.split('\n'):
            if 'Total DockQ over' in line:
                # Extract score from line like "Total DockQ over 1 native interfaces: 0.099"
                match = re.search(r'Total DockQ over.*:\s*([\d.]+)', line)
                if match:
                    return float(match.group(1))
        
        # Method 2: Look for standalone "DockQ: X.XXX" line (per-chain results)
        dockq_scores = []
        lines = output.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('DockQ:'):
                # Extract score from line like "        DockQ: 0.099"
                match = re.search(r'DockQ:\s*([\d.]+)', line)
                if match:
                    dockq_scores.append(float(match.group(1)))
        
        if dockq_scores:
            # Return the first (or average if multiple interfaces)
            return dockq_scores[0] if len(dockq_scores) == 1 else np.mean(dockq_scores)
        
        print(f"Could not parse DockQ score from output for {model_file}")
        print(f"Output preview:\n{output[:500]}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"DockQ timed out for {model_file}")
        return None
    except Exception as e:
        print(f"Error running DockQ for {model_file}: {e}")
        return None

def get_af2_dockq_scores(protein_list):
    """
    Calculate DockQ scores for AlphaFold2 predictions
    Only for proteins in the provided list
    """
    af2_scores = {}
    af2_path = Path(AF2_FILES_DIR)
    
    if not os.path.exists(AF2_FILES_DIR):
        print(f"Warning: {AF2_FILES_DIR} directory not found!")
        return af2_scores
    
    print("\nCalculating AlphaFold2 DockQ scores...")
    print("="*60)
    
    for protein in protein_list:
        # Find AF2 structure file
        af2_file = find_af2_structure(protein, AF2_FILES_DIR)
        
        if af2_file is None:
            print(f"Could not find AF2 structure for {protein}")
            continue
        
        # Find corresponding native structure
        native_file = find_native_structure(protein, NATIVE_FILES_DIR)
        
        if native_file is None:
            print(f"Could not find native structure for AF2 {protein}")
            continue
        
        print(f"Processing AF2 {protein}...")
        print(f"  AF2 Model: {af2_file}")
        print(f"  Native: {native_file}")
        
        # Run DockQ
        dockq_score = run_dockq(af2_file, native_file)
        
        if dockq_score is not None:
            af2_scores[protein.lower()] = dockq_score
            print(f"  DockQ Score: {dockq_score:.3f}")
        else:
            print(f"  Failed to get DockQ score")
        
        print()
    
    return af2_scores

def analyze_multi_seed_predictions():
    """
    Main analysis function
    """
    results = []
    
    multi_seed_path = Path(MULTI_SEED_DIR)
    
    # Process each model file
    for model_file in multi_seed_path.glob("fold_*_model_0.cif"):
        protein, fold = parse_filename(model_file.name)
        
        if protein is None:
            print(f"Could not parse filename: {model_file.name}")
            continue
        
        # Find native structure
        native_file = find_native_structure(protein, NATIVE_FILES_DIR)
        
        if native_file is None:
            print(f"Could not find native structure for {protein}")
            continue
        
        print(f"Processing {protein} fold {fold}...")
        print(f"  Model: {model_file}")
        print(f"  Native: {native_file}")
        
        # Run DockQ
        dockq_score = run_dockq(str(model_file), native_file)
        
        if dockq_score is not None:
            results.append({
                'protein': protein,
                'fold': fold,
                'model_file': model_file.name,
                'native_file': os.path.basename(native_file),
                'dockq_score': dockq_score,
                'method': 'AF3'
            })
            print(f"  DockQ Score: {dockq_score:.3f}")
        else:
            print(f"  Failed to get DockQ score")
        
        print()
    
    return pd.DataFrame(results)

def visualize_results(df, af2_scores):
    """
    Create visualizations of DockQ score variability including AF2 comparison
    """
    if df.empty:
        print("No results to visualize")
        return
    
    # Scatter plot in the style of the provided image
    plt.figure(figsize=(12, 8))
    
    # Get unique proteins and assign colors
    proteins = sorted(df['protein'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(proteins)))
    
    # Create scatter plot for each protein
    for i, protein in enumerate(proteins):
        protein_data = df[df['protein'] == protein]
        
        # Add some jitter to x-axis for better visualization
        x_positions = np.ones(len(protein_data)) * i
        x_jitter = np.random.normal(0, 0.04, len(protein_data))
        x_plot = x_positions + x_jitter
        
        # Plot AF3 points
        plt.scatter(x_plot, protein_data['dockq_score'], 
                   color=colors[i], s=150, alpha=0.6, edgecolors='black',
                   label=f'{protein} (AF3)' if i == 0 else '')
        
        # Plot AF2 point if available
        protein_lower = protein.lower()
        if protein_lower in af2_scores:
            plt.scatter(i, af2_scores[protein_lower], 
                       color=colors[i], s=150, alpha=0.9, 
                       marker='D', edgecolors='black', linewidths=1.5,
                       label=f'{protein} (AF2)' if i == 0 else '')
    
    # Customize plot
    plt.xlim(-0.5, len(proteins) - 0.5)
    plt.ylim(0, 1.0)
    plt.xticks(range(len(proteins)), proteins, fontsize=12)
    plt.ylabel('DockQ Score', fontsize=14)
    plt.xlabel('Protein', fontsize=14)
    
    # Add horizontal grid lines
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0)
    
    # Add reference lines for DockQ thresholds
    plt.axhline(y=0.23, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    plt.axhline(y=0.49, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    plt.axhline(y=0.80, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markersize=10, alpha=0.6, markeredgecolor='black', label='AF3 (multi-seed)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', 
               markersize=10, alpha=0.9, markeredgecolor='black', markeredgewidth=1.5,
               label='AF2')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dockq_scatter.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/dockq_scatter.png', format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/dockq_scatter.pdf")
    print(f"Saved: {OUTPUT_DIR}/dockq_scatter.png")
    
    # Statistical summary
    summary = df.groupby('protein')['dockq_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('range', lambda x: x.max() - x.min())
    ]).round(3)
    
    # Add AF2 scores to summary
    if af2_scores:
        af2_summary = pd.DataFrame([
            {'protein': k, 'af2_dockq': v} 
            for k, v in af2_scores.items()
        ])
        af2_summary['protein'] = af2_summary['protein'].str.lower()
        summary = summary.reset_index()
        summary['protein'] = summary['protein'].str.lower()
        summary = summary.merge(af2_summary, on='protein', how='left')
        summary = summary.set_index('protein')
    
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    print(summary)
    print("\nOverall AF3 Statistics:")
    print(f"Mean DockQ Score: {df['dockq_score'].mean():.3f}")
    print(f"Std Dev: {df['dockq_score'].std():.3f}")
    print(f"Min: {df['dockq_score'].min():.3f}")
    print(f"Max: {df['dockq_score'].max():.3f}")
    
    if af2_scores:
        print("\nAlphaFold2 Statistics:")
        af2_values = list(af2_scores.values())
        print(f"Mean DockQ Score: {np.mean(af2_values):.3f}")
        print(f"Std Dev: {np.std(af2_values):.3f}")
        print(f"Min: {np.min(af2_values):.3f}")
        print(f"Max: {np.max(af2_values):.3f}")
    
    # Save summary
    summary.to_csv(f'{OUTPUT_DIR}/dockq_summary.csv')
    print(f"\nSaved: {OUTPUT_DIR}/dockq_summary.csv")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("AlphaFold3 Multi-Seed DockQ Analysis with AF2 Comparison")
    print("="*60)
    print()
    
    # Check if directories exist
    if not os.path.exists(MULTI_SEED_DIR):
        print(f"Error: {MULTI_SEED_DIR} directory not found!")
        return
    
    if not os.path.exists(NATIVE_FILES_DIR):
        print(f"Error: {NATIVE_FILES_DIR} directory not found!")
        print("Note: Script assumes 'naive_files' - check if it should be 'native_files'")
        return
    
    # Run AF3 analysis
    print("\n" + "="*60)
    print("Analyzing AF3 predictions and running DockQ...")
    print("="*60)
    df = analyze_multi_seed_predictions()
    
    if df.empty:
        print("\nNo results obtained. Please check:")
        print("1. DockQ is installed and accessible")
        print("2. File paths are correct")
        print("3. Native structure files exist")
        return
    
    # Get unique proteins from AF3 results
    proteins_in_multiseed = df['protein'].unique()
    print(f"\nProteins found in multi-seed dataset: {list(proteins_in_multiseed)}")
    
    # Get AF2 scores only for these proteins
    af2_scores = get_af2_dockq_scores(proteins_in_multiseed)
    
    # Save raw results
    df.to_csv(f'{OUTPUT_DIR}/dockq_results.csv', index=False)
    print(f"\nSaved raw results: {OUTPUT_DIR}/dockq_results.csv")
    
    # Save AF2 scores
    if af2_scores:
        af2_df = pd.DataFrame([
            {'protein': k, 'dockq_score': v, 'method': 'AF2'} 
            for k, v in af2_scores.items()
        ])
        af2_df.to_csv(f'{OUTPUT_DIR}/af2_dockq_results.csv', index=False)
        print(f"Saved AF2 results: {OUTPUT_DIR}/af2_dockq_results.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(df, af2_scores)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()