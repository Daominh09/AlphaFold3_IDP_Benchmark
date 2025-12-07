import os
import subprocess
import pandas as pd
import re

# === BASE FOLDER CONFIGURATION ===
BASE_DIR = "dataset"
NAIVE_DIR = os.path.join(BASE_DIR, "naive_files")
ALPHAFOLD_DIR = os.path.join(BASE_DIR, "alphafold2_files")
REFERENCE_CSV = os.path.join(BASE_DIR, "af2v2_dockq_data.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "af2v3_dockq_data.csv")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(NAIVE_DIR, exist_ok=True)
os.makedirs(ALPHAFOLD_DIR, exist_ok=True)


def parse_dockq_output(output_text):
    """
    Parse DockQ output to extract metrics.
    Returns: dict with metric values
    """
    metrics = {
        'dockq': None,
        'fnat': None,
        'fnonnat': None,
        'irms': None,
        'lrms': None
    }

    lines = output_text.split('\n')
    for line in lines:
        line = line.strip()

        if 'DockQ' in line and ':' in line:
            try:
                metrics['dockq'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'fnat' in line and ':' in line:
            try:
                metrics['fnat'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'fnonnat' in line and ':' in line:
            try:
                metrics['fnonnat'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'iRMSD' in line and ':' in line:
            try:
                metrics['irms'] = float(line.split(':')[1].strip())
            except:
                pass
        elif line.startswith('LRMSD') and ':' in line:
            try:
                metrics['lrms'] = float(line.split(':')[1].strip())
            except:
                pass

    return metrics


def run_dockq(native_file, model_file):
    """
    Run DockQ command and parse results.
    """
    try:
        # DockQ command format example:
        # DockQ model_file native_file --allowed_mismatches 999
        cmd = [
            'DockQ',
            model_file,
            native_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            metrics = parse_dockq_output(result.stdout)
            return metrics
        else:
            print(f"DockQ error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"DockQ timeout for {native_file} vs {model_file}")
        return None
    except Exception as e:
        print(f"Error running DockQ: {e}")
        return None


def main():
    """
    Main function to process all files and generate results.
    """
    reference_csv = REFERENCE_CSV
    naive_dir = NAIVE_DIR
    alphafold_dir = ALPHAFOLD_DIR
    output_csv = OUTPUT_CSV

    # Check if reference CSV exists
    if not os.path.exists(reference_csv):
        print(f"Error: {reference_csv} not found!")
        return

    # Check if directories exist
    if not os.path.exists(naive_dir):
        print(f"Error: {naive_dir} directory not found!")
        return
    if not os.path.exists(alphafold_dir):
        print(f"Error: {alphafold_dir} directory not found!")
        return

    # Read reference CSV to get chain IDs
    print(f"Reading {reference_csv}...")
    ref_df = pd.read_csv(reference_csv)

    # Create mapping of pdb_id -> (idp_id, receptor_id)
    chain_mapping = {}
    for idx, row in ref_df.iterrows():
        chain_mapping[row['pdb_id']] = (row['idp_id'], row['receptor_id'])

    print(f"Loaded chain mappings for {len(chain_mapping)} PDB IDs")

    # Get list of files
    naive_files = {
        f.rsplit('.', 1)[0]: os.path.join(naive_dir, f)
        for f in os.listdir(naive_dir)
        if f.endswith(('.cif', '.pdb'))
    }
    alphafold_files = {
        f.replace('.pdb', ''): os.path.join(alphafold_dir, f)
        for f in os.listdir(alphafold_dir)
        if f.endswith('.pdb')
    }

    print(f"Found {len(naive_files)} files in {naive_dir}")
    print(f"Found {len(alphafold_files)} files in {alphafold_dir}")

    # Find PDB IDs that have both files and chain mapping
    common_pdbs = set(naive_files.keys()) & set(alphafold_files.keys()) & set(chain_mapping.keys())
    print(f"\nPDB IDs with both files and chain mapping: {len(common_pdbs)}")

    if len(common_pdbs) == 0:
        print("No matching files found!")
        return

    results = []

    print("\nProcessing files...")
    print("=" * 70)

    for pdb_id in sorted(common_pdbs):
        print(f"\nProcessing {pdb_id}...")

        naive_file = naive_files[pdb_id]
        alphafold_file = alphafold_files[pdb_id]
        idp_id, receptor_id = chain_mapping[pdb_id]

        print(f"  Chains: IDP={idp_id}, Receptor={receptor_id}")

        # Run DockQ
        print(f"  Running DockQ...")
        metrics = run_dockq(naive_file, alphafold_file)

        if metrics and metrics['dockq'] is not None:
            result = {
                'pdb_id': pdb_id,
                'idp_id': idp_id,
                'receptor_id': receptor_id,
                'dockq': metrics['dockq'],
                'fnat': metrics['fnat'],
                'fnonnat': metrics['fnonnat'],
                'irms': metrics['irms'],
                'lrms': metrics['lrms'],
                'model_name': 'alphafold2 multimer v3',
            }
            results.append(result)
            print(f"  ✓ DockQ: {metrics['dockq']:.3f}")
        else:
            print(f"  ✗ DockQ failed")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to {output_csv}")
        print(f"Total entries: {len(results)}")
        print(f"\nDockQ Statistics:")
        print(df['dockq'].describe())
    else:
        print("\nNo results generated!")


if __name__ == "__main__":
    main()
