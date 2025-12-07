from Bio.PDB import PDBParser, MMCIFParser, Superimposer
from Bio import pairwise2
import os, glob, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from matplotlib.patches import Patch

# === CONFIG ===
native_dir = "dataset/native_mfib_dataset"
af2_pred_dir = "dataset/af2_mfib_multi_dataset"
af3_pred_dir = "dataset/af3_mfib_multi_dataset"
output_csv = "results/interface_ligand_rmsd_af2_af3_comparison.csv"
distance_cutoff = 5.0  # Ångström

# === PARSERS ===
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

# === HELPER FUNCTIONS ===
def get_sequence(chain):
    """Extract amino acid sequence from a chain"""
    aa_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
               'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    seq = []
    for res in chain:
        if res.get_resname() in aa_dict:
            seq.append(aa_dict[res.get_resname()])
    return ''.join(seq)

def detect_ligand_receptor(structure):
    """Detect receptor (longer) and ligand (shorter) chains"""
    chains = list(structure.get_chains())
    chain_lengths = {chain.id: sum(1 for _ in chain.get_atoms()) for chain in chains}
    receptor_chain = max(chain_lengths, key=chain_lengths.get)
    ligand_chain = min(chain_lengths, key=chain_lengths.get)
    return receptor_chain, ligand_chain

def map_chains_by_sequence(pred_struct, native_struct, nat_rep_id, nat_lig_id):
    """Map predicted chains to native chains based on sequence similarity"""
    pred_chains = {c.id: c for c in pred_struct.get_chains()}
    native_chains = {c.id: c for c in native_struct.get_chains()}
    mapping = {}
    
    for pred_id, pred_chain in pred_chains.items():
        pred_seq = get_sequence(pred_chain)
        best_match, best_score = None, 0
        
        for nat_id, nat_chain in native_chains.items():
            if nat_id in mapping.values() and len(native_chains) > 1:
                continue
            nat_seq = get_sequence(nat_chain)
            alignments = pairwise2.align.globalxx(pred_seq, nat_seq)
            if alignments and alignments[0].score > best_score:
                best_score = alignments[0].score
                best_match = nat_id
        if best_match:
            mapping[pred_id] = best_match
    return mapping

def reverse_mapping(forward_mapping):
    return {v: k for k, v in forward_mapping.items()}

def build_residue_mapping(pred_chain, native_chain):
    """Map predicted residues to native residues by sequence alignment."""
    pred_seq = get_sequence(pred_chain)
    native_seq = get_sequence(native_chain)
    
    aa_dict = {'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 
               'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET'}
    
    pred_residues = [res for res in pred_chain if res.get_resname() in aa_dict]
    native_residues = [res for res in native_chain if res.get_resname() in aa_dict]
    
    alignments = pairwise2.align.globalxx(pred_seq, native_seq)
    if not alignments:
        return {}
    
    alignment = alignments[0]
    aligned_pred, aligned_native = alignment.seqA, alignment.seqB
    
    residue_mapping = {}
    pred_idx, native_idx = 0, 0
    
    for i in range(len(aligned_pred)):
        pred_char, native_char = aligned_pred[i], aligned_native[i]
        if pred_char != '-' and native_char != '-':
            if pred_idx < len(pred_residues) and native_idx < len(native_residues):
                residue_mapping[pred_residues[pred_idx]] = native_residues[native_idx]
            pred_idx += 1
            native_idx += 1
        elif pred_char == '-':
            native_idx += 1
        else:
            pred_idx += 1
    return residue_mapping

def get_interface_residues(chain1, chain2, cutoff):
    """Get residues in chain1 within cutoff distance of chain2"""
    interface_residues = []
    coords_chain2 = np.array([atom.get_coord() for atom in chain2.get_atoms()])
    
    for res in chain1:
        res_atoms = list(res.get_atoms())
        if not res_atoms:
            continue
        for atom in res_atoms:
            distances = np.linalg.norm(coords_chain2 - atom.get_coord(), axis=1)
            if np.min(distances) < cutoff:
                interface_residues.append(res)
                break
    return interface_residues

def get_ca_atoms_mapped(residues, residue_mapping):
    ca_dict = {}
    for pred_res in residues:
        if 'CA' in pred_res and pred_res in residue_mapping:
            ca_dict[residue_mapping[pred_res]] = pred_res['CA']
    return ca_dict

def get_heavy_atoms_mapped(chain, residue_mapping):
    atom_dict = {}
    aa_dict = {'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 
               'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET'}
    for pred_res in chain:
        if pred_res.get_resname() not in aa_dict or pred_res not in residue_mapping:
            continue
        native_res = residue_mapping[pred_res]
        for atom in pred_res:
            if atom.element != 'H':
                atom_dict[(native_res, atom.get_name())] = atom
    return atom_dict

def get_ca_atoms_native(residues):
    return {res: res['CA'] for res in residues if 'CA' in res}

def get_heavy_atoms_native(chain):
    atom_dict = {}
    aa_dict = {'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 
               'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET'}
    for res in chain:
        if res.get_resname() not in aa_dict:
            continue
        for atom in res:
            if atom.element != 'H':
                atom_dict[(res, atom.get_name())] = atom
    return atom_dict

def calculate_rmsd(atoms1, atoms2):
    coords1 = np.array([atom.get_coord() for atom in atoms1])
    coords2 = np.array([atom.get_coord() for atom in atoms2])
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def load_structure(filepath, parser_type='auto'):
    """Load structure from PDB or CIF file"""
    ext = os.path.splitext(filepath)[1].lower()
    if parser_type == 'auto':
        parser_type = 'cif' if ext == '.cif' else 'pdb'
    
    if parser_type == 'cif':
        return cif_parser.get_structure("struct", filepath)
    else:
        return pdb_parser.get_structure("struct", filepath)

def process_prediction(native, predicted, name, source_name):
    """
    Process a single native vs predicted comparison.
    Returns a dict with results or None if processing failed.
    """
    native_chains = list(native.get_chains())
    predicted_chains = list(predicted.get_chains())

    if len(native_chains) != 2 or len(predicted_chains) != 2:
        print(f"  {source_name}: Skipping - need exactly 2 chains in both structures")
        return None

    # Detect receptor/ligand from NATIVE structure
    try:
        native_receptor_id, native_ligand_id = detect_ligand_receptor(native)
    except ValueError as e:
        print(f"  {source_name}: {e}")
        return {'error': str(e)}

    # Map predicted chains to native chains
    chain_mapping_pred_to_native = map_chains_by_sequence(predicted[0], native[0], 
                                                          native_receptor_id, native_ligand_id)
    chain_mapping_native_to_pred = reverse_mapping(chain_mapping_pred_to_native)
    
    if native_receptor_id not in chain_mapping_native_to_pred or \
       native_ligand_id not in chain_mapping_native_to_pred:
        print(f"  {source_name}: Could not map chains")
        return {'error': 'Chain mapping failed'}
    
    # Get predicted chain IDs
    if len(chain_mapping_native_to_pred) == 1:
        pred_receptor_id, pred_ligand_id = "A", "B"
    else:
        pred_receptor_id = chain_mapping_native_to_pred[native_receptor_id]
        pred_ligand_id = chain_mapping_native_to_pred[native_ligand_id]

    native_receptor = native[0][native_receptor_id]
    native_ligand = native[0][native_ligand_id]
    pred_receptor = predicted[0][pred_receptor_id]
    pred_ligand = predicted[0][pred_ligand_id]

    # Build residue-level mappings
    receptor_res_mapping = build_residue_mapping(pred_receptor, native_receptor)
    ligand_res_mapping = build_residue_mapping(pred_ligand, native_ligand)

    # Interface residues
    iface_native = get_interface_residues(native_receptor, native_ligand, distance_cutoff)
    iface_pred = get_interface_residues(pred_receptor, pred_ligand, distance_cutoff)

    # Get CA atoms
    ca_native_dict = get_ca_atoms_native(iface_native)
    ca_pred_dict = get_ca_atoms_mapped(iface_pred, receptor_res_mapping)

    # Find common residues
    common_res = set(ca_native_dict.keys()) & set(ca_pred_dict.keys())
    
    if len(common_res) == 0:
        print(f"  {source_name}: No common interface Cα atoms found")
        return {'error': 'No common interface atoms'}

    common_res = sorted(common_res, key=lambda r: r.get_id()[1])
    ca_native = [ca_native_dict[res] for res in common_res]
    ca_pred = [ca_pred_dict[res] for res in common_res]

    # Interface RMSD with Superposition
    si = Superimposer()
    si.set_atoms(ca_native, ca_pred)
    iface_rmsd = si.rms
    
    # Apply transformation to all predicted atoms
    si.apply(list(predicted[0].get_atoms()))

    # Ligand RMSD (after superposition)
    lig_native_dict = get_heavy_atoms_native(native_ligand)
    lig_pred_dict = get_heavy_atoms_mapped(pred_ligand, ligand_res_mapping)
    common_lig_atoms = set(lig_native_dict.keys()) & set(lig_pred_dict.keys())

    if len(common_lig_atoms) == 0:
        lig_rmsd = float('nan')
    else:
        common_lig_atoms = sorted(common_lig_atoms, key=lambda k: (k[0].get_id()[1], k[1]))
        lig_native = [lig_native_dict[atom_id] for atom_id in common_lig_atoms]
        lig_pred = [lig_pred_dict[atom_id] for atom_id in common_lig_atoms]
        lig_rmsd = calculate_rmsd(lig_native, lig_pred)

    return {
        'native_receptor_id': native_receptor_id,
        'native_ligand_id': native_ligand_id,
        'pred_receptor_id': pred_receptor_id,
        'pred_ligand_id': pred_ligand_id,
        'iface_rmsd': iface_rmsd,
        'lig_rmsd': lig_rmsd,
        'common_res_count': len(common_res)
    }

# === FILENAME PARSING ===
def extract_pdb_id_from_native(filename):
    """Extract PDB ID from native filename (e.g., '1b2p.pdb' -> '1B2P')"""
    basename = os.path.basename(filename).replace(".pdb", "")
    # Native files are lowercase PDB IDs, convert to uppercase for matching
    return basename.upper()

def extract_pdb_id_from_af2(filename):
    """Extract PDB ID from AF2 filename 
    e.g., '1B2P_A_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb' -> '1B2P'
    """
    basename = os.path.basename(filename).replace(".pdb", "")
    # PDB ID is the first part before underscore (may include chain info like 1B2P_A)
    parts = basename.split("_")
    # First part is PDB ID (4 characters)
    return parts[0].upper()

def extract_pdb_id_from_af3(filename):
    """Extract PDB ID from AF3 filename
    e.g., 'fold_1b2p_model_0.cif' -> '1B2P'
    """
    basename = os.path.basename(filename).replace(".cif", "")
    # Format: fold_<pdbid>_model_0
    parts = basename.split("_")
    if len(parts) >= 2:
        return parts[1].upper()
    return basename.upper()

# === MAIN PROCESSING ===
def main():
    # Get all native files
    native_files = sorted(glob.glob(os.path.join(native_dir, "*.pdb")))
    
    # Build lookup dictionaries for predictions (keyed by uppercase PDB ID)
    af2_files = {}
    for f in glob.glob(os.path.join(af2_pred_dir, "*.pdb")):
        pdb_id = extract_pdb_id_from_af2(f)
        af2_files[pdb_id] = f
    
    af3_files = {}
    for f in glob.glob(os.path.join(af3_pred_dir, "*.cif")):
        pdb_id = extract_pdb_id_from_af3(f)
        af3_files[pdb_id] = f
    
    print(f"Found {len(native_files)} native structures")
    print(f"Found {len(af2_files)} AF2 predictions: {list(af2_files.keys())[:5]}...")
    print(f"Found {len(af3_files)} AF3 predictions: {list(af3_files.keys())[:5]}...")
    
    os.makedirs("results", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Complex_ID", 
            "Native_Receptor", "Native_Ligand",
            "AF2_Pred_Receptor", "AF2_Pred_Ligand",
            "AF2_Interface_RMSD(Å)", "AF2_Ligand_RMSD(Å)", "AF2_Common_Interface_Res",
            "AF3_Pred_Receptor", "AF3_Pred_Ligand",
            "AF3_Interface_RMSD(Å)", "AF3_Ligand_RMSD(Å)", "AF3_Common_Interface_Res"
        ])

        for native_file in native_files:
            name = extract_pdb_id_from_native(native_file)
            print(f"\n=== Processing {name} ===")
            
            native = pdb_parser.get_structure("native", native_file)
            
            # Process AF2 prediction
            af2_result = None
            if name in af2_files:
                print(f"  Processing AF2...")
                af2_pred = load_structure(af2_files[name], 'pdb')
                af2_result = process_prediction(native, af2_pred, name, "AF2")
            else:
                print(f"  AF2 prediction not found")
            
            # Reload native (since superimposer modifies atoms in place)
            native = pdb_parser.get_structure("native", native_file)
            
            # Process AF3 prediction
            af3_result = None
            if name in af3_files:
                print(f"  Processing AF3...")
                af3_pred = load_structure(af3_files[name], 'cif')
                af3_result = process_prediction(native, af3_pred, name, "AF3")
            else:
                print(f"  AF3 prediction not found")
            
            # Build output row
            row = [name]
            
            # Native chain info (use from whichever result is available)
            result = af2_result or af3_result
            if result and 'error' not in result:
                row.extend([result['native_receptor_id'], result['native_ligand_id']])
            else:
                row.extend(["N/A", "N/A"])
            
            # AF2 results
            if af2_result and 'error' not in af2_result:
                row.extend([
                    af2_result['pred_receptor_id'],
                    af2_result['pred_ligand_id'],
                    f"{af2_result['iface_rmsd']:.3f}",
                    f"{af2_result['lig_rmsd']:.3f}" if not np.isnan(af2_result['lig_rmsd']) else "N/A",
                    af2_result['common_res_count']
                ])
                print(f"  AF2: Interface RMSD={af2_result['iface_rmsd']:.3f}Å, "
                      f"Ligand RMSD={af2_result['lig_rmsd']:.3f}Å")
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A", 0])
            
            # AF3 results
            if af3_result and 'error' not in af3_result:
                row.extend([
                    af3_result['pred_receptor_id'],
                    af3_result['pred_ligand_id'],
                    f"{af3_result['iface_rmsd']:.3f}",
                    f"{af3_result['lig_rmsd']:.3f}" if not np.isnan(af3_result['lig_rmsd']) else "N/A",
                    af3_result['common_res_count']
                ])
                print(f"  AF3: Interface RMSD={af3_result['iface_rmsd']:.3f}Å, "
                      f"Ligand RMSD={af3_result['lig_rmsd']:.3f}Å")
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A", 0])
            
            writer.writerow(row)

    print(f"\nResults saved to {output_csv}")
    
    # Remove rows with N/A values from the CSV
    df = pd.read_csv(output_csv)
    df = df.replace('N/A', np.nan)
    df = df.dropna()
    df.to_csv(output_csv, index=False)
    print(f"Removed rows with N/A values from {output_csv}")

def visualize_results():
    # Read the CSV data
    df = pd.read_csv(output_csv)
    
    # Clean N/A values from the dataframe
    df = df.replace('N/A', np.nan)

    # === DATA PREPARATION ===
    def prepare_data(df, prefix):
        """Extract and clean data for AF2 or AF3"""
        cols = {
            'iface': f'{prefix}_Interface_RMSD(Å)',
            'lig': f'{prefix}_Ligand_RMSD(Å)',
            'res': f'{prefix}_Common_Interface_Res'
        }
        
        subset = df[['Complex_ID'] + list(cols.values())].copy()
        subset = subset[subset[cols['iface']] != 'N/A']
        subset = subset[subset[cols['lig']] != 'N/A']
        
        subset[cols['iface']] = pd.to_numeric(subset[cols['iface']])
        subset[cols['lig']] = pd.to_numeric(subset[cols['lig']])
        subset[cols['res']] = pd.to_numeric(subset[cols['res']])
        
        return subset, cols

    df_af2, cols_af2 = prepare_data(df, 'AF2')
    df_af3, cols_af3 = prepare_data(df, 'AF3')

    # === CATEGORIZATION ===
    def categorize_prediction(iface, lig):
        if iface < 1 and lig < 2:
            return 'Excellent (I<1Å, L<2Å)'
        elif iface < 2 and lig < 3:
            return 'Good (I<2Å, L<3Å)'
        elif iface < 5 and lig < 5:
            return 'Moderate (I<5Å, L<5Å)'
        else:
            return 'Poor (I≥5Å or L≥5Å)'

    df_af2['Category'] = df_af2.apply(lambda r: categorize_prediction(r[cols_af2['iface']], r[cols_af2['lig']]), axis=1)
    df_af3['Category'] = df_af3.apply(lambda r: categorize_prediction(r[cols_af3['iface']], r[cols_af3['lig']]), axis=1)

    # === FIND COMMON COMPLEXES FOR PAIRED COMPARISON ===
    common_ids = set(df_af2['Complex_ID']) & set(df_af3['Complex_ID'])
    df_af2_common = df_af2[df_af2['Complex_ID'].isin(common_ids)].sort_values('Complex_ID').reset_index(drop=True)
    df_af3_common = df_af3[df_af3['Complex_ID'].isin(common_ids)].sort_values('Complex_ID').reset_index(drop=True)

    # === STATISTICAL TESTS ===
    # Mann-Whitney U test for Interface RMSD
    stat_iface, p_iface = mannwhitneyu(
        df_af2_common[cols_af2['iface']].values,
        df_af3_common[cols_af3['iface']].values,
        alternative='two-sided'
    )

    # Mann-Whitney U test for Ligand RMSD
    stat_lig, p_lig = mannwhitneyu(
        df_af2_common[cols_af2['lig']].values,
        df_af3_common[cols_af3['lig']].values,
        alternative='two-sided'
    )

    # === STATISTICS ===
    stats = {
        'AF2': {
            'total': len(df_af2),
            'median_iface': df_af2[cols_af2['iface']].median(),
            'median_lig': df_af2[cols_af2['lig']].median(),
            'mean_iface': df_af2[cols_af2['iface']].mean(),
            'mean_lig': df_af2[cols_af2['lig']].mean(),
        },
        'AF3': {
            'total': len(df_af3),
            'median_iface': df_af3[cols_af3['iface']].median(),
            'median_lig': df_af3[cols_af3['lig']].median(),
            'mean_iface': df_af3[cols_af3['iface']].mean(),
            'mean_lig': df_af3[cols_af3['lig']].mean(),
        }
    }

    categories = [
        'Poor (I≥5Å or L≥5Å)',
        'Moderate (I<5Å, L<5Å)',
        'Good (I<2Å, L<3Å)',
        'Excellent (I<1Å, L<2Å)'
    ]

    counts_af2 = df_af2['Category'].value_counts().reindex(categories, fill_value=0)
    counts_af3 = df_af3['Category'].value_counts().reindex(categories, fill_value=0)

    # === VISUALIZATION ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    color_af2 = 'steelblue'
    color_af3 = 'indianred'

    # --- Plot 1: Side-by-Side Comparison ---
    ax1 = axes[0]
    x = np.arange(len(categories))
    width = 0.35

    bars_af2 = ax1.bar(x - width/2, counts_af2.values, width, label='AlphaFold2-Multimer', 
                       color=color_af2, edgecolor='black', linewidth=1.2)
    bars_af3 = ax1.bar(x + width/2, counts_af3.values, width, label='AlphaFold3', 
                       color=color_af3, edgecolor='black', linewidth=1.2)

    for bar in bars_af2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{int(h)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color=color_af2)
    for bar in bars_af3:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{int(h)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color=color_af3)

    ax1.set_xlabel('Prediction Quality Category', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Structures', fontsize=11, fontweight='bold')
    ax1.set_title('Quality Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add panel label A
    ax1.text(-0.08, 1.12, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

    # --- Plot 2: Violin Plot - Interface and Ligand RMSD ---
    ax2 = axes[1]

    # Prepare data for violin plot
    iface_data = [df_af2_common[cols_af2['iface']].values, df_af3_common[cols_af3['iface']].values]
    lig_data = [df_af2_common[cols_af2['lig']].values, df_af3_common[cols_af3['lig']].values]

    positions_iface = [1, 2]
    positions_lig = [4, 5]

    # Create violin plots for Interface RMSD
    vp1 = ax2.violinplot(iface_data, positions=positions_iface, showmeans=True, showmedians=True)
    vp1['bodies'][0].set_facecolor(color_af2)
    vp1['bodies'][0].set_alpha(0.7)
    vp1['bodies'][1].set_facecolor(color_af3)
    vp1['bodies'][1].set_alpha(0.7)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        vp1[partname].set_edgecolor('black')
    vp1['cmeans'].set_linestyle('--')

    # Create violin plots for Ligand RMSD
    vp2 = ax2.violinplot(lig_data, positions=positions_lig, showmeans=True, showmedians=True)
    vp2['bodies'][0].set_facecolor(color_af2)
    vp2['bodies'][0].set_alpha(0.7)
    vp2['bodies'][1].set_facecolor(color_af3)
    vp2['bodies'][1].set_alpha(0.7)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        vp2[partname].set_edgecolor('black')
    vp2['cmeans'].set_linestyle('--')

    # Add p-value annotations
    def format_pvalue(p):
        if p < 0.001:
            return 'p < 0.001'
        else:
            return f'p = {p:.3f}'

    # Interface RMSD p-value
    ax2.text(1.5, ax2.get_ylim()[1] * 0.2, format_pvalue(p_iface),
             ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))

    # Ligand RMSD p-value
    ax2.text(4.5, ax2.get_ylim()[1] * 0.2, format_pvalue(p_lig),
             ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))

    # Customize axes
    ax2.set_xticks([1.5, 4.5])
    ax2.set_xticklabels(['Interface RMSD', 'Ligand RMSD'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('RMSD (Å)', fontsize=11, fontweight='bold')
    ax2.set_title('RMSD Distribution Comparison', fontsize=13, fontweight='bold')

    # Add legend manually
    legend_elements = [
        Patch(facecolor=color_af2, alpha=0.7, edgecolor='black', label='AlphaFold2-Multimer'),
        Patch(facecolor=color_af3, alpha=0.7, edgecolor='black', label='AlphaFold3')
    ]
    ax2.legend(handles=legend_elements, fontsize=10, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add panel label B
    ax2.text(-0.08, 1.12, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')

    # === SUMMARY STATISTICS BOX ===
    summary_text = (
        f"{'='*50}\n"
        f"SUMMARY STATISTICS (n={len(df_af2_common)} common structures)\n"
        f"{'='*50}\n"
        f"{'Metric':<30} {'AF2-Multi':>10} {'AF3':>8}\n"
        f"{'-'*50}\n"
        f"{'Total Structures':<30} {stats['AF2']['total']:>10} {stats['AF3']['total']:>8}\n"
        f"{'Median Interface RMSD':<30} {stats['AF2']['median_iface']:>9.2f}Å {stats['AF3']['median_iface']:>7.2f}Å\n"
        f"{'Median Ligand RMSD':<30} {stats['AF2']['median_lig']:>9.2f}Å {stats['AF3']['median_lig']:>7.2f}Å\n"
        f"{'Mean Interface RMSD':<30} {stats['AF2']['mean_iface']:>9.2f}Å {stats['AF3']['mean_iface']:>7.2f}Å\n"
        f"{'Mean Ligand RMSD':<30} {stats['AF2']['mean_lig']:>9.2f}Å {stats['AF3']['mean_lig']:>7.2f}Å\n"
        f"{'='*50}\n"
        f"MANN-WHITNEY U TEST RESULTS\n"
        f"{'='*50}\n"
        f"Interface RMSD:\n"
        f"  U-statistic: {stat_iface:.2f}\n"
        f"  p-value: {p_iface:.4e}\n"
        f"  Significant: {'Yes' if p_iface < 0.05 else 'No'} (α=0.05)\n"
        f"\n"
        f"Ligand RMSD:\n"
        f"  U-statistic: {stat_lig:.2f}\n"
        f"  p-value: {p_lig:.4e}\n"
        f"  Significant: {'Yes' if p_lig < 0.05 else 'No'} (α=0.05)\n"
        f"{'='*50}"
    )
    print(summary_text)

    # Add legend text to figure
    legend_text = 'I = Interface RMSD (Cα)\nL = Ligand RMSD (Heavy Atoms)\n— Median  -- Mean\nMann-Whitney U test p-values'
    fig.text(0.78, 0.05, legend_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('results/af2_af3_mfib_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('results/af2_af3_mfib_comparison.png', format='png', bbox_inches='tight', dpi=300)
    print("\nFigures saved to:")
    print("  - results/af2_af3_mfib_comparison.pdf")
    print("  - results/af2_af3_mfib_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
    visualize_results()