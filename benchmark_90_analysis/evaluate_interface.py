import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict
import warnings
from scipy.stats import mannwhitneyu

# Try to import DSSP
try:
    from Bio.PDB.DSSP import DSSP
    DSSP_AVAILABLE = True
except ImportError:
    DSSP_AVAILABLE = False
    print("WARNING: DSSP not available. Install with: conda install -c salilab dssp")

warnings.filterwarnings('ignore')


class SequenceMapper:
    """Map residue positions between structures with different numbering"""
    
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        self.pp_builder = PPBuilder()
    
    def load_structure(self, file_path, structure_id="structure"):
        """Load structure from PDB or CIF file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Structure file not found: {file_path}")
        
        if file_path.suffix.lower() in ['.cif', '.mmcif']:
            return self.cif_parser.get_structure(structure_id, str(file_path))
        elif file_path.suffix.lower() in ['.pdb', '.ent']:
            return self.pdb_parser.get_structure(structure_id, str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def extract_sequence(self, structure, chain_id):
        """Extract sequence and residue numbering from a chain"""
        model = structure[0]
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id} not found in structure")
        
        chain = model[chain_id]
        sequence = []
        residue_numbers = []
        
        for residue in chain:
            if residue.id[0] == ' ':
                resname = residue.resname
                aa = self._three_to_one(resname)
                if aa:
                    sequence.append(aa)
                    residue_numbers.append(residue.id[1])
        
        if not sequence:
            raise ValueError(f"No valid residues found in chain {chain_id}")
        
        return ''.join(sequence), residue_numbers
    
    def _three_to_one(self, resname):
        """Convert three-letter amino acid code to one-letter"""
        conversion = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return conversion.get(resname, None)
    
    def align_sequences(self, seq1, seq2):
        """Align two sequences using pairwise alignment"""
        if not seq1 or not seq2:
            raise ValueError("Cannot align empty sequences")
        
        alignments = pairwise2.align.globalxx(seq1, seq2)
        
        if not alignments:
            raise ValueError("Could not align sequences")
        
        best_alignment = alignments[0]
        return best_alignment.seqA, best_alignment.seqB, best_alignment.score
    
    def create_residue_mapping(self, ref_seq, ref_nums, pred_seq, pred_nums):
        """Create mapping between reference and predicted residue numbers"""
        aligned_ref, aligned_pred, score = self.align_sequences(ref_seq, pred_seq)
        
        mapping = {}
        ref_idx = 0
        pred_idx = 0
        
        for i in range(len(aligned_ref)):
            ref_char = aligned_ref[i]
            pred_char = aligned_pred[i]
            
            if ref_char != '-' and pred_char != '-':
                if ref_char == pred_char:
                    mapping[ref_nums[ref_idx]] = pred_nums[pred_idx]
            
            if ref_char != '-':
                ref_idx += 1
            if pred_char != '-':
                pred_idx += 1
        
        return mapping


class InterfaceExtractor:
    """Extract interface residues from protein complex structures"""
    
    def __init__(self, distance_cutoff=5.0):
        self.distance_cutoff = distance_cutoff
    
    def get_interface_residues(self, structure, chain_ids, valid_residues=None):
        """Extract interface residues between specified chains"""
        if len(chain_ids) != 2:
            raise ValueError("Please provide exactly 2 chain IDs")
        
        model = structure[0]
        
        if chain_ids[0] not in model or chain_ids[1] not in model:
            raise ValueError(f"One or both chains not found: {chain_ids}")
        
        chain1 = model[chain_ids[0]]
        chain2 = model[chain_ids[1]]
        
        def get_atoms(chain, chain_id):
            atoms = []
            for res in chain:
                if res.id[0] == ' ':
                    if valid_residues is None or res.id[1] in valid_residues.get(chain_id, set()):
                        atoms.extend([atom for atom in res.get_atoms() if atom.element != 'H'])
            return atoms
        
        atoms_chain1 = get_atoms(chain1, chain_ids[0])
        atoms_chain2 = get_atoms(chain2, chain_ids[1])
        
        if not atoms_chain1 or not atoms_chain2:
            print(f"Warning: No valid atoms found in one or both chains")
            return {chain_ids[0]: set(), chain_ids[1]: set()}
        
        ns = NeighborSearch(atoms_chain1 + atoms_chain2)
        interface_residues = {chain_ids[0]: set(), chain_ids[1]: set()}
        
        for atom1 in atoms_chain1:
            nearby = ns.search(atom1.coord, self.distance_cutoff, level='A')
            for atom2 in nearby:
                if atom2.get_parent().get_parent().id == chain_ids[1]:
                    interface_residues[chain_ids[0]].add(atom1.get_parent().id[1])
                    interface_residues[chain_ids[1]].add(atom2.get_parent().id[1])
        
        return interface_residues
    
    def extract_plddt_scores(self, structure, chain_id):
        """Extract pLDDT scores from B-factor column (CA atom only)"""
        model = structure[0]
        
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id} not found in structure")
        
        chain = model[chain_id]
        plddt_scores = {}
        
        for residue in chain:
            if residue.id[0] == ' ':
                if 'CA' in residue:
                    plddt_scores[residue.id[1]] = residue['CA'].bfactor
        
        return plddt_scores


class DisorderAnalyzer:
    """Analyze disorder at protein interfaces"""
    
    def __init__(self, af2_threshold=68.0, af3_threshold=70.0):
        self.af2_threshold = af2_threshold
        self.af3_threshold = af3_threshold
    
    def calculate_interface_disorder(self, interface_residues, plddt_scores, threshold):
        """Calculate fraction of disordered residues at interface"""
        interface_plddts = [plddt_scores[res] for res in interface_residues if res in plddt_scores]
        
        if not interface_plddts:
            return 0.0
        
        disordered_count = sum(1 for score in interface_plddts if score < threshold)
        return disordered_count / len(interface_plddts)
    
    def calculate_interface_quality(self, interface_residues, plddt_scores):
        """Calculate average pLDDT at interface"""
        interface_plddts = [plddt_scores[res] for res in interface_residues if res in plddt_scores]
        
        if not interface_plddts:
            return 0.0
        
        return interface_plddts


class PAEAnalyzer:
    """Analyze PAE (Predicted Aligned Error) at interfaces"""
    
    def __init__(self, af2_pae_dir, af3_pae_dir):
        self.af2_pae_dir = Path(af2_pae_dir)
        self.af3_pae_dir = Path(af3_pae_dir)
    
    def load_pae_matrix(self, pdb_id, alphafold_version='af2'):
        """Load PAE matrix from JSON file"""
        if alphafold_version == 'af2':
            pae_file = self.af2_pae_dir / f"{pdb_id}.json"
        else:
            pae_file = self.af3_pae_dir / f"{pdb_id}.json"
        
        if not pae_file.exists():
            raise FileNotFoundError(f"PAE file not found: {pae_file}")
        
        with open(pae_file, 'r') as f:
            pae_data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(pae_data, list):
            pae_matrix = np.array(pae_data[0]['pae'])
        elif 'pae' in pae_data:
            pae_matrix = np.array(pae_data['pae'])
        elif 'predicted_aligned_error' in pae_data:
            pae_matrix = np.array(pae_data['predicted_aligned_error'])
        else:
            raise ValueError(f"Unknown PAE JSON format in {pae_file}")
        
        return pae_matrix
    
    def calculate_interface_pae(self, pae_matrix, chain1_interface, chain2_interface, 
                                chain1_offset=0, chain2_offset=0):
        """Calculate PAE statistics for interface residues"""
        chain1_indices = [int(res) - 1 + chain1_offset for res in sorted(chain1_interface)]
        chain2_indices = [int(res) - 1 + chain2_offset for res in sorted(chain2_interface)]
        
        if not chain1_indices or not chain2_indices:
            return {
                'mean_pae': 0.0,
                'median_pae': 0.0,
                'max_pae': 0.0,
                'min_pae': 0.0,
                'std_pae': 0.0,
                'interface_pairs': 0
            }
        
        pae_values = []
        
        for i in chain1_indices:
            for j in chain2_indices:
                if i < pae_matrix.shape[0] and j < pae_matrix.shape[1]:
                    pae_values.append(pae_matrix[i, j])
                    pae_values.append(pae_matrix[j, i])
        
        if not pae_values:
            return {
                'mean_pae': 0.0,
                'median_pae': 0.0,
                'max_pae': 0.0,
                'min_pae': 0.0,
                'std_pae': 0.0,
                'interface_pairs': 0
            }
        
        return {
            'mean_pae': float(np.mean(pae_values)),
            'median_pae': float(np.median(pae_values)),
            'max_pae': float(np.max(pae_values)),
            'min_pae': float(np.min(pae_values)),
            'std_pae': float(np.std(pae_values)),
            'interface_pairs': len(pae_values)
        }


class DSSPAnalyzer:
    """Analyze secondary structure using DSSP"""
    
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        
        if not DSSP_AVAILABLE:
            print("WARNING: DSSP not available for secondary structure analysis")
    
    def load_structure(self, file_path, structure_id="structure"):
        """Load structure from PDB or CIF file"""
        file_path = Path(file_path)
        if file_path.suffix.lower() in ['.cif', '.mmcif']:
            return self.cif_parser.get_structure(structure_id, str(file_path))
        else:
            return self.pdb_parser.get_structure(structure_id, str(file_path))
    
    def calculate_dssp(self, structure_file, chain_id, interface_residues):
        """Calculate DSSP secondary structure for interface residues"""
        if not DSSP_AVAILABLE:
            return {
                'helix_fraction': 0.0,
                'sheet_fraction': 0.0,
                'coil_fraction': 0.0,
                'structure_dict': {},
                'available': False
            }
        
        try:
            structure = self.load_structure(structure_file)
            model = structure[0]
            
            dssp = DSSP(model, str(structure_file), dssp='mkdssp')
            
            ss_counts = {'H': 0, 'E': 0, 'C': 0}
            ss_dict = {}
            
            interface_residues_int = {int(res) for res in interface_residues}
            
            for residue in interface_residues_int:
                key = (chain_id, (' ', residue, ' '))
                if key in dssp:
                    ss_code = dssp[key][2]
                    
                    if ss_code in ['H', 'G', 'I']:
                        ss_type = 'H'
                    elif ss_code in ['E', 'B']:
                        ss_type = 'E'
                    else:
                        ss_type = 'C'
                    
                    ss_counts[ss_type] += 1
                    ss_dict[residue] = ss_type
            
            total = sum(ss_counts.values())
            if total == 0:
                return {
                    'helix_fraction': 0.0,
                    'sheet_fraction': 0.0,
                    'coil_fraction': 0.0,
                    'structure_dict': {},
                    'available': True
                }
            
            return {
                'helix_fraction': ss_counts['H'] / total,
                'sheet_fraction': ss_counts['E'] / total,
                'coil_fraction': ss_counts['C'] / total,
                'structure_dict': ss_dict,
                'available': True
            }
            
        except Exception as e:
            print(f"  DSSP calculation failed: {str(e)}")
            return {
                'helix_fraction': 0.0,
                'sheet_fraction': 0.0,
                'coil_fraction': 0.0,
                'structure_dict': {},
                'available': False
            }


class HydropathyAnalyzer:
    """Analyze hydropathy (hydrophobicity) at interfaces"""
    
    HYDROPATHY_SCALE = {
        'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
        'X': 0.0
    }
    
    def calculate_hydropathy(self, residue_amino_acids):
        """Calculate hydropathy statistics for interface residues"""
        if not residue_amino_acids:
            return {
                'mean_hydropathy': 0.0,
                'median_hydropathy': 0.0,
                'std_hydropathy': 0.0,
                'hydrophobic_fraction': 0.0,
                'hydrophilic_fraction': 0.0,
                'residue_scores': {}
            }
        
        hydropathy_values = []
        residue_scores = {}
        
        for res_num, aa in residue_amino_acids.items():
            score = self.HYDROPATHY_SCALE.get(aa, 0.0)
            hydropathy_values.append(score)
            residue_scores[res_num] = score
        
        hydrophobic_count = sum(1 for v in hydropathy_values if v > 0)
        hydrophilic_count = sum(1 for v in hydropathy_values if v < 0)
        
        return {
            'mean_hydropathy': float(np.mean(hydropathy_values)),
            'median_hydropathy': float(np.median(hydropathy_values)),
            'std_hydropathy': float(np.std(hydropathy_values)),
            'hydrophobic_fraction': hydrophobic_count / len(hydropathy_values),
            'hydrophilic_fraction': hydrophilic_count / len(hydropathy_values),
            'residue_scores': residue_scores
        }


def load_chain_mapping(af2_csv_file, af3_csv_file):
    """Load chain ID mappings, DockQ scores, and BSA from pre-computed CSV files"""
    if not Path(af2_csv_file).exists():
        raise FileNotFoundError(f"AF2 CSV file not found: {af2_csv_file}")
    if not Path(af3_csv_file).exists():
        raise FileNotFoundError(f"AF3 CSV file not found: {af3_csv_file}")
    
    df_af2 = pd.read_csv(af2_csv_file)
    df_af3 = pd.read_csv(af3_csv_file)
    
    chain_mapping = {}
    
    for _, row in df_af2.iterrows():
        pdb_id = row['pdb_id']
        chain_mapping[pdb_id] = {
            'ref_idp': row['idp_id'],
            'ref_receptor': row['receptor_id'],
            'af2_idp': 'A',
            'af2_receptor': 'B',
            'af3_idp': 'A',
            'af3_receptor': 'B',
            'af2_dockq': float(row['dockq']) if 'dockq' in row and pd.notna(row['dockq']) else None,
            'af2_bsa': float(row['bsa']) if 'bsa' in row and pd.notna(row['bsa']) else None,  # ADD THIS
            'af3_dockq': None,
            'af3_bsa': None  # ADD THIS
        }
    
    for _, row in df_af3.iterrows():
        pdb_id = row['pdb_id']
        dockq_score = float(row['dockq']) if 'dockq' in row and pd.notna(row['dockq']) else None
        bsa_score = float(row['bsa']) if 'bsa' in row and pd.notna(row['bsa']) else None  # ADD THIS
        
        if pdb_id in chain_mapping:
            chain_mapping[pdb_id]['af3_dockq'] = dockq_score
            chain_mapping[pdb_id]['af3_bsa'] = bsa_score  # ADD THIS
        else:
            chain_mapping[pdb_id] = {
                'ref_idp': row['idp_id'],
                'ref_receptor': row['receptor_id'],
                'af2_idp': 'A',
                'af2_receptor': 'B',
                'af3_idp': 'A',
                'af3_receptor': 'B',
                'af2_dockq': None,
                'af2_bsa': None,  # ADD THIS
                'af3_dockq': dockq_score,
                'af3_bsa': bsa_score  # ADD THIS
            }
    
    return chain_mapping


def get_matching_files(ref_dir, af2_dir, af3_dir, chain_mapping):
    """Find matching files across the three directories"""
    ref_dir = Path(ref_dir)
    af2_dir = Path(af2_dir)
    af3_dir = Path(af3_dir)
    
    matching_files = []
    
    for protein_id, chain_info in chain_mapping.items():
        ref_path = None
        for ext in ['.pdb', '.cif']:
            candidate = ref_dir / f"{protein_id}{ext}"
            if candidate.exists():
                ref_path = candidate
                break
        
        af2_path = af2_dir / f"{protein_id}.pdb"
        af3_path = af3_dir / f"{protein_id}.cif"
        
        if ref_path and af2_path.exists() and af3_path.exists():
            matching_files.append((protein_id, ref_path, af2_path, af3_path, chain_info))
        else:
            print(f"Warning: Missing files for {protein_id}")
    
    return matching_files


def get_residue_amino_acids(structure, chain_id, residue_numbers):
    """Get amino acid one-letter codes for specified residues"""
    model = structure[0]
    if chain_id not in model:
        return {}
    
    chain = model[chain_id]
    residue_aa = {}
    
    conversion = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    for residue in chain:
        if residue.id[0] == ' ' and residue.id[1] in residue_numbers:
            resname = residue.resname
            aa = conversion.get(resname, 'X')
            residue_aa[residue.id[1]] = aa
    
    return residue_aa


def get_chain_length(structure_file, chain_id):
    """Get the length of a chain from structure file"""
    parser = PDBParser(QUIET=True) if structure_file.suffix == '.pdb' else MMCIFParser(QUIET=True)
    structure = parser.get_structure('temp', str(structure_file))
    chain = structure[0][chain_id]
    return len([res for res in chain if res.id[0] == ' '])


def analyze_single_protein(protein_id, ref_pdb, af2_pdb, af3_pdb, chain_info,
                          pae_analyzer, dssp_analyzer, hydropathy_analyzer, af3_iptm_dir, af2_pae_dir):
    """
    Comprehensive analysis for a single protein:
    - Interface extraction DIRECTLY from AF2 and AF3 structures
    - Filter to only include residues that exist in reference mapping
    - Disorder analysis
    - PAE analysis
    - DSSP analysis
    - Hydropathy analysis
    """
    try:
        mapper = SequenceMapper()
        extractor = InterfaceExtractor(distance_cutoff=5.0)
        disorder_analyzer = DisorderAnalyzer(af2_threshold=68.0, af3_threshold=70.0)
        
        # Load structures
        ref_struct = mapper.load_structure(ref_pdb, "reference")
        af2_struct = mapper.load_structure(af2_pdb, "af2")
        af3_struct = mapper.load_structure(af3_pdb, "af3")
        
        ref_chains = [chain_info['ref_idp'], chain_info['ref_receptor']]
        
        # ========================================
        # SEQUENCE MAPPING (to know which residues exist in reference)
        # ========================================
        ref_to_af2_mapping = {}
        ref_to_af3_mapping = {}
        
        # Map IDP chain
        ref_idp_seq, ref_idp_nums = mapper.extract_sequence(ref_struct, chain_info['ref_idp'])
        af2_idp_seq, af2_idp_nums = mapper.extract_sequence(af2_struct, chain_info['af2_idp'])
        af3_idp_seq, af3_idp_nums = mapper.extract_sequence(af3_struct, chain_info['af3_idp'])
        
        ref_to_af2_mapping[chain_info['ref_idp']] = mapper.create_residue_mapping(
            ref_idp_seq, ref_idp_nums, af2_idp_seq, af2_idp_nums
        )
        ref_to_af3_mapping[chain_info['ref_idp']] = mapper.create_residue_mapping(
            ref_idp_seq, ref_idp_nums, af3_idp_seq, af3_idp_nums
        )
        
        # Map Receptor chain
        ref_rec_seq, ref_rec_nums = mapper.extract_sequence(ref_struct, chain_info['ref_receptor'])
        af2_rec_seq, af2_rec_nums = mapper.extract_sequence(af2_struct, chain_info['af2_receptor'])
        af3_rec_seq, af3_rec_nums = mapper.extract_sequence(af3_struct, chain_info['af3_receptor'])
        
        ref_to_af2_mapping[chain_info['ref_receptor']] = mapper.create_residue_mapping(
            ref_rec_seq, ref_rec_nums, af2_rec_seq, af2_rec_nums
        )
        ref_to_af3_mapping[chain_info['ref_receptor']] = mapper.create_residue_mapping(
            ref_rec_seq, ref_rec_nums, af3_rec_seq, af3_rec_nums
        )
        
        # Create REVERSE mappings (AF2/AF3 -> Reference)
        af2_to_ref_mapping = {}
        af3_to_ref_mapping = {}
        
        # Reverse IDP mapping
        af2_to_ref_mapping[chain_info['af2_idp']] = {
            v: k for k, v in ref_to_af2_mapping[chain_info['ref_idp']].items()
        }
        af3_to_ref_mapping[chain_info['af3_idp']] = {
            v: k for k, v in ref_to_af3_mapping[chain_info['ref_idp']].items()
        }
        
        # Reverse Receptor mapping
        af2_to_ref_mapping[chain_info['af2_receptor']] = {
            v: k for k, v in ref_to_af2_mapping[chain_info['ref_receptor']].items()
        }
        af3_to_ref_mapping[chain_info['af3_receptor']] = {
            v: k for k, v in ref_to_af3_mapping[chain_info['ref_receptor']].items()
        }
        
        # ========================================
        # EXTRACT INTERFACE DIRECTLY FROM AF2
        # ========================================
        print(f"  Extracting AF2 interface directly from structure...")
        af2_chains = [chain_info['af2_idp'], chain_info['af2_receptor']]
        af2_interface_raw = extractor.get_interface_residues(
            af2_struct, af2_chains, valid_residues=None  # No filtering yet
        )
        
        # Filter AF2 interface to only include residues that exist in reference
        af2_interface = {
            chain_info['af2_idp']: set(),
            chain_info['af2_receptor']: set()
        }
        
        for res in af2_interface_raw[chain_info['af2_idp']]:
            if res in af2_to_ref_mapping[chain_info['af2_idp']]:
                af2_interface[chain_info['af2_idp']].add(res)
        
        for res in af2_interface_raw[chain_info['af2_receptor']]:
            if res in af2_to_ref_mapping[chain_info['af2_receptor']]:
                af2_interface[chain_info['af2_receptor']].add(res)
        
        print(f"    AF2 IDP: {len(af2_interface_raw[chain_info['af2_idp']])} -> {len(af2_interface[chain_info['af2_idp']])} (after filtering)")
        print(f"    AF2 Receptor: {len(af2_interface_raw[chain_info['af2_receptor']])} -> {len(af2_interface[chain_info['af2_receptor']])} (after filtering)")
        
        # ========================================
        # EXTRACT INTERFACE DIRECTLY FROM AF3
        # ========================================
        print(f"  Extracting AF3 interface directly from structure...")
        af3_chains = [chain_info['af3_idp'], chain_info['af3_receptor']]
        af3_interface_raw = extractor.get_interface_residues(
            af3_struct, af3_chains, valid_residues=None  # No filtering yet
        )
        
        # Filter AF3 interface to only include residues that exist in reference
        af3_interface = {
            chain_info['af3_idp']: set(),
            chain_info['af3_receptor']: set()
        }
        
        for res in af3_interface_raw[chain_info['af3_idp']]:
            if res in af3_to_ref_mapping[chain_info['af3_idp']]:
                af3_interface[chain_info['af3_idp']].add(res)
        
        for res in af3_interface_raw[chain_info['af3_receptor']]:
            if res in af3_to_ref_mapping[chain_info['af3_receptor']]:
                af3_interface[chain_info['af3_receptor']].add(res)
        
        print(f"    AF3 IDP: {len(af3_interface_raw[chain_info['af3_idp']])} -> {len(af3_interface[chain_info['af3_idp']])} (after filtering)")
        print(f"    AF3 Receptor: {len(af3_interface_raw[chain_info['af3_receptor']])} -> {len(af3_interface[chain_info['af3_receptor']])} (after filtering)")
        
        # ========================================
        # EXTRACT REFERENCE INTERFACE (for comparison)
        # ========================================
        ref_interface = extractor.get_interface_residues(
            ref_struct, ref_chains, valid_residues=None
        )
        
        # Check if any interface was found
        total_af2_interface = sum(len(af2_interface[cid]) for cid in af2_chains)
        total_af3_interface = sum(len(af3_interface[cid]) for cid in af3_chains)
        
        if total_af2_interface == 0 and total_af3_interface == 0:
            print(f"  Warning: No interface found for {protein_id} in both AF2 and AF3")
            return None
        
        # ========================================
        # Extract pLDDT scores
        # ========================================
        af2_plddt = {
            chain_info['af2_idp']: extractor.extract_plddt_scores(af2_struct, chain_info['af2_idp']),
            chain_info['af2_receptor']: extractor.extract_plddt_scores(af2_struct, chain_info['af2_receptor'])
        }
        
        af3_plddt = {
            chain_info['af3_idp']: extractor.extract_plddt_scores(af3_struct, chain_info['af3_idp']),
            chain_info['af3_receptor']: extractor.extract_plddt_scores(af3_struct, chain_info['af3_receptor'])
        }
        
        # ========================================
        # Get amino acids for interface residues
        # ========================================
        ref_interface_aa = {
            'idp': get_residue_amino_acids(ref_struct, chain_info['ref_idp'], 
                                           ref_interface[chain_info['ref_idp']]),
            'receptor': get_residue_amino_acids(ref_struct, chain_info['ref_receptor'], 
                                                ref_interface[chain_info['ref_receptor']])
        }
        
        af2_interface_aa = {
            'idp': get_residue_amino_acids(af2_struct, chain_info['af2_idp'], 
                                           af2_interface[chain_info['af2_idp']]),
            'receptor': get_residue_amino_acids(af2_struct, chain_info['af2_receptor'], 
                                                af2_interface[chain_info['af2_receptor']])
        }
        
        af3_interface_aa = {
            'idp': get_residue_amino_acids(af3_struct, chain_info['af3_idp'], 
                                           af3_interface[chain_info['af3_idp']]),
            'receptor': get_residue_amino_acids(af3_struct, chain_info['af3_receptor'], 
                                                af3_interface[chain_info['af3_receptor']])
        }
        
        # Get chain lengths for PAE
        af2_chain1_len = get_chain_length(af2_pdb, chain_info['af2_idp'])
        af3_chain1_len = get_chain_length(af3_pdb, chain_info['af3_idp'])
        
        # ========================================
        # INITIALIZE RESULTS STRUCTURE
        # ========================================
        results = {
            'protein_id': protein_id,
            'chain_info': chain_info,
            'reference_interface': {
                'idp': ref_interface_aa['idp'],
                'receptor': ref_interface_aa['receptor']
            },
            'af2_interface': {
                'idp': af2_interface_aa['idp'],
                'receptor': af2_interface_aa['receptor']
            },
            'af3_interface': {
                'idp': af3_interface_aa['idp'],
                'receptor': af3_interface_aa['receptor']
            },
            'af2_dockq': chain_info.get('af2_dockq'),
            'af3_dockq': chain_info.get('af3_dockq'),
            'af2_bsa': chain_info.get('af2_bsa'),
            'af3_bsa': chain_info.get('af3_bsa'),
            'af2': {},
            'af3': {},
            'reference': {}
        }


        results['mappings'] = {
            'af2_to_ref_idp': af2_to_ref_mapping[chain_info['af2_idp']],
            'af2_to_ref_rec': af2_to_ref_mapping[chain_info['af2_receptor']],
            'af3_to_ref_idp': af3_to_ref_mapping[chain_info['af3_idp']],
            'af3_to_ref_rec': af3_to_ref_mapping[chain_info['af3_receptor']],
            'ref_to_af2_idp': ref_to_af2_mapping[chain_info['ref_idp']],
            'ref_to_af2_rec': ref_to_af2_mapping[chain_info['ref_receptor']],
            'ref_to_af3_idp': ref_to_af3_mapping[chain_info['ref_idp']],
            'ref_to_af3_rec': ref_to_af3_mapping[chain_info['ref_receptor']]
        }

        # ===== AF2 DISORDER ANALYSIS =====
        results['af2']['idp'] = {
            'disorder_fraction': disorder_analyzer.calculate_interface_disorder(
                af2_interface[chain_info['af2_idp']], af2_plddt[chain_info['af2_idp']], threshold=68.0
            ),
            'quality_score': disorder_analyzer.calculate_interface_quality(
                af2_interface[chain_info['af2_idp']], af2_plddt[chain_info['af2_idp']]
            ),
            'interface_residues': len(af2_interface[chain_info['af2_idp']])
        }
        
        results['af2']['receptor'] = {
            'disorder_fraction': disorder_analyzer.calculate_interface_disorder(
                af2_interface[chain_info['af2_receptor']], af2_plddt[chain_info['af2_receptor']], threshold=68.0
            ),
            'quality_score': disorder_analyzer.calculate_interface_quality(
                af2_interface[chain_info['af2_receptor']], af2_plddt[chain_info['af2_receptor']]
            ),
            'interface_residues': len(af2_interface[chain_info['af2_receptor']])
        }
        
        # ===== AF3 DISORDER ANALYSIS =====
        results['af3']['idp'] = {
            'disorder_fraction': disorder_analyzer.calculate_interface_disorder(
                af3_interface[chain_info['af3_idp']], af3_plddt[chain_info['af3_idp']], threshold=70.0
            ),
            'quality_score': disorder_analyzer.calculate_interface_quality(
                af3_interface[chain_info['af3_idp']], af3_plddt[chain_info['af3_idp']]
            ),
            'interface_residues': len(af3_interface[chain_info['af3_idp']])
        }
        
        results['af3']['receptor'] = {
            'disorder_fraction': disorder_analyzer.calculate_interface_disorder(
                af3_interface[chain_info['af3_receptor']], af3_plddt[chain_info['af3_receptor']], threshold=70.0
            ),
            'quality_score': disorder_analyzer.calculate_interface_quality(
                af3_interface[chain_info['af3_receptor']], af3_plddt[chain_info['af3_receptor']]
            ),
            'interface_residues': len(af3_interface[chain_info['af3_receptor']])
        }
        
        # ===== AF2 PAE ANALYSIS =====
        try:
            af2_pae_matrix = pae_analyzer.load_pae_matrix(protein_id, 'af2')
            af2_interface_idp = set(af2_interface[chain_info['af2_idp']])
            af2_interface_rec = set(af2_interface[chain_info['af2_receptor']])
            
            af2_pae_stats = pae_analyzer.calculate_interface_pae(
                af2_pae_matrix, af2_interface_idp, af2_interface_rec,
                chain1_offset=0, chain2_offset=af2_chain1_len
            )
            results['af2']['pae'] = af2_pae_stats
        except Exception as e:
            print(f"  AF2 PAE failed: {str(e)}")
            results['af2']['pae'] = None
        
        # ===== AF3 PAE ANALYSIS =====
        try:
            af3_pae_matrix = pae_analyzer.load_pae_matrix(protein_id, 'af3')
            af3_interface_idp = set(af3_interface[chain_info['af3_idp']])
            af3_interface_rec = set(af3_interface[chain_info['af3_receptor']])
            
            af3_pae_stats = pae_analyzer.calculate_interface_pae(
                af3_pae_matrix, af3_interface_idp, af3_interface_rec,
                chain1_offset=0, chain2_offset=af3_chain1_len
            )
            results['af3']['pae'] = af3_pae_stats
        except Exception as e:
            print(f"  AF3 PAE failed: {str(e)}")
            results['af3']['pae'] = None
        
        # ===== AF2 DSSP ANALYSIS =====
        af2_interface_idp_set = set(af2_interface[chain_info['af2_idp']])
        af2_interface_rec_set = set(af2_interface[chain_info['af2_receptor']])
        
        af2_dssp_idp = dssp_analyzer.calculate_dssp(
            af2_pdb, chain_info['af2_idp'], af2_interface_idp_set
        )
        af2_dssp_rec = dssp_analyzer.calculate_dssp(
            af2_pdb, chain_info['af2_receptor'], af2_interface_rec_set
        )
        results['af2']['dssp'] = {
            'idp': af2_dssp_idp,
            'receptor': af2_dssp_rec
        }
        
        # ===== AF3 DSSP ANALYSIS =====
        af3_interface_idp_set = set(af3_interface[chain_info['af3_idp']])
        af3_interface_rec_set = set(af3_interface[chain_info['af3_receptor']])
        
        af3_dssp_idp = dssp_analyzer.calculate_dssp(
            af3_pdb, chain_info['af3_idp'], af3_interface_idp_set
        )
        af3_dssp_rec = dssp_analyzer.calculate_dssp(
            af3_pdb, chain_info['af3_receptor'], af3_interface_rec_set
        )
        results['af3']['dssp'] = {
            'idp': af3_dssp_idp,
            'receptor': af3_dssp_rec
        }
        
        # ===== AF2 HYDROPATHY ANALYSIS =====
        af2_hydro_idp = hydropathy_analyzer.calculate_hydropathy(
            af2_interface_aa['idp']
        )
        af2_hydro_rec = hydropathy_analyzer.calculate_hydropathy(
            af2_interface_aa['receptor']
        )
        results['af2']['hydropathy'] = {
            'idp': af2_hydro_idp,
            'receptor': af2_hydro_rec
        }
        
        # ===== AF3 HYDROPATHY ANALYSIS =====
        af3_hydro_idp = hydropathy_analyzer.calculate_hydropathy(
            af3_interface_aa['idp']
        )
        af3_hydro_rec = hydropathy_analyzer.calculate_hydropathy(
            af3_interface_aa['receptor']
        )
        results['af3']['hydropathy'] = {
            'idp': af3_hydro_idp,
            'receptor': af3_hydro_rec
        }
        
        # ===== REFERENCE DSSP ANALYSIS =====
        ref_interface_idp_set = set(ref_interface[chain_info['ref_idp']])
        ref_interface_rec_set = set(ref_interface[chain_info['ref_receptor']])
        
        ref_dssp_idp = dssp_analyzer.calculate_dssp(
            ref_pdb, chain_info['ref_idp'], ref_interface_idp_set
        )
        ref_dssp_rec = dssp_analyzer.calculate_dssp(
            ref_pdb, chain_info['ref_receptor'], ref_interface_rec_set
        )
        results['reference']['dssp'] = {
            'idp': ref_dssp_idp,
            'receptor': ref_dssp_rec
        }

        # ===== REFERENCE HYDROPATHY ANALYSIS =====
        ref_hydro_idp = hydropathy_analyzer.calculate_hydropathy(
            ref_interface_aa['idp']
        )
        ref_hydro_rec = hydropathy_analyzer.calculate_hydropathy(
            ref_interface_aa['receptor']
        )
        results['reference']['hydropathy'] = {
            'idp': ref_hydro_idp,
            'receptor': ref_hydro_rec
        }

        # ===== AF2 ipTM ANALYSIS =====
        try:
            af2_json_path = Path(af2_pae_dir) / f"{protein_id}.json"
            with open(af2_json_path, 'r') as f:
                af2_json_data = json.load(f)
            
            # Extract ipTM score (handle different JSON formats)
            if isinstance(af2_json_data, list):
                af2_iptm = af2_json_data[0].get('iptm', None)
            elif 'iptm' in af2_json_data:
                af2_iptm = af2_json_data['iptm']
            else:
                af2_iptm = None
            
            results['af2']['iptm'] = af2_iptm
        except Exception as e:
            print(f"  AF2 ipTM failed: {str(e)}")
            results['af2']['iptm'] = None

        # ===== AF3 ipTM ANALYSIS =====
        try:
            af3_iptm_path = Path(af3_iptm_dir) / f"{protein_id}.json"
            with open(af3_iptm_path, 'r') as f:
                af3_json_data = json.load(f)
            
            # Extract ipTM score (handle different JSON formats)
            if 'iptm' in af3_json_data:
                af3_iptm = af3_json_data['iptm']
            elif 'interface_ptm' in af3_json_data:
                af3_iptm = af3_json_data['interface_ptm']
            else:
                af3_iptm = None
            
            results['af3']['iptm'] = af3_iptm
        except Exception as e:
            print(f"  AF3 ipTM failed: {str(e)}")
            results['af3']['iptm'] = None
        
        return results
        
    except Exception as e:
        print(f"  Error analyzing {protein_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def batch_analyze(ref_dir, af2_dir, af3_dir, af2_csv_file, af3_csv_file,
                 af2_pae_dir, af3_pae_dir, af3_iptm_dir, output_dir="interface_results"):
    """
    Batch process all proteins with comprehensive analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Unified Interface Analysis for AlphaFold Predictions")
    print("="*60)
    print()
    
    # Initialize analyzers
    pae_analyzer = PAEAnalyzer(af2_pae_dir, af3_pae_dir)
    dssp_analyzer = DSSPAnalyzer()
    hydropathy_analyzer = HydropathyAnalyzer()
    
    if not DSSP_AVAILABLE:
        print("⚠ WARNING: DSSP is not available!")
        print("DSSP analysis will be skipped.\n")
    
    print("Loading chain mappings and DockQ scores from CSV files...")
    chain_mapping = load_chain_mapping(af2_csv_file, af3_csv_file)
    print(f"Loaded mappings for {len(chain_mapping)} proteins\n")
    
    print("Finding matching files...")
    matching_files = get_matching_files(ref_dir, af2_dir, af3_dir, chain_mapping)
    print(f"Found {len(matching_files)} matching protein complexes\n")
    
    if not matching_files:
        print("No matching files found. Please check your directories and CSV files.")
        return []
    
    all_results = []
    failed_proteins = []
    
    for i, (protein_id, ref_path, af2_path, af3_path, chain_info) in enumerate(matching_files, 1):
        print(f"[{i}/{len(matching_files)}] Processing {protein_id}...")
        print(f"  Reference chains: IDP={chain_info['ref_idp']}, Receptor={chain_info['ref_receptor']}")
        print(f"  AF2 DockQ: {chain_info.get('af2_dockq', 'N/A')}, AF3 DockQ: {chain_info.get('af3_dockq', 'N/A')}")
        
        results = analyze_single_protein(
            protein_id, ref_path, af2_path, af3_path, chain_info,
            pae_analyzer, dssp_analyzer, hydropathy_analyzer, af3_iptm_dir, af2_pae_dir
        )
        
        if results:
            all_results.append(results)
            with open(output_dir / f"{protein_id}.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Success")
        else:
            failed_proteins.append(protein_id)
            print(f"  ✗ Failed")
        
    
    # Save summary
    summary = {
        'total_proteins': len(matching_files),
        'successful': len(all_results),
        'failed': len(failed_proteins),
        'failed_proteins': failed_proteins,
        'dssp_available': DSSP_AVAILABLE,
        'results': all_results
    }
    
    with open(output_dir / "b90_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create CSV summary
    create_csv_summary(all_results, output_dir / "b90_summary.csv")
    
    print(f"\n{'='*60}")
    print(f"Interface analysis complete!")
    print(f"Successfully analyzed: {len(all_results)}/{len(matching_files)}")
    if failed_proteins:
        print(f"Failed proteins: {', '.join(failed_proteins)}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return all_results


def create_csv_summary(results_list, output_file):
    """Create CSV summary table of all metrics"""
    
    rows = []
    for result in results_list:
        pdb_id = result['protein_id']
        
        row = {
            'pdb_id': pdb_id,
            'af2_dockq': result.get('af2_dockq'),
            'af3_dockq': result.get('af3_dockq'),
            'af2_bsa': result.get('af2_bsa'),  # ADD THIS LINE
            'af3_bsa': result.get('af3_bsa'),  # ADD THIS LINE
        }
        
        # AF2 disorder metrics
        row['af2_idp_disorder_fraction'] = result['af2']['idp']['disorder_fraction']
        row['af2_idp_quality_score'] = result['af2']['idp']['quality_score']
        row['af2_idp_interface_residues'] = result['af2']['idp']['interface_residues']
        row['af2_rec_disorder_fraction'] = result['af2']['receptor']['disorder_fraction']
        row['af2_rec_quality_score'] = result['af2']['receptor']['quality_score']
        row['af2_rec_interface_residues'] = result['af2']['receptor']['interface_residues']
        
        # AF3 disorder metrics
        row['af3_idp_disorder_fraction'] = result['af3']['idp']['disorder_fraction']
        row['af3_idp_quality_score'] = result['af3']['idp']['quality_score']
        row['af3_idp_interface_residues'] = result['af3']['idp']['interface_residues']
        row['af3_rec_disorder_fraction'] = result['af3']['receptor']['disorder_fraction']
        row['af3_rec_quality_score'] = result['af3']['receptor']['quality_score']
        row['af3_rec_interface_residues'] = result['af3']['receptor']['interface_residues']
        
        # AF2 PAE metrics
        if result['af2'].get('pae'):
            row['af2_mean_pae'] = result['af2']['pae']['mean_pae']
            row['af2_median_pae'] = result['af2']['pae']['median_pae']
            row['af2_std_pae'] = result['af2']['pae']['std_pae']
        
        # AF3 PAE metrics
        if result['af3'].get('pae'):
            row['af3_mean_pae'] = result['af3']['pae']['mean_pae']
            row['af3_median_pae'] = result['af3']['pae']['median_pae']
            row['af3_std_pae'] = result['af3']['pae']['std_pae']
        
        # AF2 DSSP metrics
        if result['af2']['dssp']['idp']['available']:
            row['af2_idp_helix'] = result['af2']['dssp']['idp']['helix_fraction']
            row['af2_idp_sheet'] = result['af2']['dssp']['idp']['sheet_fraction']
            row['af2_idp_coil'] = result['af2']['dssp']['idp']['coil_fraction']
            row['af2_rec_helix'] = result['af2']['dssp']['receptor']['helix_fraction']
            row['af2_rec_sheet'] = result['af2']['dssp']['receptor']['sheet_fraction']
            row['af2_rec_coil'] = result['af2']['dssp']['receptor']['coil_fraction']
        
        # AF3 DSSP metrics
        if result['af3']['dssp']['idp']['available']:
            row['af3_idp_helix'] = result['af3']['dssp']['idp']['helix_fraction']
            row['af3_idp_sheet'] = result['af3']['dssp']['idp']['sheet_fraction']
            row['af3_idp_coil'] = result['af3']['dssp']['idp']['coil_fraction']
            row['af3_rec_helix'] = result['af3']['dssp']['receptor']['helix_fraction']
            row['af3_rec_sheet'] = result['af3']['dssp']['receptor']['sheet_fraction']
            row['af3_rec_coil'] = result['af3']['dssp']['receptor']['coil_fraction']
        
        # AF2 Hydropathy metrics
        row['af2_idp_hydropathy'] = result['af2']['hydropathy']['idp']['mean_hydropathy']
        row['af2_idp_hydrophobic_frac'] = result['af2']['hydropathy']['idp']['hydrophobic_fraction']
        row['af2_rec_hydropathy'] = result['af2']['hydropathy']['receptor']['mean_hydropathy']
        row['af2_rec_hydrophobic_frac'] = result['af2']['hydropathy']['receptor']['hydrophobic_fraction']
        
        # AF3 Hydropathy metrics
        row['af3_idp_hydropathy'] = result['af3']['hydropathy']['idp']['mean_hydropathy']
        row['af3_idp_hydrophobic_frac'] = result['af3']['hydropathy']['idp']['hydrophobic_fraction']
        row['af3_rec_hydropathy'] = result['af3']['hydropathy']['receptor']['mean_hydropathy']
        row['af3_rec_hydrophobic_frac'] = result['af3']['hydropathy']['receptor']['hydrophobic_fraction']
        
        # Reference Hydropathy metrics
        row['ref_idp_hydropathy'] = result['reference']['hydropathy']['idp']['mean_hydropathy']
        row['ref_idp_hydrophobic_frac'] = result['reference']['hydropathy']['idp']['hydrophobic_fraction']
        row['ref_rec_hydropathy'] = result['reference']['hydropathy']['receptor']['mean_hydropathy']
        row['ref_rec_hydrophobic_frac'] = result['reference']['hydropathy']['receptor']['hydrophobic_fraction']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"CSV summary saved to: {output_file}")
    print(f"Contains {len(df)} proteins with {len(df.columns)} metrics")
    return df

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


def analyze_disorder_dssp_regression(results_list, alphafold_version='af2'):
    """
    Perform regression analysis to understand how disorder fraction and DSSP matching
    contribute to predicting DockQ scores.
    
    Parameters:
    -----------
    results_list : list
        List of analysis results from the main script
    alphafold_version : str
        Either 'af2' or 'af3' to specify which model's data to analyze
    
    Returns:
    --------
    dict : Dictionary containing regression results, R² values, and raw data
    """
    
    # Initialize lists to store data
    disorder_fractions = []
    dssp_matching_fractions = []
    dockq_scores = []
    protein_ids = []
    
    # Extract data from results_list
    for results in results_list:
        # Get the appropriate data based on alphafold_version
        if alphafold_version == 'af2':
            af_key = 'af2'
            dockq_key = 'af2_dockq'
            af_to_ref_idp_key = 'af2_to_ref_idp'
            af_to_ref_rec_key = 'af2_to_ref_rec'
        else:  # af3
            af_key = 'af3'
            dockq_key = 'af3_dockq'
            af_to_ref_idp_key = 'af3_to_ref_idp'
            af_to_ref_rec_key = 'af3_to_ref_rec'
        
        # Get DockQ score
        dockq = results.get(dockq_key)
        if dockq is None:
            continue
        
        # Calculate weighted disorder fraction
        idp_interface = results[af_key]['idp']['interface_residues']
        rec_interface = results[af_key]['receptor']['interface_residues']
        idp_disorder_frac = results[af_key]['idp']['disorder_fraction']
        rec_disorder_frac = results[af_key]['receptor']['disorder_fraction']
        
        idp_disordered = idp_disorder_frac * idp_interface
        rec_disordered = rec_disorder_frac * rec_interface
        total_disordered = idp_disordered + rec_disordered
        total_interface = idp_interface + rec_interface
        
        if total_interface == 0:
            continue
        
        disorder_frac = total_disordered / total_interface
        
        # Calculate DSSP matching fraction
        if (results[af_key]['dssp']['idp']['available'] and 
            results[af_key]['dssp']['receptor']['available'] and
            results['reference']['dssp']['idp']['available'] and
            results['reference']['dssp']['receptor']['available'] and
            'mappings' in results):
            
            # Get DSSP structure dicts
            af_idp_dssp = results[af_key]['dssp']['idp']['structure_dict']
            af_rec_dssp = results[af_key]['dssp']['receptor']['structure_dict']
            ref_idp_dssp = results['reference']['dssp']['idp']['structure_dict']
            ref_rec_dssp = results['reference']['dssp']['receptor']['structure_dict']
            
            # Get mappings from results
            af_to_ref_idp = results['mappings'][af_to_ref_idp_key]
            af_to_ref_rec = results['mappings'][af_to_ref_rec_key]
            
            # Calculate matches for IDP chain
            idp_matches = 0
            idp_total = 0
            for af_res_id in af_idp_dssp.keys():
                if af_res_id in af_to_ref_idp:
                    ref_res_id = str(af_to_ref_idp[af_res_id])
                    if ref_res_id in ref_idp_dssp:
                        idp_total += 1
                        if af_idp_dssp[af_res_id] == ref_idp_dssp[ref_res_id]:
                            idp_matches += 1
            
            # Calculate matches for Receptor chain
            rec_matches = 0
            rec_total = 0
            for af_res_id in af_rec_dssp.keys():
                if af_res_id in af_to_ref_rec:
                    ref_res_id = str(af_to_ref_rec[af_res_id])
                    if ref_res_id in ref_rec_dssp:
                        rec_total += 1
                        if af_rec_dssp[af_res_id] == ref_rec_dssp[ref_res_id]:
                            rec_matches += 1
            
            # Combined matching fraction
            total_matches = idp_matches + rec_matches
            total_residues = idp_total + rec_total
            
            if total_residues > 0:
                dssp_match_frac = total_matches / total_residues
                
                # Add to lists
                disorder_fractions.append(disorder_frac)
                dssp_matching_fractions.append(dssp_match_frac)
                dockq_scores.append(dockq)
                protein_ids.append(results['protein_id'])
    
    # Convert to numpy arrays
    disorder_fraction = np.array(disorder_fractions)
    dssp_accuracy = np.array(dssp_matching_fractions)
    dockq_array = np.array(dockq_scores)
    
    if len(dockq_array) == 0:
        print(f"No valid data points found for {alphafold_version.upper()}")
        return None
    
    print(f"\n{'='*60}")
    print(f"REGRESSION ANALYSIS: {alphafold_version.upper()}")
    print(f"{'='*60}")
    print(f"Number of data points: {len(dockq_array)}")
    print(f"\nDATA RANGES:")
    print(f"  Disorder fraction:     {disorder_fraction.min():.3f} - {disorder_fraction.max():.3f}")
    print(f"  DSSP matching:         {dssp_accuracy.min():.3f} - {dssp_accuracy.max():.3f}")
    print(f"  DockQ scores:          {dockq_array.min():.3f} - {dockq_array.max():.3f}")
    
    # Calculate Pearson correlations
    corr_disorder, p_disorder = pearsonr(disorder_fraction, dockq_array)
    corr_dssp, p_dssp = pearsonr(dssp_accuracy, dockq_array)
    
    print(f"\nPEARSON CORRELATIONS:")
    print(f"  Disorder vs DockQ:     r = {corr_disorder:.3f}, p = {p_disorder:.4e}")
    print(f"  DSSP vs DockQ:         r = {corr_dssp:.3f}, p = {p_dssp:.4e}")
    
    # Model 1: Disorder only
    X1 = disorder_fraction.reshape(-1, 1)
    model1 = LinearRegression().fit(X1, dockq_array)
    r2_disorder_only = model1.score(X1, dockq_array)
    
    # Model 2: DSSP only
    X2 = dssp_accuracy.reshape(-1, 1)
    model2 = LinearRegression().fit(X2, dockq_array)
    r2_dssp_only = model2.score(X2, dockq_array)
    
    # Model 3: Both together
    X3 = np.column_stack([disorder_fraction, dssp_accuracy])
    model3 = LinearRegression().fit(X3, dockq_array)
    r2_both = model3.score(X3, dockq_array)
    
    print(f"\nVARIANCE EXPLAINED (R²):")
    print(f"  Disorder alone:        R² = {r2_disorder_only:.3f}")
    print(f"  DSSP alone:            R² = {r2_dssp_only:.3f}")
    print(f"  Both together:         R² = {r2_both:.3f}")
    
    print(f"\nIMPROVEMENT:")
    print(f"  Adding DSSP to Disorder:    ΔR² = {r2_both - r2_disorder_only:.3f}")
    print(f"  Adding Disorder to DSSP:    ΔR² = {r2_both - r2_dssp_only:.3f}")
    
    print(f"\nMODEL COEFFICIENTS:")
    print(f"  Model 1 (Disorder only):")
    print(f"    Intercept: {model1.intercept_:.3f}")
    print(f"    Coefficient: {model1.coef_[0]:.3f}")
    print(f"\n  Model 2 (DSSP only):")
    print(f"    Intercept: {model2.intercept_:.3f}")
    print(f"    Coefficient: {model2.coef_[0]:.3f}")
    print(f"\n  Model 3 (Both):")
    print(f"    Intercept: {model3.intercept_:.3f}")
    print(f"    Disorder coefficient: {model3.coef_[0]:.3f}")
    print(f"    DSSP coefficient: {model3.coef_[1]:.3f}")
    print(f"{'='*60}\n")
    
    # Return results dictionary
    return {
        'alphafold_version': alphafold_version,
        'n_samples': len(dockq_array),
        'disorder_fraction': disorder_fraction,
        'dssp_accuracy': dssp_accuracy,
        'dockq_scores': dockq_array,
        'protein_ids': protein_ids,
        'correlations': {
            'disorder_vs_dockq': {'r': corr_disorder, 'p': p_disorder},
            'dssp_vs_dockq': {'r': corr_dssp, 'p': p_dssp}
        },
        'models': {
            'disorder_only': {
                'model': model1,
                'r2': r2_disorder_only,
                'intercept': model1.intercept_,
                'coefficient': model1.coef_[0]
            },
            'dssp_only': {
                'model': model2,
                'r2': r2_dssp_only,
                'intercept': model2.intercept_,
                'coefficient': model2.coef_[0]
            },
            'both': {
                'model': model3,
                'r2': r2_both,
                'intercept': model3.intercept_,
                'coefficients': {
                    'disorder': model3.coef_[0],
                    'dssp': model3.coef_[1]
                }
            }
        },
        'improvements': {
            'dssp_added_to_disorder': r2_both - r2_disorder_only,
            'disorder_added_to_dssp': r2_both - r2_dssp_only
        }
    }



def analyze_disorder_dssp_regression_combined(results_list, output_dir="interface_results", selected="", top_n = 10):
    """
    Run regression analysis for both AF2 and AF3, compare results, and save to output directory.
    
    Parameters:
    -----------
    results_list : list
        List of analysis results from the main script
    output_dir : str or Path
        Directory to save output files
    
    Returns:
    --------
    dict : Dictionary containing results for both AF2 and AF3
    """
    import json
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Analyze AF2
    print("\n" + "="*60)
    print("RUNNING REGRESSION ANALYSIS")
    print("="*60)

    if selected != "":
        dockq_diffs = []
        for result in results_list:
            if result["protein_id"] == "7NMI" or result["protein_id"] == "7UZU" or result["protein_id"] == "7OS1":
                continue
            if result.get('af2_dockq') is not None and result.get('af3_dockq') is not None:
                diff = result['af3_dockq'] - result['af2_dockq']
                dockq_diffs.append((result['protein_id'], diff, result))
        
        # Sort by absolute difference
        dockq_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Select top N cases
        selected_cases = dockq_diffs[:top_n]
        results_list = []
        for case in selected_cases:
            results_list.append(case[2])

    af2_results = analyze_disorder_dssp_regression(results_list, 'af2')
    
    # Analyze AF3
    af3_results = analyze_disorder_dssp_regression(results_list, 'af3')
    
    # Print comparison summary
    if af2_results and af3_results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY: AF2 vs AF3")
        print(f"{'='*60}")
        print(f"{'Metric':<40} {'AF2':>8} {'AF3':>8}")
        print(f"{'-'*60}")
        print(f"{'Number of samples':<40} {af2_results['n_samples']:>8} {af3_results['n_samples']:>8}")
        print(f"\n{'R² Values:':<40}")
        print(f"{'  Disorder alone':<40} {af2_results['models']['disorder_only']['r2']:>8.3f} {af3_results['models']['disorder_only']['r2']:>8.3f}")
        print(f"{'  DSSP alone':<40} {af2_results['models']['dssp_only']['r2']:>8.3f} {af3_results['models']['dssp_only']['r2']:>8.3f}")
        print(f"{'  Both together':<40} {af2_results['models']['both']['r2']:>8.3f} {af3_results['models']['both']['r2']:>8.3f}")
        print(f"\n{'Improvement from adding DSSP:':<40} {af2_results['improvements']['dssp_added_to_disorder']:>8.3f} {af3_results['improvements']['dssp_added_to_disorder']:>8.3f}")
        print(f"{'Improvement from adding Disorder:':<40} {af2_results['improvements']['disorder_added_to_dssp']:>8.3f} {af3_results['improvements']['disorder_added_to_dssp']:>8.3f}")
        print(f"{'='*60}\n")
    
    combined_results = {
        'af2': af2_results,
        'af3': af3_results
    }
    
    # Save results to JSON (excluding sklearn models which can't be serialized)
    save_results = {}
    for version in ['af2', 'af3']:
        if combined_results[version] is not None:
            save_results[version] = {
                'alphafold_version': combined_results[version]['alphafold_version'],
                'n_samples': combined_results[version]['n_samples'],
                'protein_ids': combined_results[version]['protein_ids'],
                'disorder_fraction': combined_results[version]['disorder_fraction'].tolist(),
                'dssp_accuracy': combined_results[version]['dssp_accuracy'].tolist(),
                'dockq_scores': combined_results[version]['dockq_scores'].tolist(),
                'correlations': combined_results[version]['correlations'],
                'models': {
                    'disorder_only': {
                        'r2': combined_results[version]['models']['disorder_only']['r2'],
                        'intercept': float(combined_results[version]['models']['disorder_only']['intercept']),
                        'coefficient': float(combined_results[version]['models']['disorder_only']['coefficient'])
                    },
                    'dssp_only': {
                        'r2': combined_results[version]['models']['dssp_only']['r2'],
                        'intercept': float(combined_results[version]['models']['dssp_only']['intercept']),
                        'coefficient': float(combined_results[version]['models']['dssp_only']['coefficient'])
                    },
                    'both': {
                        'r2': combined_results[version]['models']['both']['r2'],
                        'intercept': float(combined_results[version]['models']['both']['intercept']),
                        'coefficients': {
                            'disorder': float(combined_results[version]['models']['both']['coefficients']['disorder']),
                            'dssp': float(combined_results[version]['models']['both']['coefficients']['dssp'])
                        }
                    }
                },
                'improvements': combined_results[version]['improvements']
            }
    
    # Save to JSON file
    json_file = output_dir / f"regression_analysis{selected}.json"
    with open(json_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"✓ Regression results saved to: {json_file}")
    
    # Save summary table to text file
    summary_file = output_dir / f"regression_summary{selected}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("REGRESSION ANALYSIS SUMMARY: AF2 vs AF3\n")
        f.write("="*60 + "\n\n")
        
        if af2_results and af3_results:
            f.write(f"{'Metric':<40} {'AF2':>8} {'AF3':>8}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Number of samples':<40} {af2_results['n_samples']:>8} {af3_results['n_samples']:>8}\n\n")
            
            f.write("R² Values:\n")
            f.write(f"{'  Disorder alone':<40} {af2_results['models']['disorder_only']['r2']:>8.3f} {af3_results['models']['disorder_only']['r2']:>8.3f}\n")
            f.write(f"{'  DSSP alone':<40} {af2_results['models']['dssp_only']['r2']:>8.3f} {af3_results['models']['dssp_only']['r2']:>8.3f}\n")
            f.write(f"{'  Both together':<40} {af2_results['models']['both']['r2']:>8.3f} {af3_results['models']['both']['r2']:>8.3f}\n\n")
            
            f.write("Pearson Correlations:\n")
            f.write(f"{'  Disorder vs DockQ (AF2)':<40} {af2_results['correlations']['disorder_vs_dockq']['r']:>8.3f} (p={af2_results['correlations']['disorder_vs_dockq']['p']:.4e})\n")
            f.write(f"{'  DSSP vs DockQ (AF2)':<40} {af2_results['correlations']['dssp_vs_dockq']['r']:>8.3f} (p={af2_results['correlations']['dssp_vs_dockq']['p']:.4e})\n")
            f.write(f"{'  Disorder vs DockQ (AF3)':<40} {af3_results['correlations']['disorder_vs_dockq']['r']:>8.3f} (p={af3_results['correlations']['disorder_vs_dockq']['p']:.4e})\n")
            f.write(f"{'  DSSP vs DockQ (AF3)':<40} {af3_results['correlations']['dssp_vs_dockq']['r']:>8.3f} (p={af3_results['correlations']['dssp_vs_dockq']['p']:.4e})\n\n")
            
            f.write("Improvement (ΔR²):\n")
            f.write(f"{'  Adding DSSP to Disorder':<40} {af2_results['improvements']['dssp_added_to_disorder']:>8.3f} {af3_results['improvements']['dssp_added_to_disorder']:>8.3f}\n")
            f.write(f"{'  Adding Disorder to DSSP':<40} {af2_results['improvements']['disorder_added_to_dssp']:>8.3f} {af3_results['improvements']['disorder_added_to_dssp']:>8.3f}\n\n")
            
            f.write("="*60 + "\n\n")
            
            f.write("MODEL COEFFICIENTS:\n\n")
            for version, res in [('AF2', af2_results), ('AF3', af3_results)]:
                f.write(f"{version}:\n")
                f.write(f"  Model 1 (Disorder only): DockQ = {res['models']['disorder_only']['intercept']:.3f} + {res['models']['disorder_only']['coefficient']:.3f} × Disorder\n")
                f.write(f"  Model 2 (DSSP only):     DockQ = {res['models']['dssp_only']['intercept']:.3f} + {res['models']['dssp_only']['coefficient']:.3f} × DSSP\n")
                f.write(f"  Model 3 (Both):          DockQ = {res['models']['both']['intercept']:.3f} + {res['models']['both']['coefficients']['disorder']:.3f} × Disorder + {res['models']['both']['coefficients']['dssp']:.3f} × DSSP\n\n")
    
    print(f"✓ Regression summary saved to: {summary_file}")
    
    # Save CSV with per-protein data
    import pandas as pd
    
    for version in ['af2', 'af3']:
        if combined_results[version] is not None:
            df = pd.DataFrame({
                'protein_id': combined_results[version]['protein_ids'],
                'disorder_fraction': combined_results[version]['disorder_fraction'],
                'dssp_accuracy': combined_results[version]['dssp_accuracy'],
                'dockq_score': combined_results[version]['dockq_scores']
            })
            csv_file = output_dir / f"regression_data_{version}{selected}.csv"
            df.to_csv(csv_file, index=False)
            print(f"✓ {version.upper()} data saved to: {csv_file}")
    
    print(f"\n{'='*60}")
    print("REGRESSION ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return combined_results


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import pandas as pd
from pathlib import Path
import json


def analyze_af3_extended_regression(results_list, output_dir="extended_results"):
    """
    Extended regression analysis for AF3 to predict DockQ scores.
    Tests combinations of: Disorder, DSSP, Interface Size, BSA, Hydrophobicity
    
    Parameters:
    -----------
    results_list : list
        List of analysis results from the main script
    output_dir : str or Path
        Directory to save output files
    
    Returns:
    --------
    dict : Dictionary containing all regression results and model comparisons
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("EXTENDED REGRESSION ANALYSIS: AF3")
    print(f"{'='*70}")
    
    # ========================================
    # DATA EXTRACTION
    # ========================================
    
    disorder_fractions = []
    dssp_matching_fractions = []
    interface_sizes = []
    bsa_values = []
    hydrophobicity_scores = []
    dockq_scores = []
    protein_ids = []
    
    for results in results_list:
        if results["protein_id"] == "7UZU":
            continue
        # Get DockQ score
        dockq = results.get('af3_dockq')
        if dockq is None:
            continue
        
        # Get BSA
        bsa = results.get('af3_bsa')
        if bsa is None:
            continue
        
        # Calculate weighted disorder fraction
        idp_interface = results['af3']['idp']['interface_residues']
        rec_interface = results['af3']['receptor']['interface_residues']
        idp_disorder_frac = results['af3']['idp']['disorder_fraction']
        rec_disorder_frac = results['af3']['receptor']['disorder_fraction']
        
        idp_disordered = idp_disorder_frac * idp_interface
        rec_disordered = rec_disorder_frac * rec_interface
        total_disordered = idp_disordered + rec_disordered
        total_interface = idp_interface + rec_interface
        
        if total_interface == 0:
            continue
        
        disorder_frac = total_disordered / total_interface
        
        # Calculate DSSP matching fraction
        if (results['af3']['dssp']['idp']['available'] and 
            results['af3']['dssp']['receptor']['available'] and
            results['reference']['dssp']['idp']['available'] and
            results['reference']['dssp']['receptor']['available'] and
            'mappings' in results):
            
            # Get DSSP structure dicts
            af3_idp_dssp = results['af3']['dssp']['idp']['structure_dict']
            af3_rec_dssp = results['af3']['dssp']['receptor']['structure_dict']
            ref_idp_dssp = results['reference']['dssp']['idp']['structure_dict']
            ref_rec_dssp = results['reference']['dssp']['receptor']['structure_dict']
            
            # Get mappings
            af3_to_ref_idp = results['mappings']['af3_to_ref_idp']
            af3_to_ref_rec = results['mappings']['af3_to_ref_rec']
            
            # Calculate matches for IDP chain
            idp_matches = 0
            idp_total = 0
            for af3_res_id in af3_idp_dssp.keys():
                if af3_res_id in af3_to_ref_idp:
                    ref_res_id = str(af3_to_ref_idp[af3_res_id])
                    if ref_res_id in ref_idp_dssp:
                        idp_total += 1
                        if af3_idp_dssp[af3_res_id] == ref_idp_dssp[ref_res_id]:
                            idp_matches += 1
            
            # Calculate matches for Receptor chain
            rec_matches = 0
            rec_total = 0
            for af3_res_id in af3_rec_dssp.keys():
                if af3_res_id in af3_to_ref_rec:
                    ref_res_id = str(af3_to_ref_rec[af3_res_id])
                    if ref_res_id in ref_rec_dssp:
                        rec_total += 1
                        if af3_rec_dssp[af3_res_id] == ref_rec_dssp[ref_res_id]:
                            rec_matches += 1
            
            # Combined matching fraction
            total_matches = idp_matches + rec_matches
            total_residues = idp_total + rec_total
            
            if total_residues == 0:
                continue
            
            dssp_match_frac = total_matches / total_residues
        else:
            continue
        
        # Calculate weighted hydrophobicity
        idp_hydro = results['af3']['hydropathy']['idp']['mean_hydropathy']
        rec_hydro = results['af3']['hydropathy']['receptor']['mean_hydropathy']
        weighted_hydro = (idp_hydro * idp_interface + rec_hydro * rec_interface) / total_interface
        
        # Add to lists
        disorder_fractions.append(disorder_frac)
        dssp_matching_fractions.append(dssp_match_frac)
        interface_sizes.append(total_interface)
        bsa_values.append(bsa)
        hydrophobicity_scores.append(weighted_hydro)
        dockq_scores.append(dockq)
        protein_ids.append(results['protein_id'])
    
    # Convert to numpy arrays
    disorder = np.array(disorder_fractions)
    dssp = np.array(dssp_matching_fractions)
    interface_size = np.array(interface_sizes)
    bsa = np.array(bsa_values)
    hydrophobicity = np.array(hydrophobicity_scores)
    dockq = np.array(dockq_scores)
    
    if len(dockq) == 0:
        print("ERROR: No valid data points found for AF3 extended analysis")
        return None
    
    print(f"\nData Summary:")
    print(f"  Number of samples: {len(dockq)}")
    print(f"\n  Variable Ranges:")
    print(f"    Disorder fraction:     {disorder.min():.3f} - {disorder.max():.3f}")
    print(f"    DSSP matching:         {dssp.min():.3f} - {dssp.max():.3f}")
    print(f"    Interface size:        {interface_size.min():.0f} - {interface_size.max():.0f} residues")
    print(f"    BSA:                   {bsa.min():.1f} - {bsa.max():.1f} Ų")
    print(f"    Hydrophobicity:        {hydrophobicity.min():.3f} - {hydrophobicity.max():.3f}")
    print(f"    DockQ scores:          {dockq.min():.3f} - {dockq.max():.3f}")
    
    # ========================================
    # CORRELATION ANALYSIS
    # ========================================
    
    print(f"\n{'-'*70}")
    print("PEARSON CORRELATIONS WITH DOCKQ")
    print(f"{'-'*70}")
    
    variables = {
        'Disorder': disorder,
        'DSSP': dssp,
        'Interface Size': interface_size,
        'BSA': bsa,
        'Hydrophobicity': hydrophobicity
    }
    
    correlations = {}
    for var_name, var_data in variables.items():
        r, p = pearsonr(var_data, dockq)
        correlations[var_name] = {'r': r, 'p': p}
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var_name:<20} r = {r:>7.4f}, p = {p:.4e} {significance}")
    
    # ========================================
    # REGRESSION MODELS
    # ========================================
    
    print(f"\n{'-'*70}")
    print("REGRESSION MODELS")
    print(f"{'-'*70}")
    
    models = {}
    
    # Model 1: Baseline - Disorder + DSSP
    X1 = np.column_stack([disorder, dssp])
    model1 = LinearRegression().fit(X1, dockq)
    models['baseline'] = {
        'name': 'Disorder + DSSP (Baseline)',
        'features': ['Disorder', 'DSSP'],
        'X': X1,
        'model': model1,
        'r2': model1.score(X1, dockq),
        'coefficients': model1.coef_,
        'intercept': model1.intercept_
    }
    
    # Model 2: + Interface Size
    X2 = np.column_stack([disorder, dssp, interface_size])
    model2 = LinearRegression().fit(X2, dockq)
    models['add_size'] = {
        'name': 'Baseline + Interface Size',
        'features': ['Disorder', 'DSSP', 'Size'],
        'X': X2,
        'model': model2,
        'r2': model2.score(X2, dockq),
        'coefficients': model2.coef_,
        'intercept': model2.intercept_
    }
    
    # Model 3: + BSA
    X3 = np.column_stack([disorder, dssp, bsa])
    model3 = LinearRegression().fit(X3, dockq)
    models['add_bsa'] = {
        'name': 'Baseline + BSA',
        'features': ['Disorder', 'DSSP', 'BSA'],
        'X': X3,
        'model': model3,
        'r2': model3.score(X3, dockq),
        'coefficients': model3.coef_,
        'intercept': model3.intercept_
    }
    
    # Model 4: + Hydrophobicity
    X4 = np.column_stack([disorder, dssp, hydrophobicity])
    model4 = LinearRegression().fit(X4, dockq)
    models['add_hydro'] = {
        'name': 'Baseline + Hydrophobicity',
        'features': ['Disorder', 'DSSP', 'Hydro'],
        'X': X4,
        'model': model4,
        'r2': model4.score(X4, dockq),
        'coefficients': model4.coef_,
        'intercept': model4.intercept_
    }
    
    # Model 5: + Size + BSA
    X5 = np.column_stack([disorder, dssp, interface_size, bsa])
    model5 = LinearRegression().fit(X5, dockq)
    models['add_size_bsa'] = {
        'name': 'Baseline + Size + BSA',
        'features': ['Disorder', 'DSSP', 'Size', 'BSA'],
        'X': X5,
        'model': model5,
        'r2': model5.score(X5, dockq),
        'coefficients': model5.coef_,
        'intercept': model5.intercept_
    }
    
    # Model 6: All features
    X6 = np.column_stack([disorder, dssp, interface_size, bsa, hydrophobicity])
    model6 = LinearRegression().fit(X6, dockq)
    models['all_features'] = {
        'name': 'All Features',
        'features': ['Disorder', 'DSSP', 'Size', 'BSA', 'Hydro'],
        'X': X6,
        'model': model6,
        'r2': model6.score(X6, dockq),
        'coefficients': model6.coef_,
        'intercept': model6.intercept_
    }
    
    # Print model comparison
    print(f"\nModel Performance Comparison:")
    print(f"{'Model':<45} {'R²':>8} {'ΔR²':>10} {'Features':>3}")
    print(f"{'-'*70}")
    
    baseline_r2 = models['baseline']['r2']
    for model_key, model_info in models.items():
        improvement = model_info['r2'] - baseline_r2
        n_features = len(model_info['features'])
        print(f"{model_info['name']:<45} {model_info['r2']:>8.4f} {improvement:>10.4f} {n_features:>3}")
    
    # Find best model
    best_model_key = max(models.keys(), key=lambda k: models[k]['r2'])
    best_model = models[best_model_key]
    
    print(f"\n{'-'*70}")
    print(f"BEST MODEL: {best_model['name']}")
    print(f"{'-'*70}")
    print(f"  R² = {best_model['r2']:.4f}")
    print(f"  Improvement over baseline: ΔR² = {best_model['r2'] - baseline_r2:.4f}")
    print(f"\n  Model Equation:")
    print(f"  DockQ = {best_model['intercept']:.4f}", end="")
    for i, feature in enumerate(best_model['features']):
        coef = best_model['coefficients'][i]
        sign = "+" if coef >= 0 else "-"
        print(f" {sign} {abs(coef):.4f}×{feature}", end="")
    print()
    
    print(f"\n  Individual Coefficients:")
    for i, feature in enumerate(best_model['features']):
        print(f"    {feature:<20} {best_model['coefficients'][i]:>10.6f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    
    # Save to JSON
    save_results = {
        'n_samples': len(dockq),
        'protein_ids': protein_ids,
        'data_ranges': {
            'disorder': {'min': float(disorder.min()), 'max': float(disorder.max())},
            'dssp': {'min': float(dssp.min()), 'max': float(dssp.max())},
            'interface_size': {'min': float(interface_size.min()), 'max': float(interface_size.max())},
            'bsa': {'min': float(bsa.min()), 'max': float(bsa.max())},
            'hydrophobicity': {'min': float(hydrophobicity.min()), 'max': float(hydrophobicity.max())},
            'dockq': {'min': float(dockq.min()), 'max': float(dockq.max())}
        },
        'correlations': {k: {'r': float(v['r']), 'p': float(v['p'])} for k, v in correlations.items()},
        'models': {
            k: {
                'name': v['name'],
                'features': v['features'],
                'r2': float(v['r2']),
                'intercept': float(v['intercept']),
                'coefficients': [float(c) for c in v['coefficients']],
                'delta_r2': float(v['r2'] - baseline_r2)
            } for k, v in models.items()
        },
        'best_model': best_model_key,
        'baseline_r2': float(baseline_r2),
        'best_r2': float(best_model['r2']),
        'improvement': float(best_model['r2'] - baseline_r2)
    }
    
    json_file = output_dir / "af3_extended_regression.json"
    with open(json_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n✓ Results saved to: {json_file}")
    
    # Save summary CSV
    csv_data = {
        'protein_id': protein_ids,
        'disorder_fraction': disorder,
        'dssp_accuracy': dssp,
        'interface_size': interface_size,
        'bsa': bsa,
        'hydrophobicity': hydrophobicity,
        'dockq_actual': dockq,
        'dockq_predicted_baseline': models['baseline']['model'].predict(models['baseline']['X']),
        'dockq_predicted_best': best_model['model'].predict(best_model['X'])
    }
    
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / "af3_extended_regression_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Data saved to: {csv_file}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    # Return complete results for plotting
    return {
        'n_samples': len(dockq),
        'protein_ids': protein_ids,
        'data': {
            'disorder': disorder,
            'dssp': dssp,
            'interface_size': interface_size,
            'bsa': bsa,
            'hydrophobicity': hydrophobicity,
            'dockq': dockq
        },
        'correlations': correlations,
        'models': models,
        'best_model_key': best_model_key,
        'baseline_r2': baseline_r2
    }


def plot_af3_extended_regression(results_dict, output_dir="extended_results"):
    """
    Create comprehensive visualization of AF3 extended regression analysis
    with layout:
      Row 1: Disorder, DSSP, Interface Size
      Row 2: BSA, Hydrophobicity, Best model predictions
      Row 3: Model comparison, Feature importance
    """
    if results_dict is None:
        print("ERROR: No results to plot")
        return
    
    output_dir = Path(output_dir)
    
    # Extract data - these are already numpy arrays in the results_dict
    data = results_dict['data']
    disorder = data['disorder']
    dssp = data['dssp']
    interface_size = data['interface_size']
    bsa = data['bsa']
    hydrophobicity = data['hydrophobicity']
    dockq = data['dockq']
    
    models = results_dict['models']
    correlations = results_dict['correlations']
    best_model_key = results_dict['best_model_key']
    baseline_r2 = results_dict['baseline_r2']
    best_model = models[best_model_key]
    
    # Validation
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATION")
    print(f"{'='*70}")
    print(f"Data shapes and ranges:")
    print(f"  disorder: shape={disorder.shape}, range=[{disorder.min():.4f}, {disorder.max():.4f}]")
    print(f"  dssp: shape={dssp.shape}, range=[{dssp.min():.4f}, {dssp.max():.4f}]")
    print(f"  interface_size: shape={interface_size.shape}, range=[{interface_size.min():.1f}, {interface_size.max():.1f}]")
    print(f"  bsa: shape={bsa.shape}, range=[{bsa.min():.1f}, {bsa.max():.1f}]")
    print(f"  hydrophobicity: shape={hydrophobicity.shape}, range=[{hydrophobicity.min():.4f}, {hydrophobicity.max():.4f}]")
    print(f"  dockq: shape={dockq.shape}, range=[{dockq.min():.4f}, {dockq.max():.4f}]")
    print(f"\nNumber of valid samples: {len(dockq)}")
    
    if len(dockq) == 0:
        print("ERROR: No valid data points!")
        return
    
    # Create figure with 3 rows, 3 columns
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        3, 3, hspace=0.35, wspace=0.35,
        left=0.08, right=0.96, top=0.94, bottom=0.06
    )
    
    # Common predictor metadata
    predictors_data = [
        ('Disorder Fraction', disorder, 'Blues'),
        ('DSSP Matching', dssp, 'Greens'),
        ('Interface Size (res)', interface_size, 'Oranges'),
        ('BSA (Ų)', bsa, 'Purples'),
        ('Hydrophobicity', hydrophobicity, 'Reds')
    ]
    
    corr_keys = {
        'Disorder Fraction': 'Disorder',
        'DSSP Matching': 'DSSP',
        'Interface Size (res)': 'Interface Size',
        'BSA (Ų)': 'BSA',
        'Hydrophobicity': 'Hydrophobicity'
    }
    
    # ========================================
    # ROW 1: Disorder, DSSP, Interface Size
    # ========================================
    top_axes = []
    
    for idx in range(3):
        label, pred_data, cmap = predictors_data[idx]
        ax = fig.add_subplot(gs[0, idx])
        top_axes.append(ax)
        
        print(f"\nPlotting {label}:")
        print(f"  X range: [{pred_data.min():.4f}, {pred_data.max():.4f}]")
        print(f"  Y range: [{dockq.min():.4f}, {dockq.max():.4f}]")
        print(f"  Number of points: {len(pred_data)}")
        
        # Scatter
        ax.scatter(
            pred_data, dockq, alpha=0.7, s=80,
            c=pred_data, cmap=cmap,
            edgecolors='black', linewidth=0.8, zorder=3
        )
        
        # Regression line
        X = pred_data.reshape(-1, 1)
        model = LinearRegression().fit(X, dockq)
        x_line = np.linspace(pred_data.min(), pred_data.max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, zorder=2)
        
        # Correlation info
        key = corr_keys[label]
        r = correlations[key]['r']
        p = correlations[key]['p']
        r2 = model.score(X, dockq)
        
        ax.text(
            0.05, 0.95,
            f"r = {r:.3f}\nR² = {r2:.3f}\np = {p:.2e}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_xlabel(label, fontsize=10, fontweight='bold')
        ax.set_ylabel('DockQ Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{label} → DockQ', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        x_margin = (pred_data.max() - pred_data.min()) * 0.1
        ax.set_xlim(pred_data.min() - x_margin, pred_data.max() + x_margin)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    
    # ========================================
    # ROW 2: BSA, Hydrophobicity, Best model predictions
    # ========================================
    mid_axes = []
    
    # BSA and Hydrophobicity scatter
    for idx in range(2):
        plot_idx = idx + 3  # 3 -> BSA, 4 -> Hydrophobicity
        label, pred_data, cmap = predictors_data[plot_idx]
        ax = fig.add_subplot(gs[1, idx])
        mid_axes.append(ax)
        
        print(f"\nPlotting {label}:")
        print(f"  X range: [{pred_data.min():.4f}, {pred_data.max():.4f}]")
        print(f"  Y range: [{dockq.min():.4f}, {dockq.max():.4f}]")
        print(f"  Number of points: {len(pred_data)}")
        
        ax.scatter(
            pred_data, dockq, alpha=0.7, s=80,
            c=pred_data, cmap=cmap,
            edgecolors='black', linewidth=0.8, zorder=3
        )
        
        X = pred_data.reshape(-1, 1)
        model = LinearRegression().fit(X, dockq)
        x_line = np.linspace(pred_data.min(), pred_data.max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, zorder=2)
        
        key = corr_keys[label]
        r = correlations[key]['r']
        p = correlations[key]['p']
        r2 = model.score(X, dockq)
        
        ax.text(
            0.05, 0.95,
            f"r = {r:.3f}\nR² = {r2:.3f}\np = {p:.2e}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_xlabel(label, fontsize=10, fontweight='bold')
        ax.set_ylabel('DockQ Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{label} → DockQ', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        x_margin = (pred_data.max() - pred_data.min()) * 0.1
        ax.set_xlim(pred_data.min() - x_margin, pred_data.max() + x_margin)
        ax.set_ylim(-0.05, 1.05)
    
    # Best model predictions vs actual (right plot of row 2)
    ax_best = fig.add_subplot(gs[1, 2])
    mid_axes.append(ax_best)
    
    best_pred = best_model['model'].predict(best_model['X'])
    
    scatter_best = ax_best.scatter(
        dockq, best_pred, alpha=0.6, s=60,
        c=bsa, cmap='viridis',
        edgecolors='black', linewidth=0.5
    )
    
    min_val = min(dockq.min(), best_pred.min())
    max_val = max(dockq.max(), best_pred.max())
    ax_best.plot(
        [min_val, max_val], [min_val, max_val], 'r--',
        linewidth=2.5, alpha=0.8, label='Perfect'
    )
    
    improvement = best_model['r2'] - baseline_r2
    ax_best.text(
        0.05, 0.95,
        f"Best: {best_model['name']}\nR² = {best_model['r2']:.4f}\nΔR² = +{improvement:.4f}",
        transform=ax_best.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax_best.set_xlabel('Actual DockQ', fontsize=10, fontweight='bold')
    ax_best.set_ylabel('Predicted DockQ', fontsize=10, fontweight='bold')
    ax_best.set_title('Best Model Predictions', fontsize=11, fontweight='bold')
    ax_best.grid(True, alpha=0.3)
    ax_best.legend(fontsize=9)
    
    cbar_best = plt.colorbar(scatter_best, ax=ax_best)
    cbar_best.set_label('BSA (Ų)', fontsize=9)
    
    # ========================================
    # ROW 3: Model comparison (R²) and Coefficients/Correlations
    # ========================================
    
    # Model comparison bar chart (span 2 columns)
    ax_r2 = fig.add_subplot(gs[2, 0:2])
    
    model_keys = list(models.keys())
    model_names = [models[k]['name'] for k in model_keys]
    r2_values = [models[k]['r2'] for k in model_keys]
    improvements = [models[k]['r2'] - baseline_r2 for k in model_keys]
    
    x = np.arange(len(model_names))
    colors = []
    for k in model_keys:
        if k == 'baseline':
            colors.append('steelblue')
        elif k == best_model_key:
            colors.append('indianred')
        else:
            colors.append('lightgray')
    
    bars = ax_r2.bar(x, r2_values, alpha=0.7, edgecolor='black', linewidth=1.2, color=colors)
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax_r2.text(
            bar.get_x() + bar.get_width() / 2., height + 0.01,
            f'{height:.4f}\n(+{imp:.4f})',
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )
    
    ax_r2.set_ylabel('R² (Variance Explained)', fontsize=11, fontweight='bold')
    ax_r2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(
        [name.replace(' + ', '\n+\n').replace('Baseline', 'Base') for name in model_names],
        fontsize=7
    )
    ax_r2.grid(True, alpha=0.3, axis='y')
    ax_r2.set_ylim(0, max(r2_values) * 1.25)
    ax_r2.axhline(
        y=baseline_r2, color='steelblue', linestyle='--',
        linewidth=2, alpha=0.7, label='Baseline'
    )
    ax_r2.legend(fontsize=9, loc='upper left')
    
    # Coefficients and correlations (right of row 3)
    ax_coeff = fig.add_subplot(gs[2, 2])
    
    features = best_model['features']
    coefficients = best_model['coefficients']
    
    feature_to_corr = {
        'Disorder': 'Disorder',
        'DSSP': 'DSSP',
        'Size': 'Interface Size',
        'BSA': 'BSA',
        'Hydro': 'Hydrophobicity'
    }
    corr_values = [correlations[feature_to_corr[f]]['r'] for f in features]
    
    y_pos = np.arange(len(features))
    
    colors_coef = ['steelblue' if c > 0 else 'indianred' for c in coefficients]
    ax_coeff.barh(
        y_pos + 0.2, coefficients, height=0.35,
        color=colors_coef, alpha=0.7,
        edgecolor='black', linewidth=1.2, label='Coefficient'
    )
    
    colors_corr = ['steelblue' if c > 0 else 'indianred' for c in corr_values]
    ax_coeff.barh(
        y_pos - 0.2, corr_values, height=0.35,
        color=colors_corr, alpha=0.4,
        edgecolor='black', linewidth=1.2, label='Correlation (r)'
    )
    
    for i, (coef, corr) in enumerate(zip(coefficients, corr_values)):
        ax_coeff.text(
            coef, i + 0.2, f' {coef:.4f}',
            ha='left' if -0.7 < coef < 0.4 else 'right',
            va='center', fontsize=8, fontweight='bold'
        )
        ax_coeff.text(
            corr, i - 0.2, f' {corr:.3f}',
            ha='left' if -0.7 < corr < 0.4 else 'right',
            va='center', fontsize=8, fontweight='bold'
        )
    
    ax_coeff.set_yticks(y_pos)
    ax_coeff.set_yticklabels(features, fontsize=10)
    ax_coeff.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax_coeff.set_xlabel('Value', fontsize=10, fontweight='bold')
    ax_coeff.set_title('Best Model:\nCoefficients & Correlations', fontsize=11, fontweight='bold')
    ax_coeff.legend(fontsize=8, loc='best')
    ax_coeff.grid(True, alpha=0.3, axis='x')
    
    # ========================================
    # PANEL LABELS (A–H) using existing axes
    # ========================================
    all_axes = [
        *top_axes,          # Row 1: A, B, C
        *mid_axes,          # Row 2: D, E, F
        ax_r2, ax_coeff     # Row 3: G, H
    ]
    
    for i, ax in enumerate(all_axes):
        label = chr(65 + i)  # 'A'..'H'
        ax.text(
            0.0, 1.12, label,
            transform=ax.transAxes,
            fontsize=16, fontweight='bold',
            va='top', ha='right'
        )
    
    # Overall title
    fig.suptitle(
        'AF3 Extended Regression Analysis: Predicting DockQ from Multiple Interface Properties',
        fontsize=14, fontweight='bold', y=0.99
    )
    
    # Save figure
    output_file = output_dir / "af3_extended_regression.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\n✓ Regression visualization saved to: {output_file}")
    
    plt.close()
    
    # Summary
    print(f"\n{'='*70}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Samples plotted: {len(dockq)}")
    print(f"Baseline R²: {baseline_r2:.4f}")
    print(f"Best model: {best_model['name']}")
    print(f"Best R²: {best_model['r2']:.4f}")
    print(f"Improvement: ΔR² = +{best_model['r2'] - baseline_r2:.4f}")
    print("\nPlot layout:")
    print("  Row 1 (A–C): Disorder, DSSP, Interface Size")
    print("  Row 2 (D–F): BSA, Hydrophobicity, Best model predictions")
    print("  Row 3 (G–H): Model comparison (R²), Coefficients & correlations")
    print(f"{'='*70}\n")



def plot_disorder_dssp_regression(json_file="interface_results/regression_analysis.json", 
                                   output_dir="interface_results", selected=""):
    """
    Create comprehensive visualization of disorder-DSSP regression analysis
    
    Parameters:
    -----------
    json_file : str or Path
        Path to regression_analysis.json file
    output_dir : str or Path
        Directory to save output figures
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract AF2 and AF3 data
    af2_data = data.get('af2')
    af3_data = data.get('af3')
    
    if not af2_data or not af3_data:
        print("Error: Missing AF2 or AF3 data in JSON file")
        return
    
    # Convert to numpy arrays
    af2_disorder = np.array(af2_data['disorder_fraction'])
    af2_dssp = np.array(af2_data['dssp_accuracy'])
    af2_dockq = np.array(af2_data['dockq_scores'])
    
    af3_disorder = np.array(af3_data['disorder_fraction'])
    af3_dssp = np.array(af3_data['dssp_accuracy'])
    af3_dockq = np.array(af3_data['dockq_scores'])
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    af2_color = '#2E86AB'  # Blue
    af3_color = '#A23B72'  # Purple
    
    # ============================================================
    # ROW 1: Scatter plots with regression lines
    # ============================================================
    
    # Plot 1: AF2 Disorder vs DockQ
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(af2_disorder, af2_dockq, alpha=0.6, s=60, 
                          c=af2_disorder, cmap='Blues', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    X = af2_disorder.reshape(-1, 1)
    model = LinearRegression().fit(X, af2_dockq)
    x_line = np.linspace(af2_disorder.min(), af2_disorder.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax1.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, label='Linear fit')
    
    # Add statistics
    r2 = af2_data['models']['disorder_only']['r2']
    corr = af2_data['correlations']['disorder_vs_dockq']['r']
    ax1.text(0.05, 0.95, f"R² = {r2:.3f}\nr = {corr:.3f}", 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Disorder Fraction', fontsize=11, fontweight='bold')
    ax1.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
    ax1.set_title('AF2: Disorder → DockQ', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='lower left')
    
    # Plot 2: AF2 DSSP vs DockQ
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(af2_dssp, af2_dockq, alpha=0.6, s=60,
                          c=af2_dssp, cmap='Greens', edgecolors='black', linewidth=0.5)
    
    X = af2_dssp.reshape(-1, 1)
    model = LinearRegression().fit(X, af2_dockq)
    x_line = np.linspace(af2_dssp.min(), af2_dssp.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax2.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, label='Linear fit')
    
    r2 = af2_data['models']['dssp_only']['r2']
    corr = af2_data['correlations']['dssp_vs_dockq']['r']
    ax2.text(0.05, 0.95, f"R² = {r2:.3f}\nr = {corr:.3f}",
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('DSSP Matching Accuracy', fontsize=11, fontweight='bold')
    ax2.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
    ax2.set_title('AF2: DSSP → DockQ', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='lower right')
    
    # Plot 3: AF3 Disorder vs DockQ
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(af3_disorder, af3_dockq, alpha=0.6, s=60, marker='^',
                          c=af3_disorder, cmap='Purples', edgecolors='black', linewidth=0.5)
    
    X = af3_disorder.reshape(-1, 1)
    model = LinearRegression().fit(X, af3_dockq)
    x_line = np.linspace(af3_disorder.min(), af3_disorder.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax3.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, label='Linear fit')
    
    r2 = af3_data['models']['disorder_only']['r2']
    corr = af3_data['correlations']['disorder_vs_dockq']['r']
    ax3.text(0.05, 0.95, f"R² = {r2:.3f}\nr = {corr:.3f}",
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Disorder Fraction', fontsize=11, fontweight='bold')
    ax3.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
    ax3.set_title('AF3: Disorder → DockQ', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='lower left')
    
    # Plot 4: AF3 DSSP vs DockQ
    ax4 = fig.add_subplot(gs[0, 3])
    scatter4 = ax4.scatter(af3_dssp, af3_dockq, alpha=0.6, s=60, marker='^',
                          c=af3_dssp, cmap='Oranges', edgecolors='black', linewidth=0.5)
    
    X = af3_dssp.reshape(-1, 1)
    model = LinearRegression().fit(X, af3_dockq)
    x_line = np.linspace(af3_dssp.min(), af3_dssp.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax4.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, label='Linear fit')
    
    r2 = af3_data['models']['dssp_only']['r2']
    corr = af3_data['correlations']['dssp_vs_dockq']['r']
    ax4.text(0.05, 0.95, f"R² = {r2:.3f}\nr = {corr:.3f}",
            transform=ax4.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('DSSP Matching Accuracy', fontsize=11, fontweight='bold')
    ax4.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
    ax4.set_title('AF3: DSSP → DockQ', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='lower right')
    
    # ============================================================
    # ROW 2: R² comparison and model coefficients
    # ============================================================
    
    # Plot 5: R² Comparison Bar Chart
    ax5 = fig.add_subplot(gs[1, 0:2])
    
    models = ['Disorder\nOnly', 'DSSP\nOnly', 'Both\nTogether']
    af2_r2 = [af2_data['models']['disorder_only']['r2'],
              af2_data['models']['dssp_only']['r2'],
              af2_data['models']['both']['r2']]
    af3_r2 = [af3_data['models']['disorder_only']['r2'],
              af3_data['models']['dssp_only']['r2'],
              af3_data['models']['both']['r2']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, af2_r2, width, label='AF2 Multimer',
                   color=af2_color, alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax5.bar(x + width/2, af3_r2, width, label='AF3',
                   color=af3_color, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5.set_ylabel('R² (Variance Explained)', fontsize=11, fontweight='bold')
    ax5.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, fontsize=10)
    ax5.legend(fontsize=10, loc='upper left')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, max(max(af2_r2), max(af3_r2)) * 1.15])
    
    # Plot 6: Improvement Analysis
    ax6 = fig.add_subplot(gs[1, 2:4])
    
    improvements = ['DSSP Added\nto Disorder', 'Disorder Added\nto DSSP']
    af2_improve = [af2_data['improvements']['dssp_added_to_disorder'],
                   af2_data['improvements']['disorder_added_to_dssp']]
    af3_improve = [af3_data['improvements']['dssp_added_to_disorder'],
                   af3_data['improvements']['disorder_added_to_dssp']]
    
    x = np.arange(len(improvements))
    
    bars1 = ax6.bar(x - width/2, af2_improve, width, label='AF2 Multimer',
                   color=af2_color, alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax6.bar(x + width/2, af3_improve, width, label='AF3',
                   color=af3_color, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax6.set_ylabel('ΔR² (Improvement)', fontsize=11, fontweight='bold')
    ax6.set_title('Added Value of Combined Model', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(improvements, fontsize=10)
    ax6.legend(fontsize=10, loc='upper right')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # ============================================================
    # ROW 3: Combined model predictions and residuals
    # ============================================================
    
    # Plot 7: AF2 Combined Model - Predicted vs Actual
    ax7 = fig.add_subplot(gs[2, 0:2])
    
    # Recreate combined model predictions
    X_af2 = np.column_stack([af2_disorder, af2_dssp])
    model_af2 = LinearRegression().fit(X_af2, af2_dockq)
    af2_predicted = model_af2.predict(X_af2)
    
    scatter7 = ax7.scatter(af2_dockq, af2_predicted, alpha=0.6, s=60,
                          c=af2_disorder, cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(af2_dockq.min(), af2_predicted.min())
    max_val = max(af2_dockq.max(), af2_predicted.max())
    ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect prediction')
    
    r2 = af2_data['models']['both']['r2']
    ax7.text(0.05, 0.95, f"R² = {r2:.3f}\nn = {len(af2_dockq)}",
            transform=ax7.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax7.set_xlabel('Actual DockQ Score', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Predicted DockQ Score', fontsize=11, fontweight='bold')
    ax7.set_title('AF2: Combined Model (Disorder + DSSP)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=9)
    
    cbar7 = plt.colorbar(scatter7, ax=ax7)
    cbar7.set_label('Disorder Fraction', fontsize=9)
    
    # Plot 8: AF3 Combined Model - Predicted vs Actual
    ax8 = fig.add_subplot(gs[2, 2:4])
    
    X_af3 = np.column_stack([af3_disorder, af3_dssp])
    model_af3 = LinearRegression().fit(X_af3, af3_dockq)
    af3_predicted = model_af3.predict(X_af3)
    
    scatter8 = ax8.scatter(af3_dockq, af3_predicted, alpha=0.6, s=60, marker='^',
                          c=af3_disorder, cmap='plasma', edgecolors='black', linewidth=0.5)
    
    min_val = min(af3_dockq.min(), af3_predicted.min())
    max_val = max(af3_dockq.max(), af3_predicted.max())
    ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect prediction')
    
    r2 = af3_data['models']['both']['r2']
    ax8.text(0.05, 0.95, f"R² = {r2:.3f}\nn = {len(af3_dockq)}",
            transform=ax8.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax8.set_xlabel('Actual DockQ Score', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Predicted DockQ Score', fontsize=11, fontweight='bold')
    ax8.set_title('AF3: Combined Model (Disorder + DSSP)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)
    
    cbar8 = plt.colorbar(scatter8, ax=ax8)
    cbar8.set_label('Disorder Fraction', fontsize=9)
    
    # Add panel labels
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B, C, D, E, F, G, H
        ax.text(0, 1.12, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')
    
    # Overall title
    # fig.suptitle('Disorder-DSSP Regression Analysis: Predicting DockQ from Interface Properties',
    #              fontsize=16, fontweight='bold', y=0.95)
    
    # Save figure
    output_file = output_dir / f"disorder_dssp_regression{selected}.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\n✓ Regression visualization saved to: {output_file}")
    
    
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"AF2 Multimer: {len(af2_dockq)} samples")
    print(f"AF3: {len(af3_dockq)} samples")
    print(f"\nBest R² values:")
    print(f"  AF2 (Both): {af2_data['models']['both']['r2']:.3f}")
    print(f"  AF3 (Both): {af3_data['models']['both']['r2']:.3f}")
    print(f"{'='*60}\n")


def plot_pae_analysis(results_list, output_dir="extended_results"):
    """Create comprehensive PAE analysis figure"""
    
    output_dir = Path(output_dir)
    
    # Extract PAE data
    af2_pae_mean = []
    af2_pae_median = []
    af3_pae_mean = []
    af3_pae_median = []
    af2_dockq = []
    af3_dockq = []
    
    for result in results_list:
        if result['af2'].get('pae'):
            af2_pae_mean.append(result['af2']['pae']['mean_pae'])
            af2_pae_median.append(result['af2']['pae']['median_pae'])
            if result.get('af2_dockq') is not None:
                af2_dockq.append(result['af2_dockq'])
        
        if result['af3'].get('pae'):
            af3_pae_mean.append(result['af3']['pae']['mean_pae'])
            af3_pae_median.append(result['af3']['pae']['median_pae'])
            if result.get('af3_dockq') is not None:
                af3_dockq.append(result['af3_dockq'])
    
    # Statistical test for PAE comparison
    stat_pae, p_pae = None, None
    if af2_pae_mean and af3_pae_mean:
        stat_pae, p_pae = mannwhitneyu(af2_pae_mean, af3_pae_mean, alternative='two-sided')
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    af2_color = 'steelblue'
    af3_color = 'indianred'
    
    # Plot 1: AF2 PAE Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if af2_pae_mean:
        n, bins, patches = ax1.hist(af2_pae_mean, bins=20, alpha=0.7, color=af2_color, 
                                     edgecolor='black', linewidth=1.2)
        mean_val = np.mean(af2_pae_mean)
        median_val = np.median(af2_pae_mean)
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_val:.2f} Å')
        ax1.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Median: {median_val:.2f} Å')
        ax1.set_xlabel('Mean Interface PAE (Å)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('AF2 Multimer: Interface PAE Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.text(0.02, 0.98, f'n = {len(af2_pae_mean)}', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.5))
    
    # Plot 2: AF3 PAE Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if af3_pae_mean:
        n, bins, patches = ax2.hist(af3_pae_mean, bins=20, alpha=0.7, color=af3_color,
                                     edgecolor='black', linewidth=1.2)
        mean_val = np.mean(af3_pae_mean)
        median_val = np.median(af3_pae_mean)
        ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean: {mean_val:.2f} Å')
        ax2.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Median: {median_val:.2f} Å')
        ax2.set_xlabel('Mean Interface PAE (Å)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_title('AF3: Interface PAE Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.text(0.02, 0.98, f'n = {len(af3_pae_mean)}', transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.5))
    
    # Plot 3: PAE Comparison Boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    if af2_pae_mean and af3_pae_mean:
        box_data = [af2_pae_mean, af3_pae_mean]
        bp = ax3.boxplot(box_data, labels=['AF2\nMultimer', 'AF3'], patch_artist=True,
                        widths=0.6, showmeans=True, showfliers=True,
                        meanprops=dict(marker='D', markerfacecolor='gold',
                                      markeredgecolor='black', markersize=8),
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
        bp['boxes'][0].set_facecolor(af2_color)
        bp['boxes'][1].set_facecolor(af3_color)
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax3.set_ylabel('Mean Interface PAE (Å)', fontsize=11, fontweight='bold')
        ax3.set_title('PAE Comparison', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add p-value annotation
        if p_pae is not None:
            p_text = f'p < 0.001' if p_pae < 0.001 else f'p = {p_pae:.3f}'
            ax3.text(0.5, 0.97, f'Mann-Whitney U\n{p_text}', 
                    transform=ax3.transAxes, ha='center', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', 
                    alpha=0.6, edgecolor='gray'))
    
    # Plot 4: AF2 PAE vs DockQ
    ax4 = fig.add_subplot(gs[1, 0])
    if af2_pae_mean and af2_dockq:
        min_len = min(len(af2_pae_mean), len(af2_dockq))
        scatter = ax4.scatter(af2_pae_mean[:min_len], af2_dockq[:min_len],
                            alpha=0.7, s=80, c=af2_pae_mean[:min_len],
                            cmap='coolwarm_r', edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('PAE (Å)', fontsize=9)
        ax4.set_xlabel('Mean Interface PAE (Å)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
        ax4.set_title('AF2 Multimer: PAE vs Model Quality', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.23, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax4.axhline(y=0.49, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax4.axhline(y=0.80, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    
    # Plot 5: AF3 PAE vs DockQ
    ax5 = fig.add_subplot(gs[1, 1])
    if af3_pae_mean and af3_dockq:
        min_len = min(len(af3_pae_mean), len(af3_dockq))
        scatter = ax5.scatter(af3_pae_mean[:min_len], af3_dockq[:min_len],
                            alpha=0.7, s=80, c=af3_pae_mean[:min_len],
                            cmap='coolwarm_r', edgecolors='black', linewidth=0.5, marker='^')
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('PAE (Å)', fontsize=9)
        ax5.set_xlabel('Mean Interface PAE (Å)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
        ax5.set_title('AF3: PAE vs Model Quality', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0.23, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax5.axhline(y=0.49, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax5.axhline(y=0.80, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    
    # Plot 6: Mean vs Median PAE Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if af2_pae_mean and af3_pae_mean:
        x = np.arange(2)
        width = 0.35
        
        af2_mean_avg = np.mean(af2_pae_mean)
        af2_median_avg = np.mean(af2_pae_median)
        af3_mean_avg = np.mean(af3_pae_mean)
        af3_median_avg = np.mean(af3_pae_median)
        
        ax6.bar(x - width/2, [af2_mean_avg, af2_median_avg], width,
               label='AF2 Multimer', color=af2_color, alpha=0.7, edgecolor='black')
        ax6.bar(x + width/2, [af3_mean_avg, af3_median_avg], width,
               label='AF3', color=af3_color, alpha=0.7, edgecolor='black')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Mean PAE', 'Median PAE'], fontsize=10)
        ax6.set_ylabel('PAE Value (Å)', fontsize=11, fontweight='bold')
        ax6.set_title('Average PAE Statistics', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')

    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B, C, D, E, F
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    
    
    plt.savefig(output_dir / "pae_analysis.pdf", format='pdf', bbox_inches='tight')
    print(f"PAE analysis plot saved to: {output_dir}/pae_analysis.pdf")
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pathlib import Path

def plot_dssp_analysis(results_list, output_dir="extended_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract DSSP data
    af2_helix_idp = []
    af2_sheet_idp = []
    af2_coil_idp = []
    af3_helix_idp = []
    af3_sheet_idp = []
    af3_coil_idp = []
    
    af2_helix_rec = []
    af2_sheet_rec = []
    af2_coil_rec = []
    af3_helix_rec = []
    af3_sheet_rec = []
    af3_coil_rec = []
    
    for result in results_list:
        if result['af2']['dssp']['idp']['available']:
            af2_helix_idp.append(result['af2']['dssp']['idp']['helix_fraction'])
            af2_sheet_idp.append(result['af2']['dssp']['idp']['sheet_fraction'])
            af2_coil_idp.append(result['af2']['dssp']['idp']['coil_fraction'])
            
            af2_helix_rec.append(result['af2']['dssp']['receptor']['helix_fraction'])
            af2_sheet_rec.append(result['af2']['dssp']['receptor']['sheet_fraction'])
            af2_coil_rec.append(result['af2']['dssp']['receptor']['coil_fraction'])
        
        if result['af3']['dssp']['idp']['available']:
            af3_helix_idp.append(result['af3']['dssp']['idp']['helix_fraction'])
            af3_sheet_idp.append(result['af3']['dssp']['idp']['sheet_fraction'])
            af3_coil_idp.append(result['af3']['dssp']['idp']['coil_fraction'])
            
            af3_helix_rec.append(result['af3']['dssp']['receptor']['helix_fraction'])
            af3_sheet_rec.append(result['af3']['dssp']['receptor']['sheet_fraction'])
            af3_coil_rec.append(result['af3']['dssp']['receptor']['coil_fraction'])
    
    if not af2_helix_idp and not af3_helix_idp:
        print("No DSSP data available. Skipping DSSP plot.")
        return
    
    # ========================================
    # STATISTICAL TESTS FOR IDP CHAINS
    # ========================================
    mwu_results = {}
    
    # Helix comparison
    if af2_helix_idp and af3_helix_idp:
        stat_h, p_h = mannwhitneyu(af2_helix_idp, af3_helix_idp, alternative='two-sided')
        mwu_results['helix'] = {'stat': stat_h, 'p_val': p_h}
    else:
        mwu_results['helix'] = {'stat': np.nan, 'p_val': np.nan}
    
    # Sheet comparison
    if af2_sheet_idp and af3_sheet_idp:
        stat_s, p_s = mannwhitneyu(af2_sheet_idp, af3_sheet_idp, alternative='two-sided')
        mwu_results['sheet'] = {'stat': stat_s, 'p_val': p_s}
    else:
        mwu_results['sheet'] = {'stat': np.nan, 'p_val': np.nan}
    
    # Coil comparison
    if af2_coil_idp and af3_coil_idp:
        stat_c, p_c = mannwhitneyu(af2_coil_idp, af3_coil_idp, alternative='two-sided')
        mwu_results['coil'] = {'stat': stat_c, 'p_val': p_c}
    else:
        mwu_results['coil'] = {'stat': np.nan, 'p_val': np.nan}
    
    # ========================================
    # CREATE FIGURE
    # ========================================
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.2,
                          left=0.06, right=0.98, top=0.92, bottom=0.08)
    
    af2_color = 'steelblue'
    af3_color = 'indianred'
    helix_color = '#E74C3C'
    sheet_color = '#F39C12'
    coil_color = '#95A5A6'
    
    # ========================================
    # Plot 1: AF2 IDP Secondary Structure Distribution
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    if af2_helix_idp:
        x = np.arange(3)
        means = [np.mean(af2_helix_idp), np.mean(af2_sheet_idp), np.mean(af2_coil_idp)]
        medians = [np.median(af2_helix_idp), np.median(af2_sheet_idp), np.median(af2_coil_idp)]
        stds = [np.std(af2_helix_idp), np.std(af2_sheet_idp), np.std(af2_coil_idp)]
        
        bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                      color=[helix_color, sheet_color, coil_color], edgecolor='black', linewidth=1.2)
        ax1.scatter(x, medians, color='darkgreen', s=100, marker='D', zorder=5,
                   label='Median', edgecolors='black', linewidth=1)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Helix', 'Sheet', 'Coil'], fontsize=10)
        ax1.set_ylabel('Fraction', fontsize=11, fontweight='bold')
        ax1.set_title('AF2 Multimer: IDP Secondary Structure', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)
    
    # ========================================
    # Plot 2: AF3 IDP Secondary Structure Distribution
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    if af3_helix_idp:
        x = np.arange(3)
        means = [np.mean(af3_helix_idp), np.mean(af3_sheet_idp), np.mean(af3_coil_idp)]
        medians = [np.median(af3_helix_idp), np.median(af3_sheet_idp), np.median(af3_coil_idp)]
        stds = [np.std(af3_helix_idp), np.std(af3_sheet_idp), np.std(af3_coil_idp)]
        
        bars = ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                      color=[helix_color, sheet_color, coil_color], edgecolor='black', linewidth=1.2)
        ax2.scatter(x, medians, color='darkgreen', s=100, marker='D', zorder=5,
                   label='Median', edgecolors='black', linewidth=1)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Helix', 'Sheet', 'Coil'], fontsize=10)
        ax2.set_ylabel('Fraction', fontsize=11, fontweight='bold')
        ax2.set_title('AF3: IDP Secondary Structure', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)
    
    # ========================================
    # Plot 3: IDP Secondary Structure Comparison (WITH MANN-WHITNEY U)
    # ========================================
    ax3 = fig.add_subplot(gs[0, 2])
    if af2_helix_idp and af3_helix_idp:
        x = np.arange(3)
        width = 0.35
        
        af2_means = [np.mean(af2_helix_idp), np.mean(af2_sheet_idp), np.mean(af2_coil_idp)]
        af3_means = [np.mean(af3_helix_idp), np.mean(af3_sheet_idp), np.mean(af3_coil_idp)]
        
        af2_stds = [np.std(af2_helix_idp), np.std(af2_sheet_idp), np.std(af2_coil_idp)]
        af3_stds = [np.std(af3_helix_idp), np.std(af3_sheet_idp), np.std(af3_coil_idp)]
        
        bars1 = ax3.bar(x - width/2, af2_means, width, yerr=af2_stds, capsize=4,
                       label='AF2 Multimer', color=af2_color, alpha=0.7,
                       edgecolor='black', linewidth=1.2)
        bars2 = ax3.bar(x + width/2, af3_means, width, yerr=af3_stds, capsize=4,
                       label='AF3', color=af3_color, alpha=0.7,
                       edgecolor='black', linewidth=1.2)
        
        # Add Mann-Whitney U p-values above bars
        for i, struct_type in enumerate(['helix', 'sheet', 'coil']):
            p_val = mwu_results[struct_type]['p_val']
            if not np.isnan(p_val):
                sig_marker = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                y_pos = max(af2_means[i] + af2_stds[i], af3_means[i] + af3_stds[i]) - 0.14
                ax3.text(i, y_pos, f'p={p_val:.2f}\n{sig_marker}', 
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.3))
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Helix', 'Sheet', 'Coil'], fontsize=10)
        ax3.set_ylabel('Fraction', fontsize=11, fontweight='bold')
        ax3.set_title('IDP Secondary Structure Comparison', 
                     fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1.0)
        
        # Add significance legend at bottom
        ax3.text(0.5, -0.07, 'Mann-Whitney U: ***p<0.001, **p<0.01, *p<0.05, ns=not significant',
                transform=ax3.transAxes, ha='center', va='top', fontsize=9,
                style='italic')
    
    # ========================================
    # Plot 4: AF2 Receptor Secondary Structure
    # ========================================
    ax4 = fig.add_subplot(gs[1, 0])
    if af2_helix_rec:
        x = np.arange(3)
        means = [np.mean(af2_helix_rec), np.mean(af2_sheet_rec), np.mean(af2_coil_rec)]
        medians = [np.median(af2_helix_rec), np.median(af2_sheet_rec), np.median(af2_coil_rec)]
        stds = [np.std(af2_helix_rec), np.std(af2_sheet_rec), np.std(af2_coil_rec)]
        
        bars = ax4.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                      color=[helix_color, sheet_color, coil_color], edgecolor='black', linewidth=1.2)
        ax4.scatter(x, medians, color='darkgreen', s=100, marker='D', zorder=5,
                   label='Median', edgecolors='black', linewidth=1)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Helix', 'Sheet', 'Coil'], fontsize=10)
        ax4.set_ylabel('Fraction', fontsize=11, fontweight='bold')
        ax4.set_title('AF2 Multimer: Receptor Secondary Structure', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
    
    # ========================================
    # Plot 5: AF3 Receptor Secondary Structure
    # ========================================
    ax5 = fig.add_subplot(gs[1, 1])
    if af3_helix_rec:
        x = np.arange(3)
        means = [np.mean(af3_helix_rec), np.mean(af3_sheet_rec), np.mean(af3_coil_rec)]
        medians = [np.median(af3_helix_rec), np.median(af3_sheet_rec), np.median(af3_coil_rec)]
        stds = [np.std(af3_helix_rec), np.std(af3_sheet_rec), np.std(af3_coil_rec)]
        
        bars = ax5.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                      color=[helix_color, sheet_color, coil_color], edgecolor='black', linewidth=1.2)
        ax5.scatter(x, medians, color='darkgreen', s=100, marker='D', zorder=5,
                   label='Median', edgecolors='black', linewidth=1)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Helix', 'Sheet', 'Coil'], fontsize=10)
        ax5.set_ylabel('Fraction', fontsize=11, fontweight='bold')
        ax5.set_title('AF3: Receptor Secondary Structure', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, 1)
    
    # ========================================
    # Plot 6: Overall Secondary Structure Comparison
    # ========================================
    ax6 = fig.add_subplot(gs[1, 2])
    if af2_helix_idp and af3_helix_idp:
        x = np.arange(2)
        width = 0.25
        
        # Calculate combined means (IDP + Receptor)
        af2_helix_combined = af2_helix_idp + af2_helix_rec
        af2_sheet_combined = af2_sheet_idp + af2_sheet_rec
        af3_helix_combined = af3_helix_idp + af3_helix_rec
        af3_sheet_combined = af3_sheet_idp + af3_sheet_rec
        
        af2_means = [np.mean(af2_helix_combined), np.mean(af2_sheet_combined)]
        af3_means = [np.mean(af3_helix_combined), np.mean(af3_sheet_combined)]
        
        ax6.bar(x - width/2, af2_means, width, label='AF2 Multimer',
               color=af2_color, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax6.bar(x + width/2, af3_means, width, label='AF3',
               color=af3_color, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Helix', 'Sheet'], fontsize=10)
        ax6.set_ylabel('Average Fraction', fontsize=11, fontweight='bold')
        ax6.set_title('Overall Secondary Structure Comparison', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(0, 1)
    
    # Add panel labels
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B, C, D, E, F
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    
    
    output_file = output_dir / "dssp_analysis.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    
    plt.close()

from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt

def plot_comprehensive_results(results_list, output_plot="interface_comprehensive.pdf"):
    fig = plt.figure(figsize=(17, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35, 
                          left=0.06, right=0.98, top=0.96, bottom=0.05)
    
    # Extract data with weighted calculation
    af2_disorder = []
    af2_dockq = []
    af3_disorder = []
    af3_dockq = []
    interface_sizes_af2 = []
    interface_sizes_af3 = []
    
    # NEW: DSSP matching data
    af2_dssp_match = []
    af3_dssp_match = []
    af2_dockq_dssp = []
    af3_dockq_dssp = []

    for results in results_list:
        # Weighted disorder calculation (existing code)
        idp_interface_af2 = results['af2']['idp']['interface_residues']
        rec_interface_af2 = results['af2']['receptor']['interface_residues']
        idp_disorder_frac_af2 = results['af2']['idp']['disorder_fraction']
        rec_disorder_frac_af2 = results['af2']['receptor']['disorder_fraction']
        
        idp_disordered_af2 = idp_disorder_frac_af2 * idp_interface_af2
        rec_disordered_af2 = rec_disorder_frac_af2 * rec_interface_af2
        total_disordered_af2 = idp_disordered_af2 + rec_disordered_af2
        total_interface_af2 = idp_interface_af2 + rec_interface_af2
        
        af2_d = (total_disordered_af2 / total_interface_af2) if total_interface_af2 > 0 else 0.0
        
        idp_interface_af3 = results['af3']['idp']['interface_residues']
        rec_interface_af3 = results['af3']['receptor']['interface_residues']
        idp_disorder_frac_af3 = results['af3']['idp']['disorder_fraction']
        rec_disorder_frac_af3 = results['af3']['receptor']['disorder_fraction']
        
        idp_disordered_af3 = idp_disorder_frac_af3 * idp_interface_af3
        rec_disordered_af3 = rec_disorder_frac_af3 * rec_interface_af3
        total_disordered_af3 = idp_disordered_af3 + rec_disordered_af3
        total_interface_af3 = idp_interface_af3 + rec_interface_af3
        
        af3_d = (total_disordered_af3 / total_interface_af3) if total_interface_af3 > 0 else 0.0
        
        af2_q = results.get('af2_dockq')
        af3_q = results.get('af3_dockq')
        
        if af2_q is not None:
            af2_disorder.append(af2_d * 100)
            af2_dockq.append(af2_q)
            interface_sizes_af2.append(total_interface_af2)
        
        if af3_q is not None:
            af3_disorder.append(af3_d * 100)
            af3_dockq.append(af3_q)
            interface_sizes_af3.append(total_interface_af3)
        
        # DSSP matching calculation
        if (results['af2']['dssp']['idp']['available'] and 
            results['af2']['dssp']['receptor']['available'] and
            results['reference']['dssp']['idp']['available'] and
            results['reference']['dssp']['receptor']['available'] and
            'mappings' in results and
            af2_q is not None):
            
            af2_idp_dssp = results['af2']['dssp']['idp']['structure_dict']
            af2_rec_dssp = results['af2']['dssp']['receptor']['structure_dict']
            ref_idp_dssp = results['reference']['dssp']['idp']['structure_dict']
            ref_rec_dssp = results['reference']['dssp']['receptor']['structure_dict']

            af2_to_ref_idp = results['mappings']['af2_to_ref_idp']
            af2_to_ref_rec = results['mappings']['af2_to_ref_rec']
            
            idp_matches = 0
            idp_total = 0
            for af2_res_id in af2_idp_dssp.keys():
                if af2_res_id in af2_to_ref_idp:
                    ref_res_id = str(af2_to_ref_idp[af2_res_id])
                    if ref_res_id in ref_idp_dssp:
                        idp_total += 1
                        if af2_idp_dssp[af2_res_id] == ref_idp_dssp[ref_res_id]:
                            idp_matches += 1
            
            rec_matches = 0
            rec_total = 0
            for af2_res_id in af2_rec_dssp.keys():
                if af2_res_id in af2_to_ref_rec:
                    ref_res_id = str(af2_to_ref_rec[af2_res_id])
                    if ref_res_id in ref_rec_dssp:
                        rec_total += 1
                        if af2_rec_dssp[af2_res_id] == ref_rec_dssp[ref_res_id]:
                            rec_matches += 1
            
            total_matches = idp_matches + rec_matches
            total_residues = idp_total + rec_total
            if total_residues > 0:
                af2_match_frac = total_matches / total_residues
                af2_dssp_match.append(af2_match_frac * 100)
                af2_dockq_dssp.append(af2_q)

        if (results['af3']['dssp']['idp']['available'] and 
            results['af3']['dssp']['receptor']['available'] and
            results['reference']['dssp']['idp']['available'] and
            results['reference']['dssp']['receptor']['available'] and
            'mappings' in results and
            af3_q is not None):
            
            af3_idp_dssp = results['af3']['dssp']['idp']['structure_dict']
            af3_rec_dssp = results['af3']['dssp']['receptor']['structure_dict']
            ref_idp_dssp = results['reference']['dssp']['idp']['structure_dict']
            ref_rec_dssp = results['reference']['dssp']['receptor']['structure_dict']
            
            af3_to_ref_idp = results['mappings']['af3_to_ref_idp']
            af3_to_ref_rec = results['mappings']['af3_to_ref_rec']
            
            idp_matches = 0
            idp_total = 0
            for af3_res_id in af3_idp_dssp.keys():
                if af3_res_id in af3_to_ref_idp:
                    ref_res_id = str(af3_to_ref_idp[af3_res_id])
                    if ref_res_id in ref_idp_dssp:
                        idp_total += 1
                        if af3_idp_dssp[af3_res_id] == ref_idp_dssp[ref_res_id]:
                            idp_matches += 1
            
            rec_matches = 0
            rec_total = 0
            for af3_res_id in af3_rec_dssp.keys():
                if af3_res_id in af3_to_ref_rec:
                    ref_res_id = str(af3_to_ref_rec[af3_res_id])
                    if ref_res_id in ref_rec_dssp:
                        rec_total += 1
                        if af3_rec_dssp[af3_res_id] == ref_rec_dssp[ref_res_id]:
                            rec_matches += 1
            
            total_matches = idp_matches + rec_matches
            total_residues = idp_total + rec_total
            
            if total_residues > 0:
                af3_match_frac = total_matches / total_residues
                af3_dssp_match.append(af3_match_frac * 100)
                af3_dockq_dssp.append(af3_q)

    # Colors
    af2_color = 'steelblue'
    af3_color = 'indianred'
    
    # Categorize data for Mann-Whitney U tests
    af2_by_cat = {'low': [], 'med': [], 'high': []}
    af3_by_cat = {'low': [], 'med': [], 'high': []}
    
    for d, q in zip(af2_disorder, af2_dockq):
        if d < 20:
            af2_by_cat['low'].append(q)
        elif d < 40:
            af2_by_cat['med'].append(q)
        else:
            af2_by_cat['high'].append(q)
    
    for d, q in zip(af3_disorder, af3_dockq):
        if d < 20:
            af3_by_cat['low'].append(q)
        elif d < 40:
            af3_by_cat['med'].append(q)
        else:
            af3_by_cat['high'].append(q)
    
    # Calculate Mann-Whitney U tests for each category
    mwu_results = {}
    for cat, cat_name in [('low', 'Low'), ('med', 'Medium'), ('high', 'High')]:
        if af2_by_cat[cat] and af3_by_cat[cat]:
            stat, p_val = mannwhitneyu(af2_by_cat[cat], af3_by_cat[cat], alternative='two-sided')
            mwu_results[cat] = {'stat': stat, 'p_val': p_val, 'name': cat_name}
        else:
            mwu_results[cat] = {'stat': np.nan, 'p_val': np.nan, 'name': cat_name}
    
    # ========================================
    # ROW 1: Plot 1 and 2
    # ========================================
    
    ax1 = fig.add_subplot(gs[1, :2])
    if af2_disorder:
        ax1.scatter(af2_disorder, af2_dockq, label='AF2 Multimer', 
                   alpha=0.7, s=60, color=af2_color, edgecolors='none')
    if af3_disorder:
        ax1.scatter(af3_disorder, af3_dockq, label='AF3', 
                   alpha=0.7, s=60, color=af3_color, marker='^', edgecolors='none')
    
    ax1.axhline(y=0.23, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.axhline(y=0.49, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.axhline(y=0.80, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.text(105, 0.23, 'Acceptable (0.23)', fontsize=8, va='center', color='gray')
    ax1.text(105, 0.49, 'Medium (0.49)', fontsize=8, va='center', color='gray')
    ax1.text(105, 0.80, 'High (0.80)', fontsize=8, va='center', color='gray')
    
    ax1.set_xlabel('Disorder Fraction at Interface (%)', fontsize=10)
    ax1.set_ylabel('DockQ Score', fontsize=10)
    ax1.set_title('Model Performance vs Interface Disorder (Weighted)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.set_xlim(-5, 110)
    ax1.set_ylim(0, 1.05)
    
    ax2 = fig.add_subplot(gs[1, 2])
    box_data = [af2_dockq, af3_dockq]
    bp = ax2.boxplot(box_data, labels=['AF2\nMultimer', 'AF3'], patch_artist=True,
                     widths=0.5, showmeans=True, meanprops=dict(marker='D', markerfacecolor='gold', 
                     markeredgecolor='black', markersize=7))
    bp['boxes'][0].set_facecolor(af2_color)
    bp['boxes'][1].set_facecolor(af3_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax2.set_ylabel('DockQ Score', fontsize=10)
    ax2.set_title('DockQ Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    
    # ========================================
    # ROW 2: Plot 3 and 4 (WITH MANN-WHITNEY U P-VALUES)
    # ========================================
    
    ax3 = fig.add_subplot(gs[2, :2])
    categories = ['Low\n(<20%)', 'Medium\n(20-40%)', 'High\n(≥40%)']
    
    x = np.arange(len(categories))
    width = 0.35
    
    af2_means = [np.mean(af2_by_cat['low']) if af2_by_cat['low'] else 0,
                 np.mean(af2_by_cat['med']) if af2_by_cat['med'] else 0,
                 np.mean(af2_by_cat['high']) if af2_by_cat['high'] else 0]
    af3_means = [np.mean(af3_by_cat['low']) if af3_by_cat['low'] else 0,
                 np.mean(af3_by_cat['med']) if af3_by_cat['med'] else 0,
                 np.mean(af3_by_cat['high']) if af3_by_cat['high'] else 0]
    
    af2_stds = [np.std(af2_by_cat['low']) if len(af2_by_cat['low']) > 1 else 0,
                np.std(af2_by_cat['med']) if len(af2_by_cat['med']) > 1 else 0,
                np.std(af2_by_cat['high']) if len(af2_by_cat['high']) > 1 else 0]
    af3_stds = [np.std(af3_by_cat['low']) if len(af3_by_cat['low']) > 1 else 0,
                np.std(af3_by_cat['med']) if len(af3_by_cat['med']) > 1 else 0,
                np.std(af3_by_cat['high']) if len(af3_by_cat['high']) > 1 else 0]
    
    bars1 = ax3.bar(x - width/2, af2_means, width, label='AF2 Multimer', 
                    color=af2_color, alpha=0.7, yerr=af2_stds, capsize=4, error_kw={'linewidth': 1.5})
    bars2 = ax3.bar(x + width/2, af3_means, width, label='AF3', 
                    color=af3_color, alpha=0.7, yerr=af3_stds, capsize=4, error_kw={'linewidth': 1.5})
    
    for i, (cat_key, label) in enumerate(zip(['low', 'med', 'high'], categories)):
        n_af2 = len(af2_by_cat[cat_key])
        n_af3 = len(af3_by_cat[cat_key])
        if n_af2 > 0:
            ax3.text(i - width/2, af2_means[i] + af2_stds[i] + 0.05, f'n={n_af2}', 
                    ha='center', va='bottom', fontsize=8)
        if n_af3 > 0:
            ax3.text(i + width/2, af3_means[i] + af3_stds[i] + 0.05, f'n={n_af3}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Add Mann-Whitney U p-value
        p_val = mwu_results[cat_key]['p_val']
        if not np.isnan(p_val):
            sig_marker = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns')
            ax3.text(i, max(af2_means[i] + af2_stds[i], af3_means[i] + af3_stds[i]) - 0.1, 
                    f'p={p_val:.3f}\n{sig_marker}', ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.3))
    
    ax3.set_ylabel('Mean DockQ Score', fontsize=10)
    ax3.set_xlabel('Disorder Level at Interface', fontsize=10)
    ax3.set_title('Model Performance by Disorder Category\n(Mann-Whitney U: **p<0.01, *p<0.05, ns=not significant)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
    
    # 4. AF3 vs AF2 Performance Delta
    ax4 = fig.add_subplot(gs[2, 2])
    dockq_diffs = []
    for i in range(min(len(af2_dockq), len(af3_dockq))):
        if i < len(results_list):
            if results_list[i].get('af2_dockq') is not None and results_list[i].get('af3_dockq') is not None:
                dockq_diffs.append(results_list[i]['af3_dockq'] - results_list[i]['af2_dockq'])
    
    ax4.hist(dockq_diffs, bins=15, color='#85C1E9', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='No change')
    mean_diff = np.mean(dockq_diffs) if dockq_diffs else 0
    ax4.axvline(x=mean_diff, color='green', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_diff:.3f}')
    ax4.set_xlabel('DockQ Difference (AF3 - AF2)', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('AF3 vs AF2 Performance Δ', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
    ax4.set_ylim(0, max(ax4.get_ylim()[1], 14))
    
    # ========================================
    # ROW 3: Plot 5, 6, and 7
    # ========================================
    
    ax5 = fig.add_subplot(gs[3, 0])
    categories_sr = ['Low\n(<20%)', 'Med\n(20-40%)', 'High\n(≥40%)']
    
    af2_success = {'low': 0, 'med': 0, 'high': 0}
    af2_total = {'low': 0, 'med': 0, 'high': 0}
    af3_success = {'low': 0, 'med': 0, 'high': 0}
    af3_total = {'low': 0, 'med': 0, 'high': 0}
    
    for d, q in zip(af2_disorder, af2_dockq):
        cat = 'low' if d < 20 else ('med' if d < 40 else 'high')
        af2_total[cat] += 1
        if q >= 0.23:
            af2_success[cat] += 1
    
    for d, q in zip(af3_disorder, af3_dockq):
        cat = 'low' if d < 20 else ('med' if d < 40 else 'high')
        af3_total[cat] += 1
        if q >= 0.23:
            af3_success[cat] += 1
    
    af2_rates = [af2_success[k]/af2_total[k]*100 if af2_total[k] > 0 else 0 
                 for k in ['low', 'med', 'high']]
    af3_rates = [af3_success[k]/af3_total[k]*100 if af3_total[k] > 0 else 0 
                 for k in ['low', 'med', 'high']]
    
    x = np.arange(len(categories_sr))
    width = 0.35
    ax5.bar(x - width/2, af2_rates, width, label='AF2', color=af2_color, alpha=0.7)
    ax5.bar(x + width/2, af3_rates, width, label='AF3', color=af3_color, alpha=0.7)
    
    ax5.set_ylabel('Success Rate (%)', fontsize=10)
    ax5.set_xlabel('Disorder Level', fontsize=10)
    ax5.set_title('Success Rate by Disorder\n(DockQ ≥ 0.23)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories_sr, fontsize=9)
    ax5.legend(fontsize=8, loc='lower left')
    ax5.set_ylim(0, 105)
    ax5.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
    
    ax6 = fig.add_subplot(gs[3, 1])
    if interface_sizes_af2:
        ax6.scatter(interface_sizes_af2, af2_disorder, label='AF2', 
                   alpha=0.5, s=50, color=af2_color, edgecolors='none')
    if interface_sizes_af3:
        ax6.scatter(interface_sizes_af3, af3_disorder, label='AF3', 
                   alpha=0.5, s=50, color=af3_color, marker='^', edgecolors='none')
    
    ax6.set_xlabel('Interface Size (residues)', fontsize=10)
    ax6.set_ylabel('Disorder Fraction (%)', fontsize=10)
    ax6.set_title('Interface Size vs Disorder', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax6.set_ylim(-5, 105)
    
    ax7 = fig.add_subplot(gs[3, 2])
    bins = np.arange(0, 105, 10)
    
    ax7.hist(af2_disorder, bins=bins, alpha=0.6, 
             label='AF2 Multimer', color=af2_color, edgecolor='black', linewidth=0.8)
    ax7.hist(af3_disorder, bins=bins, alpha=0.6, 
             label='AF3', color=af3_color, edgecolor='black', linewidth=0.8)
    
    ax7.axvline(x=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                label='Low/Med (20%)')
    ax7.axvline(x=40, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                label='Med/High (40%)')
    
    ax7.set_xlabel('Disorder Fraction (%)', fontsize=10)
    ax7.set_ylabel('Count', fontsize=10)
    ax7.set_title('Distribution of Disorder Levels', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8, loc='upper right')
    ax7.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
    ax7.set_xlim(0, 100)
    
    # ========================================
    # ROW 4: Plot 8 - DSSP MATCHING PLOT
    # ========================================
    
    ax8 = fig.add_subplot(gs[0, :])
    
    if af2_dssp_match or af3_dssp_match:
        if af2_dssp_match:
            ax8.scatter(af2_dssp_match, af2_dockq_dssp, label='AF2 Multimer', 
                       alpha=0.7, s=60, color=af2_color, linewidth=0.5)
        if af3_dssp_match:
            ax8.scatter(af3_dssp_match, af3_dockq_dssp, label='AF3', 
                       alpha=0.7, s=60, color=af3_color, marker='^', linewidth=0.5)
        
        ax8.axhline(y=0.23, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax8.axhline(y=0.49, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax8.axhline(y=0.80, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        
        ax8.set_xlabel('DSSP Matching Fraction (%)', fontsize=10)
        ax8.set_ylabel('DockQ Score', fontsize=10)
        ax8.set_title('Model Performance vs Secondary Structure Accuracy', fontsize=12, fontweight='bold')
        ax8.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax8.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax8.set_xlim(-5, 105)
        ax8.set_ylim(0, 1.05)
        
        if af2_dssp_match:
            af2_corr = np.corrcoef(af2_dssp_match, af2_dockq_dssp)[0, 1] if len(af2_dssp_match) > 1 else 0
            ax8.text(0.02, 0.98, f'AF2 r = {af2_corr:.3f}', 
                    transform=ax8.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        if af3_dssp_match:
            af3_corr = np.corrcoef(af3_dssp_match, af3_dockq_dssp)[0, 1] if len(af3_dssp_match) > 1 else 0
            ax8.text(0.02, 0.88, f'AF3 r = {af3_corr:.3f}', 
                    transform=ax8.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax8.text(0.5, 0.5, 'No DSSP matching data available', 
                ha='center', va='center', fontsize=12, transform=ax8.transAxes)
        ax8.set_xlabel('DSSP Matching Fraction (%)', fontsize=10)
        ax8.set_ylabel('DockQ Score', fontsize=10)
        ax8.set_title('Model Performance vs Secondary Structure Accuracy', fontsize=12, fontweight='bold')
    
    # Add panel labels
    axes_list = [ax8, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)
        ax.text(0, 1.12, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    
    plt.savefig(output_plot, format='pdf', bbox_inches='tight')
    
    # Print summary with Mann-Whitney U results
    print(f"\nComprehensive plot saved to {output_plot}")
    print(f"\n{'='*70}")
    print("DISORDER CATEGORY COMPARISON - Mann-Whitney U Test Results")
    print(f"{'='*70}")
    
    for cat_key in ['low', 'med', 'high']:
        cat_name = mwu_results[cat_key]['name']
        n_af2 = len(af2_by_cat[cat_key])
        n_af3 = len(af3_by_cat[cat_key])
        mean_af2 = np.mean(af2_by_cat[cat_key]) if af2_by_cat[cat_key] else np.nan
        mean_af3 = np.mean(af3_by_cat[cat_key]) if af3_by_cat[cat_key] else np.nan
        p_val = mwu_results[cat_key]['p_val']
        
        print(f"\n{cat_name} Disorder (<20%, 20-40%, ≥40%):" if cat_key == 'low' else 
              f"\n{cat_name} Disorder:" if cat_key == 'med' else f"\n{cat_name} Disorder:")
        print(f"  AF2: n={n_af2}, mean DockQ = {mean_af2:.4f}")
        print(f"  AF3: n={n_af3}, mean DockQ = {mean_af3:.4f}")
        if not np.isnan(p_val):
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"  Mann-Whitney U p-value = {p_val:.4f} {sig}")
        else:
            print(f"  Mann-Whitney U p-value = N/A (insufficient data)")
    
    print(f"\n{'='*70}")
    print(f"AF2 points: {len(af2_disorder)}")
    print(f"AF3 points: {len(af3_disorder)}")
    print(f"AF2 DSSP matching points: {len(af2_dssp_match)}")
    print(f"AF3 DSSP matching points: {len(af3_dssp_match)}")
    if af2_dssp_match:
        print(f"AF2 DSSP match range: {min(af2_dssp_match):.1f}% - {max(af2_dssp_match):.1f}%")
        print(f"AF2 DSSP match mean: {np.mean(af2_dssp_match):.1f}%")
    if af3_dssp_match:
        print(f"AF3 DSSP match range: {min(af3_dssp_match):.1f}% - {max(af3_dssp_match):.1f}%")
        print(f"AF3 DSSP match mean: {np.mean(af3_dssp_match):.1f}%")
    
    plt.close()


def plot_determinants_figure(results_list, output_dir="interface_results"):
    
    output_dir = Path(output_dir)
    
    # Extract data for AF2
    af2_bsa = []
    af2_dockq = []
    af2_coil = []
    
    for result in results_list:
        if result.get('af2_dockq') is not None and result.get('af2_bsa') is not None:
            af2_bsa.append(result['af2_bsa'])
            af2_dockq.append(result['af2_dockq'])
            
            # Coil content in IDP chain
            if result['af2']['dssp']['idp']['available']:
                coil_frac = result['af2']['dssp']['idp']['coil_fraction']
                af2_coil.append(coil_frac * 100)
            else:
                af2_coil.append(None)
    
    # Extract data for AF3
    af3_bsa = []
    af3_dockq = []
    af3_hydro = []
    af3_coil = []
    
    for result in results_list:
        if result.get('af3_dockq') is not None and result.get('af3_bsa') is not None:
            af3_bsa.append(result['af3_bsa'])
            af3_dockq.append(result['af3_dockq'])
            
            # Get all residue-level hydropathy scores
            idp_hydro_dict = result['af3']['hydropathy']['idp']['residue_scores']
            rec_hydro_dict = result['af3']['hydropathy']['receptor']['residue_scores']

            # Combine all scores from both chains
            all_hydro_scores = list(idp_hydro_dict.values()) + list(rec_hydro_dict.values())

            # Calculate mean across ALL interface residues
            if all_hydro_scores:
                avg_hydro = np.mean(all_hydro_scores)
            else:
                avg_hydro = 0.0
            af3_hydro.append(avg_hydro * 100)
            
            # Coil content in IDP chain
            if result['af3']['dssp']['idp']['available']:
                coil_frac = result['af3']['dssp']['idp']['coil_fraction']
                af3_coil.append(coil_frac * 100)
            else:
                af3_coil.append(None)
    
    # Create bins (3 equally populated bins)
    def create_bins(values, dockq_values, n_bins=3):
        """Create equally populated bins"""
        sorted_indices = np.argsort(values)
        bin_size = len(values) // n_bins
        
        bins = [[] for _ in range(n_bins)]
        for i, idx in enumerate(sorted_indices):
            bin_idx = min(i // bin_size, n_bins - 1)
            bins[bin_idx].append(dockq_values[idx])
        
        return bins
    
    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35, 
                          left=0.08, right=0.96, top=0.92, bottom=0.08)
    
    # Color themes
    af3_colors = ['#F5E6D3', '#E8B4A0', '#B85450']  # Beige → Light Red → Dark Red
    af2_colors = ['#D6EAF8', '#85C1E9', '#2E86C1']  # Light Blue → Medium Blue → Dark Blue
    
    # ========== ROW 1: AF3 RESULTS ==========
    
    # ========== PLOT 1: AF3 Buried Surface Area ==========
    ax1 = fig.add_subplot(gs[0, 0])
    if af3_bsa and af3_dockq:
        af3_bsa_bins = create_bins(af3_bsa, af3_dockq)
        
        positions = [1, 2, 3]
        parts = ax1.violinplot(af3_bsa_bins, positions=positions, showmeans=False,
                               showextrema=False, widths=0.7)
        
        # Set violin colors WITHOUT black edges
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(af3_colors[i])
            pc.set_alpha(0.9)
            pc.set_edgecolor('none')
            pc.set_linewidth(0)
        
        # Add box plots with black lines
        bp = ax1.boxplot(af3_bsa_bins, positions=positions, widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2.5))
        
        ax1.set_ylabel('DockQ', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Buried Surface Area (Å²)', fontsize=12, fontweight='bold')
        ax1.set_title('AF3 - Buried Surface Area', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(['Small', 'Medium', 'Large'], fontsize=11)
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_xlim(0.5, 3.5)
        ax1.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax1.text(0.02, 0.98, f'n={len(af3_dockq)}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PLOT 2: AF3 Interface Hydrophobicity ==========
    ax2 = fig.add_subplot(gs[0, 1])
    if af3_hydro and af3_dockq:
        af3_hydro_bins = create_bins(af3_hydro, af3_dockq)
        
        positions = [1, 2, 3]
        parts = ax2.violinplot(af3_hydro_bins, positions=positions, showmeans=False,
                               showextrema=False, widths=0.7)
        
        # Set violin colors WITHOUT black edges
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(af3_colors[i])
            pc.set_alpha(0.9)
            pc.set_edgecolor('none')
            pc.set_linewidth(0)
        
        bp = ax2.boxplot(af3_hydro_bins, positions=positions, widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2.5))
        
        ax2.set_ylabel('DockQ', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Interface Hydrophobicity', fontsize=12, fontweight='bold')
        ax2.set_title('AF3 - Interface Hydrophobicity', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(['Low', 'Medium', 'High'], fontsize=11)
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_xlim(0.5, 3.5)
        ax2.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax2.text(0.02, 0.98, f'n={len(af3_dockq)}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PLOT 3: AF3 % Coil in IDP chain ==========
    ax3 = fig.add_subplot(gs[0, 2])
    af3_coil_valid = [(c, d) for c, d in zip(af3_coil, af3_dockq) if c is not None]
    if af3_coil_valid:
        af3_coil_vals = [x[0] for x in af3_coil_valid]
        af3_coil_dockq = [x[1] for x in af3_coil_valid]
        
        af3_coil_bins = create_bins(af3_coil_vals, af3_coil_dockq)
        
        positions = [1, 2, 3]
        parts = ax3.violinplot(af3_coil_bins, positions=positions, showmeans=False,
                               showextrema=False, widths=0.7)
        
        # Set violin colors WITHOUT black edges
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(af3_colors[i])
            pc.set_alpha(0.9)
            pc.set_edgecolor('none')
            pc.set_linewidth(0)
        
        bp = ax3.boxplot(af3_coil_bins, positions=positions, widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2.5))
        
        ax3.set_ylabel('DockQ', fontsize=13, fontweight='bold')
        ax3.set_xlabel('% Coil in IDP chain', fontsize=12, fontweight='bold')
        ax3.set_title('AF3 - % Coil in IDP Chain', fontsize=13, fontweight='bold', pad=10)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(['Low', 'Medium', 'High'], fontsize=11)
        ax3.set_ylim(-0.05, 1.15)
        ax3.set_xlim(0.5, 3.5)
        ax3.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax3.text(0.02, 0.98, f'n={len(af3_coil_dockq)}',
                transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== ROW 2: AF2 RESULTS ==========
    
    # ========== PLOT 4: AF2 Buried Surface Area ==========
    ax4 = fig.add_subplot(gs[1, 0])
    if af2_bsa and af2_dockq:
        af2_bsa_bins = create_bins(af2_bsa, af2_dockq)
        
        positions = [1, 2, 3]
        parts = ax4.violinplot(af2_bsa_bins, positions=positions, showmeans=False, 
                               showextrema=False, widths=0.7)
        
        # Set violin colors WITHOUT black edges
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(af2_colors[i])
            pc.set_alpha(0.9)
            pc.set_edgecolor('none')
            pc.set_linewidth(0)
        
        bp = ax4.boxplot(af2_bsa_bins, positions=positions, widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2.5))
        
        ax4.set_ylabel('DockQ', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Buried Surface Area (Å²)', fontsize=12, fontweight='bold')
        ax4.set_title('AF2-Multimer - Buried Surface Area', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xticks(positions)
        ax4.set_xticklabels(['Small', 'Medium', 'Large'], fontsize=11)
        ax4.set_ylim(-0.05, 1.15)
        ax4.set_xlim(0.5, 3.5)
        ax4.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax4.text(0.02, 0.98, f'n={len(af2_dockq)}', 
                transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PLOT 5: AF2 Interface Hydrophobicity ==========
    # Extract AF2 hydrophobicity data
    af2_hydro = []
    af2_hydro_dockq = []
    
    for result in results_list:
        if result.get('af2_dockq') is not None:
            
            # Get residue-level hydropathy scores for all interface residues
            idp_hydro_dict = result['af2']['hydropathy']['idp']['residue_scores']
            rec_hydro_dict = result['af2']['hydropathy']['receptor']['residue_scores']
            
            # Collect all hydropathy scores from both chains
            all_hydro_scores = list(idp_hydro_dict.values()) + list(rec_hydro_dict.values())
            
            # Calculate mean across all interface residues
            if all_hydro_scores:
                avg_hydro = np.mean(all_hydro_scores)
            else:
                avg_hydro = 0.0

            af2_hydro.append(avg_hydro * 100)
            af2_hydro_dockq.append(result['af2_dockq'])
    
    ax5 = fig.add_subplot(gs[1, 1])
    if af2_hydro and af2_hydro_dockq:
        af2_hydro_bins = create_bins(af2_hydro, af2_hydro_dockq)
        
        positions = [1, 2, 3]
        parts = ax5.violinplot(af2_hydro_bins, positions=positions, showmeans=False,
                               showextrema=False, widths=0.7)
        
        # Set violin colors WITHOUT black edges
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(af2_colors[i])
            pc.set_alpha(0.9)
            pc.set_edgecolor('none')
            pc.set_linewidth(0)
        
        bp = ax5.boxplot(af2_hydro_bins, positions=positions, widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2.5))
        
        ax5.set_ylabel('DockQ', fontsize=13, fontweight='bold')
        ax5.set_xlabel('Interface Hydrophobicity', fontsize=12, fontweight='bold')
        ax5.set_title('AF2-Multimer - Interface Hydrophobicity', fontsize=13, fontweight='bold', pad=10)
        ax5.set_xticks(positions)
        ax5.set_xticklabels(['Low', 'Medium', 'High'], fontsize=11)
        ax5.set_ylim(-0.05, 1.15)
        ax5.set_xlim(0.5, 3.5)
        ax5.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax5.text(0.02, 0.98, f'n={len(af2_hydro_dockq)}',
                transform=ax5.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PLOT 6: AF2 % Coil in IDP chain ==========
    ax6 = fig.add_subplot(gs[1, 2])
    af2_coil_valid = [(c, d) for c, d in zip(af2_coil, af2_dockq) if c is not None]
    if af2_coil_valid:
        af2_coil_vals = [x[0] for x in af2_coil_valid]
        af2_coil_dockq = [x[1] for x in af2_coil_valid]
        
        af2_coil_bins = create_bins(af2_coil_vals, af2_coil_dockq)
        
        positions = [1, 2, 3]
        parts = ax6.violinplot(af2_coil_bins, positions=positions, showmeans=False,
                               showextrema=False, widths=0.7)
        
        # Set violin colors WITHOUT black edges
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(af2_colors[i])
            pc.set_alpha(0.9)
            pc.set_edgecolor('none')
            pc.set_linewidth(0)
        
        bp = ax6.boxplot(af2_coil_bins, positions=positions, widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2.5))
        
        ax6.set_ylabel('DockQ', fontsize=13, fontweight='bold')
        ax6.set_xlabel('% Coil in IDP chain', fontsize=12, fontweight='bold')
        ax6.set_title('AF2-Multimer - % Coil in IDP Chain', fontsize=13, fontweight='bold', pad=10)
        ax6.set_xticks(positions)
        ax6.set_xticklabels(['Low', 'Medium', 'High'], fontsize=11)
        ax6.set_ylim(-0.05, 1.15)
        ax6.set_xlim(0.5, 3.5)
        ax6.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax6.text(0.02, 0.98, f'n={len(af2_coil_dockq)}',
                transform=ax6.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B, C, D, E, F
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    
    # Main title
    
    plt.savefig(output_dir / "determinants_analysis.pdf", format='pdf', bbox_inches='tight')
    print(f"\nDeterminants analysis plot saved to: {output_dir}/determinants_analysis.pdf")
    plt.close()
    
def plot_dataset_characteristics(results_list, output_plot="b90_dataset_characteristics.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract AF3 DockQ data (Plot 1 only)
    af3_dockq = []
    
    # Extract REFERENCE data (Plots 2, 3, 4)
    ref_interface_size = []
    ref_hydrophobicity = []
    ref_helix = []
    ref_sheet = []
    ref_coil = []
    
    for results in results_list:
        # ========================================
        # Plot 1: AF3 DockQ score
        # ========================================
        if results.get('af3_dockq') is not None:
            af3_dockq.append(results['af3_dockq'])
        
        # ========================================
        # Plot 2: REFERENCE Interface size
        # ========================================
        # Get interface size from reference structure
        ref_idp_interface = results['reference_interface']['idp']
        ref_rec_interface = results['reference_interface']['receptor']
        
        ref_idp_size = len(ref_idp_interface)
        ref_rec_size = len(ref_rec_interface)
        ref_total_size = ref_idp_size + ref_rec_size
        
        if ref_total_size > 0:
            ref_interface_size.append(ref_total_size)
        
        # ========================================
        # Plot 3: REFERENCE Hydrophobicity
        # ========================================
        if 'reference' in results and 'hydropathy' in results['reference']:
            ref_idp_hydro = results['reference']['hydropathy']['idp'].get('hydrophobic_fraction', 0)
            ref_rec_hydro = results['reference']['hydropathy']['receptor'].get('hydrophobic_fraction', 0)
            
            # Weighted average by interface size
            if ref_idp_size > 0 and ref_rec_size > 0:
                weighted_hydro = (ref_idp_hydro * ref_idp_size + ref_rec_hydro * ref_rec_size) / ref_total_size
                ref_hydrophobicity.append(weighted_hydro)
        
        # ========================================
        # Plot 4: REFERENCE Secondary structure
        # ========================================
        if 'reference' in results and 'dssp' in results['reference']:
            if results['reference']['dssp']['idp'].get('available', False):
                ref_idp_helix = results['reference']['dssp']['idp'].get('helix_fraction', 0)
                ref_idp_sheet = results['reference']['dssp']['idp'].get('sheet_fraction', 0)
                ref_idp_coil = results['reference']['dssp']['idp'].get('coil_fraction', 0)
                
                ref_rec_helix = results['reference']['dssp']['receptor'].get('helix_fraction', 0)
                ref_rec_sheet = results['reference']['dssp']['receptor'].get('sheet_fraction', 0)
                ref_rec_coil = results['reference']['dssp']['receptor'].get('coil_fraction', 0)
                
                # Weighted average by interface size
                if ref_idp_size > 0 and ref_rec_size > 0:
                    weighted_helix = (ref_idp_helix * ref_idp_size + ref_rec_helix * ref_rec_size) / ref_total_size
                    weighted_sheet = (ref_idp_sheet * ref_idp_size + ref_rec_sheet * ref_rec_size) / ref_total_size
                    weighted_coil = (ref_idp_coil * ref_idp_size + ref_rec_coil * ref_rec_size) / ref_total_size
                    
                    ref_helix.append(weighted_helix * 100)  # Convert to percentage
                    ref_sheet.append(weighted_sheet * 100)
                    ref_coil.append(weighted_coil * 100)
    
    # Color scheme
    color_main = 'indianred'  # For AF3 DockQ
    color_ref = 'steelblue'  # Green for reference data
    color_mean = '#E74C3C'  # Red
    color_median = '#3498DB'  # Blue
    
    # ========================================
    # Plot 1: AF3 DockQ Distribution
    # ========================================
    ax1 = axes[0, 0]
    if af3_dockq:
        n, bins, patches = ax1.hist(af3_dockq, bins=20, color=color_main, alpha=0.7, 
                                     edgecolor='black', linewidth=1.2)
        mean_val = np.mean(af3_dockq)
        median_val = np.median(af3_dockq)
        
        ax1.axvline(mean_val, color=color_mean, linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_val:.3f}')
        ax1.axvline(median_val, color=color_median, linestyle='--', linewidth=2.5, 
                   label=f'Median: {median_val:.3f}')
        
        # Add quality thresholds
        ax1.axvline(0.23, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.axvline(0.49, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.axvline(0.80, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.text(0.23, ax1.get_ylim()[1]*0.95, 'Acceptable', fontsize=8, 
                rotation=90, va='top', ha='right', color='gray')
        ax1.text(0.49, ax1.get_ylim()[1]*0.95, 'Medium', fontsize=8, 
                rotation=90, va='top', ha='right', color='gray')
        ax1.text(0.80, ax1.get_ylim()[1]*0.95, 'High', fontsize=8, 
                rotation=90, va='top', ha='right', color='gray')
    
    ax1.set_xlabel('DockQ Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'AF3 DockQ Distribution (n={len(af3_dockq)})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========================================
    # Plot 2: REFERENCE Interface Size Distribution
    # ========================================
    ax2 = axes[0, 1]
    if ref_interface_size:
        n, bins, patches = ax2.hist(ref_interface_size, bins=20, color=color_ref, alpha=0.7, 
                                     edgecolor='black', linewidth=1.2)
        mean_val = np.mean(ref_interface_size)
        median_val = np.median(ref_interface_size)
        
        ax2.axvline(mean_val, color=color_mean, linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_val:.1f}')
        ax2.axvline(median_val, color=color_median, linestyle='--', linewidth=2.5, 
                   label=f'Median: {median_val:.1f}')
    
    ax2.set_xlabel('Interface Size (residues)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'Reference Interface Size Distribution (n={len(ref_interface_size)})', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========================================
    # Plot 3: REFERENCE Hydrophobicity Distribution
    # ========================================
    ax3 = axes[1, 0]
    if ref_hydrophobicity:
        n, bins, patches = ax3.hist(ref_hydrophobicity, bins=20, color=color_ref, alpha=0.7, 
                                     edgecolor='black', linewidth=1.2)
        mean_val = np.mean(ref_hydrophobicity)
        median_val = np.median(ref_hydrophobicity)
        
        ax3.axvline(mean_val, color=color_mean, linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_val:.2f}')
        ax3.axvline(median_val, color=color_median, linestyle='--', linewidth=2.5, 
                   label=f'Median: {median_val:.2f}')
        ax3.axvline(0.5, color='green', linestyle=':', linewidth=1.5, alpha=0.6,
                   label='50% Hydrophobic')
    
    ax3.set_xlabel('Hydrophobic Fraction', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title(f'Reference Interface Hydrophobicity (n={len(ref_hydrophobicity)})', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========================================
    # Plot 4: REFERENCE Secondary Structure Distribution (BOX PLOT)
    # ========================================
    ax4 = axes[1, 1]
    if ref_helix:
        # Prepare data for box plot
        ss_data = [ref_helix, ref_sheet, ref_coil]
        labels = ['Helix', 'Sheet', 'Coil']
        colors_ss = ['#F39C12', '#3498DB', '#95A5A6']  # Orange, Blue, Gray
        
        # Create box plot
        bp = ax4.boxplot(ss_data, labels=labels, patch_artist=True,
                        widths=0.6,
                        medianprops=dict(color='black', linewidth=2),
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, 
                                       linestyle='none', markeredgecolor='darkred'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_ss):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean markers
        means = [np.mean(ref_helix), np.mean(ref_sheet), np.mean(ref_coil)]
        ax4.plot([1, 2, 3], means, 'D', color='white', markersize=8, 
                markeredgecolor='black', markeredgewidth=2, label='Mean', zorder=3)
        
        # Add horizontal grid
        ax4.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.set_axisbelow(True)
        
        # Add text box with statistics
        stats_text = 'Mean (Median)\n'
        stats_text += f'Helix: {np.mean(ref_helix):.1f}% ({np.median(ref_helix):.1f}%)\n'
        stats_text += f'Sheet: {np.mean(ref_sheet):.1f}% ({np.median(ref_sheet):.1f}%)\n'
        stats_text += f'Coil: {np.mean(ref_coil):.1f}% ({np.median(ref_coil):.1f}%)'
        
        ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.legend(loc='upper left', fontsize=9)
    
    ax4.set_ylabel('Secondary Structure Content (%)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Structure Type', fontsize=11, fontweight='bold')
    ax4.set_title(f'Reference Secondary Structure at Interface (n={len(ref_helix)})', 
                 fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    
    axes_list = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B, C, D
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    plt.tight_layout()
    plt.savefig(output_plot, format='pdf', bbox_inches='tight')
    print(f"\nDataset characteristics plot saved to {output_plot}")
    print(f"\nDataset Statistics:")
    print(f"  AF3 DockQ: Mean={np.mean(af3_dockq):.3f}, Median={np.median(af3_dockq):.3f}, n={len(af3_dockq)}")
    print(f"  Reference Interface Size: Mean={np.mean(ref_interface_size):.1f}, Median={np.median(ref_interface_size):.1f}, n={len(ref_interface_size)}")
    print(f"  Reference Hydrophobicity: Mean={np.mean(ref_hydrophobicity):.2f}, Median={np.median(ref_hydrophobicity):.2f}, n={len(ref_hydrophobicity)}")
    if ref_helix:
        print(f"  Reference Secondary Structure (Mean %):")
        print(f"    Helix: {np.mean(ref_helix):.1f}% (Median: {np.median(ref_helix):.1f}%)")
        print(f"    Sheet: {np.mean(ref_sheet):.1f}% (Median: {np.median(ref_sheet):.1f}%)")
        print(f"    Coil: {np.mean(ref_coil):.1f}% (Median: {np.median(ref_coil):.1f}%)")
    plt.close()

def plot_iptm_barplot(selected_cases, output_dir="interface_results"):
    output_dir = Path(output_dir)
    
    # Extract data
    pdb_ids = []
    af2_iptm = []
    af3_iptm = []
    dockq_diffs = []
    
    for pdb_id, diff, result in selected_cases:
        pdb_ids.append(pdb_id)
        af2_iptm.append(result['af2'].get('iptm', 0))
        af3_iptm.append(result['af3'].get('iptm', 0))
        dockq_diffs.append(diff)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(pdb_ids) * 1.2), 6))
    
    x = np.arange(len(pdb_ids))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, af2_iptm, width, label='AF2 Multimer', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, af3_iptm, width, label='AF3', 
                   color='indianred', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.02,
                f'{height1:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.02,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add confidence thresholds
    ax.axhline(0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Styling
    ax.set_ylabel('ipTM Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('PDB ID', fontsize=13, fontweight='bold')
    ax.set_title('Interface Predicted TM-score (ipTM) Comparison\nFor Top Cases by |ΔDockQ|',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Create x-tick labels with ΔDockQ
    labels = [f'{pdb}\n(ΔDQ:{diff:+.2f})' for pdb, diff in zip(pdb_ids, dockq_diffs)]
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    output_file = output_dir / "iptm_comparison_barplot.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"  ✓ ipTM bar plot saved to: {output_file}")
    plt.close()

def analyze_contact_stability(af2_pdb, af3_pdb, native_pdb, chain_info, distance_cutoff=5.0):
    from Bio.PDB import PDBParser, MMCIFParser
    import numpy as np
    
    try:
        print(f"      → Loading structures and creating sequence mappings...")
        
        mapper = SequenceMapper()
        extractor = InterfaceExtractor(distance_cutoff=distance_cutoff)
        
        # Load all three structures
        native_struct = mapper.load_structure(native_pdb, "reference")
        af2_struct = mapper.load_structure(af2_pdb, "af2")
        af3_struct = mapper.load_structure(af3_pdb, "af3")
        
        # IDP chain mappings
        ref_idp_seq, ref_idp_nums = mapper.extract_sequence(native_struct, chain_info['ref_idp'])
        af2_idp_seq, af2_idp_nums = mapper.extract_sequence(af2_struct, chain_info['af2_idp'])
        af3_idp_seq, af3_idp_nums = mapper.extract_sequence(af3_struct, chain_info['af3_idp'])
        
        # Receptor chain mappings
        ref_rec_seq, ref_rec_nums = mapper.extract_sequence(native_struct, chain_info['ref_receptor'])
        af2_rec_seq, af2_rec_nums = mapper.extract_sequence(af2_struct, chain_info['af2_receptor'])
        af3_rec_seq, af3_rec_nums = mapper.extract_sequence(af3_struct, chain_info['af3_receptor'])
        
        # Create reference -> predicted mappings
        ref_to_af2_idp = mapper.create_residue_mapping(ref_idp_seq, ref_idp_nums, af2_idp_seq, af2_idp_nums)
        ref_to_af3_idp = mapper.create_residue_mapping(ref_idp_seq, ref_idp_nums, af3_idp_seq, af3_idp_nums)
        ref_to_af2_rec = mapper.create_residue_mapping(ref_rec_seq, ref_rec_nums, af2_rec_seq, af2_rec_nums)
        ref_to_af3_rec = mapper.create_residue_mapping(ref_rec_seq, ref_rec_nums, af3_rec_seq, af3_rec_nums)
        
        # Create reverse mappings (predicted -> reference)
        af2_to_ref_idp = {v: k for k, v in ref_to_af2_idp.items()}
        af3_to_ref_idp = {v: k for k, v in ref_to_af3_idp.items()}
        af2_to_ref_rec = {v: k for k, v in ref_to_af2_rec.items()}
        af3_to_ref_rec = {v: k for k, v in ref_to_af3_rec.items()}
        
        print(f"        ✓ Mapped IDP: {len(ref_to_af2_idp)} (AF2), {len(ref_to_af3_idp)} (AF3) residues")
        print(f"        ✓ Mapped Receptor: {len(ref_to_af2_rec)} (AF2), {len(ref_to_af3_rec)} (AF3) residues")
        
        # ========================================
        # STEP 3: Extract pLDDT scores (using original residue numbering)
        # ========================================
        af2_plddt_idp = extractor.extract_plddt_scores(af2_struct, chain_info['af2_idp'])
        af2_plddt_rec = extractor.extract_plddt_scores(af2_struct, chain_info['af2_receptor'])
        
        af3_plddt_idp = extractor.extract_plddt_scores(af3_struct, chain_info['af3_idp'])
        af3_plddt_rec = extractor.extract_plddt_scores(af3_struct, chain_info['af3_receptor'])
        
        # ========================================
        # STEP 4: Define contact extraction function
        # ========================================
        def get_contacts_with_mapping(structure, chain1_id, chain2_id, 
                                     valid_res_chain1=None, valid_res_chain2=None,
                                     plddt_dict_chain1=None, plddt_dict_chain2=None,
                                     disorder_threshold=70):

            model = structure[0]
            chain1 = model[chain1_id]
            chain2 = model[chain2_id]
            contacts = []
            disorder_contacts = []
            
            for res1 in chain1:
                if res1.id[0] != ' ':
                    continue
                res1_id = res1.id[1] 
                # Skip if not in valid residue set (i.e., not mappable to native)
                if valid_res_chain1 is not None and res1_id not in valid_res_chain1:
                    continue

                for atom1 in res1.get_atoms():
                    if atom1.element == 'H':
                        continue
                    
                    for res2 in chain2:
                        if res2.id[0] != ' ':
                            continue
                        
                        res2_id = res2.id[1]
                        
                        # Skip if not in valid residue set
                        if valid_res_chain2 is not None and res2_id not in valid_res_chain2:
                            continue
                        
                        for atom2 in res2.get_atoms():
                            if atom2.element == 'H':
                                continue
                            
                            distance = atom1 - atom2
                            if distance <= distance_cutoff:
                                contact = (res1_id, res2_id)
                                contacts.append(contact)
                                
                                # Check if contact involves disordered residue
                                if plddt_dict_chain1 is not None and plddt_dict_chain2 is not None:
                                    plddt1 = plddt_dict_chain1.get(res1_id, 100)
                                    plddt2 = plddt_dict_chain2.get(res2_id, 100)
                                    if plddt1 < disorder_threshold or plddt2 < disorder_threshold:
                                        disorder_contacts.append(contact)
                                break
            
            # Remove duplicates
            contacts = list(set(contacts))
            disorder_contacts = list(set(disorder_contacts))
            
            return contacts, disorder_contacts
        
        print(f"      → Extracting contacts from native structure...")
        native_contacts, _ = get_contacts_with_mapping(
            native_struct, 
            chain_info['ref_idp'], 
            chain_info['ref_receptor']
        )
        native_contact_set = set(native_contacts)
        print(f"        ✓ Found {len(native_contacts)} native contacts")
        
        print(f"      → Extracting contacts from AF2...")
        
        # Valid residues = those that map to native
        af2_valid_idp = set(ref_to_af2_idp.values())  # AF2 numbering
        af2_valid_rec = set(ref_to_af2_rec.values())
        
        af2_contacts_raw, af2_disorder_contacts_raw = get_contacts_with_mapping(
            af2_struct,
            chain_info['af2_idp'],
            chain_info['af2_receptor'],
            valid_res_chain1=af2_valid_idp,
            valid_res_chain2=af2_valid_rec,
            plddt_dict_chain1=af2_plddt_idp,
            plddt_dict_chain2=af2_plddt_rec,
            disorder_threshold=68
        )
        
        # Convert AF2 contacts to reference numbering for comparison
        af2_contacts_ref = []
        for res1_af2, res2_af2 in af2_contacts_raw:
            res1_ref = af2_to_ref_idp.get(res1_af2)
            res2_ref = af2_to_ref_rec.get(res2_af2)
            if res1_ref is not None and res2_ref is not None:
                af2_contacts_ref.append((res1_ref, res2_ref))
        
        af2_contact_set_ref = set(af2_contacts_ref)
        print(f"        ✓ Found {len(af2_contacts_raw)} AF2 contacts ({len(af2_contacts_ref)} mapped to native)")
        
        print(f"      → Extracting contacts from AF3...")
        
        # Valid residues = those that map to native
        af3_valid_idp = set(ref_to_af3_idp.values())  # AF3 numbering
        af3_valid_rec = set(ref_to_af3_rec.values())
        
        af3_contacts_raw, af3_disorder_contacts_raw = get_contacts_with_mapping(
            af3_struct,
            chain_info['af3_idp'],
            chain_info['af3_receptor'],
            valid_res_chain1=af3_valid_idp,
            valid_res_chain2=af3_valid_rec,
            plddt_dict_chain1=af3_plddt_idp,
            plddt_dict_chain2=af3_plddt_rec,
            disorder_threshold=70
        )
        
        # Convert AF3 contacts to reference numbering for comparison
        af3_contacts_ref = []
        for res1_af3, res2_af3 in af3_contacts_raw:
            res1_ref = af3_to_ref_idp.get(res1_af3)
            res2_ref = af3_to_ref_rec.get(res2_af3)
            if res1_ref is not None and res2_ref is not None:
                af3_contacts_ref.append((res1_ref, res2_ref))
        
        af3_contact_set_ref = set(af3_contacts_ref)
        print(f"        ✓ Found {len(af3_contacts_raw)} AF3 contacts ({len(af3_contacts_ref)} mapped to native)")
        
        print(f"      → Calculating contact accuracy metrics...")
        
        # True positives: contacts correctly predicted
        af2_correct = len(af2_contact_set_ref & native_contact_set)
        af3_correct = len(af3_contact_set_ref & native_contact_set)
        
        
        # Recall (sensitivity): TP / (TP + FN)
        af2_contact_accuracy = af2_correct / len(native_contact_set) if len(native_contact_set) > 0 else 0
        af3_contact_accuracy = af3_correct / len(native_contact_set) if len(native_contact_set) > 0 else 0
        
        # Disorder contact fraction (in original numbering)
        af2_disorder_frac = len(af2_disorder_contacts_raw) / len(af2_contacts_raw) if len(af2_contacts_raw) > 0 else 0
        af3_disorder_frac = len(af3_disorder_contacts_raw) / len(af3_contacts_raw) if len(af3_contacts_raw) > 0 else 0
        
        # Over-contacted detection (comparing mapped contacts)
        af2_over_contacted = len(af2_contact_set_ref) > len(native_contact_set)
        af3_over_contacted = len(af3_contact_set_ref) > len(native_contact_set)
        
        return {
            # Disorder metrics
            'disorder_contact_fraction_af2': af2_disorder_frac,
            'disorder_contact_fraction_af3': af3_disorder_frac,
            'af2_disorder_contacts': len(af2_disorder_contacts_raw),
            'af3_disorder_contacts': len(af3_disorder_contacts_raw),
            
            # Over-contact detection
            'af2_over_contacted': af2_over_contacted,
            'af3_over_contacted': af3_over_contacted,
            
            # Recall (sensitivity) - main accuracy metric
            'af2_contact_accuracy': af2_contact_accuracy,
            'af3_contact_accuracy': af3_contact_accuracy,
        
            
            # Raw counts
            'af2_contact_count': len(af2_contacts_raw),
            'af3_contact_count': len(af3_contacts_raw),
            'native_contact_count': len(native_contacts),
            
            
            # True/False positives/negatives
            'af2_true_positive': af2_correct,
            'af3_true_positive': af3_correct,
            
        }
        
    except Exception as e:
        print(f"    ✗ Contact stability analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



from Bio.PDB import PDBParser, Superimposer
import numpy as np
from scipy.stats import pearsonr

def align_structures(structures, reference_idx=0):
    sup = Superimposer()
    ref_atoms = [atom for atom in structures[reference_idx].get_atoms() if atom.name == 'CA']
    
    aligned_coords = []
    for struct in structures:
        mobile_atoms = [atom for atom in struct.get_atoms() if atom.name == 'CA']
        sup.set_atoms(ref_atoms, mobile_atoms)
        sup.apply(mobile_atoms)  # Apply transformation to atoms
        
        coords = np.array([atom.coord for atom in mobile_atoms])
        aligned_coords.append(coords)
    
    return np.array(aligned_coords)

def analyze_conformational_diversity(af3_samples, plddt_threshold=70):
    """
    Calculate conformational diversity metrics across AF3 seeds
    af3_samples: list of 4 PDB/CIF files from different seeds
    """
    from Bio.PDB import MMCIFParser  # Use MMCIFParser for CIF files
    
    parser = MMCIFParser(QUIET=True)  # Changed from PDBParser to MMCIFParser
    structures = [parser.get_structure(f"seed_{i}", pdb) 
                  for i, pdb in enumerate(af3_samples)]
    
    # Align structures
    aligned_coords = align_structures(structures)  # shape: (4, n_residues, 3)
    
    # Calculate per-residue RMSD across 4 seeds
    per_residue_rmsd = np.std(aligned_coords, axis=0)  # std across seeds
    per_residue_rmsd = np.sqrt(np.sum(per_residue_rmsd**2, axis=1))  # magnitude
    
    # Get pLDDT scores from first seed
    plddt_scores = np.array([atom.bfactor for atom in structures[0].get_atoms() 
                             if atom.name == 'CA'])
    
    # Separate by pLDDT
    low_plddt_mask = plddt_scores < plddt_threshold
    high_plddt_mask = plddt_scores >= plddt_threshold
    
    low_plddt_rmsd = np.mean(per_residue_rmsd[low_plddt_mask]) if np.any(low_plddt_mask) else 0.0
    high_plddt_rmsd = np.mean(per_residue_rmsd[high_plddt_mask]) if np.any(high_plddt_mask) else 0.0
    rmsd_ratio = low_plddt_rmsd / high_plddt_rmsd if high_plddt_rmsd > 0 else 0.0
    
    # Correlation between pLDDT and variability
    corr, _ = pearsonr(plddt_scores, per_residue_rmsd)
    disorder_variability_correlation = abs(corr)
    
    # Classify ensemble behavior
    if low_plddt_rmsd < 2.0:
        diversity_category = 'collapsed'
    elif rmsd_ratio > 3.0:
        diversity_category = 'ensemble-like'
    else:
        diversity_category = 'intermediate'
    
    return {
        'low_plddt_rmsd': low_plddt_rmsd,
        'high_plddt_rmsd': high_plddt_rmsd,
        'rmsd_ratio': rmsd_ratio,
        'disorder_variability_correlation': disorder_variability_correlation,
        'af3_samples_diversity': diversity_category
    }

def get_interface_bfactors(pdb_file, interface_residues, chain_id):
    """
    Extract B-factors for interface residues from experimental structure.
    
    This is DIFFERENT from pLDDT - it measures experimental atomic mobility.
    """
    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    interface_bfactors = []
    bulk_bfactors = []
    
    for model in structure:
        if chain_id not in model:
            continue
        chain = model[chain_id]
        
        for residue in chain:
            res_id = residue.id[1]
            res_id = str(res_id)
            # Get CA atom B-factor (representative)
            if 'CA' in residue:
                bfactor = residue['CA'].get_bfactor()
                
                if res_id in interface_residues:
                    interface_bfactors.append(bfactor)
                else:
                    bulk_bfactors.append(bfactor)
    
    return {
        'mean_interface_bfactor': np.mean(interface_bfactors),
        'mean_bulk_bfactor': np.mean(bulk_bfactors),
        'bfactor_ratio': np.mean(interface_bfactors) / np.mean(bulk_bfactors),
        'interface_bfactors': interface_bfactors,
        'bulk_bfactors': bulk_bfactors
    }

def plot_residue_level_analysis(results_list, output_dir="interface_results", top_n=4,
                                ref_dir="dataset/naive_files",
                                af2_dir="dataset/alphafold2_files",
                                af3_dir="dataset/alphafold3_files",
                                af3_cf_dir="dataset/alphafold3_conformation"):
    
    output_dir = Path(output_dir)
    residue_dir = output_dir / "residue_level_plots"
    residue_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating residue-level analysis plots (top {top_n} cases)...")
    
    # Find interesting cases (largest DockQ differences)
    dockq_diffs = []
    for result in results_list:
        if result["protein_id"] == "7NMI" or result["protein_id"] == "7UZU" or result["protein_id"] == "7OS1":
            continue
        if result.get('af2_dockq') is not None and result.get('af3_dockq') is not None:
            diff = result['af3_dockq'] - result['af2_dockq']
            dockq_diffs.append((result['protein_id'], diff, result))
    
    # Sort by absolute difference
    dockq_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Select top N cases
    selected_cases = dockq_diffs[:top_n]
    
    # Create ipTM bar plot for these selected cases
    print(f"  - Creating ipTM bar plot for top {top_n} cases...")
    plot_iptm_barplot(selected_cases, output_dir)

    print(f"  - Get b-factor for top {top_n} cases...")
    bfactor_result = {}
    for pdb_id, diff, result in selected_cases:
        print(f"    • Analyzing {pdb_id}...")
        pdb_file = f'{af3_dir}/{pdb_id}.cif' 
        idp_bfactor = get_interface_bfactors(pdb_file, result["af3_interface"]["idp"].keys(), result["chain_info"]["af3_idp"])
        rec_bfactor = get_interface_bfactors(pdb_file, result["af3_interface"]["receptor"].keys(), result["chain_info"]["af3_receptor"])
        bfactor_result[pdb_id] = {
            "idp": idp_bfactor,
            "receptor": rec_bfactor
        }

    with open(output_dir / "bfactor_result.json", 'w') as f:
        json.dump(bfactor_result, f, indent=2)
    # ========================================
    # NEW: Run contact and diversity analysis
    # ========================================
    # print(f"\n  - Running contact stability analysis...")
    # contact_results = []
    
    # for result in results_list:
    #     print(f"    • Analyzing {result['protein_id']}...")
        
    #     # Find structure files
    #     ref_path = None
    #     for ext in ['.pdb', '.cif']:
    #         candidate = Path(ref_dir) / f"{result['protein_id']}{ext}"
    #         if candidate.exists():
    #             ref_path = candidate
    #             break
        
    #     af2_path = Path(af2_dir) / f"{result['protein_id']}.pdb"
    #     af3_path = Path(af3_dir) / f"{result['protein_id']}.cif"
        
    #     if ref_path and af2_path.exists() and af3_path.exists():
    #         contact_analysis = analyze_contact_stability(
    #             af2_path, af3_path, ref_path, 
    #             result['chain_info']
    #         )
            
    #         if contact_analysis:
    #             contact_results.append({
    #                 'pdb_id': result['protein_id'],
    #                 **contact_analysis
    #             })
    #             print(f"      ✓ Contact analysis complete")
    #             print(f"        AF2 accuracy: {contact_analysis['af2_contact_accuracy']:.2f}")
    #             print(f"        AF3 accuracy: {contact_analysis['af3_contact_accuracy']:.2f}")
    
    # # Save contact analysis results
    # if contact_results:
    #     contact_df = pd.DataFrame(contact_results)
    #     contact_csv = output_dir / "contact_stability_analysis.csv"
    #     contact_df.to_csv(contact_csv, index=False)
    #     print(f"\n  ✓ Contact stability results saved to: {contact_csv}")
    
    print(f"\n  - Running conformational diversity analysis...")
    diversity_results = []
    
    for pdb_id, diff, result in selected_cases:
        print(f"    • Analyzing {pdb_id}...")
        af3_samples = [
            f'{af3_cf_dir}/{pdb_id}_seed_{i}.cif' 
            for i in range(1, 5)
        ]
        
        result = analyze_conformational_diversity(af3_samples)
        result['pdb_id'] = pdb_id
        diversity_results.append(result)
        # Save diversity analysis results
        if diversity_results:
            diversity_df = pd.DataFrame(diversity_results)
            diversity_csv = output_dir / "conformational_diversity_analysis.csv"
            diversity_df.to_csv(diversity_csv, index=False)
            print(f"\n  ✓ Conformational diversity results saved to: {diversity_csv}")
        
    # ========================================
    # Create individual residue-level plots
    # ========================================
    print(f"\n  - Creating individual residue-level plots...")
    for pdb_id, diff, result in selected_cases:
        try:
            print(f"    • Creating plot for {pdb_id} (ΔDockQ = {diff:+.3f})...")
            fig = _create_comprehensive_interface_plot(pdb_id, result)
            fig.savefig(residue_dir / f'{pdb_id}_residue_analysis.pdf', 
                       format='pdf', bbox_inches='tight')
            plt.close(fig)
            print(f"      ✓ Plot saved")
        except Exception as e:
            print(f"      ✗ Error creating plot for {pdb_id}: {str(e)}")
    
    print(f"\n  ✓ Residue-level plots saved to: {residue_dir}/")
    print(f"\n{'='*60}")
    print(f"RESIDUE-LEVEL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Selected cases: {len(selected_cases)}")
    print(f"Contact analyses: {len(contact_results)}")
    print(f"Diversity analyses: {len(diversity_results)}")
    print(f"{'='*60}\n")
    


def _create_comprehensive_interface_plot(pdb_id, result):
    """
    Create 3-panel figure showing interface residue-level analysis
    
    Panel A: Secondary Structure (DSSP) - Reference, AF2, AF3
    Panel B: Disorder Propensity (from pLDDT) - AF2 vs AF3
    Panel C: Hydropathy (Kyte-Doolittle) - Reference, AF2, AF3
    
    MODIFIED: Uses reference numbering for display (x-axis) but native numbering for data extraction
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Extract interface residue data for IDP chain
    chain_type = 'idp'
    
    # Get residue positions in their NATIVE numbering systems
    ref_interface = result['reference_interface'][chain_type]  # {ref_pos_str: aa}
    af2_interface = result['af2_interface'][chain_type]        # {af2_pos_str: aa}
    af3_interface = result['af3_interface'][chain_type]        # {af3_pos_str: aa}
    
    # ========================================
    # CREATE MAPPINGS: Native -> Reference coordinate system
    # ========================================
    # We need to build mappings from the stored results
    # Note: These should ideally be stored in analyze_single_protein
    # For now, we'll reconstruct them from the chain_info
    
    # Load structures to recreate mappings
    mapper = SequenceMapper()
    
    try:
        # Find structure files
        ref_path = None
        for ext in ['.pdb', '.cif']:
            candidate = Path("dataset/naive_files") / f"{pdb_id}{ext}"
            if candidate.exists():
                ref_path = candidate
                break
        
        af2_path = Path("dataset/alphafold2_files") / f"{pdb_id}.pdb"
        af3_path = Path("dataset/alphafold3_files") / f"{pdb_id}.cif"
        
        if not (ref_path and af2_path.exists() and af3_path.exists()):
            raise FileNotFoundError("Required structure files not found")
        
        # Load structures
        ref_struct = mapper.load_structure(ref_path, "reference")
        af2_struct = mapper.load_structure(af2_path, "af2")
        af3_struct = mapper.load_structure(af3_path, "af3")
        
        chain_info = result['chain_info']
        
        # Extract sequences and create mappings
        ref_idp_seq, ref_idp_nums = mapper.extract_sequence(ref_struct, chain_info['ref_idp'])
        af2_idp_seq, af2_idp_nums = mapper.extract_sequence(af2_struct, chain_info['af2_idp'])
        af3_idp_seq, af3_idp_nums = mapper.extract_sequence(af3_struct, chain_info['af3_idp'])
        
        # Create Reference -> AF2/AF3 mappings
        ref_to_af2_mapping = mapper.create_residue_mapping(ref_idp_seq, ref_idp_nums, 
                                                           af2_idp_seq, af2_idp_nums)
        ref_to_af3_mapping = mapper.create_residue_mapping(ref_idp_seq, ref_idp_nums, 
                                                           af3_idp_seq, af3_idp_nums)
        
        # Create REVERSE mappings (AF2/AF3 -> Reference) for display
        af2_to_ref_mapping = {v: k for k, v in ref_to_af2_mapping.items()}
        af3_to_ref_mapping = {v: k for k, v in ref_to_af3_mapping.items()}
        
    except Exception as e:
        print(f"Warning: Could not create mappings for {pdb_id}, using identity mapping: {e}")
        # Fallback: assume identity mapping
        af2_to_ref_mapping = {int(pos): int(pos) for pos in af2_interface.keys()}
        af3_to_ref_mapping = {int(pos): int(pos) for pos in af3_interface.keys()}
    
    # ========================================
    # CREATE UNIFIED REFERENCE COORDINATE SPACE
    # ========================================
    # All positions in REFERENCE numbering
    ref_positions_set = {int(pos) for pos in ref_interface.keys()}
    
    # Map AF2 interface positions to reference coordinates
    af2_positions_in_ref_coords = set()
    for af2_pos_str in af2_interface.keys():
        af2_pos_int = int(af2_pos_str)
        ref_pos = af2_to_ref_mapping.get(af2_pos_int)
        if ref_pos is not None:
            af2_positions_in_ref_coords.add(ref_pos)
    
    # Map AF3 interface positions to reference coordinates
    af3_positions_in_ref_coords = set()
    for af3_pos_str in af3_interface.keys():
        af3_pos_int = int(af3_pos_str)
        ref_pos = af3_to_ref_mapping.get(af3_pos_int)
        if ref_pos is not None:
            af3_positions_in_ref_coords.add(ref_pos)
    
    # UNION of all positions in reference coordinates
    all_ref_positions = sorted(ref_positions_set | af2_positions_in_ref_coords | af3_positions_in_ref_coords)
    
    if not all_ref_positions:
        raise ValueError(f"No interface data for {pdb_id}")
    
    print(f"    Unified coordinate space for {pdb_id}:")
    print(f"      Reference positions: {len(ref_positions_set)}")
    print(f"      AF2 positions (in ref coords): {len(af2_positions_in_ref_coords)}")
    print(f"      AF3 positions (in ref coords): {len(af3_positions_in_ref_coords)}")
    print(f"      Total unique positions: {len(all_ref_positions)}")
    
    # Create REVERSE lookup: Reference position -> Native AF2/AF3 positions
    ref_to_af2_native = {v: k for k, v in af2_to_ref_mapping.items()}
    ref_to_af3_native = {v: k for k, v in af3_to_ref_mapping.items()}
    
    # Get amino acids for display (prefer AF3 > AF2 > Ref)
    residues = []
    for ref_pos in all_ref_positions:
        ref_pos_str = str(ref_pos)
        
        # Try AF3 first
        af3_native_pos = ref_to_af3_native.get(ref_pos)
        if af3_native_pos is not None and str(af3_native_pos) in af3_interface:
            residues.append(af3_interface[str(af3_native_pos)])
        # Then AF2
        elif (af2_native_pos := ref_to_af2_native.get(ref_pos)) is not None and str(af2_native_pos) in af2_interface:
            residues.append(af2_interface[str(af2_native_pos)])
        # Finally reference
        elif ref_pos_str in ref_interface:
            residues.append(ref_interface[ref_pos_str])
        else:
            residues.append('X')
    
    x = np.arange(len(all_ref_positions))
    
    # ========================================
    # Panel A: DSSP (Secondary Structure)
    # ========================================
    ax = axes[0]
    
    # DSSP color scheme
    ss_colors_map = {
        'G': '#FF00FF', 'H': '#00FF00', 'I': '#FFFF00',
        'E': '#FF0000', 'B': '#FFA500', 'T': '#00FFFF',
        'S': '#0000FF', 'C': '#D3D3D3', '-': 'white'
    }
    
    ref_dssp_available = result['reference']['dssp'][chain_type].get('available', False)
    af2_dssp_available = result['af2']['dssp'][chain_type].get('available', False)
    af3_dssp_available = result['af3']['dssp'][chain_type].get('available', False)
    
    if af2_dssp_available or af3_dssp_available or ref_dssp_available:
        # Get DSSP dictionaries in NATIVE numbering
        ref_ss_dict = result['reference']['dssp'][chain_type].get('structure_dict', {})
        af2_ss_dict = result['af2']['dssp'][chain_type].get('structure_dict', {})
        af3_ss_dict = result['af3']['dssp'][chain_type].get('structure_dict', {})
        
        ref_colors_ss = []
        af2_colors_ss = []
        af3_colors_ss = []
        
        for ref_pos in all_ref_positions:
            ref_pos_str = str(ref_pos)
            
            # Reference DSSP
            if ref_dssp_available and ref_pos_str in ref_ss_dict:
                ref_ss = ref_ss_dict[ref_pos_str]
                ref_colors_ss.append(ss_colors_map.get(ref_ss, '#D3D3D3'))
            else:
                ref_colors_ss.append('white')
            
            # AF2 DSSP (convert ref_pos to AF2 native position)
            af2_native_pos = ref_to_af2_native.get(ref_pos)
            if af2_dssp_available and af2_native_pos is not None:
                af2_native_pos_str = str(af2_native_pos)
                if af2_native_pos_str in af2_ss_dict:
                    af2_ss = af2_ss_dict[af2_native_pos_str]
                    af2_colors_ss.append(ss_colors_map.get(af2_ss, '#D3D3D3'))
                else:
                    af2_colors_ss.append('white')
            else:
                af2_colors_ss.append('white')
            
            # AF3 DSSP (convert ref_pos to AF3 native position)
            af3_native_pos = ref_to_af3_native.get(ref_pos)
            if af3_dssp_available and af3_native_pos is not None:
                af3_native_pos_str = str(af3_native_pos)
                if af3_native_pos_str in af3_ss_dict:
                    af3_ss = af3_ss_dict[af3_native_pos_str]
                    af3_colors_ss.append(ss_colors_map.get(af3_ss, '#D3D3D3'))
                else:
                    af3_colors_ss.append('white')
            else:
                af3_colors_ss.append('white')
        
        bar_height = 1.0 / 3.0
        
        ax.bar(x, [bar_height]*len(x), bottom=2*bar_height, color=ref_colors_ss, 
               edgecolor='black', linewidth=0.3, width=1.0, label='Reference')
        ax.bar(x, [bar_height]*len(x), bottom=bar_height, color=af2_colors_ss, 
               edgecolor='black', linewidth=0.3, width=1.0, label='AF2')
        ax.bar(x, [bar_height]*len(x), bottom=0, color=af3_colors_ss, 
               edgecolor='black', linewidth=0.3, width=1.0, label='AF3')
        
        ax.axhline(bar_height, color='black', linewidth=1.5, alpha=0.7)
        ax.axhline(2*bar_height, color='black', linewidth=1.5, alpha=0.7)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF00FF', label='π-Helix'),
            Patch(facecolor='#00FF00', label='α-Helix'),
            Patch(facecolor='#FFFF00', label='310-Helix'),
            Patch(facecolor='#FF0000', label='β-Strand'),
            Patch(facecolor='#FFA500', label='β-Bridge'),
            Patch(facecolor='#00FFFF', label='Turn'),
            Patch(facecolor='#0000FF', label='Bend'),
            Patch(facecolor='#D3D3D3', label='Loop'),
            Patch(facecolor='white', edgecolor='black', label='Not in interface')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=5)
        
        # Add statistics
        stats_text = ''
        if ref_dssp_available:
            ref_helix = result['reference']['dssp'][chain_type]['helix_fraction']
            ref_sheet = result['reference']['dssp'][chain_type]['sheet_fraction']
            ref_coil = result['reference']['dssp'][chain_type]['coil_fraction']
            stats_text += f'Ref ({len(ref_positions_set)} res): H:{ref_helix:.1%} E:{ref_sheet:.1%} C:{ref_coil:.1%}\n'
        else:
            stats_text += 'Ref: DSSP N/A\n'
        
        if af2_dssp_available:
            af2_helix = result['af2']['dssp'][chain_type]['helix_fraction']
            af2_sheet = result['af2']['dssp'][chain_type]['sheet_fraction']
            af2_coil = result['af2']['dssp'][chain_type]['coil_fraction']
            stats_text += f'AF2 ({len(af2_positions_in_ref_coords)} res): H:{af2_helix:.1%} E:{af2_sheet:.1%} C:{af2_coil:.1%}\n'
        else:
            stats_text += 'AF2: DSSP N/A\n'
        
        if af3_dssp_available:
            af3_helix = result['af3']['dssp'][chain_type]['helix_fraction']
            af3_sheet = result['af3']['dssp'][chain_type]['sheet_fraction']
            af3_coil = result['af3']['dssp'][chain_type]['coil_fraction']
            stats_text += f'AF3 ({len(af3_positions_in_ref_coords)} res): H:{af3_helix:.1%} E:{af3_sheet:.1%} C:{af3_coil:.1%}'
        else:
            stats_text += 'AF3: DSSP N/A'
        
        ax.text(0.02, 0.95, stats_text, 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(0.98, 0.83, 'Ref', transform=ax.transAxes, fontsize=10, 
                fontweight='bold', ha='right', va='center')
        ax.text(0.98, 0.50, 'AF2', transform=ax.transAxes, fontsize=10, 
                fontweight='bold', ha='right', va='center')
        ax.text(0.98, 0.17, 'AF3', transform=ax.transAxes, fontsize=10, 
                fontweight='bold', ha='right', va='center')
    else:
        ax.text(0.5, 0.5, 'DSSP data not available', 
                transform=ax.transAxes, fontsize=12, ha='center', va='center')
        ax.bar(x, [1]*len(x), color='lightgray', edgecolor='black', linewidth=0.5, width=1.0)
    
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.17, 0.50, 0.83])
    ax.set_yticklabels(['AF3', 'AF2', 'Ref'], fontsize=10)
    ax.set_ylabel('Secondary\nStructure', fontsize=12, fontweight='bold')
    
    title = f'{pdb_id}: Interface Residue-Level Analysis (IDP Chain, Reference Coordinates)'
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.1)
    
    # ========================================
    # Panel B: Disorder Propensity
    # ========================================
    ax = axes[1]
    
    # Get pLDDT scores in NATIVE numbering
    af2_quality_list = result['af2'][chain_type].get('quality_score', [])
    af3_quality_list = result['af3'][chain_type].get('quality_score', [])
    
    af2_interface_positions_sorted = sorted([int(p) for p in af2_interface.keys()])
    af3_interface_positions_sorted = sorted([int(p) for p in af3_interface.keys()])
    
    # Create dictionaries: Native position -> pLDDT
    af2_plddt_dict = {}
    if len(af2_quality_list) == len(af2_interface_positions_sorted):
        for pos, plddt in zip(af2_interface_positions_sorted, af2_quality_list):
            af2_plddt_dict[pos] = plddt
    
    af3_plddt_dict = {}
    if len(af3_quality_list) == len(af3_interface_positions_sorted):
        for pos, plddt in zip(af3_interface_positions_sorted, af3_quality_list):
            af3_plddt_dict[pos] = plddt
    
    # Extract pLDDT values aligned to reference coordinates
    af2_plddt_values = []
    af3_plddt_values = []
    
    for ref_pos in all_ref_positions:
        # AF2: convert ref_pos to AF2 native position, then get pLDDT
        af2_native_pos = ref_to_af2_native.get(ref_pos)
        if af2_native_pos is not None and af2_native_pos in af2_plddt_dict:
            plddt_af2 = af2_plddt_dict[af2_native_pos]
        else:
            plddt_af2 = 0
        af2_plddt_values.append(plddt_af2)
        
        # AF3: convert ref_pos to AF3 native position, then get pLDDT
        af3_native_pos = ref_to_af3_native.get(ref_pos)
        if af3_native_pos is not None and af3_native_pos in af3_plddt_dict:
            plddt_af3 = af3_plddt_dict[af3_native_pos]
        else:
            plddt_af3 = 0
        af3_plddt_values.append(plddt_af3)
    
    # Convert to disorder scores
    af2_disorder_scores = [1.0 - (plddt / 100.0) if plddt > 0 else 0 
                          for plddt in af2_plddt_values]
    af3_disorder_scores = [1.0 - (plddt / 100.0) if plddt > 0 else 0 
                          for plddt in af3_plddt_values]
    
    # Plot
    ax.fill_between(x, 0, af2_disorder_scores, color='steelblue', alpha=0.4, label='AF2 disorder')
    ax.fill_between(x, 0, af3_disorder_scores, color='indianred', alpha=0.4, label='AF3 disorder')
    
    af2_mask = np.array(af2_disorder_scores) > 0
    af3_mask = np.array(af3_disorder_scores) > 0
    
    if af2_mask.any():
        ax.plot(x[af2_mask], np.array(af2_disorder_scores)[af2_mask], 
               color='steelblue', linewidth=1.5, alpha=0.8, marker='o', markersize=3)
    if af3_mask.any():
        ax.plot(x[af3_mask], np.array(af3_disorder_scores)[af3_mask], 
               color='indianred', linewidth=1.5, alpha=0.8, marker='^', markersize=3)
    
    ax.axhline(0.3, linestyle='--', color='red', alpha=0.7, linewidth=1.5, 
               label='Disorder threshold (pLDDT<70)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Disorder\nPropensity', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Statistics
    af2_disorder_frac = result['af2'][chain_type]['disorder_fraction']
    af3_disorder_frac = result['af3'][chain_type]['disorder_fraction']
    
    af2_mean_disorder = np.mean([d for d, m in zip(af2_disorder_scores, af2_mask) if m]) if af2_mask.any() else 0
    af3_mean_disorder = np.mean([d for d, m in zip(af3_disorder_scores, af3_mask) if m]) if af3_mask.any() else 0
    
    stats_text = f'AF2 ({len(af2_positions_in_ref_coords)} res): Frac={af2_disorder_frac:.1%}, Mean={af2_mean_disorder:.2f}\n'
    stats_text += f'AF3 ({len(af3_positions_in_ref_coords)} res): Frac={af3_disorder_frac:.1%}, Mean={af3_mean_disorder:.2f}'
    
    ax.text(0.02, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================
    # Panel C: Hydropathy
    # ========================================
    ax = axes[2]
    
    # Get hydropathy dictionaries in NATIVE numbering
    ref_hydro_dict = result['reference']['hydropathy'][chain_type]['residue_scores']
    af2_hydro_dict = result['af2']['hydropathy'][chain_type]['residue_scores']
    af3_hydro_dict = result['af3']['hydropathy'][chain_type]['residue_scores']
    
    hydropathy_values = []
    source_markers = []
    
    for ref_pos in all_ref_positions:
        ref_pos_str = str(ref_pos)
        
        # Try AF3 first
        af3_native_pos = ref_to_af3_native.get(ref_pos)
        if af3_native_pos is not None and str(af3_native_pos) in af3_hydro_dict:
            hydropathy_values.append(af3_hydro_dict[str(af3_native_pos)])
            source_markers.append('af3')
        # Then AF2
        elif (af2_native_pos := ref_to_af2_native.get(ref_pos)) is not None and str(af2_native_pos) in af2_hydro_dict:
            hydropathy_values.append(af2_hydro_dict[str(af2_native_pos)])
            source_markers.append('af2')
        # Finally reference
        elif ref_pos_str in ref_hydro_dict:
            hydropathy_values.append(ref_hydro_dict[ref_pos_str])
            source_markers.append('ref')
        else:
            hydropathy_values.append(0)
            source_markers.append('none')
    
    # Color coding
    colors_hydro = []
    for h, source in zip(hydropathy_values, source_markers):
        if source == 'af3':
            colors_hydro.append('#8B4513' if h > 0 else '#4169E1')
        elif source == 'af2':
            colors_hydro.append('#D2691E' if h > 0 else '#87CEEB')
        elif source == 'ref':
            colors_hydro.append('#F4A460' if h > 0 else '#B0E0E6')
        else:
            colors_hydro.append('white')
    
    ax.bar(x, hydropathy_values, color=colors_hydro, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Hydropathy\n(Kyte-Doolittle)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Residue Position (Reference Numbering)', fontsize=12, fontweight='bold')
    ax.set_ylim(-5, 5)
    ax.grid(axis='y', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8B4513', label='Hydrophobic (+) [AF3]'),
        Patch(facecolor='#4169E1', label='Hydrophilic (−) [AF3]'),
        Patch(facecolor='#D2691E', label='Hydrophobic (+) [AF2]'),
        Patch(facecolor='#87CEEB', label='Hydrophilic (−) [AF2]'),
        Patch(facecolor='#F4A460', label='Hydrophobic (+) [Ref]'),
        Patch(facecolor='#B0E0E6', label='Hydrophilic (−) [Ref]')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=3)
    
    # Statistics
    ref_mean_hydro = result['reference']['hydropathy'][chain_type]['mean_hydropathy']
    ref_hydrophobic_frac = result['reference']['hydropathy'][chain_type]['hydrophobic_fraction']
    af3_mean_hydro = result['af3']['hydropathy'][chain_type]['mean_hydropathy']
    af3_hydrophobic_frac = result['af3']['hydropathy'][chain_type]['hydrophobic_fraction']
    
    stats_text = f'Ref: Mean={ref_mean_hydro:.2f} | Hydrophobic={ref_hydrophobic_frac:.1%}\n'
    stats_text += f'AF3: Mean={af3_mean_hydro:.2f} | Hydrophobic={af3_hydrophobic_frac:.1%}'
    
    ax.text(0.02, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================
    # X-axis labels (Reference positions with source markers)
    # ========================================
    n_ticks = min(20, len(all_ref_positions))
    tick_step = max(1, len(all_ref_positions) // n_ticks)
    tick_indices = list(range(0, len(all_ref_positions), tick_step))
    ax.set_xticks([x[i] for i in tick_indices])
    
    labels = []
    for i in tick_indices:
        ref_pos = all_ref_positions[i]
        res = residues[i]
        
        # Check presence in each structure
        in_ref = ref_pos in ref_positions_set
        in_af2 = ref_pos in af2_positions_in_ref_coords
        in_af3 = ref_pos in af3_positions_in_ref_coords
        
        # Create marker
        if in_ref and in_af2 and in_af3:
            marker = ''  # All three
        elif in_af2 and in_af3:
            marker = '²³'
        elif in_ref and in_af3:
            marker = 'ʳ³'
        elif in_ref and in_af2:
            marker = 'ʳ²'
        elif in_ref:
            marker = 'ʳ'
        elif in_af2:
            marker = '²'
        elif in_af3:
            marker = '³'
        else:
            marker = '?'
        
        labels.append(f'{res}{ref_pos}{marker}')
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Add legend for markers
    marker_text = 'Markers: ʳ=Ref only, ²=AF2 only, ³=AF3 only, ʳ²=Ref+AF2, ʳ³=Ref+AF3, ²³=AF2+AF3, (none)=All three'
    fig.text(0.5, 0.001, marker_text, ha='center', fontsize=8, style='italic', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    labels = ['A', 'B', 'C']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        # Add letter label in top-left corner
        ax.text(-0.02, 1.02, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    return fig

def plot_contact_stability_analysis(contact_results_csv, output_dir="interface_results"):
    """
    Plot 1: Scatter plot showing contact accuracy vs disorder contact fraction
    Side-by-side panels for AF2 and AF3, color-coded by over-contacted status
    
    Args:
        contact_results_csv: Path to contact_stability_analysis.csv
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    
    # Load data
    df = pd.read_csv(contact_results_csv)
    
    print(f"\nGenerating contact stability analysis plot...")
    print(f"  Loaded {len(df)} proteins from {contact_results_csv}")
    
    # Create figure with 2 side-by-side panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for over-contacted status
    color_map = {True: '#E74C3C', False: '#3498DB'}  # Red=True, Blue=False
    
    # ========================================
    # Panel 1: AF2 Performance
    # ========================================
    ax1 = axes[0]
    
    # Separate by over-contacted status
    df_af2_normal = df[df['af2_over_contacted'] == False]
    df_af2_over = df[df['af2_over_contacted'] == True]
    
    # Plot normal cases
    ax1.scatter(df_af2_normal['disorder_contact_fraction_af2'] * 100,
               df_af2_normal['af2_contact_accuracy'],
               c=color_map[False], s=80, alpha=0.7, 
               edgecolors='black', linewidth=0.8,
               label=f'Normal contacts (n={len(df_af2_normal)})')
    
    # Plot over-contacted cases
    if len(df_af2_over) > 0:
        ax1.scatter(df_af2_over['disorder_contact_fraction_af2'] * 100,
                   df_af2_over['af2_contact_accuracy'],
                   c=color_map[True], s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.8, marker='^',
                   label=f'Over-contacted (n={len(df_af2_over)})')
    
    # Styling
    ax1.set_xlabel('Disorder Contact Fraction (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Contact Accuracy (Recall)', fontsize=12, fontweight='bold')
    ax1.set_title('AF2 Multimer: Contact Prediction Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add reference lines
    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='50% accuracy')
    ax1.axvline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='50% disorder')
    
    # Add statistics box
    af2_mean_accuracy = df['af2_contact_accuracy'].mean()
    af2_mean_disorder = df['disorder_contact_fraction_af2'].mean() * 100
    af2_over_frac = (df['af2_over_contacted'].sum() / len(df)) * 100
    
    stats_text = f'Mean Accuracy: {af2_mean_accuracy:.2f}\n'
    stats_text += f'Mean Disorder %: {af2_mean_disorder:.1f}%\n'
    stats_text += f'Over-contacted: {af2_over_frac:.1f}%'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================
    # Panel 2: AF3 Performance
    # ========================================
    ax2 = axes[1]
    
    # Separate by over-contacted status
    df_af3_normal = df[df['af3_over_contacted'] == False]
    df_af3_over = df[df['af3_over_contacted'] == True]
    
    # Plot normal cases
    ax2.scatter(df_af3_normal['disorder_contact_fraction_af3'] * 100,
               df_af3_normal['af3_contact_accuracy'],
               c=color_map[False], s=80, alpha=0.7,
               edgecolors='black', linewidth=0.8,
               label=f'Normal contacts (n={len(df_af3_normal)})')
    
    # Plot over-contacted cases
    if len(df_af3_over) > 0:
        ax2.scatter(df_af3_over['disorder_contact_fraction_af3'] * 100,
                   df_af3_over['af3_contact_accuracy'],
                   c=color_map[True], s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.8, marker='^',
                   label=f'Over-contacted (n={len(df_af3_over)})')
    
    # Styling
    ax2.set_xlabel('Disorder Contact Fraction (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Contact Accuracy (Recall)', fontsize=12, fontweight='bold')
    ax2.set_title('AF3: Contact Prediction Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add reference lines
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.axvline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Add statistics box
    af3_mean_accuracy = df['af3_contact_accuracy'].mean()
    af3_mean_disorder = df['disorder_contact_fraction_af3'].mean() * 100
    af3_over_frac = (df['af3_over_contacted'].sum() / len(df)) * 100
    
    stats_text = f'Mean Accuracy: {af3_mean_accuracy:.2f}\n'
    stats_text += f'Mean Disorder %: {af3_mean_disorder:.1f}%\n'
    stats_text += f'Over-contacted: {af3_over_frac:.1f}%'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes_list = [ax1, ax2]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    
    # Overall title
    
    plt.tight_layout()
    
    output_file = output_dir / "contact_stability_analysis.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"  ✓ Contact stability scatter plot saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n  Summary Statistics:")
    print(f"    AF2: Accuracy={af2_mean_accuracy:.2f}, Disorder={af2_mean_disorder:.1f}%, Over-contacted={af2_over_frac:.1f}%")
    print(f"    AF3: Accuracy={af3_mean_accuracy:.2f}, Disorder={af3_mean_disorder:.1f}%, Over-contacted={af3_over_frac:.1f}%")
    
    plt.close()


def plot_conformational_diversity_heatmap(diversity_results_csv, output_dir="interface_results"):
    """
    Plot 2: Heatmap table showing conformational diversity metrics
    Rows: PDB IDs (6 selected AF3 failure cases)
    Columns: low_plddt_rmsd, high_plddt_rmsd, rmsd_ratio, correlation, diversity_behavior
    
    Args:
        diversity_results_csv: Path to conformational_diversity_analysis.csv
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    
    # Load data
    df = pd.read_csv(diversity_results_csv)
    
    print(f"\nGenerating conformational diversity heatmap...")
    print(f"  Loaded {len(df)} proteins from {diversity_results_csv}")
    
    # Select 6 examples (prefer variety in diversity_behavior)
    if len(df) > 6:
        # Try to get 2 of each type if possible
        collapsed = df[df['af3_samples_diversity'] == 'collapsed']
        intermediate = df[df['af3_samples_diversity'] == 'intermediate']
        ensemble = df[df['af3_samples_diversity'] == 'ensemble-like']
        
        selected_indices = []
        for group in [collapsed, intermediate, ensemble]:
            n_select = min(2, len(group))
            selected_indices.extend(group.head(n_select).index.tolist())
        
        # Fill up to 6 if needed
        if len(selected_indices) < 6:
            remaining = df.drop(selected_indices).head(6 - len(selected_indices))
            selected_indices.extend(remaining.index.tolist())
        
        df_selected = df.loc[selected_indices[:6]]
    else:
        df_selected = df
    
    print(f"  Selected {len(df_selected)} cases for visualization")
    
    # Prepare data for heatmap
    pdb_ids = df_selected['pdb_id'].values
    
    # Calculate RMSD ratio
    rmsd_ratio = df_selected['low_plddt_rmsd'] / df_selected['high_plddt_rmsd']
    rmsd_ratio = rmsd_ratio.fillna(0)  # Handle division by zero
    
    # Create data matrix (excluding categorical column)
    heatmap_data = pd.DataFrame({
        'Low-pLDDT RMSD (Å)': df_selected['low_plddt_rmsd'].values,
        'High-pLDDT RMSD (Å)': df_selected['high_plddt_rmsd'].values,
        'RMSD Ratio\n(Low/High)': rmsd_ratio.values,
        'Disorder-RMSD\nCorrelation': df_selected['disorder_variability_correlation'].values,
    }, index=pdb_ids)
    
    # Create figure
    fig, (ax_heatmap, ax_behavior) = plt.subplots(1, 2, figsize=(14, len(df_selected) * 0.8 + 2),
                                                   gridspec_kw={'width_ratios': [4, 1]})
    
    # ========================================
    # Left Panel: Numeric Heatmap
    # ========================================
    
    # Normalize data for coloring (higher values = more concerning)
    # For correlation: negative correlation is worse (red), positive is better (green)
    heatmap_normalized = heatmap_data.copy()
    
    # Invert correlation so negative = high value (red)
    heatmap_normalized['Disorder-RMSD\nCorrelation'] = -heatmap_normalized['Disorder-RMSD\nCorrelation']
    
    # Create custom colormap (white -> yellow -> red for concerning values)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    
    # Plot heatmap with values
    im = ax_heatmap.imshow(heatmap_normalized.values, cmap=cmap, aspect='auto', vmin=0)
    
    # Set ticks and labels
    ax_heatmap.set_xticks(np.arange(len(heatmap_data.columns)))
    ax_heatmap.set_yticks(np.arange(len(heatmap_data)))
    ax_heatmap.set_xticklabels(heatmap_data.columns, fontsize=10, fontweight='bold')
    ax_heatmap.set_yticklabels(pdb_ids, fontsize=11, fontweight='bold')
    
    # Rotate x-axis labels
    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values as text
    for i in range(len(heatmap_data)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            
            # Format based on column
            if j < 2:  # RMSD values
                text = f'{value:.2f}'
            elif j == 2:  # Ratio
                text = f'{value:.2f}x'
            else:  # Correlation
                text = f'{value:.2f}'
            
            # Choose text color based on background
            bg_value = heatmap_normalized.iloc[i, j]
            text_color = 'white' if bg_value > heatmap_normalized.values.max() * 0.6 else 'black'
            
            ax_heatmap.text(j, i, text, ha="center", va="center",
                          color=text_color, fontsize=10, fontweight='bold')
    
    ax_heatmap.set_title('Conformational Diversity Metrics', fontsize=13, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Concern Level\n(Red=High, Green=Low)', fontsize=9, fontweight='bold')
    
    # ========================================
    # Right Panel: Diversity Behavior
    # ========================================
    
    # Get diversity behaviors
    behaviors = df_selected['af3_samples_diversity'].values
    
    # Color map for behaviors
    behavior_colors = {
        'collapsed': '#3498DB',      # Blue (good)
        'intermediate': '#F39C12',   # Orange (warning)
        'ensemble-like': '#E74C3C'   # Red (concerning)
    }
    
    # Convert behaviors to numeric values for plotting
    behavior_to_num = {
        'collapsed': 0,
        'intermediate': 1,
        'ensemble-like': 2
    }
    
    # Create numeric matrix for imshow
    behavior_numeric = np.array([[behavior_to_num.get(b, 0)] for b in behaviors])
    
    # Create custom colormap for behaviors
    from matplotlib.colors import ListedColormap
    behavior_cmap = ListedColormap([behavior_colors['collapsed'], 
                                   behavior_colors['intermediate'], 
                                   behavior_colors['ensemble-like']])
    
    # Plot as image with numeric values
    im_behavior = ax_behavior.imshow(behavior_numeric, aspect='auto', cmap=behavior_cmap, 
                                     vmin=0, vmax=2)
    
    # Remove x ticks, keep y ticks
    ax_behavior.set_xticks([])
    ax_behavior.set_yticks(np.arange(len(behaviors)))
    ax_behavior.set_yticklabels(behaviors, fontsize=10, fontweight='bold')
    
    ax_behavior.set_title('Diversity\nBehavior', fontsize=11, fontweight='bold', pad=10)
    
    # Add legend for behaviors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=behavior_colors['collapsed'], label='Collapsed'),
        Patch(facecolor=behavior_colors['intermediate'], label='Intermediate'),
        Patch(facecolor=behavior_colors['ensemble-like'], label='Ensemble-like')
    ]
    ax_behavior.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=1)
    
    # Overall title
    fig.suptitle('AF3 Conformational Diversity Analysis\n(Selected Failure Cases)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_file = output_dir / "conformational_diversity_heatmap.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"  ✓ Conformational diversity heatmap saved to: {output_file}")
    
    # Print summary
    print(f"\n  Diversity Behavior Summary:")
    print(f"    Collapsed: {(behaviors == 'collapsed').sum()}")
    print(f"    Intermediate: {(behaviors == 'intermediate').sum()}")
    print(f"    Ensemble-like: {(behaviors == 'ensemble-like').sum()}")
    
    plt.close()

def plot_iptm_vs_dockq(results_list, output_dir="interface_results"):
    """
    Create comprehensive ipTM vs DockQ analysis plot for AF2 and AF3
    
    Layout (2x2):
    - Top Left: AF2 ipTM vs DockQ
    - Top Right: AF3 ipTM vs DockQ
    - Bottom Left: Combined scatter plot
    - Bottom Right: ipTM vs DockQ Delta analysis
    """
    output_dir = Path(output_dir)
    
    # Extract data
    af2_iptm = []
    af2_dockq = []
    af3_iptm = []
    af3_dockq = []
    protein_ids = []
    
    for result in results_list:
        protein_id = result['protein_id']
        
        # AF2 data
        if result['af2'].get('iptm') is not None and result.get('af2_dockq') is not None:
            af2_iptm.append(result['af2']['iptm'])
            af2_dockq.append(result['af2_dockq'])
        
        # AF3 data
        if result['af3'].get('iptm') is not None and result.get('af3_dockq') is not None:
            af3_iptm.append(result['af3']['iptm'])
            af3_dockq.append(result['af3_dockq'])
            protein_ids.append(protein_id)
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    af2_color = 'steelblue'
    af3_color = 'indianred'
    
    # ========================================
    # Plot 1: AF2 ipTM vs DockQ (Top Left)
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    if af2_iptm and af2_dockq:
        scatter = ax1.scatter(af2_iptm, af2_dockq, alpha=0.7, s=80, 
                             c=af2_iptm, cmap='Blues', edgecolors='black', 
                             linewidth=0.5, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('ipTM Score', fontsize=9)
        
        # DockQ quality thresholds
        ax1.axhline(y=0.23, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax1.axhline(y=0.49, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax1.axhline(y=0.80, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax1.text(0.02, 0.23, 'Acceptable', fontsize=8, va='bottom', color='gray')
        ax1.text(0.02, 0.49, 'Medium', fontsize=8, va='bottom', color='gray')
        ax1.text(0.02, 0.80, 'High', fontsize=8, va='bottom', color='gray')
        
        # Calculate correlation
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(af2_iptm, af2_dockq)
        spearman_r, spearman_p = spearmanr(af2_iptm, af2_dockq)
        
        # Add correlation text
        ax1.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('ipTM Score', fontsize=11, fontweight='bold')
        ax1.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
        ax1.set_title('AF2 Multimer: ipTM vs DockQ', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        
        # Add sample count
        ax1.text(0.95, 0.05, f'n = {len(af2_iptm)}', transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================
    # Plot 2: AF3 ipTM vs DockQ (Top Right)
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    if af3_iptm and af3_dockq:
        scatter = ax2.scatter(af3_iptm, af3_dockq, alpha=0.7, s=80, 
                             c=af3_iptm, cmap='Reds', edgecolors='black', 
                             linewidth=0.5, marker='^', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('ipTM Score', fontsize=9)
        
        # DockQ quality thresholds
        ax2.axhline(y=0.23, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax2.axhline(y=0.49, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax2.axhline(y=0.80, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax2.text(0.02, 0.23, 'Acceptable', fontsize=8, va='bottom', color='gray')
        ax2.text(0.02, 0.49, 'Medium', fontsize=8, va='bottom', color='gray')
        ax2.text(0.02, 0.80, 'High', fontsize=8, va='bottom', color='gray')
        
        # Calculate correlation
        pearson_r, pearson_p = pearsonr(af3_iptm, af3_dockq)
        spearman_r, spearman_p = spearmanr(af3_iptm, af3_dockq)
        
        # Add correlation text
        ax2.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('ipTM Score', fontsize=11, fontweight='bold')
        ax2.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
        ax2.set_title('AF3: ipTM vs DockQ', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        
        # Add sample count
        ax2.text(0.95, 0.05, f'n = {len(af3_iptm)}', transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================
    # Plot 3: Combined Scatter (Bottom Left)
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    if af2_iptm and af3_iptm:
        ax3.scatter(af2_iptm, af2_dockq, label='AF2 Multimer', alpha=0.6, 
                   s=70, color=af2_color, edgecolors='black', linewidth=0.5)
        ax3.scatter(af3_iptm, af3_dockq, label='AF3', alpha=0.6, s=70, 
                   color=af3_color, marker='^', edgecolors='black', linewidth=0.5)
        
        # DockQ quality thresholds
        ax3.axhline(y=0.23, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax3.axhline(y=0.49, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax3.axhline(y=0.80, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        
        ax3.set_xlabel('ipTM Score', fontsize=11, fontweight='bold')
        ax3.set_ylabel('DockQ Score', fontsize=11, fontweight='bold')
        ax3.set_title('Combined: ipTM vs DockQ', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
    
    # ========================================
    # Plot 4: ipTM vs DockQ Delta (Bottom Right)
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    if af2_iptm and af3_iptm and af2_dockq and af3_dockq:
        # Calculate differences for matching proteins
        min_len = min(len(af2_iptm), len(af3_iptm), len(af2_dockq), len(af3_dockq))
        iptm_diff = [af3_iptm[i] - af2_iptm[i] for i in range(min_len)]
        dockq_diff = [af3_dockq[i] - af2_dockq[i] for i in range(min_len)]
        
        scatter = ax4.scatter(iptm_diff, dockq_diff, alpha=0.7, s=70, 
                             c=dockq_diff, cmap='RdYlGn', edgecolors='black', 
                             linewidth=0.5, vmin=-0.5, vmax=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('DockQ Δ', fontsize=9)
        
        # Add reference lines
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Calculate correlation
        if len(iptm_diff) > 2:
            pearson_r, _ = pearsonr(iptm_diff, dockq_diff)
            ax4.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}',
                    transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('ipTM Difference (AF3 - AF2)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('DockQ Difference (AF3 - AF2)', fontsize=11, fontweight='bold')
        ax4.set_title('Performance Improvement Analysis', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Count quadrants
        q1 = sum(1 for i, d in zip(iptm_diff, dockq_diff) if i > 0 and d > 0)  # Both improved
        q2 = sum(1 for i, d in zip(iptm_diff, dockq_diff) if i < 0 and d > 0)  # DockQ improved
        q3 = sum(1 for i, d in zip(iptm_diff, dockq_diff) if i < 0 and d < 0)  # Both worse
        q4 = sum(1 for i, d in zip(iptm_diff, dockq_diff) if i > 0 and d < 0)  # ipTM improved
        
        ax4.text(0.95, 0.05, f'Q1: {q1}\nQ2: {q2}\nQ3: {q3}\nQ4: {q4}',
                transform=ax4.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add panel labels
    axes_list = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B, C, D
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    
    # Main title
    fig.suptitle('Interface Predicted TM-score (ipTM) vs DockQ Analysis: AF2 Multimer vs AF3',
                 fontsize=15, fontweight='bold', y=0.95)
    
    # Save figure
    plt.savefig(output_dir / "iptm_vs_dockq_analysis.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print(f"\nipTM vs DockQ analysis plot saved to: {output_dir}/iptm_vs_dockq_analysis.pdf")
    print(f"  AF2 points: {len(af2_iptm)} (ipTM + DockQ available)")
    print(f"  AF3 points: {len(af3_iptm)} (ipTM + DockQ available)")
    
    # Print statistics
    if af2_iptm and af3_iptm:
        print(f"\nStatistics:")
        print(f"  AF2 ipTM: mean={np.mean(af2_iptm):.3f}, median={np.median(af2_iptm):.3f}, std={np.std(af2_iptm):.3f}")
        print(f"  AF3 ipTM: mean={np.mean(af3_iptm):.3f}, median={np.median(af3_iptm):.3f}, std={np.std(af3_iptm):.3f}")
        print(f"  AF2 DockQ: mean={np.mean(af2_dockq):.3f}, median={np.median(af2_dockq):.3f}, std={np.std(af2_dockq):.3f}")
        print(f"  AF3 DockQ: mean={np.mean(af3_dockq):.3f}, median={np.median(af3_dockq):.3f}, std={np.std(af3_dockq):.3f}")
    
    plt.close()

def plot_all_analysis(results_list, output_dir="interface_results"):
    """
    Create all comprehensive visualization plots
    """
    print("\nGenerating visualization plots...")
    
    output_dir = Path(output_dir)
    
    try:
        # Disorder analysis plot
        if 'plot_comprehensive_results' in globals():
            print("  - Creating disorder analysis plot...")
            plot_comprehensive_results(results_list, output_plot=str(output_dir / "interface_comparison.pdf"))
        else:
            print("  ⚠ plot_comprehensive_results() not defined - skipping disorder plot")
    except Exception as e:
        print(f"  ✗ Error creating disorder plot: {str(e)}")
    
    try:
        # Dataset analysis plot
        if 'plot_dataset_characteristics' in globals():
            print("  - Creating dataset analysis plot...")
            plot_dataset_characteristics(results_list, output_plot=str(output_dir / "b90_dataset_characteristics.pdf"))
        else:
            print("  ⚠ plot_dataset_characteristics() not defined - skipping dataset plot")
    except Exception as e:
        print(f"  ✗ Error creating dataset analysis plot: {str(e)}")

    try:
        # PAE analysis plot
        if 'plot_pae_analysis' in globals():
            print("  - Creating PAE analysis plot...")
            plot_pae_analysis(results_list, output_dir)
        else:
            print("  ⚠ plot_pae_analysis() not defined - skipping PAE plot")
    except Exception as e:
        print(f"  ✗ Error creating PAE plot: {str(e)}")
    
    try:
        # DSSP analysis plot
        if 'plot_dssp_analysis' in globals():
            print("  - Creating DSSP analysis plot...")
            plot_dssp_analysis(results_list, output_dir)
        else:
            print("  ⚠ plot_dssp_analysis() not defined - skipping DSSP plot")
    except Exception as e:
        print(f"  ✗ Error creating DSSP plot: {str(e)}")

    try:
        # Determinants analysis plot
        if 'plot_determinants_figure' in globals():
            print("  - Creating determinants analysis plot...")
            plot_determinants_figure(results_list, output_dir)
        else:
            print("  ⚠ plot_determinants_figure() not defined - skipping determinants plot")
    except Exception as e:
        print(f"  ✗ Error creating determinants plot: {str(e)}")

    try:
        # Residue-level analysis plots (now includes ipTM histogram)
        if 'plot_residue_level_analysis' in globals():
            print("  - Creating residue-level analysis plots and ipTM histogram...")
            plot_residue_level_analysis(results_list, output_dir, top_n=6)
        else:
            print("  ⚠ plot_residue_level_analysis() not defined - skipping residue plots")
    except Exception as e:
        print(f"  ✗ Error creating residue-level plots: {str(e)}")

    try:
        plot_contact_stability_analysis("interface_results/contact_stability_analysis.csv", output_dir)
    except Exception as e:
        print(f"  ✗ Error creating contact stability plot: {str(e)}")
    
    try:
        plot_conformational_diversity_heatmap("interface_results/conformational_diversity_analysis.csv", output_dir)
    except Exception as e:
        print(f"  ✗ Error creating conformational diversity plot: {str(e)}")
    print("\nVisualization plots generation complete!")

    try:
        plot_iptm_vs_dockq(results_list, output_dir)
    except Exception as e:
        print(f"  ✗ Error creating iptm vs dockq plot: {str(e)}")

    try:
        analyze_disorder_dssp_regression_combined(results_list, output_dir)
        plot_disorder_dssp_regression(json_file="interface_results/regression_analysis.json", output_dir="interface_results")
    except Exception as e:
        print(f"  ✗ Error analyzing disorder dssp regression: {str(e)}")

    try:
        analyze_disorder_dssp_regression_combined(results_list, output_dir, selected="_selected", top_n=10)
        plot_disorder_dssp_regression(json_file="interface_results/regression_analysis_selected.json", output_dir="interface_results", selected="_selected")
    except Exception as e:
        print(f"  ✗ Error analyzing disorder dssp regression: {str(e)}")

    try:
        results = analyze_af3_extended_regression(results_list, "interface_results")
        plot_af3_extended_regression(results, "interface_results")
    except Exception as e:
        print(f"  ✗ Error analyzing disorder dssp regression: {str(e)}")

    print("\nVisualization plots generation complete!")

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    REF_DIR = "dataset/naive_files"
    AF2_DIR = "dataset/alphafold2_files"
    AF3_DIR = "dataset/alphafold3_files"
    AF2_CSV_FILE = "dataset/af2v3_dockq_data.csv"
    AF3_CSV_FILE = "dataset/af3_dockq_data.csv"
    AF2_PAE_DIR = "dataset/alphafold2_scores"
    AF3_PAE_DIR = "dataset/alphafold3_full_scores"
    AF3_IPTM_DIR = "dataset/alphafold3_scores"
    OUTPUT_DIR = "interface_results"
    # =======================================================
    
    print("\n" + "="*60)
    print("UNIFIED INTERFACE ANALYSIS FOR ALPHAFOLD PREDICTIONS")
    print("="*60)
    print("Analysis includes:")
    print("  ✓ Interface residue extraction")
    print("  ✓ Disorder analysis (pLDDT)")
    print("  ✓ PAE (Predicted Aligned Error)")
    print("  ✓ DSSP secondary structure")
    print("  ✓ Hydropathy scores")
    print("  ✓ DockQ quality assessment")
    print("="*60 + "\n")
    
    # Check DSSP availability
    if not DSSP_AVAILABLE:
        print("⚠ WARNING: DSSP is not available!")
        print("DSSP secondary structure analysis will be skipped.")
        print("\nTo install DSSP:")
        print("  - Conda:        conda install -c salilab dssp")
        print("  - Ubuntu/Debian: sudo apt-get install dssp")
        print("  - macOS:        brew install brewsci/bio/dssp")
        print()
    
    # Run batch analysis
    print("Starting batch analysis...\n")
    result_path = os.path.join(OUTPUT_DIR, "b90_summary.json")
    if os.path.isfile(result_path):
        results_list = json.load(open(result_path))['results']
    else:
        results_list = batch_analyze(
            ref_dir=REF_DIR,
            af2_dir=AF2_DIR,
            af3_dir=AF3_DIR,
            af2_csv_file=AF2_CSV_FILE,
            af3_csv_file=AF3_CSV_FILE,
            af2_pae_dir=AF2_PAE_DIR,
            af3_pae_dir=AF3_PAE_DIR,
            af3_iptm_dir=AF3_IPTM_DIR,
            output_dir=OUTPUT_DIR
        )
    # Process results
    if results_list:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Successfully analyzed: {len(results_list)} proteins")
        print(f"\nOutput directory: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        print("  📄 Individual JSON: {pdb_id}.json")
        print("  📄 Summary JSON:    b90_summary.json")
        print("  📊 CSV table:       b90_summary.csv")
        
        # Create visualization plots
        print("\n" + "-"*60)
        plot_all_analysis(results_list, OUTPUT_DIR)
        print("-"*60)
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Check the CSV file for all metrics:")
        print(f"   {OUTPUT_DIR}/b90_summary.csv")
        print("\n2. Use results_list for custom plotting:")
        print("   results_list = json.load(open('interface_results/b90_summary.json'))['results']")
        print("\n3. Available plotting functions (define separately):")
        print("   - plot_comprehensive_results(results_list, output_plot)")
        print("   - plot_pae_analysis(results_list, output_dir)")
        print("   - plot_dssp_analysis(results_list, output_dir)")
        print("="*60 + "\n")
        
    else:
        print("\n" + "="*60)
        print("NO SUCCESSFUL ANALYSES")
        print("="*60)
        print("Please check:")
        print("  1. Input directory paths exist")
        print("  2. CSV files contain valid data")
        print("  3. Structure files are accessible")
        print("  4. PAE JSON files are present")
        print("="*60 + "\n")