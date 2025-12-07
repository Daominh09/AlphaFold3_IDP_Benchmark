import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Bio.PDB import PDBParser, MMCIFParser


class MoRFExtractor:
    
    # Mapping from one-letter to three-letter amino acid codes
    AA_CODE_MAP = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        'U': 'SEC', 'O': 'PYL', 'B': 'ASX', 'Z': 'GLX', 'X': 'XAA'
    }
    
    def __init__(self, folder_path, mcw_threshold=0.72):
        self.folder_path = folder_path
        self.mcw_threshold = mcw_threshold
        self.morf_results = {}
    
    def _convert_to_three_letter(self, one_letter_code):
        """Convert one-letter amino acid code to three-letter code."""
        return self.AA_CODE_MAP.get(one_letter_code.upper(), one_letter_code)
        
    def extract_morf_regions(self):
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {self.folder_path}")
            return {}
        
        self.morf_results = {}
        
        for file_path in txt_files:
            protein_name = os.path.basename(file_path).replace('.txt', '')[1:]
            
            try:
                protein_data = self._process_file(file_path)
                if protein_data and protein_data['morf_regions']:
                    self.morf_results[protein_name] = protein_data
            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return self.morf_results
    
    def _process_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        protein_id = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('>'):
                protein_id = line.strip()[1:]
                data_start_idx = i + 1
                break
        
        residue_data = []
        for line in lines[data_start_idx:]:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    residue_idx = int(parts[0])
                    residue = parts[1]
                    mcw_score = float(parts[2])
                    residue_data.append({
                        'residue_id': residue_idx,
                        'residue': self._convert_to_three_letter(residue),
                        'MCW': mcw_score
                    })
        
        morf_regions = [r for r in residue_data if r['MCW'] > self.mcw_threshold]
        
        return {
            'protein_id': protein_id if protein_id else os.path.basename(file_path),
            'morf_regions': morf_regions,
            'total_residues': len(residue_data),
            'morf_residue_count': len(morf_regions)
        }
    
    def get_morf_regions(self, protein_name):
        if protein_name in self.morf_results:
            return self.morf_results[protein_name]['morf_regions']
        return []
    
    def get_all_proteins(self):
        return list(self.morf_results.keys())
    
    def get_protein_info(self, protein_name):
        return self.morf_results.get(protein_name, None)
    
    def get_summary_stats(self):
        if not self.morf_results:
            return {}
        
        total_proteins = len(self.morf_results)
        total_morf_residues = sum(
            data['morf_residue_count'] for data in self.morf_results.values()
        )
        avg_morf_residues = total_morf_residues / total_proteins if total_proteins > 0 else 0
        
        return {
            'total_proteins': total_proteins,
            'total_morf_residues': total_morf_residues,
            'avg_morf_residues_per_protein': avg_morf_residues,
            'mcw_threshold': self.mcw_threshold
        }

class PLDDTExtractor:
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.plddt_results = {}
    
    def extract_plddt_scores(self):
        pdb_files = glob.glob(os.path.join(self.folder_path, "*.pdb"))
        cif_files = glob.glob(os.path.join(self.folder_path, "*.cif"))
        
        all_files = pdb_files + cif_files
        
        if not all_files:
            print(f"No .pdb or .cif files found in {self.folder_path}")
            return {}
        
        self.plddt_results = {}
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            file_name_no_ext = file_name.replace('.pdb', '').replace('.cif', '')
            
            if not file_name_no_ext.endswith('_A'):
                continue
            
            protein_name = file_name_no_ext.replace('_A', '')
            
            try:
                if file_path.endswith('.pdb'):
                    protein_data = self._process_pdb_file(file_path)
                else:
                    protein_data = self._process_cif_file(file_path)
                
                if protein_data and protein_data['ca_atoms']:
                    self.plddt_results[protein_name] = protein_data
            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return self.plddt_results
    
    def _process_pdb_file(self, file_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        ca_atoms.append({
                            'residue_id': residue.id[1],
                            'residue': residue.get_resname(),
                            'pLDDT': ca_atom.get_bfactor()
                        })
        
        return {
            'ca_atoms': ca_atoms,
            'total_residues': len(ca_atoms)
        }
    
    def _process_cif_file(self, file_path):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        ca_atoms.append({
                            'residue_id': residue.id[1],
                            'residue': residue.get_resname(),
                            'pLDDT': ca_atom.get_bfactor()
                        })
        
        return {
            'ca_atoms': ca_atoms,
            'total_residues': len(ca_atoms)
        }
    
    def get_plddt_scores(self, protein_name):
        if protein_name in self.plddt_results:
            return self.plddt_results[protein_name]['ca_atoms']
        return []
    
    def get_all_proteins(self):
        return list(self.plddt_results.keys())
    
    def get_protein_info(self, protein_name):
        return self.plddt_results.get(protein_name, None)
    
    def get_summary_stats(self):
        if not self.plddt_results:
            return {}
        
        total_proteins = len(self.plddt_results)
        total_residues = sum(
            data['total_residues'] for data in self.plddt_results.values()
        )
        avg_residues = total_residues / total_proteins if total_proteins > 0 else 0
        
        all_scores = []
        for data in self.plddt_results.values():
            all_scores.extend([atom['pLDDT'] for atom in data['ca_atoms']])
        
        avg_plddt = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return {
            'total_proteins': total_proteins,
            'total_residues': total_residues,
            'avg_residues_per_protein': avg_residues,
            'avg_plddt_score': avg_plddt
        }


class PLDDTMultimerExtractor:
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.plddt_results = {}
    
    def extract_plddt_scores(self):
        pdb_files = glob.glob(os.path.join(self.folder_path, "*.pdb"))
        cif_files = glob.glob(os.path.join(self.folder_path, "*.cif"))
        
        all_files = pdb_files + cif_files
        
        if not all_files:
            print(f"No .pdb or .cif files found in {self.folder_path}")
            return {}
        
        self.plddt_results = {}
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            protein_name = file_name.replace('.pdb', '').replace('.cif', '')
            
            try:
                if file_path.endswith('.pdb'):
                    protein_data = self._process_pdb_file(file_path)
                else:
                    protein_data = self._process_cif_file(file_path)
                
                if protein_data:
                    self.plddt_results[protein_name] = protein_data
            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return self.plddt_results
    
    def _process_pdb_file(self, file_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        
        chains_data = {}
        for model in structure:
            for chain in model:
                chain_id = chain.id
                ca_atoms = []
                
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        ca_atoms.append({
                            'residue_id': residue.id[1],
                            'residue': residue.get_resname(),
                            'pLDDT': ca_atom.get_bfactor()
                        })
                
                if ca_atoms:
                    chains_data[chain_id] = {
                        'ca_atoms': ca_atoms,
                        'total_residues': len(ca_atoms)
                    }
        
        return chains_data
    
    def _process_cif_file(self, file_path):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        
        chains_data = {}
        for model in structure:
            for chain in model:
                chain_id = chain.id
                ca_atoms = []
                
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        ca_atoms.append({
                            'residue_id': residue.id[1],
                            'residue': residue.get_resname(),
                            'pLDDT': ca_atom.get_bfactor()
                        })
                
                if ca_atoms:
                    chains_data[chain_id] = {
                        'ca_atoms': ca_atoms,
                        'total_residues': len(ca_atoms)
                    }
        
        return chains_data
    
    def get_protein_chains(self, protein_name):
        if protein_name in self.plddt_results:
            return list(self.plddt_results[protein_name].keys())
        return []
    
    def get_chain_data(self, protein_name, chain_id):
        if protein_name in self.plddt_results:
            if chain_id in self.plddt_results[protein_name]:
                return self.plddt_results[protein_name][chain_id]
        return None
    
    def get_all_proteins(self):
        return list(self.plddt_results.keys())
    
    def get_protein_info(self, protein_name):
        return self.plddt_results.get(protein_name, None)
    
    def get_summary_stats(self):
        if not self.plddt_results:
            return {}
        
        total_proteins = len(self.plddt_results)
        total_chains = sum(len(chains) for chains in self.plddt_results.values())
        total_residues = 0
        all_scores = []
        
        for protein_data in self.plddt_results.values():
            for chain_data in protein_data.values():
                total_residues += chain_data['total_residues']
                all_scores.extend([atom['pLDDT'] for atom in chain_data['ca_atoms']])
        
        avg_chains_per_protein = total_chains / total_proteins if total_proteins > 0 else 0
        avg_residues_per_chain = total_residues / total_chains if total_chains > 0 else 0
        avg_plddt = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return {
            'total_proteins': total_proteins,
            'total_chains': total_chains,
            'total_residues': total_residues,
            'avg_chains_per_protein': avg_chains_per_protein,
            'avg_residues_per_chain': avg_residues_per_chain,
            'avg_plddt_score': avg_plddt
        }


from scipy.stats import mannwhitneyu
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class MoRFPLDDTAnalyzer:
    
    def __init__(self, morf_folder, mono_af2_folder, mono_af3_folder, 
                 multi_af2_folder, multi_af3_folder, mcw_threshold=0.72):
        self.morf_extractor = MoRFExtractor(morf_folder, mcw_threshold)
        self.mono_af2 = PLDDTExtractor(mono_af2_folder)
        self.mono_af3 = PLDDTExtractor(mono_af3_folder)
        self.multi_af2 = PLDDTMultimerExtractor(multi_af2_folder)
        self.multi_af3 = PLDDTMultimerExtractor(multi_af3_folder)
        
        self.morf_results = {}
        self.mono_af2_results = {}
        self.mono_af3_results = {}
        self.multi_af2_results = {}
        self.multi_af3_results = {}
        
        self.delta_plddt_af2 = {}
        self.delta_plddt_af3 = {}
        
    def run_full_analysis(self):
        print("Extracting MoRF regions...")
        self.morf_results = self.morf_extractor.extract_morf_regions()
        
        print("Extracting AF2 monomer pLDDT scores...")
        self.mono_af2_results = self.mono_af2.extract_plddt_scores()
        
        print("Extracting AF3 monomer pLDDT scores...")
        self.mono_af3_results = self.mono_af3.extract_plddt_scores()

        print("Extracting AF2 multimer pLDDT scores...")
        self.multi_af2_results = self.multi_af2.extract_plddt_scores()
        print("Extracting AF3 multimer pLDDT scores...")
        self.multi_af3_results = self.multi_af3.extract_plddt_scores()

        print("\nCalculating delta pLDDT scores...")
        self.delta_plddt_af2 = self._calculate_delta_plddt('AF2')
        self.delta_plddt_af3 = self._calculate_delta_plddt('AF3')
        
        print("\nGenerating plots...")
        self._plot_combined_analysis()
        
        print("\nAnalysis complete!")
        
    def _calculate_delta_plddt(self, version):
            if version == 'AF2':
                mono_results = self.mono_af2_results
                multi_results = self.multi_af2_results
            else:
                mono_results = self.mono_af3_results
                multi_results = self.multi_af3_results
            
            delta_results = {}
            
            for protein_name in self.morf_results.keys():
                if protein_name not in mono_results or protein_name not in multi_results:
                    print(f"[{version}] Skipping {protein_name}: not found in both mono and multi datasets")
                    continue
                
                mono_data = mono_results[protein_name]['ca_atoms']
                
                if 'A' not in multi_results[protein_name]:
                    print(f"[{version}] Skipping {protein_name}: chain A not found in multimer")
                    continue
                multi_data = multi_results[protein_name]['A']['ca_atoms']
                if len(mono_data) != len(multi_data):
                    print(f"[{version}] WARNING - {protein_name}: Length mismatch!")
                    print(f"  Monomer: {len(mono_data)} residues")
                    print(f"  Multimer chain A: {len(multi_data)} residues")
                    print(f"  Change to process chain B")
                    multi_data = multi_results[protein_name]['B']['ca_atoms']
                    if len(mono_data) != len(multi_data):
                        print(f"  Monomer: {len(mono_data)} residues")
                        print(f"  Multimer chain B: {len(multi_data)} residues")
                
                mismatch_found = False
                for i, (mono, multi) in enumerate(zip(mono_data, multi_data)):
                    if mono['residue_id'] != multi['residue_id']:
                        if not mismatch_found:
                            print(f"[{version}] RESIDUE ID MISMATCH in {protein_name}:")
                            mismatch_found = True
                        print(f"  Position {i}: Mono ID={mono['residue_id']} ({mono['residue']}) vs Multi ID={multi['residue_id']} ({multi['residue']})")
                    elif mono['residue'] != multi['residue']:
                        if not mismatch_found:
                            print(f"[{version}] RESIDUE TYPE MISMATCH in {protein_name}:")
                            mismatch_found = True
                        print(f"  Position {i}, ID={mono['residue_id']}: Mono={mono['residue']} vs Multi={multi['residue']}")
                
                mono_dict = {(atom['residue_id'], atom['residue']): atom['pLDDT'] for atom in mono_data}
                multi_dict = {(atom['residue_id'], atom['residue']): atom['pLDDT'] for atom in multi_data}
                
                morf_regions = self.morf_results[protein_name]['morf_regions']
                
                delta_morf_residues = []
                for morf_residue in morf_regions:
                    res_id = morf_residue['residue_id']
                    res = morf_residue['residue']
                    if (res_id, res) in mono_dict and (res_id, res) in multi_dict:
                        delta = multi_dict[(res_id, res)] - mono_dict[(res_id, res)]
                        delta_morf_residues.append({
                            'residue_id': res_id,
                            'residue': morf_residue['residue'],
                            'MCW': morf_residue['MCW'],
                            'mono_plddt': mono_dict[(res_id, res)],
                            'multi_plddt': multi_dict[(res_id, res)],
                            'delta_plddt': delta
                        })
                    else:
                        print(f"[{version}] WARNING - {protein_name}: MoRF residue {res_id} not found in {'monomer' if res_id not in mono_dict else 'multimer'}")
                    
                if delta_morf_residues:
                    avg_delta = np.mean([r['delta_plddt'] for r in delta_morf_residues])
                    delta_results[protein_name] = {
                        'residues': delta_morf_residues,
                        'avg_delta_plddt': avg_delta,
                        'num_morf_residues': len(delta_morf_residues)
                    }
                else:
                    print(f"[{version}] WARNING - {protein_name}: No valid MoRF residues found for delta calculation")
            
            return delta_results

    def _calculate_disorder_order_accuracy(self, version):
        """
        Calculate accuracy of disorder-to-order transitions in MoRF regions.
        AF2 threshold: 68 (disorder if pLDDT < 68)
        AF3 threshold: 70 (disorder if pLDDT < 70)
        Accuracy = (number of residues changing from disorder to order) / (total MoRF residues)
        """
        if version == 'AF2':
            mono_results = self.mono_af2_results
            multi_results = self.multi_af2_results
            threshold = 68
        else:  # AF3
            mono_results = self.mono_af3_results
            multi_results = self.multi_af3_results
            threshold = 70
        
        accuracy_results = {}
        
        for protein_name in self.morf_results.keys():
            if protein_name not in mono_results or protein_name not in multi_results:
                continue
            
            mono_data = mono_results[protein_name]['ca_atoms']
            
            # Try chain A first, then chain B
            if 'A' not in multi_results[protein_name]:
                continue
            multi_data = multi_results[protein_name]['A']['ca_atoms']
            
            if len(mono_data) != len(multi_data):
                if 'B' in multi_results[protein_name]:
                    multi_data = multi_results[protein_name]['B']['ca_atoms']
            
            # Create dictionaries for quick lookup
            mono_dict = {(atom['residue_id'], atom['residue']): atom['pLDDT'] for atom in mono_data}
            multi_dict = {(atom['residue_id'], atom['residue']): atom['pLDDT'] for atom in multi_data}
            
            morf_regions = self.morf_results[protein_name]['morf_regions']
            
            total_morf_residues = 0
            disorder_to_order_count = 0
            transition_details = []
            
            for morf_residue in morf_regions:
                res_id = morf_residue['residue_id']
                res = morf_residue['residue']
                
                if (res_id, res) in mono_dict and (res_id, res) in multi_dict:
                    mono_plddt = mono_dict[(res_id, res)]
                    multi_plddt = multi_dict[(res_id, res)]
                    
                    # Classify: 1 = disorder (pLDDT < threshold), 0 = order (pLDDT >= threshold)
                    mono_state = 1 if mono_plddt < threshold else 0
                    multi_state = 1 if multi_plddt < threshold else 0
                    
                    total_morf_residues += 1
                    
                    # Check if transition from disorder (1) to order (0)
                    if mono_state == 1 and multi_state == 0:
                        disorder_to_order_count += 1
                        
                    transition_details.append({
                        'residue_id': res_id,
                        'residue': res,
                        'mono_plddt': mono_plddt,
                        'multi_plddt': multi_plddt,
                        'mono_state': mono_state,
                        'multi_state': multi_state,
                        'transition': 'disorder_to_order' if (mono_state == 1 and multi_state == 0) else 'no_transition'
                    })
            
            if total_morf_residues > 0:
                accuracy = disorder_to_order_count / total_morf_residues
                accuracy_results[protein_name] = {
                    'total_morf_residues': total_morf_residues,
                    'disorder_to_order_count': disorder_to_order_count,
                    'accuracy': accuracy,
                    'transition_details': transition_details
                }
        
        return accuracy_results

    def _plot_combined_analysis(self):
        """
        Create a combined figure with all three analyses:
        Row 1: Delta pLDDT distributions (AF2 and AF3)
        Row 2: Disorder-to-order accuracy distributions (AF2 and AF3)
        Row 3: MCW correlation line graph (spans both columns)
        """
        # Create figure with 3 rows and 2 columns
        fig = plt.figure(figsize=(16, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
        
        # ========== ROW 1: Delta pLDDT Distributions ==========
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get common proteins for paired comparison
        common_proteins_delta = set(self.delta_plddt_af2.keys()) & set(self.delta_plddt_af3.keys())
        af2_avg_deltas = [self.delta_plddt_af2[p]['avg_delta_plddt'] for p in common_proteins_delta]
        af3_avg_deltas = [self.delta_plddt_af3[p]['avg_delta_plddt'] for p in common_proteins_delta]
        
        # Statistical test for Delta pLDDT
        stat_delta, p_delta = mannwhitneyu(af2_avg_deltas, af3_avg_deltas, alternative='two-sided')
        
        # AF2 histogram
        ax1.hist(af2_avg_deltas, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.mean(af2_avg_deltas), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(af2_avg_deltas):.2f}')
        ax1.set_xlabel('Average Delta pLDDT (MoRF regions)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'AF2: Distribution of Avg Delta pLDDT\n(n={len(af2_avg_deltas)} proteins)', 
                    fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')
        
        # AF3 histogram
        ax2.hist(af3_avg_deltas, bins=30, alpha=0.7, color='indianred', edgecolor='black')
        ax2.axvline(np.mean(af3_avg_deltas), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(af3_avg_deltas):.2f}')
        ax2.set_xlabel('Average Delta pLDDT (MoRF regions)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'AF3: Distribution of Avg Delta pLDDT\n(n={len(af3_avg_deltas)} proteins)', 
                    fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')
        
        # Add p-value annotation for Delta pLDDT
        p_text_delta = f'p < 0.001' if p_delta < 0.001 else f'p = {p_delta:.3f}'
        fig.text(0.5, 0.64, f'Mann-Whitney U: {p_text_delta}', 
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))
        
        # ========== ROW 2: Disorder-to-Order Accuracy Distributions ==========
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate accuracies
        af2_accuracy = self._calculate_disorder_order_accuracy('AF2')
        af3_accuracy = self._calculate_disorder_order_accuracy('AF3')
        
        # Store results
        self.disorder_order_af2 = af2_accuracy
        self.disorder_order_af3 = af3_accuracy
        
        # Get common proteins
        common_proteins_acc = set(af2_accuracy.keys()) & set(af3_accuracy.keys())
        
        if common_proteins_acc:
            protein_names_sorted = sorted(common_proteins_acc, key=lambda p: af2_accuracy[p]['accuracy'])
            af2_accuracies = [af2_accuracy[p]['accuracy'] * 100 for p in protein_names_sorted]
            af3_accuracies = [af3_accuracy[p]['accuracy'] * 100 for p in protein_names_sorted]
            
            # Statistical test for Accuracy
            stat_acc, p_acc = mannwhitneyu(af2_accuracies, af3_accuracies, alternative='two-sided')
            
            # AF2 accuracy histogram
            ax3.hist(af2_accuracies, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            mean_af2 = np.mean(af2_accuracies)
            ax3.axvline(mean_af2, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_af2:.2f}%')
            ax3.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax3.set_title(f'AF2 Accuracy Distribution (threshold=68)\n(n={len(protein_names_sorted)} proteins)', 
                        fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold')
            
            # AF3 accuracy histogram
            ax4.hist(af3_accuracies, bins=20, alpha=0.7, color='indianred', edgecolor='black')
            mean_af3 = np.mean(af3_accuracies)
            ax4.axvline(mean_af3, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_af3:.2f}%')
            ax4.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax4.set_title(f'AF3 Accuracy Distribution (threshold=70)\n(n={len(protein_names_sorted)} proteins)', 
                        fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold')
            
            # Add p-value annotation for Accuracy
            p_text_acc = f'p < 0.001' if p_acc < 0.001 else f'p = {p_acc:.3f}'
            fig.text(0.5, 0.362, f'Mann-Whitney U: {p_text_acc}', 
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))
        
        # ========== ROW 3: MCW Correlation (spans both columns) ==========
        ax5 = fig.add_subplot(gs[2, :])
        
        # Collect data for each protein
        protein_data = {}
        common_proteins = set(self.delta_plddt_af2.keys()) & set(self.delta_plddt_af3.keys())
        
        for protein_name in common_proteins:
            af2_residues = self.delta_plddt_af2[protein_name]['residues']
            mcw_scores = [r['MCW'] for r in af2_residues]
            avg_mcw = np.mean(mcw_scores)
            
            af2_deltas = [r['delta_plddt'] for r in af2_residues]
            avg_delta_af2 = np.mean(af2_deltas)
            
            af3_residues = self.delta_plddt_af3[protein_name]['residues']
            af3_deltas = [r['delta_plddt'] for r in af3_residues]
            avg_delta_af3 = np.mean(af3_deltas)
            
            protein_data[protein_name] = {
                'avg_mcw': avg_mcw,
                'avg_delta_af2': avg_delta_af2,
                'avg_delta_af3': avg_delta_af3
            }
        
        # Sort by average MCW score
        sorted_proteins = sorted(protein_data.items(), key=lambda x: x[1]['avg_mcw'])
        
        avg_mcw_list = [data['avg_mcw'] for _, data in sorted_proteins]
        avg_delta_af2_list = [data['avg_delta_af2'] for _, data in sorted_proteins]
        avg_delta_af3_list = [data['avg_delta_af3'] for _, data in sorted_proteins]
        
        x_axis = range(len(sorted_proteins))
        
        # Plot three lines
        line1 = ax5.plot(x_axis, avg_mcw_list, 'o-', color='#6A5FB4', 
                        linewidth=2, markersize=4, label='Avg MCW Score', alpha=0.7)
        
        # Create secondary y-axis for delta pLDDT
        ax5_twin = ax5.twinx()
        line2 = ax5_twin.plot(x_axis, avg_delta_af2_list, 's-', color='steelblue', 
                        linewidth=2, markersize=4, label='Avg Δ pLDDT (AF2)', alpha=0.7)
        line3 = ax5_twin.plot(x_axis, avg_delta_af3_list, '^-', color='indianred', 
                        linewidth=2, markersize=4, label='Avg Δ pLDDT (AF3)', alpha=0.7)
        
        # Labels and title
        ax5.set_xlabel('Proteins (sorted by MCW score)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Average MCW Score', fontsize=12, fontweight='bold')
        ax5_twin.set_ylabel('Average Δ pLDDT', fontsize=12, fontweight='bold')
        
        ax5.set_title(f'Average MCW Score vs Average Δ pLDDT\n(n={len(sorted_proteins)} proteins)', 
                    fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='best', fontsize=10)
        
        ax5.grid(True, alpha=0.3)
        
        # Calculate and display correlations
        r_af2, p_af2 = stats.pearsonr(avg_mcw_list, avg_delta_af2_list)
        r_af3, p_af3 = stats.pearsonr(avg_mcw_list, avg_delta_af3_list)
        
        textstr = f'AF2: r = {r_af2:.3f}, p = {p_af2:.2e}\nAF3: r = {r_af3:.3f}, p = {p_af3:.2e}'
        ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax5.text(-0.045, 1.05, 'E', transform=ax5.transAxes, fontsize=16, fontweight='bold')
        
        # Save the combined figure
        plt.savefig('results/morf_analysis_figure.pdf', format='pdf', bbox_inches='tight')
        print("Saved: results/morf_analysis_figure.pdf")
        plt.close()
        
        # Print summary statistics
        print(f"{'='*70}")
        print("PLOT SUMMARY WITH STATISTICAL TESTS")
        print(f"{'='*70}")
        print(f"Row 1: Delta pLDDT Distributions")
        print(f"  Common proteins: {len(common_proteins_delta)}")
        print(f"  AF2 Mean: {np.mean(af2_avg_deltas):.3f}")
        print(f"  AF3 Mean: {np.mean(af3_avg_deltas):.3f}")
        print(f"  Mann-Whitney U statistic: {stat_delta:.2f}")
        print(f"  p-value: {p_delta:.4e}")
        print(f"  Significant: {'Yes' if p_delta < 0.05 else 'No'} (α=0.05)")
        
        if common_proteins_acc:
            print(f"\nRow 2: Disorder-to-Order Accuracy")
            print(f"  Common proteins: {len(protein_names_sorted)}")
            print(f"  AF2 Mean Accuracy: {mean_af2:.2f}%")
            print(f"  AF3 Mean Accuracy: {mean_af3:.2f}%")
            print(f"  Mann-Whitney U statistic: {stat_acc:.2f}")
            print(f"  p-value: {p_acc:.4e}")
            print(f"  Significant: {'Yes' if p_acc < 0.05 else 'No'} (α=0.05)")
        
        print(f"\nRow 3: MCW Correlation")
        print(f"  Proteins: {len(sorted_proteins)}")
        print(f"  AF2 - Pearson r: {r_af2:.3f}, p-value: {p_af2:.2e}")
        print(f"  AF3 - Pearson r: {r_af3:.3f}, p-value: {p_af3:.2e}")
        print(f"{'='*70}")
        


    def get_summary_statistics(self):
        summary = {
            'AF2': {
                'num_proteins': len(self.delta_plddt_af2),
                'avg_delta_plddt_mean': np.mean([d['avg_delta_plddt'] for d in self.delta_plddt_af2.values()]),
                'avg_delta_plddt_std': np.std([d['avg_delta_plddt'] for d in self.delta_plddt_af2.values()]),
            },
            'AF3': {
                'num_proteins': len(self.delta_plddt_af3),
                'avg_delta_plddt_mean': np.mean([d['avg_delta_plddt'] for d in self.delta_plddt_af3.values()]),
                'avg_delta_plddt_std': np.std([d['avg_delta_plddt'] for d in self.delta_plddt_af3.values()]),
            }
        }
        
        # Add disorder-to-order accuracy statistics if available
        if hasattr(self, 'disorder_order_af2') and hasattr(self, 'disorder_order_af3'):
            common_proteins = set(self.disorder_order_af2.keys()) & set(self.disorder_order_af3.keys())
            
            if common_proteins:
                af2_accuracies = [self.disorder_order_af2[p]['accuracy'] * 100 for p in common_proteins]
                af3_accuracies = [self.disorder_order_af3[p]['accuracy'] * 100 for p in common_proteins]
                
                summary['disorder_order_accuracy'] = {
                    'num_proteins': len(common_proteins),
                    'AF2': {
                        'mean': np.mean(af2_accuracies),
                        'std': np.std(af2_accuracies),
                        'median': np.median(af2_accuracies),
                        'min': np.min(af2_accuracies),
                        'max': np.max(af2_accuracies),
                        'threshold': 68
                    },
                    'AF3': {
                        'mean': np.mean(af3_accuracies),
                        'std': np.std(af3_accuracies),
                        'median': np.median(af3_accuracies),
                        'min': np.min(af3_accuracies),
                        'max': np.max(af3_accuracies),
                        'threshold': 70
                    }
                }
                
                # Perform paired t-test
                from scipy.stats import ttest_rel
                t_stat, p_value = ttest_rel(af2_accuracies, af3_accuracies)
                summary['disorder_order_accuracy']['t_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # Count comparison
                af3_better = sum(1 for a2, a3 in zip(af2_accuracies, af3_accuracies) if a3 > a2)
                af2_better = sum(1 for a2, a3 in zip(af2_accuracies, af3_accuracies) if a2 > a3)
                equal = sum(1 for a2, a3 in zip(af2_accuracies, af3_accuracies) if a2 == a3)
                
                summary['disorder_order_accuracy']['comparison'] = {
                    'af3_better': af3_better,
                    'af2_better': af2_better,
                    'equal': equal
                }
        
        return summary


if __name__ == "__main__":
    analyzer = MoRFPLDDTAnalyzer(
        morf_folder="dataset/morf_mfib_dataset",
        mono_af2_folder="dataset/af2_mfib_mono_dataset",
        mono_af3_folder="dataset/af3_mfib_mono_dataset",
        multi_af2_folder="dataset/af2_mfib_multi_dataset",
        multi_af3_folder="dataset/af3_mfib_multi_dataset",
        mcw_threshold=0.72
    )
    
    analyzer.run_full_analysis()
    
    summary = analyzer.get_summary_statistics()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Delta pLDDT statistics
    for version in ['AF2', 'AF3']:
        print(f"\n{version}:")
        print(f"  Proteins analyzed: {summary[version]['num_proteins']}")
        print(f"  Mean avg delta pLDDT: {summary[version]['avg_delta_plddt_mean']:.3f}")
        print(f"  Std avg delta pLDDT: {summary[version]['avg_delta_plddt_std']:.3f}")
    
