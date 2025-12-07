import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

class ProteinDataAnalysis:
    def __init__(self, other_data_path, newmodel_data_path):
        self.other_data_path = other_data_path
        self.newmodel_data_path = newmodel_data_path

        self.other_field_names = [
            "uniprot_disprot_id", "amino_acid_sequence", "disprot_pdb_annotation", "disprot_annotation",
            "dssp_assignment", "plddt_scores", "aucpred_disorder_score", "aucpred_binary_prediction",
            "aucpred_np_disorder_score", "aucpred_np_binary_prediction", "disomine_disorder_score",
            "disomine_binary_prediction", "espritz_d_disorder_score", "espritz_d_binary_prediction",
            "fidpnn_disorder_score", "fidpnn_binary_prediction", "fidplr_disorder_score",
            "fidplr_binary_prediction", "predisorder_disorder_score", "predisorder_binary_prediction",
            "rawmsa_disorder_score", "rawmsa_binary_prediction", "spot_disorder1_disorder_score",
            "spot_disorder1_binary_prediction", "spot_disorder_s_disorder_score", "spot_disorder_s_binary_prediction",
            "spot_disorder2_disorder_score", "spot_disorder2_binary_prediction"
        ]
        # newmodel has 4 lines per entry
        self.newmodel_field_names = [
            "uniprot_disprot_id", "amino_acid_sequence",
            "alphafold3_plddt_scores", "esmfold_plddt_scores"
        ]

        self.combined_data = None

    # ---------- low-level parsers ----------
    def read_file(self, type):
        with open(self.other_data_path if type == "other" else self.newmodel_data_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def get_uniprot_id(self, line):
        return line[1:].split('|')[0].lower() if line.startswith('>') else None

    def str_to_float_list(self, s):
        return [float(x) for x in s.strip().split('|') if x]

    def annotation_to_int_list(self, annotation_str):
        # digits are labels; any non-digit (e.g., '-') becomes -1 (masked)
        return [int(ch) if ch.isdigit() else -1 for ch in annotation_str.strip()]

    def create_mask(self, annotation_list):
        return np.array([i for i, v in enumerate(annotation_list) if v != -1])

    def apply_mask(self, data_list, mask):
        data_array = np.array(data_list)
        return data_array[mask]

    def parse_data(self, type):
        """Parse 'other' (28-line blocks) or 'newmodel' (4-line blocks)."""
        if type == "other":
            num_line = 28
            field_names = self.other_field_names
        else:
            num_line = 4
            field_names = self.newmodel_field_names

        lines = self.read_file(type=type)
        result = {}
        for i in range(0, len(lines) + num_line - len(lines)%num_line, num_line):
            if i + num_line - 1 < len(lines) and lines[i].startswith('>'):
                uniprot_id = self.get_uniprot_id(lines[i])
                entry_dict = {}
                for j in range(1, num_line):
                    entry_dict[field_names[j]] = lines[i + j]
                result[uniprot_id] = entry_dict
        return result

    # ---------- data join ----------
    def load_and_combine_data(self):
        """Join 'other' and 'newmodel' on UniProt with exact sequence match."""
        other_data = self.parse_data("other")
        nm_data = self.parse_data("newmodel")

        self.combined_data = {}
        common_ids = set(other_data.keys()) & set(nm_data.keys())
        for uid in common_ids:
            if other_data[uid]['amino_acid_sequence'] == nm_data[uid]['amino_acid_sequence']:
                combined = other_data[uid].copy()
                combined['alphafold3_plddt_scores'] = nm_data[uid]['alphafold3_plddt_scores']
                combined['esmfold_plddt_scores']   = nm_data[uid]['esmfold_plddt_scores']
                self.combined_data[uid] = combined
        return len(self.combined_data)

    def prepare_evaluation_data(self, dataset_type='disprot_pdb', model='alphafold2'):
        if self.combined_data is None:
            self.load_and_combine_data()

        field_name = 'disprot_pdb_annotation' if dataset_type == 'disprot_pdb' else 'disprot_annotation'
        
        # Map model names to their corresponding pLDDT field names
        plddt_mapping = {
            'alphafold2': 'plddt_scores',           # From combined.dat line 6
            'alphafold3': 'alphafold3_plddt_scores', # From newmodel.fasta line 3
            'esmfold': 'esmfold_plddt_scores'       # From newmodel.fasta line 4
        }
        
        if model not in plddt_mapping:
            raise ValueError(f"model must be one of {list(plddt_mapping.keys())}, got '{model}'")
        
        plddt_key = plddt_mapping[model]

        all_true = []
        all_tplddt = []
        all_plddt = []
        protein_data = {}

        for uid, data in self.combined_data.items():
            # require target pLDDT to be present
            if plddt_key not in data:
                continue

            annotation_str = data.get(field_name, 'None')
            if annotation_str == 'None':
                continue

            ann_list = self.annotation_to_int_list(annotation_str)
            plddt = self.str_to_float_list(data[plddt_key])
            tplddt = [1.0 - (x / 100.0) for x in plddt]

            if dataset_type == 'disprot_pdb':
                mask = self.create_mask(ann_list)
                if len(mask) == 0:
                    continue
                # ensure lengths are long enough for mask
                max_idx = int(mask.max())
                if len(plddt) <= max_idx:
                    continue

                true_vals = self.apply_mask(ann_list, mask)
                tplddt_vals = self.apply_mask(tplddt, mask)
                plddt_vals = self.apply_mask(plddt, mask)

                all_true.extend(true_vals)
                all_tplddt.extend(tplddt_vals)
                all_plddt.extend(plddt_vals)

                protein_data[uid] = {
                    'true_disorder': true_vals,
                    'model_tplddt': tplddt_vals,
                    'model_plddt': plddt_vals,
                    'mask': mask,
                    'sequence_length': len(true_vals),
                    'disorder_content': np.mean(true_vals)
                }
            else:
                # 'disprot' assumes same length
                true_vals = np.array(ann_list)
                tplddt_vals = np.array(tplddt[:len(true_vals)])
                plddt_vals = np.array(plddt[:len(true_vals)])

                all_true.extend(true_vals)
                all_tplddt.extend(tplddt_vals)
                all_plddt.extend(plddt_vals)

                protein_data[uid] = {
                    'true_disorder': true_vals,
                    'model_tplddt': tplddt_vals,
                    'model_plddt': plddt_vals,
                    'mask': None,
                    'sequence_length': len(true_vals),
                    'disorder_content': np.mean(true_vals)
                }

        return {
            'all_true': np.array(all_true),
            'all_model_tplddt': np.array(all_tplddt),
            'all_model_plddt': np.array(all_plddt),
            'protein_data': protein_data
        }

    # Update all method signatures to use model='alphafold2' as default

    def find_optimal_threshold(self, dataset_type='disprot_pdb', model='alphafold2'):
        ev = self.prepare_evaluation_data(dataset_type, model)
        y_true = ev['all_true']
        plddt = ev['all_model_plddt']

        best_mcc, best_thr = -1.0, 0
        for thr in range(1, 100):
            # low pLDDT => disordered (1)
            y_pred = 1 - (plddt >= thr).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc, best_thr = mcc, thr
        return best_thr, best_mcc

    def calculate_roc_metrics(self, dataset_type='disprot_pdb', model='alphafold2'):
        ev = self.prepare_evaluation_data(dataset_type, model)
        fpr, tpr, th = roc_curve(ev['all_true'], ev['all_model_tplddt'])
        return {'fpr': fpr, 'tpr': tpr, 'thresholds': th, 'auc': auc(fpr, tpr)}

    def calculate_pr_metrics(self, dataset_type='disprot_pdb', model='alphafold2'):
        ev = self.prepare_evaluation_data(dataset_type, model)
        precision, recall, th = precision_recall_curve(ev['all_true'], ev['all_model_tplddt'])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return {'precision': precision, 'recall': recall, 'thresholds': th, 'fmax': float(np.max(f1))}

    def calculate_mcc(self, dataset_type='disprot_pdb', model='alphafold2', threshold=None):
        if threshold is None:
            threshold, _ = self.find_optimal_threshold(dataset_type, model)
        ev = self.prepare_evaluation_data(dataset_type, model)
        y_true = ev['all_true']
        plddt = ev['all_model_plddt']
        y_pred = 1 - (plddt >= threshold).astype(int)
        return matthews_corrcoef(y_true, y_pred), threshold

    def calculate_rmsd_per_protein(self, dataset_type='disprot_pdb', model='alphafold2'):
        ev = self.prepare_evaluation_data(dataset_type, model)
        protein_data = ev['protein_data']
        rmsd = {}
        for uid, d in protein_data.items():
            y = d['true_disorder']
            p = d['model_tplddt']
            rmsd[uid] = float(np.sqrt(np.mean((y - p) ** 2)))
        return rmsd

    def calculate_rmsd_by_class(self, dataset_type='disprot_pdb', model='alphafold2'):
        ev = self.prepare_evaluation_data(dataset_type, model)
        protein_data = ev['protein_data']
        classes = {'highly_ordered': [], 'mixed': [], 'highly_disordered': []}
        for _, d in protein_data.items():
            dc = d['disorder_content']
            y = d['true_disorder']
            p = d['model_tplddt']
            r = float(np.sqrt(np.mean((y - p) ** 2)))
            if dc < 0.1:
                classes['highly_ordered'].append(r)
            elif dc > 0.9:
                classes['highly_disordered'].append(r)
            else:
                classes['mixed'].append(r)
        return {k: np.array(v) for k, v in classes.items()}

    def compare_with_other_predictors(self, dataset_type='disprot_pdb', metric='auc', model='alphafold2'):
        """Keeps your previous comparison logic, using the same masked proteins as the chosen model."""
        ev = self.prepare_evaluation_data(dataset_type, model)
        protein_data = ev['protein_data']

        predictors = ['aucpred', 'aucpred_np', 'disomine', 'espritz_d', 'fidpnn',
                    'fidplr', 'predisorder', 'rawmsa', 'spot_disorder1',
                    'spot_disorder_s', 'spot_disorder2']

        results = {f'{model}_tplddt': None}

        # baseline: chosen model
        if metric == 'auc':
            results[f'{model}_tplddt'] = self.calculate_roc_metrics(dataset_type, model)['auc']
        elif metric == 'fmax':
            results[f'{model}_tplddt'] = self.calculate_pr_metrics(dataset_type, model)['fmax']
        elif metric == 'mcc':
            results[f'{model}_tplddt'] = self.calculate_mcc(dataset_type, model)[0]

        # others
        for predictor in predictors:
            try:
                all_scores, all_binary, all_true_pred = [], [], []
                for uid, pinfo in protein_data.items():
                    data = self.combined_data[uid]
                    try:
                        scores = self.str_to_float_list(data[f'{predictor}_disorder_score'])
                        binary = [int(c) for c in data[f'{predictor}_binary_prediction'].strip()]
                    except (KeyError, ValueError):
                        continue

                    if dataset_type == 'disprot_pdb':
                        mask = pinfo['mask']
                        if mask is None or len(mask) == 0:
                            continue
                        max_idx = int(mask.max())
                        if len(scores) <= max_idx or len(binary) <= max_idx:
                            continue
                        valid_scores = self.apply_mask(scores, mask)
                        valid_binary = self.apply_mask(binary, mask)
                        valid_true = pinfo['true_disorder']
                    else:
                        L = pinfo['sequence_length']
                        if len(scores) < L or len(binary) < L:
                            continue
                        valid_scores = np.array(scores[:L])
                        valid_binary = np.array(binary[:L])
                        valid_true = pinfo['true_disorder']

                    all_scores.extend(valid_scores)
                    all_binary.extend(valid_binary)
                    all_true_pred.extend(valid_true)

                all_scores = np.array(all_scores)
                all_binary = np.array(all_binary)
                all_true_pred = np.array(all_true_pred)
                if len(all_scores) == 0:
                    results[predictor] = None
                    continue

                if metric == 'auc':
                    fpr, tpr, _ = roc_curve(all_true_pred, all_scores)
                    results[predictor] = auc(fpr, tpr)
                elif metric == 'fmax':
                    precision, recall, _ = precision_recall_curve(all_true_pred, all_scores)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                    results[predictor] = float(np.max(f1))
                elif metric == 'mcc':
                    results[predictor] = matthews_corrcoef(all_true_pred, all_binary)
            except Exception as e:
                print(f"Error calculating {metric} for {predictor}: {e}")
                results[predictor] = None

        return results

    def make_pld_table(self, dataset_type='disprot_pdb', model='alphafold2',
                    thr_start=60, thr_end=80, step=2,
                    per_threshold_metrics=True):

        # pooled eval data
        ev = self.prepare_evaluation_data(dataset_type, model)
        y_true = ev['all_true'].astype(int)
        plddt  = ev['all_model_plddt']
        tplddt = ev['all_model_tplddt']
        cov = len(ev['protein_data'])

        # continuous (for the non-per-threshold mode)
        cont_auc  = self.calculate_roc_metrics(dataset_type, model)['auc']
        cont_fmax = self.calculate_pr_metrics(dataset_type, model)['fmax']

        rows = []
        for thr in range(thr_start, thr_end + 1, step):
            # Binary decision for this threshold: disordered if pLDDT < thr
            y_pred = 1 - (plddt >= thr).astype(int)

            # Confusion matrix elements
            TP = int(((y_true == 1) & (y_pred == 1)).sum())
            TN = int(((y_true == 0) & (y_pred == 0)).sum())
            FP = int(((y_true == 0) & (y_pred == 1)).sum())
            FN = int(((y_true == 1) & (y_pred == 0)).sum())

            # Rates
            TPR = TP / (TP + FN) if (TP + FN) > 0 else float('nan')  # recall
            FPR = FP / (FP + TN) if (FP + TN) > 0 else float('nan')
            TNR = TN / (TN + FP) if (TN + FP) > 0 else float('nan')  # specificity
            PPV = TP / (TP + FP) if (TP + FP) > 0 else float('nan')  # precision
            BAC = (TPR + TNR) / 2 if (not np.isnan(TPR) and not np.isnan(TNR)) else float('nan')
            MCC = matthews_corrcoef(y_true, y_pred) if (TP + TN + FP + FN) > 0 else float('nan')

            if per_threshold_metrics:
                # AUC from the single operating point (equivalent to BAC)
                try:
                    fpr_, tpr_, _ = roc_curve(y_true, y_pred)
                    AUC_val = auc(fpr_, tpr_)
                except Exception:
                    AUC_val = float('nan')

                # F1 at this threshold (NOT max)
                if (TP + FP) > 0 and (TP + FN) > 0:
                    F1_thr = 2 * PPV * TPR / (PPV + TPR) if (PPV + TPR) > 0 else 0.0
                else:
                    F1_thr = float('nan')

                AUC_out, Fmax_out = float(AUC_val), float(F1_thr)
            else:
                # Classic, threshold-agnostic
                AUC_out, Fmax_out = float(cont_auc), float(cont_fmax)

            rows.append({
                'Pred': f'pLD{thr}',
                'MCC':  float(MCC),
                'AUC':  AUC_out,
                'Fmax': Fmax_out,
                'TPR':  float(TPR),
                'FPR':  float(FPR),
                'TNR':  float(TNR),
                'PPV':  float(PPV),
                'BAC':  float(BAC),
                'COV':  int(cov),
            })

        cols = ['Pred', 'MCC', 'AUC', 'Fmax', 'TPR', 'FPR', 'TNR', 'PPV', 'BAC', 'COV']
        return pd.DataFrame(rows, columns=cols)

    def generate_performance_report(self, dataset_type='disprot_pdb', model='alphafold2'):
        print(f"\n=== Disorder Prediction Performance Report ({dataset_type}, model={model}) ===")
        ev = self.prepare_evaluation_data(dataset_type, model)
        print(f"Dataset: {len(ev['protein_data'])} proteins")
        print(f"Total residues: {len(ev['all_true'])}")
        print(f"Disorder content: {np.mean(ev['all_true']):.3f}")

        thr, best_mcc = self.find_optimal_threshold(dataset_type, model)
        print(f"\nOptimal pLDDT threshold: {thr} (MCC: {best_mcc:.3f})")

        roc_metrics = self.calculate_roc_metrics(dataset_type, model)
        pr_metrics  = self.calculate_pr_metrics(dataset_type, model)

        print(f"\n{model} Performance Metrics:")
        print(f"AUC:  {roc_metrics['auc']:.3f}")
        print(f"Fmax: {pr_metrics['fmax']:.3f}")
        print(f"MCC:  {best_mcc:.3f}")

        rmsd_by_class = self.calculate_rmsd_by_class(dataset_type, model)
        print(f"\nRMSD by protein class:")
        for cls, arr in rmsd_by_class.items():
            if len(arr) > 0:
                print(f"{cls}: {np.mean(arr):.3f} ± {np.std(arr):.3f} ({len(arr)} proteins)")

        print(f"\nComparison with other predictors (AUC):")
        auc_cmp = self.compare_with_other_predictors(dataset_type, 'auc', model)
        sorted_items = sorted(auc_cmp.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        for i, (name, score) in enumerate(sorted_items):
            if score is not None:
                status = "🏆" if i == 0 else "📈" if name == f'{model}_tplddt' else ""
                print(f"{name}: {score:.3f} {status}")

        return {
            'optimal_threshold': thr,
            'auc': roc_metrics['auc'],
            'fmax': pr_metrics['fmax'],
            'mcc': best_mcc,
            'rmsd_by_class': rmsd_by_class,
            'comparison': auc_cmp
        }

def plot_structure_model_comparison(pda, dataset_type='disprot_pdb', figsize=(12, 9)):
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Define colors for structure prediction models
    structure_colors = {
        'alphafold2': '#E69F00',  # Orange
        'alphafold3': '#D55E00',  # Red-orange  
        'esmfold': '#CC79A7'      # Pink
    }
    
    # Define markers for structure models
    structure_markers = {
        'alphafold2': 'o',
        'alphafold3': 's',
        'esmfold': '^'
    }
    
    # Model labels
    structure_labels = {
        'alphafold2': 'pLD (AF2)',
        'alphafold3': 'pLD (AF3)',
        'esmfold': 'pLD (ESM)'
    }
    
    # ========== TOP LEFT: ROC for Structure Models ==========
    ax_roc_struct = axes[0, 0]
    
    structure_aucs = {}
    for model in ['alphafold2', 'alphafold3', 'esmfold']:
        try:
            roc_metrics = pda.calculate_roc_metrics(dataset_type, model)
            fpr, tpr = roc_metrics['fpr'], roc_metrics['tpr']
            roc_auc = roc_metrics['auc']
            structure_aucs[model] = roc_auc
            
            ax_roc_struct.plot(fpr, tpr, 
                             color=structure_colors[model],
                             linewidth=2.5,
                             label=f"{structure_labels[model]} (AUC={roc_auc:.2f})",
                             alpha=0.8)
            
            # Add marker at optimal threshold
            thr, _ = pda.find_optimal_threshold(dataset_type, model)
            ev = pda.prepare_evaluation_data(dataset_type, model)
            plddt = ev['all_model_plddt']
            y_true = ev['all_true']
            y_pred_binary = 1 - (plddt >= thr).astype(int)
            
            # Find FPR and TPR at optimal threshold
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            fpr_opt = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            ax_roc_struct.plot(fpr_opt, tpr_opt, 
                             marker=structure_markers[model],
                             markersize=10,
                             color=structure_colors[model],
                             markeredgecolor='white',
                             markeredgewidth=1.5)
            
        except Exception as e:
            print(f"Error plotting ROC for {model}: {e}")
    
    # Diagonal line
    ax_roc_struct.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax_roc_struct.set_xlabel('False positive rate', fontsize=12, fontweight='bold')
    ax_roc_struct.set_ylabel('True positive rate', fontsize=12, fontweight='bold')
    ax_roc_struct.set_xlim([0.0, 1.0])
    ax_roc_struct.set_ylim([0.0, 1.0])
    ax_roc_struct.legend(loc='upper left', bbox_to_anchor=(0.56, 0.25), frameon=True, 
                        fancybox=True, shadow=False, fontsize=10)
    ax_roc_struct.grid(True, alpha=0.3)
    
    # ========== TOP RIGHT: ROC for Other Predictors ==========
    ax_roc_other = axes[0, 1]
    
    predictors = ['aucpred', 'aucpred_np', 'disomine', 'espritz_d', 'fidpnn',
                  'fidplr', 'predisorder', 'rawmsa', 'spot_disorder1',
                  'spot_disorder_s', 'spot_disorder2']
    
    # Use seaborn color palette
    predictor_colors = sns.color_palette("tab10", n_colors=len(predictors))
    
    # Get protein data for masking consistency (use alphafold2 as reference)
    ev_ref = pda.prepare_evaluation_data(dataset_type, 'alphafold2')
    protein_data = ev_ref['protein_data']
    
    predictor_aucs = {}
    for idx, predictor in enumerate(predictors):
        try:
            all_scores = []
            all_true = []
            
            for uid, pinfo in protein_data.items():
                data = pda.combined_data[uid]
                try:
                    scores = pda.str_to_float_list(data[f'{predictor}_disorder_score'])
                except (KeyError, ValueError):
                    continue
                
                if dataset_type == 'disprot_pdb':
                    mask = pinfo['mask']
                    if mask is None or len(mask) == 0:
                        continue
                    max_idx = int(mask.max())
                    if len(scores) <= max_idx:
                        continue
                    valid_scores = pda.apply_mask(scores, mask)
                    valid_true = pinfo['true_disorder']
                else:
                    L = pinfo['sequence_length']
                    if len(scores) < L:
                        continue
                    valid_scores = np.array(scores[:L])
                    valid_true = pinfo['true_disorder']
                
                all_scores.extend(valid_scores)
                all_true.extend(valid_true)
            
            if len(all_scores) > 0:
                all_scores = np.array(all_scores)
                all_true = np.array(all_true)
                
                fpr, tpr, _ = roc_curve(all_true, all_scores)
                roc_auc = auc(fpr, tpr)
                predictor_aucs[predictor] = roc_auc
                
                ax_roc_other.plot(fpr, tpr,
                                color=predictor_colors[idx],
                                linewidth=2,
                                label=f"{predictor}",
                                alpha=0.7)
        except Exception as e:
            print(f"Error plotting ROC for {predictor}: {e}")
    
    # Diagonal line
    ax_roc_other.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax_roc_other.set_xlabel('False positive rate', fontsize=12, fontweight='bold')
    ax_roc_other.set_ylabel('True positive rate', fontsize=12, fontweight='bold')
    ax_roc_other.set_xlim([0.0, 1.0])
    ax_roc_other.set_ylim([0.0, 1.0])
    
    # Add AUC bar chart inset
    if predictor_aucs:
        ax_inset_roc = ax_roc_other.inset_axes([0.52, 0.05, 0.45, 0.30])
        sorted_predictors = sorted(predictor_aucs.items(), key=lambda x: x[1], reverse=True)
        pred_names = [p[0] for p in sorted_predictors]
        pred_values = [p[1] for p in sorted_predictors]
        
        bars = ax_inset_roc.barh(range(len(pred_names)), pred_values,
                                  color=[predictor_colors[predictors.index(p)] for p in pred_names],
                                  alpha=0.7)
        ax_inset_roc.set_yticks(range(len(pred_names)))
        ax_inset_roc.set_yticklabels(pred_names, fontsize=10)
        ax_inset_roc.xaxis.set_label_position('top')
        ax_inset_roc.set_xlabel('AUC', fontsize=10, fontweight='bold')
        ax_inset_roc.set_xlim([0.7, 1.0])
        ax_inset_roc.tick_params(axis='x', labelsize=8, pad = 0.1)
        ax_inset_roc.grid(True, alpha=0.3, axis='x')
    
    ax_roc_other.grid(True, alpha=0.3)
    
    # ========== BOTTOM LEFT: PR for Structure Models ==========
    ax_pr_struct = axes[1, 0]
    
    structure_fmaxs = {}
    for model in ['alphafold2', 'alphafold3', 'esmfold']:
        try:
            pr_metrics = pda.calculate_pr_metrics(dataset_type, model)
            precision, recall = pr_metrics['precision'], pr_metrics['recall']
            fmax = pr_metrics['fmax']
            structure_fmaxs[model] = fmax
            
            ax_pr_struct.plot(recall, precision,
                            color=structure_colors[model],
                            linewidth=2.5,
                            label=f"{structure_labels[model]} (F$_{{max}}$={fmax:.2f})",
                            alpha=0.8)
            
            # Add marker at F-max
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            max_idx = np.argmax(f1_scores)
            ax_pr_struct.plot(recall[max_idx], precision[max_idx],
                            marker=structure_markers[model],
                            markersize=10,
                            color=structure_colors[model],
                            markeredgecolor='white',
                            markeredgewidth=1.5)
            
        except Exception as e:
            print(f"Error plotting PR for {model}: {e}")
    
    ax_pr_struct.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax_pr_struct.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_pr_struct.set_xlim([0.0, 1.0])
    ax_pr_struct.set_ylim([0.0, 1.0])
    ax_pr_struct.legend(loc='upper left', bbox_to_anchor=(0.56, 0.25), frameon=True, 
                       fancybox=True, shadow=False, fontsize=10)
    ax_pr_struct.grid(True, alpha=0.3)
    
    # ========== BOTTOM RIGHT: PR for Other Predictors ==========
    ax_pr_other = axes[1, 1]
    
    predictor_fmaxs = {}
    for idx, predictor in enumerate(predictors):
        try:
            all_scores = []
            all_true = []
            
            for uid, pinfo in protein_data.items():
                data = pda.combined_data[uid]
                try:
                    scores = pda.str_to_float_list(data[f'{predictor}_disorder_score'])
                except (KeyError, ValueError):
                    continue
                
                if dataset_type == 'disprot_pdb':
                    mask = pinfo['mask']
                    if mask is None or len(mask) == 0:
                        continue
                    max_idx = int(mask.max())
                    if len(scores) <= max_idx:
                        continue
                    valid_scores = pda.apply_mask(scores, mask)
                    valid_true = pinfo['true_disorder']
                else:
                    L = pinfo['sequence_length']
                    if len(scores) < L:
                        continue
                    valid_scores = np.array(scores[:L])
                    valid_true = pinfo['true_disorder']
                
                all_scores.extend(valid_scores)
                all_true.extend(valid_true)
            
            if len(all_scores) > 0:
                all_scores = np.array(all_scores)
                all_true = np.array(all_true)
                
                precision, recall, _ = precision_recall_curve(all_true, all_scores)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                fmax = float(np.max(f1_scores))
                predictor_fmaxs[predictor] = fmax
                
                ax_pr_other.plot(recall, precision,
                               color=predictor_colors[idx],
                               linewidth=2,
                               label=f"{predictor}",
                               alpha=0.7)
        except Exception as e:
            print(f"Error plotting PR for {predictor}: {e}")
    
    ax_pr_other.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax_pr_other.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_pr_other.set_xlim([0.0, 1.0])
    ax_pr_other.set_ylim([0.0, 1.0])
    
    # Add Fmax bar chart inset
    if predictor_fmaxs:
        ax_inset_pr = ax_pr_other.inset_axes([0.52, 0.05, 0.45, 0.30])
        sorted_predictors = sorted(predictor_fmaxs.items(), key=lambda x: x[1], reverse=True)
        pred_names = [p[0] for p in sorted_predictors]
        pred_values = [p[1] for p in sorted_predictors]
        
        bars = ax_inset_pr.barh(range(len(pred_names)), pred_values,
                                 color=[predictor_colors[predictors.index(p)] for p in pred_names],
                                 alpha=0.7)
        ax_inset_pr.set_yticks(range(len(pred_names)))
        ax_inset_pr.set_yticklabels(pred_names, fontsize=10)
        ax_inset_pr.xaxis.set_label_position('top')
        ax_inset_pr.set_xlabel('F$_{max}$', fontsize=10, fontweight='bold')
        ax_inset_pr.set_xlim([0.5, 1.0])
        ax_inset_pr.tick_params(axis='x', labelsize=8, pad = 0.1)
        ax_inset_pr.grid(True, alpha=0.3, axis='x')
    
    ax_pr_other.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Performance Summary ({dataset_type})")
    print(f"{'='*60}")
    
    print("\nStructure Prediction Models:")
    print(f"{'Model':<15} {'AUC':<10} {'F-max':<10}")
    print("-" * 35)
    for model in ['alphafold2', 'alphafold3', 'esmfold']:
        auc_val = structure_aucs.get(model, 0)
        fmax_val = structure_fmaxs.get(model, 0)
        print(f"{structure_labels[model]:<15} {auc_val:.3f}      {fmax_val:.3f}")
    
    print("\nTop 5 Other Predictors (by AUC):")
    sorted_by_auc = sorted(predictor_aucs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"{'Predictor':<15} {'AUC':<10} {'F-max':<10}")
    print("-" * 35)
    for pred, auc_val in sorted_by_auc:
        fmax_val = predictor_fmaxs.get(pred, 0)
        print(f"{pred:<15} {auc_val:.3f}      {fmax_val:.3f}")
    
    return fig, axes


def plot_mcc_comparison(pda, dataset_type='disprot_pdb', figsize=(10, 6)):
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Structure prediction models
    structure_models = ['alphafold2', 'alphafold3', 'esmfold']
    structure_labels = {
        'alphafold2': 'pLD (AF2)',
        'alphafold3': 'pLD (AF3)',
        'esmfold': 'pLD (ESM)'
    }
    
    # Disorder predictors
    predictors = ['aucpred', 'aucpred_np', 'disomine', 'espritz_d', 'fidpnn',
                  'fidplr', 'predisorder', 'rawmsa', 'spot_disorder1',
                  'spot_disorder_s', 'spot_disorder2']
    
    mcc_scores = {}
    
    # Calculate MCC for structure models (using optimal threshold)
    for model in structure_models:
        try:
            mcc, threshold = pda.calculate_mcc(dataset_type, model, threshold=None)
            mcc_scores[structure_labels[model]] = mcc
            print(f"{structure_labels[model]}: MCC={mcc:.3f} at threshold={threshold}")
        except Exception as e:
            print(f"Error calculating MCC for {model}: {e}")
            mcc_scores[structure_labels[model]] = 0
    
    # Calculate MCC for other predictors (using their binary predictions)
    ev_ref = pda.prepare_evaluation_data(dataset_type, 'alphafold2')
    protein_data = ev_ref['protein_data']
    
    for predictor in predictors:
        try:
            all_binary = []
            all_true = []
            
            for uid, pinfo in protein_data.items():
                data = pda.combined_data[uid]
                try:
                    binary = [int(c) for c in data[f'{predictor}_binary_prediction'].strip()]
                except (KeyError, ValueError):
                    continue
                
                if dataset_type == 'disprot_pdb':
                    mask = pinfo['mask']
                    if mask is None or len(mask) == 0:
                        continue
                    max_idx = int(mask.max())
                    if len(binary) <= max_idx:
                        continue
                    valid_binary = pda.apply_mask(binary, mask)
                    valid_true = pinfo['true_disorder']
                else:
                    L = pinfo['sequence_length']
                    if len(binary) < L:
                        continue
                    valid_binary = np.array(binary[:L])
                    valid_true = pinfo['true_disorder']
                
                all_binary.extend(valid_binary)
                all_true.extend(valid_true)
            
            if len(all_binary) > 0:
                all_binary = np.array(all_binary)
                all_true = np.array(all_true)
                mcc = matthews_corrcoef(all_true, all_binary)
                mcc_scores[predictor] = mcc
            else:
                mcc_scores[predictor] = 0
        except Exception as e:
            print(f"Error calculating MCC for {predictor}: {e}")
            mcc_scores[predictor] = 0
    
    # Sort by MCC score
    sorted_items = sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Color structure models differently
    colors = []
    for name in names:
        if 'pLD' in name:
            if 'AF2' in name:
                colors.append('#E69F00')  # Orange
            elif 'AF3' in name:
                colors.append('#D55E00')  # Red-orange
            else:  # ESM
                colors.append('#CC79A7')  # Pink
        else:
            colors.append('#0072B2')  # Blue for other predictors
    
    # Create bar chart
    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, max(scores) * 1.1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, i, f'{score:.3f}', 
               va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MCC Comparison Summary ({dataset_type})")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Method':<20} {'MCC':<10}")
    print("-" * 60)
    for i, (name, score) in enumerate(sorted_items, 1):
        print(f"{i:<6} {name:<20} {score:.3f}")
    
    return fig, ax

def plot_rmsd_comparison(pda, dataset_type='disprot_pdb', figsize=(14, 5)):
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Structure prediction models
    structure_models = ['alphafold2', 'alphafold3', 'esmfold']
    structure_labels = {
        'alphafold2': 'pLD (AF2)',
        'alphafold3': 'pLD (AF3)',
        'esmfold': 'pLD (ESM)'
    }
    structure_colors = {
        'alphafold2': '#E69F00',
        'alphafold3': '#D55E00',
        'esmfold': '#CC79A7'
    }
    
    # Disorder predictors
    predictors = ['aucpred', 'aucpred_np', 'disomine', 'espritz_d', 'fidpnn',
                  'fidplr', 'predisorder', 'rawmsa', 'spot_disorder1',
                  'spot_disorder_s', 'spot_disorder2']
    
    # Use tab10 palette for predictors
    predictor_palette = sns.color_palette("tab10", n_colors=len(predictors))
    predictor_colors = {pred: predictor_palette[i] for i, pred in enumerate(predictors)}
    
    # Calculate RMSD for structure models
    rmsd_data = {
        'all': {},
        'highly_disordered': {},
        'highly_ordered': {}
    }
    
    for model in structure_models:
        try:
            # Get per-protein RMSD
            ev = pda.prepare_evaluation_data(dataset_type, model)
            protein_data = ev['protein_data']
            
            all_rmsd = []
            highly_disordered_rmsd = []
            highly_ordered_rmsd = []
            
            for uid, d in protein_data.items():
                y = d['true_disorder']
                p = d['model_tplddt']
                rmsd = float(np.sqrt(np.mean((y - p) ** 2)))
                disorder_content = d['disorder_content']
                
                all_rmsd.append(rmsd)
                
                if disorder_content > 0.9:
                    highly_disordered_rmsd.append(rmsd)
                elif disorder_content < 0.1:
                    highly_ordered_rmsd.append(rmsd)
            
            rmsd_data['all'][structure_labels[model]] = np.array(all_rmsd)
            rmsd_data['highly_disordered'][structure_labels[model]] = np.array(highly_disordered_rmsd)
            rmsd_data['highly_ordered'][structure_labels[model]] = np.array(highly_ordered_rmsd)
            
        except Exception as e:
            print(f"Error calculating RMSD for {model}: {e}")
    
    # Calculate RMSD for other predictors
    ev_ref = pda.prepare_evaluation_data(dataset_type, 'alphafold2')
    protein_data_ref = ev_ref['protein_data']
    
    for predictor in predictors:
        try:
            all_rmsd = []
            highly_disordered_rmsd = []
            highly_ordered_rmsd = []
            
            for uid, pinfo in protein_data_ref.items():
                data = pda.combined_data[uid]
                try:
                    scores = pda.str_to_float_list(data[f'{predictor}_disorder_score'])
                except (KeyError, ValueError):
                    continue
                
                if dataset_type == 'disprot_pdb':
                    mask = pinfo['mask']
                    if mask is None or len(mask) == 0:
                        continue
                    max_idx = int(mask.max())
                    if len(scores) <= max_idx:
                        continue
                    valid_scores = pda.apply_mask(scores, mask)
                    valid_true = pinfo['true_disorder']
                else:
                    L = pinfo['sequence_length']
                    if len(scores) < L:
                        continue
                    valid_scores = np.array(scores[:L])
                    valid_true = pinfo['true_disorder']
                
                rmsd = float(np.sqrt(np.mean((valid_true - valid_scores) ** 2)))
                disorder_content = pinfo['disorder_content']
                
                all_rmsd.append(rmsd)
                
                if disorder_content > 0.9:
                    highly_disordered_rmsd.append(rmsd)
                elif disorder_content < 0.1:
                    highly_ordered_rmsd.append(rmsd)
            
            if len(all_rmsd) > 0:
                rmsd_data['all'][predictor] = np.array(all_rmsd)
                rmsd_data['highly_disordered'][predictor] = np.array(highly_disordered_rmsd)
                rmsd_data['highly_ordered'][predictor] = np.array(highly_ordered_rmsd)
                
        except Exception as e:
            print(f"Error calculating RMSD for {predictor}: {e}")
    
    # Plot three panels
    panel_titles = ['All proteins', 'Highly disordered proteins', 'Highly ordered proteins']
    panel_keys = ['all', 'highly_disordered', 'highly_ordered']
    
    # First pass: calculate global max for y-axis scaling
    global_max = 0
    plot_data = []
    
    for title, key in zip(panel_titles, panel_keys):
        data_dict = rmsd_data[key]
        
        # Sort by mean RMSD
        sorted_items = sorted(data_dict.items(), key=lambda x: np.mean(x[1]))
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Calculate means and standard errors
        means = [np.mean(v) for v in values]
        stds = [np.std(v) for v in values]
        
        # Update global max (mean + std for error bars)
        for m, s in zip(means, stds):
            global_max = max(global_max, m + s)
        
        # Assign colors
        colors = []
        for name in names:
            if 'pLD' in name:
                if 'AF2' in name:
                    colors.append(structure_colors['alphafold2'])
                elif 'AF3' in name:
                    colors.append(structure_colors['alphafold3'])
                else:
                    colors.append(structure_colors['esmfold'])
            else:
                colors.append(predictor_colors[name])
        
        plot_data.append({
            'names': names,
            'means': means,
            'stds': stds,
            'colors': colors,
            'title': title
        })
    
    # Second pass: plot with fixed unified y-axis (0.0 to 1.0)
    for ax, pdata in zip(axes, plot_data):
        # Create bar chart
        x_pos = np.arange(len(pdata['names']))
        bars = ax.bar(x_pos, pdata['means'], yerr=pdata['stds'], capsize=3, 
                     color=pdata['colors'], alpha=0.8, edgecolor='black', linewidth=0.5,
                     error_kw={'linewidth': 1, 'ecolor': 'black'})
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pdata['names'], rotation=90, ha='right', fontsize=9)
        ax.set_ylabel('RMSD', fontsize=11, fontweight='bold')
        ax.set_title(pdata['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0, 1.0])  # Fixed y-axis from 0.0 to 1.0
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"RMSD Comparison Summary ({dataset_type})")
    print(f"{'='*80}")
    
    for panel_name, key in zip(panel_titles, panel_keys):
        print(f"\n{panel_name}:")
        print(f"{'Method':<20} {'Mean RMSD':<12} {'Std Dev':<12} {'N proteins':<12}")
        print("-" * 80)
        
        data_dict = rmsd_data[key]
        sorted_items = sorted(data_dict.items(), key=lambda x: np.mean(x[1]))
        
        for name, values in sorted_items:
            mean_val = np.mean(values)
            std_val = np.std(values)
            n_proteins = len(values)
            print(f"{name:<20} {mean_val:<12.3f} {std_val:<12.3f} {n_proteins:<12}")
    
    return fig, axes

def plot_order_disorder_rate_distribution(pda, dataset_type='disprot_pdb', figsize=(14, 10)):
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    # Create figure with 2x3 grid (2 rows: TPR and TNR, 3 cols: models)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Structure prediction models
    structure_models = ['alphafold2', 'alphafold3', 'esmfold']
    structure_labels = {
        'alphafold2': 'AlphaFold2',
        'alphafold3': 'AlphaFold3',
        'esmfold': 'ESMFold'
    }
    structure_colors = {
        'alphafold2': '#E69F00',
        'alphafold3': '#D55E00',
        'esmfold': '#CC79A7'
    }
    
    # Store results for each model
    model_results = {}
    
    for model in structure_models:
        # Get optimal threshold for this model
        optimal_threshold, _ = pda.find_optimal_threshold(dataset_type, model)
        
        # Get evaluation data
        ev = pda.prepare_evaluation_data(dataset_type, model)
        protein_data = ev['protein_data']
        
        # Store per-protein TPR and TNR
        tpr_list = []
        tnr_list = []
        protein_ids = []
        
        for uid, pinfo in protein_data.items():
            y_true = pinfo['true_disorder']
            plddt = pinfo['model_plddt']
            
            # Convert to binary predictions using optimal threshold
            # Disordered (1) if pLDDT < threshold
            y_pred = 1 - (plddt >= optimal_threshold).astype(int)
            
            # Calculate TPR for this protein (disordered regions)
            disordered_mask = (y_true == 1)
            if disordered_mask.sum() > 0:
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                protein_tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            else:
                protein_tpr = np.nan
            
            # Calculate TNR for this protein (ordered regions)
            ordered_mask = (y_true == 0)
            if ordered_mask.sum() > 0:
                tn = ((y_true == 0) & (y_pred == 0)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                protein_tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            else:
                protein_tnr = np.nan
            
            # Only add if we have valid values
            if not np.isnan(protein_tpr):
                tpr_list.append(protein_tpr)
            if not np.isnan(protein_tnr):
                tnr_list.append(protein_tnr)
            
            protein_ids.append(uid)
        
        model_results[model] = {
            'tpr': np.array(tpr_list),
            'tnr': np.array(tnr_list),
            'threshold': optimal_threshold,
            'protein_ids': protein_ids
        }
    
    # Plot distributions
    for col_idx, model in enumerate(structure_models):
        results = model_results[model]
        color = structure_colors[model]
        label = structure_labels[model]
        
        # Top row: TPR distribution
        ax_tpr = axes[0, col_idx]
        if len(results['tpr']) > 0:
            ax_tpr.hist(results['tpr'], bins=30, color=color, alpha=0.7, 
                       edgecolor='black', linewidth=0.5)
            ax_tpr.axvline(np.mean(results['tpr']), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(results["tpr"]):.3f}')
            ax_tpr.axvline(np.median(results['tpr']), color='blue', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(results["tpr"]):.3f}')
        
        ax_tpr.set_xlabel('True Positive Rate (Predicted/True Disorder Region)', fontsize=11, fontweight='bold')
        ax_tpr.set_ylabel('Number of proteins', fontsize=11, fontweight='bold')
        ax_tpr.set_title(f'{label}\n(threshold={results["threshold"]})', 
                        fontsize=12, fontweight='bold')
        ax_tpr.set_xlim([0, 1])
        ax_tpr.legend(loc='upper left', fontsize=9)
        ax_tpr.grid(True, alpha=0.3)
        
        # Bottom row: TNR distribution
        ax_tnr = axes[1, col_idx]
        if len(results['tnr']) > 0:
            ax_tnr.hist(results['tnr'], bins=30, color=color, alpha=0.7, 
                       edgecolor='black', linewidth=0.5)
            ax_tnr.axvline(np.mean(results['tnr']), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(results["tnr"]):.3f}')
            ax_tnr.axvline(np.median(results['tnr']), color='blue', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(results["tnr"]):.3f}')
        
        ax_tnr.set_xlabel('True Negative Rate (Predicted/True Order Region)', fontsize=11, fontweight='bold')
        ax_tnr.set_ylabel('Number of proteins', fontsize=11, fontweight='bold')
        ax_tnr.legend(loc='upper left', fontsize=9)
        ax_tnr.grid(True, alpha=0.3)
        ax_tnr.set_xlim([0, 1])
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Per-Protein TPR and TNR Distribution Summary ({dataset_type})")
    print(f"{'='*80}")
    
    for model in structure_models:
        results = model_results[model]
        label = structure_labels[model]
        
        print(f"\n{label} (Threshold = {results['threshold']}):")
        print(f"{'Metric':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'N':<10}")
        print("-" * 80)
        
        if len(results['tpr']) > 0:
            print(f"{'TPR':<15} {np.mean(results['tpr']):<10.3f} {np.median(results['tpr']):<10.3f} "
                  f"{np.std(results['tpr']):<10.3f} {np.min(results['tpr']):<10.3f} "
                  f"{np.max(results['tpr']):<10.3f} {len(results['tpr']):<10}")
        
        if len(results['tnr']) > 0:
            print(f"{'TNR':<15} {np.mean(results['tnr']):<10.3f} {np.median(results['tnr']):<10.3f} "
                  f"{np.std(results['tnr']):<10.3f} {np.min(results['tnr']):<10.3f} "
                  f"{np.max(results['tnr']):<10.3f} {len(results['tnr']):<10}")
    
    return fig, axes, model_results


def plot_amino_acid_disorder_frequency(pda, dataset_type='disprot_pdb', figsize=(16, 8)):
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Standard amino acid order
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Structure prediction models
    structure_models = ['alphafold2', 'alphafold3', 'esmfold']
    structure_labels = {
        'alphafold2': 'AlphaFold2',
        'alphafold3': 'AlphaFold3',
        'esmfold': 'ESMFold'
    }
    
    # Initialize storage for amino acid disorder counts
    aa_disorder_counts = {aa: {'ground_truth': {'disordered': 0, 'total': 0}} for aa in amino_acids}
    for model in structure_models:
        for aa in amino_acids:
            aa_disorder_counts[aa][model] = {'disordered': 0, 'total': 0}
    
    # Process each model
    for model in structure_models:
        # Get optimal threshold for this model
        optimal_threshold, _ = pda.find_optimal_threshold(dataset_type, model)
        
        # Get evaluation data
        ev = pda.prepare_evaluation_data(dataset_type, model)
        protein_data = ev['protein_data']
        
        # Process each protein
        for uid, pinfo in protein_data.items():
            # Get sequence
            sequence = pda.combined_data[uid]['amino_acid_sequence']
            
            # Get ground truth and predictions
            y_true = pinfo['true_disorder']
            plddt = pinfo['model_plddt']
            y_pred = 1 - (plddt >= optimal_threshold).astype(int)
            
            # Get mask if applicable
            mask = pinfo['mask']
            
            # Process each residue
            if dataset_type == 'disprot_pdb' and mask is not None:
                # Use mask to select valid positions
                for idx in mask:
                    idx = int(idx)
                    if idx < len(sequence):
                        aa = sequence[idx].upper()
                        if aa in amino_acids:
                            # Find position in masked arrays
                            mask_pos = np.where(mask == idx)[0]
                            if len(mask_pos) > 0:
                                mask_pos = mask_pos[0]
                                
                                # Ground truth (only count once per model loop, use first model)
                                if model == structure_models[0]:
                                    aa_disorder_counts[aa]['ground_truth']['total'] += 1
                                    if y_true[mask_pos] == 1:
                                        aa_disorder_counts[aa]['ground_truth']['disordered'] += 1
                                
                                # Model prediction
                                aa_disorder_counts[aa][model]['total'] += 1
                                if y_pred[mask_pos] == 1:
                                    aa_disorder_counts[aa][model]['disordered'] += 1
            else:
                # No mask, use full sequence
                for idx in range(len(y_true)):
                    if idx < len(sequence):
                        aa = sequence[idx].upper()
                        if aa in amino_acids:
                            # Ground truth (only count once per model loop)
                            if model == structure_models[0]:
                                aa_disorder_counts[aa]['ground_truth']['total'] += 1
                                if y_true[idx] == 1:
                                    aa_disorder_counts[aa]['ground_truth']['disordered'] += 1
                            
                            # Model prediction
                            aa_disorder_counts[aa][model]['total'] += 1
                            if y_pred[idx] == 1:
                                aa_disorder_counts[aa][model]['disordered'] += 1
    
    # Calculate frequencies
    ground_truth_freq = []
    af2_freq = []
    af3_freq = []
    esm_freq = []
    
    for aa in amino_acids:
        # Ground truth frequency
        if aa_disorder_counts[aa]['ground_truth']['total'] > 0:
            gt_freq = aa_disorder_counts[aa]['ground_truth']['disordered'] / aa_disorder_counts[aa]['ground_truth']['total']
        else:
            gt_freq = 0
        ground_truth_freq.append(gt_freq)
        
        # AlphaFold2 frequency
        if aa_disorder_counts[aa]['alphafold2']['total'] > 0:
            af2_f = aa_disorder_counts[aa]['alphafold2']['disordered'] / aa_disorder_counts[aa]['alphafold2']['total']
        else:
            af2_f = 0
        af2_freq.append(af2_f)
        
        # AlphaFold3 frequency
        if aa_disorder_counts[aa]['alphafold3']['total'] > 0:
            af3_f = aa_disorder_counts[aa]['alphafold3']['disordered'] / aa_disorder_counts[aa]['alphafold3']['total']
        else:
            af3_f = 0
        af3_freq.append(af3_f)
        
        # ESMFold frequency
        if aa_disorder_counts[aa]['esmfold']['total'] > 0:
            esm_f = aa_disorder_counts[aa]['esmfold']['disordered'] / aa_disorder_counts[aa]['esmfold']['total']
        else:
            esm_f = 0
        esm_freq.append(esm_f)
    
    # Calculate correlations with ground truth
    gt_array = np.array(ground_truth_freq)
    af2_corr = np.corrcoef(gt_array, af2_freq)[0, 1]
    af3_corr = np.corrcoef(gt_array, af3_freq)[0, 1]
    esm_corr = np.corrcoef(gt_array, esm_freq)[0, 1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(amino_acids))
    width = 0.2
    
    # Plot bars
    bars1 = ax.bar(x - 1.5*width, ground_truth_freq, width, label='Ground Truth', 
                   color='#2E86AB', alpha=0.9, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, af2_freq, width, 
                   label=f'AlphaFold2 (r={af2_corr:.3f})', 
                   color='#E69F00', alpha=0.9, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, af3_freq, width, 
                   label=f'AlphaFold3 (r={af3_corr:.3f})', 
                   color='#D55E00', alpha=0.9, edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, esm_freq, width, 
                   label=f'ESMFold (r={esm_corr:.3f})', 
                   color='#CC79A7', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Amino Acid', fontsize=14, fontweight='bold')
    ax.set_ylabel('Disorder Frequency', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(amino_acids, fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{'='*100}")
    print(f"Amino Acid Disorder Frequency Summary ({dataset_type})")
    print(f"{'='*100}")
    print(f"{'AA':<4} {'Ground Truth':<15} {'AlphaFold2':<15} {'AlphaFold3':<15} {'ESMFold':<15} "
          f"{'GT Count':<12} {'Total':<10}")
    print("-" * 100)
    
    for i, aa in enumerate(amino_acids):
        gt_count = aa_disorder_counts[aa]['ground_truth']['disordered']
        total_count = aa_disorder_counts[aa]['ground_truth']['total']
        print(f"{aa:<4} {ground_truth_freq[i]:<15.3f} {af2_freq[i]:<15.3f} {af3_freq[i]:<15.3f} "
              f"{esm_freq[i]:<15.3f} {gt_count:<12} {total_count:<10}")
    
    # Calculate correlation between ground truth and each model
    print(f"\n{'='*60}")
    print("Correlation with Ground Truth:")
    print(f"{'='*60}")
    print(f"AlphaFold2: {af2_corr:.3f}")
    print(f"AlphaFold3: {af3_corr:.3f}")
    print(f"ESMFold:    {esm_corr:.3f}")
    
    return fig, ax, aa_disorder_counts

# ---------- paths ----------
OTHER_DATA_PATH    = "dataset/combined.dat"     # 28-line blocks per entry (your 'other' file)
NEWMODEL_DATA_PATH = "dataset/newmodel.fasta"   # 4-line blocks per entry (header, seq, AF pLDDT, ESM pLDDT)

def main():
    dataset_type = "disprot_pdb"
    model = "alphafold3"

    pda = ProteinDataAnalysis(
        other_data_path=OTHER_DATA_PATH,
        newmodel_data_path=NEWMODEL_DATA_PATH,
    )

    n_common = pda.load_and_combine_data()
    print(f"Loaded {n_common} overlapping proteins")

    # Generate report for any model
    report = pda.generate_performance_report(dataset_type, model='alphafold3')
    
    # Generate comparison figure (ROC and PR curves)
    print("\n" + "="*60)
    print("Generating ROC and PR curve comparison...")
    print("="*60)
    fig, axes = plot_structure_model_comparison(pda, dataset_type='disprot_pdb')
    plt.savefig('results/model_comparison.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: model_comparison.pdf")

    # Generate MCC comparison
    print("\n" + "="*60)
    print("Generating MCC comparison...")
    print("="*60)
    fig_mcc, ax_mcc = plot_mcc_comparison(pda, dataset_type='disprot_pdb')
    plt.savefig('results/mcc_comparison.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: mcc_comparison.pdf")

    # Generate RMSD comparison  
    print("\n" + "="*60)
    print("Generating RMSD comparison...")
    print("="*60)
    fig_rmsd, axes_rmsd = plot_rmsd_comparison(pda, dataset_type='disprot_pdb')
    plt.savefig('results/rmsd_comparison.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: rmsd_comparison.pdf")

    # Generate per-protein TPR/TNR distribution
    print("\n" + "="*60)
    print("Generating per-protein TPR and TNR distributions...")
    print("="*60)
    fig_rates, axes_rates, model_results = plot_order_disorder_rate_distribution(pda, dataset_type='disprot_pdb')
    plt.savefig('results/order_disorder_rate_distribution.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: order_disorder_rate_distribution.pdf")
    
    # Optional: Save the per-protein results to CSV files
    for model_name, results in model_results.items():
        df_results = pd.DataFrame({
            'protein_id': results['protein_ids'][:len(results['tpr'])],
            'TPR': results['tpr'],
        })
        # Add TNR if available
        if len(results['tnr']) == len(results['tpr']):
            df_results['TNR'] = results['tnr']
        
        output_file = f'results/order_disorder_rate_distribution_{model_name}.csv'
        df_results.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    # Generate per-protein TPR/TNR distribution
    print("\n" + "="*60)
    print("Generating amino acid disorder frequency...")
    print("="*60)
    fig_rates, axes_rates, model_results = plot_amino_acid_disorder_frequency(pda, dataset_type='disprot_pdb')
    plt.savefig('results/amino_acid_disorder_frequency.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: amino_acid_disorder_frequency.pdf")

    # Generate threshold table
    print("\n" + "="*60)
    print(f"Generating pLDDT threshold table (60-80) for {model}...")
    print("="*60)
    table_60_80 = pda.make_pld_table(dataset_type, model, thr_start=60, thr_end=80, step=2)
    table_60_80.to_csv(f"results/pld60_80_table_{model}.csv", index=False)
    print(f"Saved: pld60_80_table_{model}.csv")
    
    print("\n" + "="*60)
    print("All analyses complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison.pdf (ROC and PR curves)")
    print("  - mcc_comparison.pdf (MCC bar chart)")
    print("  - rmsd_comparison.pdf (RMSD by protein class)")
    print("  - order_disorder_rate_distribution.pdf (TPR/TNR distributions)")
    print("  - order_disorder_rate_distribution_*.csv (Per-protein rate data)")
    print(f"  - pld60_80_table_{model}.csv (Threshold performance table)")
    

if __name__ == "__main__":
    main()