import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === BASE FOLDERS ===
BASE_DIR = "dataset"
RESULTS_DIR = "dockq_results"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

AF2_version = "v3"
AF2_CSV = os.path.join(BASE_DIR, f"af2{AF2_version}_dockq_data.csv")
AF3_CSV = os.path.join(BASE_DIR, "af3_dockq_data.csv")

# === CAPRI CLASSIFICATION ===
def classify_dockq_to_capri(dockq_score):
    if pd.isna(dockq_score):
        return None
    if dockq_score >= 0.80:
        return 'High'
    elif dockq_score >= 0.49:
        return 'Medium'
    elif dockq_score >= 0.23:
        return 'Acceptable'
    else:
        return 'Incorrect'


def add_capri_classification(df):
    """
    Add CAPRI classification column to dataframe.
    
    Args:
        df: DataFrame with 'dockq' column
    
    Returns:
        DataFrame with added 'capri_class' column
    """
    df = df.copy()
    df['capri_class'] = df['dockq'].apply(classify_dockq_to_capri)
    return df


def calculate_capri_statistics(df, model_name):
    """
    Calculate CAPRI success rates and class distribution.
    
    Args:
        df: DataFrame with 'capri_class' column
        model_name: Name of the model (e.g., 'AF2', 'AF3')
    
    Returns:
        dict: Statistics including success rates and class counts
    """
    total = len(df[df['capri_class'].notna()])
    
    if total == 0:
        return None
    
    class_counts = df['capri_class'].value_counts()
    
    # Success rates
    acceptable_or_better = df[df['capri_class'].isin(['High', 'Medium', 'Acceptable'])].shape[0]
    medium_or_better = df[df['capri_class'].isin(['High', 'Medium'])].shape[0]
    high = df[df['capri_class'] == 'High'].shape[0]
    
    stats = {
        'model': model_name,
        'total': total,
        'high_count': class_counts.get('High', 0),
        'medium_count': class_counts.get('Medium', 0),
        'acceptable_count': class_counts.get('Acceptable', 0),
        'incorrect_count': class_counts.get('Incorrect', 0),
        'success_rate_acceptable': (acceptable_or_better / total) * 100,
        'success_rate_medium': (medium_or_better / total) * 100,
        'success_rate_high': (high / total) * 100,
    }
    
    return stats


def plot_capri_merged_analysis(df_af2, df_af3, stats_af2, stats_af3):
    """
    Create a single merged figure with CAPRI comparison analyses (2 subplots).
    
    Args:
        df_af2: AF2 DataFrame with CAPRI classification
        df_af3: AF3 DataFrame with CAPRI classification
        stats_af2: AF2 CAPRI statistics dictionary
        stats_af3: AF3 CAPRI statistics dictionary
    """
    fig = plt.figure(figsize=(16, 6))
    
    classes = ['High', 'Medium', 'Acceptable', 'Incorrect']
    
    # Get class counts
    af2_counts = df_af2['capri_class'].value_counts()
    af3_counts = df_af3['capri_class'].value_counts()
    af2_values = [af2_counts.get(c, 0) for c in classes]
    af3_values = [af3_counts.get(c, 0) for c in classes]
    
    # === SUBPLOT 1: CAPRI Classification Comparison (Left) ===
    ax1 = plt.subplot(1, 2, 1)
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, af2_values, width, label=f'AF2 Multimer {AF2_version}', 
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, af3_values, width, label='AF3', 
                   color='indianred', edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('CAPRI Class', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax1.set_title('CAPRI Classification Comparison: AF2 vs AF3', fontsize=15, fontweight='bold', y=1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=11)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # === SUBPLOT 2: Success Rates Comparison (Right) ===
    ax2 = plt.subplot(1, 2, 2)
    
    categories = ['≥ Acceptable', '≥ Medium', 'High']
    af2_rates = [
        stats_af2['success_rate_acceptable'],
        stats_af2['success_rate_medium'],
        stats_af2['success_rate_high']
    ]
    af3_rates = [
        stats_af3['success_rate_acceptable'],
        stats_af3['success_rate_medium'],
        stats_af3['success_rate_high']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, af2_rates, width, label=f'AF2 Multimer {AF2_version}', 
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, af3_rates, width, label='AF3', 
                   color='indianred', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('CAPRI Threshold', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('CAPRI Success Rates: AF2 vs AF3', fontsize=15, fontweight='bold', y=1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 110)
    
    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    axes_list = [ax1, ax2]
    for i, ax in enumerate(axes_list):
        label = chr(65 + i)  # A, B
        ax.text(-0.05, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
        
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'af2{AF2_version}_af3_capri_merged_analysis.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Merged CAPRI analysis plot saved to: {out_path}")


def load_data(af2_path, af3_path):
    """Load AF2 and AF3 CSV files."""
    print("Reading CSV files...")
    df_af2 = pd.read_csv(af2_path)
    df_af3 = pd.read_csv(af3_path)
    
    print(f"\nAF2 Total entries: {len(df_af2)}")
    print(f"AF2 Unique PDB IDs: {df_af2['pdb_id'].nunique()}")
    print(f"\nAF3 Total entries: {len(df_af3)}")
    print(f"AF3 Unique PDB IDs: {df_af3['pdb_id'].nunique()}")
    
    return df_af2, df_af3


def calculate_statistics(df, metrics):
    """Calculate statistics for given metrics."""
    stats = {}
    for metric in metrics:
        if metric in df.columns:
            data = df[metric].dropna()
            stats[metric] = {
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'min': data.min(),
                'max': data.max(),
                'count': len(data)
            }
    return stats


def print_statistics(stats, model_name):
    """Print statistics in formatted table."""
    print(f"\n{model_name} STATISTICS:")
    print("-" * 70)
    for metric, values in stats.items():
        print(f"{metric.upper()}")
        print(f"  Mean:   {values['mean']:.4f}")
        print(f"  Std:    {values['std']:.4f}")
        print(f"  Median: {values['median']:.4f}")
        print(f"  Min:    {values['min']:.4f}")
        print(f"  Max:    {values['max']:.4f}")
        print(f"  Count:  {values['count']}")
        print()

def plot_distributions(df_af2, df_af3, metrics):
    """Create distribution plots for all metrics."""
    print("Creating distribution plots...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # AF3 (top row)
    for idx, metric in enumerate(metrics):
        if metric in df_af3.columns:
            ax = axes[0, idx]
            data = df_af3[metric].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='indianred')
            mean_val, median_val = data.mean(), data.median()
            ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('Frequency')
            ax.set_title(f'AF3 - {metric.upper()} (n={len(data)})', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            # Add subplot label
            ax.text(-0.1, 1.05, chr(65 + idx), transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='top')
    
    # AF2 (bottom row)
    for idx, metric in enumerate(metrics):
        if metric in df_af2.columns:
            ax = axes[1, idx]
            data = df_af2[metric].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            mean_val, median_val = data.mean(), data.median()
            ax.axvline(mean_val, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('Frequency')
            ax.set_title(f'AF2 - {metric.upper()} (n={len(data)})', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            # Add subplot label
            ax.text(-0.1, 1.05, chr(65 + idx + 5), transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='top')
    
    
    plt.tight_layout()
    dist_plot_path = os.path.join(RESULTS_DIR, f'af2{AF2_version}_af3_distributions.pdf')
    plt.savefig(dist_plot_path, format='pdf', bbox_inches='tight')
    print(f"Distribution plots saved to: {dist_plot_path}")


def plot_metric_comparisons(df_af2, df_af3, metrics):
    """Create comparison bar plots for each metric."""
    print("\nCreating comparison bar graphs for each metric...")
    
    common_pdbs = sorted(set(df_af2['pdb_id']) & set(df_af3['pdb_id']))
    print(f"\nCommon PDB IDs: {len(common_pdbs)}")
    
    for metric in metrics:
        if metric in df_af2.columns and metric in df_af3.columns:
            fig, ax = plt.subplots(figsize=(24, 8))
            
            af2_data, af3_data, pdb_labels = [], [], []
            
            for pdb in common_pdbs:
                af2_val = df_af2[df_af2['pdb_id'] == pdb][metric].values
                af3_val = df_af3[df_af3['pdb_id'] == pdb][metric].values
                if len(af2_val) > 0 and len(af3_val) > 0:
                    af2_data.append(af2_val[0])
                    af3_data.append(af3_val[0])
                    pdb_labels.append(pdb)
            
            sorted_indices = np.argsort(af2_data)[::-1]
            af2_sorted = [af2_data[i] for i in sorted_indices]
            af3_sorted = [af3_data[i] for i in sorted_indices]
            pdb_sorted = [pdb_labels[i] for i in sorted_indices]
            
            # Calculate absolute delta for highlighting
            deltas = [abs(af3_sorted[i] - af2_sorted[i]) for i in range(len(af2_sorted))]
            
            # Find indices of top 5 highest absolute deltas
            top5_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)[:10]
            
            # Separate top 5 by who wins (AF3 > AF2 or AF2 > AF3)
            top5_af3_wins = [i for i in top5_indices if af3_sorted[i] > af2_sorted[i]]
            top5_af2_wins = [i for i in top5_indices if af2_sorted[i] > af3_sorted[i]]
            
            # Calculate means and medians
            af2_mean = np.mean(af2_sorted)
            af3_mean = np.mean(af3_sorted)
            af2_median = np.median(af2_sorted)
            af3_median = np.median(af3_sorted)
            
            x = np.arange(len(pdb_sorted))
            width = 0.5  # Increased from 0.35 to 0.4
            
            ax.bar(x - width/2, af2_sorted, width, label=f'AF2 Multimer {AF2_version}', color='steelblue', edgecolor='black')
            ax.bar(x + width/2, af3_sorted, width, label='AF3', color='indianred', edgecolor='black')
            
            # Add mean lines
            ax.axhline(y=af2_mean, color='steelblue', linestyle='--', linewidth=2, 
                      label=f'AF2 Mean: {af2_mean:.3f}', alpha=0.7)
            ax.axhline(y=af3_mean, color='indianred', linestyle='--', linewidth=2, 
                      label=f'AF3 Mean: {af3_mean:.3f}', alpha=0.7)
            
            # Add median lines
            ax.axhline(y=af2_median, color='steelblue', linestyle=':', linewidth=2, 
                      label=f'AF2 Median: {af2_median:.3f}', alpha=0.7)
            ax.axhline(y=af3_median, color='indianred', linestyle=':', linewidth=2, 
                      label=f'AF3 Median: {af3_median:.3f}', alpha=0.7)
            
            ax.set_xlabel('PDB ID', fontsize=12)
            ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
            ax.set_xticks(x)
            
            # Create x-tick labels with highlighting for top 5
            xticklabels = []
            for i, pdb in enumerate(pdb_sorted):
                xticklabels.append(pdb)
            
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=12)
            
            # Highlight top 5 x-tick labels
            # Green for AF3 > AF2, Yellow for AF2 > AF3
            for i, label in enumerate(ax.get_xticklabels()):
                if i in top5_af3_wins:
                    label.set_color('red')
                    label.set_weight('bold')
                    label.set_fontsize(9)
                elif i in top5_af2_wins:
                    label.set_color('blue')  # Using 'gold' for better visibility than 'yellow'
                    label.set_weight('bold')
                    label.set_fontsize(9)
            
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Reduce space at the beginning and end
            ax.margins(x=0.01)  # Reduce margins to 1% on each side
            
            plt.tight_layout()
            out_path = os.path.join(RESULTS_DIR, f'af2{AF2_version}_af3_{metric}_comparison.pdf')
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            print(f"  {metric.upper()} comparison saved to: {out_path}")

def save_summary_table(stats_af2, stats_af3):
    """Save summary statistics to CSV."""
    summary_af2 = pd.DataFrame(stats_af2).T[['count', 'mean', 'std', 'median', 'min', 'max']]
    summary_af2['model'] = 'AF2'
    summary_af3 = pd.DataFrame(stats_af3).T[['count', 'mean', 'std', 'median', 'min', 'max']]
    summary_af3['model'] = 'AF3'
    
    summary_combined = pd.concat([summary_af2, summary_af3]).reset_index().rename(columns={'index': 'metric'})
    print("\n" + "="*70)
    print("COMPARISON SUMMARY TABLE")
    print("="*70 + "\n")
    print(summary_combined.to_string(index=False))
    
    summary_csv_path = os.path.join(RESULTS_DIR, f'af2{AF2_version}_af3_comparison_summary.csv')
    summary_combined.to_csv(summary_csv_path, index=False)
    print(f"\nComparison summary saved to: {summary_csv_path}")


def main():
    """Main analysis pipeline."""
    metrics = ['dockq', 'fnat', 'fnonnat', 'irms', 'lrms']
    
    # Load data
    df_af2, df_af3 = load_data(AF2_CSV, AF3_CSV)
    
    # Calculate basic statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70 + "\n")
    
    stats_af2 = calculate_statistics(df_af2, metrics)
    print_statistics(stats_af2, "AF2")
    
    stats_af3 = calculate_statistics(df_af3, metrics)
    print_statistics(stats_af3, "AF3")
    
    # Create distribution plots
    plot_distributions(df_af2, df_af3, metrics)
    
    # Create metric comparison plots
    plot_metric_comparisons(df_af2, df_af3, metrics)
    
    # Save summary table
    save_summary_table(stats_af2, stats_af3)
    
    # === CAPRI ANALYSIS (MERGED) ===
    print("\n" + "="*70)
    print("CAPRI CLASSIFICATION ANALYSIS")
    print("="*70 + "\n")
    
    # Add CAPRI classification
    df_af2_capri = add_capri_classification(df_af2)
    df_af3_capri = add_capri_classification(df_af3)
    
    # Calculate CAPRI statistics
    capri_stats_af2 = calculate_capri_statistics(df_af2_capri, 'AF2')
    capri_stats_af3 = calculate_capri_statistics(df_af3_capri, 'AF3')
    
    # Print CAPRI statistics
    print("AF2 CAPRI STATISTICS:")
    print("-" * 70)
    print(f"Total predictions: {capri_stats_af2['total']}")
    print(f"\nClass Distribution:")
    print(f"  High:       {capri_stats_af2['high_count']} ({capri_stats_af2['high_count']/capri_stats_af2['total']*100:.1f}%)")
    print(f"  Medium:     {capri_stats_af2['medium_count']} ({capri_stats_af2['medium_count']/capri_stats_af2['total']*100:.1f}%)")
    print(f"  Acceptable: {capri_stats_af2['acceptable_count']} ({capri_stats_af2['acceptable_count']/capri_stats_af2['total']*100:.1f}%)")
    print(f"  Incorrect:  {capri_stats_af2['incorrect_count']} ({capri_stats_af2['incorrect_count']/capri_stats_af2['total']*100:.1f}%)")
    print(f"\nSuccess Rates:")
    print(f"  ≥ Acceptable: {capri_stats_af2['success_rate_acceptable']:.2f}%")
    print(f"  ≥ Medium:     {capri_stats_af2['success_rate_medium']:.2f}%")
    print(f"  High:         {capri_stats_af2['success_rate_high']:.2f}%")
    
    print("\n\nAF3 CAPRI STATISTICS:")
    print("-" * 70)
    print(f"Total predictions: {capri_stats_af3['total']}")
    print(f"\nClass Distribution:")
    print(f"  High:       {capri_stats_af3['high_count']} ({capri_stats_af3['high_count']/capri_stats_af3['total']*100:.1f}%)")
    print(f"  Medium:     {capri_stats_af3['medium_count']} ({capri_stats_af3['medium_count']/capri_stats_af3['total']*100:.1f}%)")
    print(f"  Acceptable: {capri_stats_af3['acceptable_count']} ({capri_stats_af3['acceptable_count']/capri_stats_af3['total']*100:.1f}%)")
    print(f"  Incorrect:  {capri_stats_af3['incorrect_count']} ({capri_stats_af3['incorrect_count']/capri_stats_af3['total']*100:.1f}%)")
    print(f"\nSuccess Rates:")
    print(f"  ≥ Acceptable: {capri_stats_af3['success_rate_acceptable']:.2f}%")
    print(f"  ≥ Medium:     {capri_stats_af3['success_rate_medium']:.2f}%")
    print(f"  High:         {capri_stats_af3['success_rate_high']:.2f}%")
    
    # Create merged CAPRI visualization
    print("\n\nCreating merged CAPRI visualization...")
    plot_capri_merged_analysis(df_af2_capri, df_af3_capri, capri_stats_af2, capri_stats_af3)
    
    # Save CAPRI statistics to CSV
    capri_summary = pd.DataFrame([capri_stats_af2, capri_stats_af3])
    capri_csv_path = os.path.join(RESULTS_DIR, f'af2{AF2_version}_af3_capri_statistics_summary.csv')
    capri_summary.to_csv(capri_csv_path, index=False)
    print(f"CAPRI statistics saved to: {capri_csv_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()