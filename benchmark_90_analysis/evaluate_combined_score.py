import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def calculate_combined_score(data):
    """Calculate 0.8 * iPTM + 0.2 * PTM"""
    iptm = data.get('iptm', 0)
    ptm = data.get('ptm', 0)
    return 0.8 * iptm + 0.2 * ptm

def load_scores_from_folder(folder_path):
    """Load all JSON files from a folder and extract scores"""
    scores = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return scores
    
    for json_file in sorted(folder.glob('*.json')):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                combined_score = calculate_combined_score(data)
                scores[json_file.stem] = {
                    'combined_score': combined_score
                }
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    return scores

def create_comparison_dataframe(af2_scores, af3_scores):
    """Create a pandas DataFrame for easier analysis"""
    common_entries = set(af2_scores.keys()) & set(af3_scores.keys())
    
    data = []
    for entry in sorted(common_entries):
        data.append({
            'AF2_Combined': af2_scores[entry]['combined_score'],
            'AF3_Combined': af3_scores[entry]['combined_score']
        })
    
    return pd.DataFrame(data)

def perform_statistical_test(df):
    """Perform Mann-Whitney U test"""
    statistic, p_value = mannwhitneyu(df['AF2_Combined'], df['AF3_Combined'], 
                                       alternative='two-sided')
    return statistic, p_value

def plot_distribution_comparison(df, p_value, output_dir='results'):
    """Create 1x3 distribution comparison figure with p-value"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # AF3 distribution
    axes[0].hist(df['AF3_Combined'], bins=20, color='indianred', edgecolor='black', alpha=0.7)
    af3_mean = df['AF3_Combined'].mean()
    af3_median = df['AF3_Combined'].median()
    axes[0].axvline(af3_mean, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {af3_mean:.3f}')
    axes[0].axvline(af3_median, color='darkgreen', linestyle='--', linewidth=2, label=f'Median: {af3_median:.3f}')
    axes[0].set_xlabel('Combined Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'AF3 (n={len(df)})', fontsize=13, fontweight='bold', y=1.05)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # AF2 distribution
    axes[1].hist(df['AF2_Combined'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    af2_mean = df['AF2_Combined'].mean()
    af2_median = df['AF2_Combined'].median()
    axes[1].axvline(af2_mean, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {af2_mean:.3f}')
    axes[1].axvline(af2_median, color='darkgreen', linestyle='--', linewidth=2, label=f'Median: {af2_median:.3f}')
    axes[1].set_xlabel('Combined Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'AF2 (n={len(df)})', fontsize=13, fontweight='bold', y=1.05)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Box plot comparison
    box_data = [df['AF2_Combined'], df['AF3_Combined']]
    bp = axes[2].boxplot(box_data, labels=['AF2', 'AF3'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('indianred')
    axes[2].set_ylabel('Combined Score', fontsize=12)
    axes[2].set_title('Box Plot Comparison', fontsize=13, fontweight='bold', y=1.05)
    axes[2].grid(axis='y', alpha=0.3)
    
    # Add p-value annotation
    if p_value < 0.001:
        p_text = 'p < 0.001'
    else:
        p_text = f'p = {p_value:.3f}'
    axes[2].text(0.5, 0.98, f'Mann-Whitney U\n{p_text}', 
                 transform=axes[2].transAxes,
                 fontsize=11, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    for i, ax in enumerate(axes):
        label = chr(65 + i)  # A, B, C
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_score_comparison.pdf', format='pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/combined_score_comparison.pdf")
    plt.close()

def main():
    # Define paths
    base_dir = "dataset"
    af2_folder = os.path.join(base_dir, "alphafold2_scores")
    af3_folder = os.path.join(base_dir, "alphafold3_scores")
    output_dir = "results"
    
    # Load scores
    print("Loading AlphaFold2 scores...")
    af2_scores = load_scores_from_folder(af2_folder)
    print(f"Loaded {len(af2_scores)} AF2 entries")
    
    print("Loading AlphaFold3 scores...")
    af3_scores = load_scores_from_folder(af3_folder)
    print(f"Loaded {len(af3_scores)} AF3 entries")
    
    if not af2_scores or not af3_scores:
        print("Error: Could not load scores from one or both folders")
        return
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(af2_scores, af3_scores)
    
    if df.empty:
        print("No common entries found between AF2 and AF3")
        return
    
    print(f"\nComparing {len(df)} common entries...")
    
    # Perform statistical test
    statistic, p_value = perform_statistical_test(df)
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("  Result: Statistically significant difference (p < 0.05)")
    else:
        print("  Result: No statistically significant difference (p >= 0.05)")
    
    # Generate plot
    plot_distribution_comparison(df, p_value, output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()