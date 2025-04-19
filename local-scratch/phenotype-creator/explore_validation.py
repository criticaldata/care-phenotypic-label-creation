import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator

def explore_validation():
    """Explore the validation metrics for phenotype creation."""
    # Load synthetic data
    data_path = Path(__file__).parent / 'synthetic_healthcare_data.csv'
    if not data_path.exists():
        print("Please run data_generator.py first to create synthetic data")
        return
    
    data = pd.read_csv(data_path)
    
    # Identify clinical factors and care patterns
    clinical_cols = [col for col in data.columns if col.startswith('clinical_factor')]
    care_pattern_cols = [col for col in data.columns if col.startswith('care_pattern')]
    
    print(f"Loaded data with {len(data)} records")
    
    # Create CarePhenotypeCreator
    creator = CarePhenotypeCreator(
        data=data,
        clinical_factors=clinical_cols,
        n_clusters=3,
        random_state=42
    )
    
    # Get phenotype labels
    labels = creator.create_phenotype_labels()
    
    # Validate phenotypes
    validation_metrics = ['clinical_separation', 'pattern_consistency', 'unexplained_variation']
    validation_results = creator.validate_phenotypes(labels, validation_metrics)
    
    # 1. Analyze clinical separation
    print("\n1. Clinical Separation Analysis:")
    clinical_separation = validation_results['clinical_separation']
    
    for factor, metrics in clinical_separation.items():
        f_stat = metrics['f_statistic']
        p_value = metrics['p_value']
        print(f"{factor}: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")
    
    # Visualize clinical separation
    f_stats = [metrics['f_statistic'] for factor, metrics in clinical_separation.items()]
    p_values = [metrics['p_value'] for factor, metrics in clinical_separation.items()]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    x = range(len(clinical_separation))
    ax1.bar(x, f_stats, color='blue', alpha=0.6, label='F-statistic')
    ax2.plot(x, p_values, 'ro-', label='p-value')
    
    ax1.set_xlabel('Clinical Factor')
    ax1.set_ylabel('F-statistic', color='blue')
    ax2.set_ylabel('p-value', color='red')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(clinical_separation.keys()), rotation=45)
    
    # Add significance threshold line
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('clinical_separation.png')
    print("\nSaved clinical separation visualization to clinical_separation.png")
    
    # 2. Analyze pattern consistency
    print("\n2. Pattern Consistency Analysis:")
    pattern_consistency = validation_results['pattern_consistency']
    
    for phenotype, metrics in pattern_consistency.items():
        if phenotype != 'overall':
            mean_corr = metrics['mean_correlation']
            stability = metrics['pattern_stability']
            print(f"Phenotype {phenotype}: Mean correlation = {mean_corr:.2f}, Stability = {stability:.2f}")
    
    print(f"\nOverall pattern correlation: {pattern_consistency['overall']['mean_pattern_correlation']:.2f}")
    print(f"Overall pattern stability: {pattern_consistency['overall']['mean_pattern_stability']:.2f}")
    
    # Visualize pattern consistency
    phenotypes = [p for p in pattern_consistency.keys() if p != 'overall']
    correlations = [metrics['mean_correlation'] for p, metrics in pattern_consistency.items() if p != 'overall']
    stabilities = [metrics['pattern_stability'] for p, metrics in pattern_consistency.items() if p != 'overall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(phenotypes))
    
    ax.bar(x - width/2, correlations, width, label='Mean Correlation')
    ax.bar(x + width/2, stabilities, width, label='Pattern Stability')
    
    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Phenotype')
    ax.set_title('Pattern Consistency by Phenotype')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Phenotype {p}' for p in phenotypes])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('pattern_consistency.png')
    print("\nSaved pattern consistency visualization to pattern_consistency.png")
    
    # 3. Analyze unexplained variation
    print("\n3. Unexplained Variation Analysis:")
    unexplained_variation = validation_results['unexplained_variation']
    
    for phenotype, metrics in unexplained_variation.items():
        if phenotype != 'overall':
            total_var = metrics['total_variance']
            explained_var = metrics['explained_variance']
            unexplained_var = metrics['unexplained_variance']
            unexplained_ratio = metrics['unexplained_ratio']
            
            print(f"Phenotype {phenotype}:")
            print(f"  Total variance: {total_var:.2f}")
            print(f"  Explained variance: {explained_var:.2f}")
            print(f"  Unexplained variance: {unexplained_var:.2f}")
            print(f"  Unexplained ratio: {unexplained_ratio:.2f}")
    
    print(f"\nMean unexplained ratio: {unexplained_variation['overall']['mean_unexplained_ratio']:.2f}")
    
    # Visualize unexplained variation
    phenotypes = [p for p in unexplained_variation.keys() if p != 'overall']
    total_vars = [metrics['total_variance'] for p, metrics in unexplained_variation.items() if p != 'overall']
    explained_vars = [metrics['explained_variance'] for p, metrics in unexplained_variation.items() if p != 'overall']
    unexplained_vars = [metrics['unexplained_variance'] for p, metrics in unexplained_variation.items() if p != 'overall']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    x = np.arange(len(phenotypes))
    
    ax.bar(x - width, total_vars, width, label='Total Variance')
    ax.bar(x, explained_vars, width, label='Explained Variance')
    ax.bar(x + width, unexplained_vars, width, label='Unexplained Variance')
    
    ax.set_ylabel('Variance')
    ax.set_xlabel('Phenotype')
    ax.set_title('Variance Components by Phenotype')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Phenotype {p}' for p in phenotypes])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('variance_components.png')
    print("\nSaved variance components visualization to variance_components.png")

if __name__ == "__main__":
    explore_validation()