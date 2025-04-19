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

def explore_clinical_factors():
    """Explore how clinical factors affect phenotype creation."""
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
    print(f"Clinical factors: {clinical_cols}")
    
    # 1. Create phenotypes with and without accounting for clinical factors
    creator_with = CarePhenotypeCreator(
        data=data,
        clinical_factors=clinical_cols,
        n_clusters=3,
        random_state=42
    )
    
    creator_without = CarePhenotypeCreator(
        data=data,
        clinical_factors=[],  # No clinical factors
        n_clusters=3,
        random_state=42
    )
    
    # Get phenotype labels
    labels_with = creator_with.create_phenotype_labels()
    labels_without = creator_without.create_phenotype_labels()
    
    # Add labels to the data
    data['phenotype_with_clinical'] = labels_with
    data['phenotype_without_clinical'] = labels_without
    
    # 2. Compare the phenotype assignments
    contingency = pd.crosstab(
        data['phenotype_with_clinical'], 
        data['phenotype_without_clinical'],
        rownames=['With Clinical'],
        colnames=['Without Clinical']
    )
    
    print("\nContingency table of phenotype assignments:")
    print(contingency)
    
    # 3. Analyze clinical separation
    separation_with = creator_with.analyze_clinical_separation()
    
    print("\nClinical separation analysis:")
    for factor, metrics in separation_with.items():
        print(f"{factor}: F-statistic = {metrics['f_statistic']:.2f}, p-value = {metrics['p_value']:.4f}")
    
    # 4. Visualize phenotypes relative to clinical factors
    fig, axes = plt.subplots(len(clinical_cols), 2, figsize=(15, 5*len(clinical_cols)))
    
    for i, clinical_col in enumerate(clinical_cols):
        # With clinical factors
        sns.boxplot(
            x='phenotype_with_clinical',
            y=clinical_col,
            data=data,
            ax=axes[i, 0]
        )
        axes[i, 0].set_title(f'{clinical_col} by Phenotype (With Clinical)')
        
        # Without clinical factors
        sns.boxplot(
            x='phenotype_without_clinical',
            y=clinical_col,
            data=data,
            ax=axes[i, 1]
        )
        axes[i, 1].set_title(f'{clinical_col} by Phenotype (Without Clinical)')
    
    plt.tight_layout()
    plt.savefig('clinical_factor_effect.png')
    print("\nSaved clinical factor visualization to clinical_factor_effect.png")
    
    # 5. Analyze unexplained variation
    variation_with = creator_with.analyze_unexplained_variation()
    variation_without = creator_without.analyze_unexplained_variation()
    
    print("\nUnexplained variation analysis:")
    print("\nWith clinical factors:")
    for phenotype, metrics in variation_with.items():
        if phenotype != 'overall':
            print(f"Phenotype {phenotype}: Unexplained ratio = {metrics['unexplained_ratio']:.2f}")
    
    print("\nWithout clinical factors:")
    for phenotype, metrics in variation_without.items():
        if phenotype != 'overall':
            print(f"Phenotype {phenotype}: Unexplained ratio = {metrics['unexplained_ratio']:.2f}")
    
    # 6. Visualize unexplained variation
    unexplained_with = [metrics['unexplained_ratio'] for phenotype, metrics in variation_with.items() 
                       if phenotype != 'overall']
    unexplained_without = [metrics['unexplained_ratio'] for phenotype, metrics in variation_without.items() 
                          if phenotype != 'overall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(unexplained_with))
    
    ax.bar(x - width/2, unexplained_with, width, label='With Clinical Factors')
    ax.bar(x + width/2, unexplained_without, width, label='Without Clinical Factors')
    
    ax.set_ylabel('Unexplained Variation Ratio')
    ax.set_xlabel('Phenotype')
    ax.set_title('Unexplained Variation by Phenotype')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Phenotype {i}' for i in range(len(unexplained_with))])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('unexplained_variation.png')
    print("\nSaved unexplained variation visualization to unexplained_variation.png")

if __name__ == "__main__":
    explore_clinical_factors()