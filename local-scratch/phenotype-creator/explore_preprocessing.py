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

def explore_preprocessing():
    """Explore the preprocessing functionality of CarePhenotypeCreator."""
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
    print(f"Care patterns: {care_pattern_cols}")
    
    # Initialize CarePhenotypeCreator
    creator = CarePhenotypeCreator(
        data=data,
        clinical_factors=clinical_cols,
        n_clusters=3,
        random_state=42
    )
    
    # Explore preprocessing with different parameters
    print("\n1. Basic preprocessing")
    preprocessed_data = creator.preprocess_data()
    print(f"Shape after preprocessing: {preprocessed_data.shape}")
    print(preprocessed_data.head())
    
    # Visualize distribution before and after preprocessing
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Distributions Before and After Preprocessing')
    
    for i, col in enumerate(care_pattern_cols[:3]):
        # Plot original data
        sns.histplot(data[col], ax=axes[0, i], kde=True)
        axes[0, i].set_title(f'Original: {col}')
        
        # Plot preprocessed data
        sns.histplot(preprocessed_data[col], ax=axes[1, i], kde=True)
        axes[1, i].set_title(f'Preprocessed: {col}')
    
    plt.tight_layout()
    plt.savefig('preprocessing_distributions.png')
    print("\nSaved distribution visualization to preprocessing_distributions.png")
    
    # Test missing value handling
    print("\n2. Testing missing value handling")
    # Introduce missing values
    data_with_missing = data.copy()
    for col in care_pattern_cols:
        mask = np.random.random(len(data)) < 0.1  # 10% missing values
        data_with_missing.loc[mask, col] = np.nan
    
    print(f"Added missing values - Total NaN count: {data_with_missing.isna().sum().sum()}")
    
    # Create new instance with missing data
    creator_missing = CarePhenotypeCreator(
        data=data_with_missing,
        clinical_factors=clinical_cols,
        n_clusters=3
    )
    
    # Test different strategies
    strategies = ['mean', 'median', 'mode']
    for strategy in strategies:
        processed = creator_missing.preprocess_data(missing_strategy=strategy)
        print(f"Missing values after {strategy} strategy: {processed.isna().sum().sum()}")
    
    # Test outlier handling
    print("\n3. Testing outlier handling")
    # Introduce outliers
    data_with_outliers = data.copy()
    for col in care_pattern_cols:
        # Add extreme values
        outlier_indices = np.random.choice(len(data), size=5, replace=False)
        data_with_outliers.loc[outlier_indices, col] = data[col].mean() + 10 * data[col].std()
    
    creator_outliers = CarePhenotypeCreator(
        data=data_with_outliers,
        clinical_factors=clinical_cols,
        n_clusters=3
    )
    
    # Test different thresholds
    thresholds = [2.0, 3.0, 4.0]
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(12, 10))
    fig.suptitle('Outlier Detection with Different Thresholds')
    
    for i, threshold in enumerate(thresholds):
        processed = creator_outliers.preprocess_data(handle_outliers=True, outlier_threshold=threshold)
        # Plot first care pattern as example
        col = care_pattern_cols[0]
        axes[i].scatter(range(len(data)), data_with_outliers[col], alpha=0.5, label='Original')
        axes[i].scatter(range(len(data)), processed[col], alpha=0.5, label='Processed')
        axes[i].set_title(f'Outlier threshold: {threshold}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('outlier_handling.png')
    print("\nSaved outlier handling visualization to outlier_handling.png")

if __name__ == "__main__":
    explore_preprocessing()