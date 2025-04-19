import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=100, n_clinical_factors=3, n_care_patterns=5, random_seed=42):
    '''Generate synthetic healthcare data for phenotype exporation'''
    np.random.seed(random_seed)

    # Create subject IDs
    subject_ids = [f'SUBJ{i:04d}' for i in range(n_samples)]

    # Create timestamps 
    base_date = datetime(2023, 1, 1)
    timestamps = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]

    # Generate clinical factors (like SOFA scores, Charlson index etc.)
    clinical_data= {}
    for i in range(n_clinical_factors):
        # Create clinical factors with different distributions 
        if i == 0:
            # Normal distribution for first factor (e.g. SOFA score)
            clinical_data[f'clinical_factor_{i}'] = np.random.normal(5, 2, n_samples).round(1)
        elif i == 1:
            # Skewed distribution for second factor (e.g. Charlson index)
            clinical_data[f'clinical_factor_{i}'] = np.random.exponential(3, n_samples).round(1)
        else:
            # Uniform distribution for other factors
            clinical_data[f'clinical_factor_{i}'] = np.random.uniform(0, 10, n_samples).round(1)

    
    # Generate care patterns 
    care_patterns = {}
    for i in range(n_care_patterns):
        # Create patterns with dependencis on clinical factors
        base_pattern = np.random.normal(10, 3, n_samples)

        # Add dependency on clinical factors
        for j in range(n_clinical_factors):
            factor_influence = 0.5 * clinical_data[f'clinical_factor_{j}']
            base_pattern += factor_influence * (0.5 + np.random.random(n_samples))/n_clinical_factors

        # Add noise
        base_pattern += np.random.normal(0, 2, n_samples)

        care_patterns[f'care_pattern_{i}'] = np.maximum(0, base_pattern).round(2)

    # Combine all data
    data = {
        'subject_id': subject_ids,
        'timestamp': timestamps,
        **clinical_data,
        **care_patterns
    }

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # Generate dataset with 200 patients
    data = generate_synthetic_data(n_samples=200)
    print(f'Generated synthetic data with {len(data)} records')
    print(f'\nData preview:')
    print(data.head())

    # Summary statistics
    print(f'\nSummary statistics:')
    print(data.describe())

    data.to_csv(r'C:\Users\tn351\code\care-phenotypic-label-creation\local-scratch\phenotype-creator\data\Synthetic_healthcare_data.csv')
