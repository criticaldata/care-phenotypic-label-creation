#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for preparing MIMIC-IV data for phenotype analysis.
This script demonstrates how to select a cohort and compute care patterns.

NOTE: This script is an ALTERNATIVE to the SQL workflow in the sql_scripts directory.
If you've already used the SQL scripts to generate cohort_data.csv, you can SKIP
this script and proceed directly to 02_create_phenotypes.py.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/data_preparation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare MIMIC data for phenotype analysis')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing MIMIC-IV CSV files')
    parser.add_argument('--output_dir', type=str, default='../processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--min_age', type=int, default=18,
                        help='Minimum patient age to include')
    parser.add_argument('--min_los', type=int, default=3,
                        help='Minimum length of stay in days')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of patients to include (for memory management)')
    return parser.parse_args()

def load_mimic_tables(data_dir):
    """Load required MIMIC-IV tables."""
    logger.info("Loading MIMIC tables from %s", data_dir)
    
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    admissions = pd.read_csv(os.path.join(data_dir, 'admissions.csv'))
    
    # For lab events and chart events, we may need to be selective because they're large
    # Load only relevant columns from labevents
    dtypes = {
        'labevent_id': 'int32',
        'subject_id': 'int32',
        'hadm_id': 'float32',  # Some values might be missing, so use float
        'specimen_id': 'int32',
        'itemid': 'int16',
        'charttime': 'str',
        'storetime': 'str',
        'value': 'str',  # Will convert specific lab values later if needed
        'valuenum': 'float32',
        'valueuom': 'str',
        'ref_range_lower': 'float32',
        'ref_range_upper': 'float32'
    }
    
    # Read labevents in chunks to manage memory
    chunks = []
    for chunk in pd.read_csv(os.path.join(data_dir, 'labevents.csv'), 
                             dtype=dtypes, 
                             chunksize=100000):
        chunks.append(chunk[['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']])
    
    labevents = pd.concat(chunks)
    
    # Similarly, load subset of chartevents
    chunks = []
    for chunk in pd.read_csv(os.path.join(data_dir, 'chartevents.csv'), 
                             usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
                             chunksize=100000):
        chunks.append(chunk)
    
    chartevents = pd.concat(chunks)
    
    logger.info("Loaded %d patients, %d admissions, %d lab events, %d chart events", 
                len(patients), len(admissions), len(labevents), len(chartevents))
    
    return patients, admissions, labevents, chartevents

def define_cohort(patients, admissions, min_age=18, min_los=3, sample_size=1000):
    """Define study cohort based on inclusion criteria."""
    logger.info("Defining cohort with min_age=%d, min_los=%d", min_age, min_los)
    
    # Calculate patient age at admission
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    patients['anchor_age'] = patients['anchor_age'].astype(float)
    
    # Join patients and admissions
    cohort = admissions.merge(patients, on='subject_id', how='inner')
    
    # Apply inclusion criteria
    adult_mask = cohort['anchor_age'] >= min_age
    
    # Calculate length of stay
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
    cohort['los_days'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / (24*3600)
    los_mask = cohort['los_days'] >= min_los
    
    # Apply all criteria
    cohort = cohort[adult_mask & los_mask]
    
    # Limit to sample size (if specified)
    if sample_size and len(cohort) > sample_size:
        cohort = cohort.sample(sample_size, random_state=42)
    
    logger.info("Defined cohort with %d admissions", len(cohort))
    
    return cohort

def calculate_clinical_factors(cohort, labevents, chartevents):
    """Calculate clinical factors that may explain variations in care."""
    logger.info("Calculating clinical factors")
    
    # Age is already available
    clinical_factors = pd.DataFrame({
        'subject_id': cohort['subject_id'],
        'hadm_id': cohort['hadm_id'],
        'age': cohort['anchor_age'],
        'los_days': cohort['los_days']
    })
    
    # Calculate a simple severity score (proxy for SOFA or similar)
    # Here we use arterial blood pressure and creatinine as examples
    
    # Common vitals - MAP (Mean Arterial Pressure) - lower means more severe
    vitals_map_itemid = 220052  # MIMIC-IV item ID for MAP
    map_values = chartevents[chartevents['itemid'] == vitals_map_itemid]
    map_mean = map_values.groupby('hadm_id')['valuenum'].mean()
    
    # Lab values - Creatinine - higher means more severe
    creatinine_itemid = 50912  # MIMIC-IV item ID for creatinine
    creatinine_values = labevents[labevents['itemid'] == creatinine_itemid]
    creatinine_max = creatinine_values.groupby('hadm_id')['valuenum'].max()
    
    # Add these to clinical factors
    clinical_factors = clinical_factors.merge(
        pd.DataFrame({'hadm_id': map_mean.index, 'map_mean': map_mean.values}),
        on='hadm_id', how='left'
    )
    
    clinical_factors = clinical_factors.merge(
        pd.DataFrame({'hadm_id': creatinine_max.index, 'creatinine_max': creatinine_max.values}),
        on='hadm_id', how='left'
    )
    
    # Fill missing values with means
    clinical_factors['map_mean'] = clinical_factors['map_mean'].fillna(
        clinical_factors['map_mean'].mean())
    clinical_factors['creatinine_max'] = clinical_factors['creatinine_max'].fillna(
        clinical_factors['creatinine_max'].mean())
    
    # Create simple severity score (z-score based)
    map_z = (clinical_factors['map_mean'] - clinical_factors['map_mean'].mean()) / \
            clinical_factors['map_mean'].std()
    creatinine_z = (clinical_factors['creatinine_max'] - clinical_factors['creatinine_max'].mean()) / \
                  clinical_factors['creatinine_max'].std()
    
    # Combine (note: low MAP is bad, high creatinine is bad)
    clinical_factors['severity_score'] = -map_z + creatinine_z
    
    # Normalize for easier interpretation
    clinical_factors['severity_score'] = (clinical_factors['severity_score'] - 
                                         clinical_factors['severity_score'].min()) / \
                                         (clinical_factors['severity_score'].max() - 
                                          clinical_factors['severity_score'].min())
    
    logger.info("Calculated clinical factors for %d admissions", len(clinical_factors))
    
    return clinical_factors

def calculate_care_patterns(cohort, labevents, chartevents):
    """Calculate care pattern metrics that reflect how care is delivered."""
    logger.info("Calculating care patterns")
    
    # Initialize care patterns dataframe
    care_patterns = pd.DataFrame({
        'subject_id': cohort['subject_id'],
        'hadm_id': cohort['hadm_id']
    })
    
    # Lab test frequency per day
    lab_counts = labevents.groupby('hadm_id').size()
    care_patterns = care_patterns.merge(
        pd.DataFrame({'hadm_id': lab_counts.index, 'lab_count': lab_counts.values}),
        on='hadm_id', how='left'
    )
    care_patterns['lab_count'] = care_patterns['lab_count'].fillna(0)
    care_patterns = care_patterns.merge(cohort[['hadm_id', 'los_days']], on='hadm_id', how='left')
    care_patterns['lab_tests_per_day'] = care_patterns['lab_count'] / care_patterns['los_days']
    
    # Vitals measurement frequency per day
    vitals_counts = chartevents.groupby('hadm_id').size()
    care_patterns = care_patterns.merge(
        pd.DataFrame({'hadm_id': vitals_counts.index, 'vitals_count': vitals_counts.values}),
        on='hadm_id', how='left'
    )
    care_patterns['vitals_count'] = care_patterns['vitals_count'].fillna(0)
    care_patterns['vital_signs_per_day'] = care_patterns['vitals_count'] / care_patterns['los_days']
    
    # Calculate specific lab test frequencies for common tests
    # Example: Complete Blood Count (CBC)
    cbc_itemids = [51301, 51300, 51218]  # Example: WBC, RBC, Hematocrit
    cbc_labs = labevents[labevents['itemid'].isin(cbc_itemids)]
    cbc_counts = cbc_labs.groupby('hadm_id').size()
    care_patterns = care_patterns.merge(
        pd.DataFrame({'hadm_id': cbc_counts.index, 'cbc_count': cbc_counts.values}),
        on='hadm_id', how='left'
    )
    care_patterns['cbc_count'] = care_patterns['cbc_count'].fillna(0)
    care_patterns['cbc_tests_per_day'] = care_patterns['cbc_count'] / care_patterns['los_days']
    
    # Example: Chemistry panel
    chem_itemids = [50912, 50928, 50971]  # Example: Creatinine, Glucose, Potassium
    chem_labs = labevents[labevents['itemid'].isin(chem_itemids)]
    chem_counts = chem_labs.groupby('hadm_id').size()
    care_patterns = care_patterns.merge(
        pd.DataFrame({'hadm_id': chem_counts.index, 'chem_count': chem_counts.values}),
        on='hadm_id', how='left'
    )
    care_patterns['chem_count'] = care_patterns['chem_count'].fillna(0)
    care_patterns['chem_tests_per_day'] = care_patterns['chem_count'] / care_patterns['los_days']
    
    logger.info("Calculated care patterns for %d admissions", len(care_patterns))
    
    return care_patterns

def prepare_final_dataset(cohort, clinical_factors, care_patterns):
    """Combine all components into the final analysis dataset."""
    logger.info("Preparing final dataset")
    
    # Merge clinical factors and care patterns
    final_dataset = clinical_factors.merge(care_patterns, on=['subject_id', 'hadm_id'], how='inner')
    
    # Add demographic information
    demographics = cohort[['subject_id', 'hadm_id', 'gender', 'insurance', 'language', 'marital_status']]
    final_dataset = final_dataset.merge(demographics, on=['subject_id', 'hadm_id'], how='inner')
    
    # Define a simplified ethnicity column
    ethnicity_mapping = {
        'WHITE': 'White',
        'BLACK/AFRICAN AMERICAN': 'Black',
        'HISPANIC/LATINO': 'Hispanic',
        'ASIAN': 'Asian',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Pacific Islander',
        'AMERICAN INDIAN/ALASKA NATIVE': 'Native American',
        # Map other values to 'Other'
    }
    
    cohort['ethnicity_simplified'] = cohort['race'].apply(
        lambda x: next((v for k, v in ethnicity_mapping.items() if k in x.upper()), 'Other')
    )
    
    final_dataset = final_dataset.merge(
        cohort[['hadm_id', 'ethnicity_simplified']], 
        on='hadm_id', 
        how='inner'
    )
    
    # For demonstration, create a simple outcome variable
    # This could be mortality, readmission, etc. in a real analysis
    np.random.seed(42)
    final_dataset['outcome_variable'] = (
        0.3 * final_dataset['severity_score'] + 
        0.1 * np.random.randn(len(final_dataset))
    ) > 0.15
    
    final_dataset['outcome_variable'] = final_dataset['outcome_variable'].astype(int)
    
    # Clean up columns - remove intermediate columns
    cols_to_drop = ['lab_count', 'vitals_count', 'cbc_count', 'chem_count', 'los_days_y']
    final_dataset = final_dataset.drop(columns=[c for c in cols_to_drop if c in final_dataset.columns])
    
    if 'los_days_x' in final_dataset.columns:
        final_dataset = final_dataset.rename(columns={'los_days_x': 'los_days'})
    
    logger.info("Final dataset prepared with %d rows and %d columns", 
               len(final_dataset), len(final_dataset.columns))
    
    return final_dataset

def main():
    """Main function to prepare MIMIC data for analysis."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
    # Load MIMIC tables
    patients, admissions, labevents, chartevents = load_mimic_tables(args.data_dir)
    
    # Define cohort
    cohort = define_cohort(
        patients, admissions, 
        min_age=args.min_age, 
        min_los=args.min_los,
        sample_size=args.sample_size
    )
    
    # Calculate clinical factors
    clinical_factors = calculate_clinical_factors(cohort, labevents, chartevents)
    
    # Calculate care patterns
    care_patterns = calculate_care_patterns(cohort, labevents, chartevents)
    
    # Prepare final dataset
    final_dataset = prepare_final_dataset(cohort, clinical_factors, care_patterns)
    
    # Save final dataset
    output_path = os.path.join(args.output_dir, 'cohort_data.csv')
    final_dataset.to_csv(output_path, index=False)
    logger.info("Saved final dataset to %s", output_path)
    
    # Save cohort information for reference
    cohort_info = {
        'total_patients': len(cohort['subject_id'].unique()),
        'total_admissions': len(cohort),
        'min_age': args.min_age,
        'min_los': args.min_los,
        'age_mean': cohort['anchor_age'].mean(),
        'age_std': cohort['anchor_age'].std(),
        'los_mean': cohort['los_days'].mean(),
        'los_std': cohort['los_days'].std(),
        'gender_distribution': cohort['gender'].value_counts().to_dict(),
        'ethnicity_distribution': cohort['ethnicity_simplified'].value_counts().to_dict()
    }
    
    import json
    with open(os.path.join(args.output_dir, 'cohort_info.json'), 'w') as f:
        json.dump(cohort_info, f, indent=2)
    
    logger.info("Data preparation completed in %.2f seconds", time.time() - start_time)

if __name__ == "__main__":
    main() 