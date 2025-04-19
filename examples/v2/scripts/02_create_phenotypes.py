#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for creating care phenotype labels based on observable care patterns.
Uses the prepared MIMIC-IV data to identify patterns of care delivery.
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
import logging
import json
import matplotlib.pyplot as plt
from care_phenotype_analyzer import CarePhenotypeCreator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/phenotype_creation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create care phenotype labels')
    parser.add_argument('--data_path', type=str, default='../processed_data/cohort_data.csv',
                        help='Path to prepared dataset')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='Number of phenotype clusters to create')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    return parser.parse_args()

def load_data(data_path):
    """Load prepared dataset."""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    data = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
    
    # Check for required columns that should exist in both SQL and Python-generated data
    required_columns = ['subject_id', 'lab_tests_per_day', 'severity_score']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from data: {missing_columns}. "
                         f"Make sure you're using the correct dataset generated either "
                         f"by the SQL workflow or by 01_prepare_data.py.")
    
    return data

def identify_care_patterns(data):
    """Identify columns representing care patterns."""
    # Care patterns are typically frequency or timing measurements
    pattern_keywords = [
        'per_day', 'frequency', 'time_to', 'tests', 'vital_signs', 
        'monitoring', 'interval', 'delay'
    ]
    
    care_patterns = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in pattern_keywords):
            care_patterns.append(col)
            
    if not care_patterns:
        logger.warning("No care pattern columns identified by keywords. Using all numeric columns.")
        care_patterns = data.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude certain columns
        exclude_keywords = ['id', 'age', 'score', 'time', 'date', 'outcome']
        care_patterns = [col for col in care_patterns 
                        if not any(keyword in col.lower() for keyword in exclude_keywords)]
    
    logger.info(f"Identified {len(care_patterns)} care pattern columns: {care_patterns}")
    return care_patterns

def identify_clinical_factors(data):
    """Identify columns representing clinical factors."""
    # Clinical factors are typically patient characteristics or severity measures
    factor_keywords = [
        'age', 'score', 'severity', 'comorbidity', 'index', 'stage',
        'risk', 'grade', 'weight', 'bmi', 'history', 'duration'
    ]
    
    clinical_factors = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in factor_keywords):
            clinical_factors.append(col)
            
    if not clinical_factors:
        logger.warning("No clinical factor columns identified by keywords.")
    
    logger.info(f"Identified {len(clinical_factors)} clinical factor columns: {clinical_factors}")
    return clinical_factors

def create_phenotypes(data, care_patterns, clinical_factors, n_clusters, random_state):
    """Create phenotype labels using CarePhenotypeCreator."""
    logger.info(f"Creating phenotypes with {n_clusters} clusters")
    
    # Select relevant columns
    phenotype_data = data[care_patterns + clinical_factors].copy()
    
    # Initialize phenotype creator
    creator = CarePhenotypeCreator(
        data=phenotype_data,
        clinical_factors=clinical_factors,
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    # Create phenotype labels
    logger.info("Generating phenotype labels")
    phenotype_labels = creator.create_phenotype_labels()
    
    # Analyze clinical separation
    logger.info("Analyzing clinical separation")
    separation_metrics = creator.analyze_clinical_separation()
    
    # Analyze unexplained variation
    logger.info("Analyzing unexplained variation")
    variation_metrics = creator.analyze_unexplained_variation()
    
    return phenotype_labels, separation_metrics, variation_metrics, creator

def save_results(phenotype_labels, separation_metrics, variation_metrics, data, output_dir):
    """Save phenotype results to files."""
    logger.info(f"Saving results to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save phenotype labels
    phenotype_labels.name = 'phenotype'
    phenotype_df = pd.DataFrame(phenotype_labels)
    phenotype_df.to_csv(os.path.join(output_dir, 'phenotype_labels.csv'))
    
    # Save data with phenotype labels
    data_with_phenotypes = data.copy()
    data_with_phenotypes['phenotype'] = phenotype_labels
    data_with_phenotypes.to_csv(os.path.join(output_dir, 'data_with_phenotypes.csv'), index=False)
    
    # Save separation metrics
    with open(os.path.join(output_dir, 'clinical_separation.json'), 'w') as f:
        json.dump(separation_metrics, f, indent=2)
    
    # Save variation metrics
    with open(os.path.join(output_dir, 'unexplained_variation.json'), 'w') as f:
        json.dump(variation_metrics, f, indent=2)
    
    # Save phenotype summary
    phenotype_summary = data_with_phenotypes.groupby('phenotype').agg({
        # Calculate means of all numeric columns
        **{col: 'mean' for col in data_with_phenotypes.select_dtypes(include=[np.number]).columns},
        # Calculate counts
        'subject_id': 'count'
    }).rename(columns={'subject_id': 'count'})
    
    phenotype_summary.to_csv(os.path.join(output_dir, 'phenotype_summary.csv'))
    
    logger.info("Results saved successfully")

def visualize_phenotypes(data, phenotype_labels, care_patterns, clinical_factors, output_dir):
    """Create visualizations of phenotype results."""
    logger.info("Creating visualizations")
    
    # Create figures directory
    figures_dir = os.path.join(output_dir, '../figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Add phenotype labels to data for visualization
    data_with_phenotypes = data.copy()
    data_with_phenotypes['phenotype'] = phenotype_labels
    
    # 1. Visualize care patterns by phenotype
    plt.figure(figsize=(15, 10))
    
    for i, pattern in enumerate(care_patterns[:5]):  # Limit to first 5 patterns for readability
        plt.subplot(2, 3, i+1)
        boxplot = data_with_phenotypes.boxplot(column=pattern, by='phenotype', 
                                             return_type='dict', showmeans=True)
        plt.title(f'{pattern} by Phenotype')
        plt.suptitle('')  # Remove default suptitle
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'care_patterns_by_phenotype.png'))
    plt.close()
    
    # 2. Visualize clinical factors by phenotype
    plt.figure(figsize=(15, 10))
    
    for i, factor in enumerate(clinical_factors[:5]):  # Limit to first 5 factors for readability
        plt.subplot(2, 3, i+1)
        boxplot = data_with_phenotypes.boxplot(column=factor, by='phenotype', 
                                             return_type='dict', showmeans=True)
        plt.title(f'{factor} by Phenotype')
        plt.suptitle('')  # Remove default suptitle
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'clinical_factors_by_phenotype.png'))
    plt.close()
    
    # 3. Create phenotype distribution chart
    plt.figure(figsize=(10, 6))
    phenotype_counts = data_with_phenotypes['phenotype'].value_counts().sort_index()
    phenotype_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Phenotypes')
    plt.xlabel('Phenotype')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'phenotype_distribution.png'))
    plt.close()
    
    # 4. Create pattern correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = data_with_phenotypes[care_patterns].corr()
    
    import seaborn as sns
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Between Care Patterns')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pattern_correlation.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {figures_dir}")

def main():
    """Main function to create phenotype labels."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
    # Load data
    data = load_data(args.data_path)
    
    # Identify care patterns and clinical factors
    care_patterns = identify_care_patterns(data)
    clinical_factors = identify_clinical_factors(data)
    
    # Create phenotypes
    phenotype_labels, separation_metrics, variation_metrics, creator = create_phenotypes(
        data, care_patterns, clinical_factors, args.n_clusters, args.random_state
    )
    
    # Save results
    save_results(phenotype_labels, separation_metrics, variation_metrics, data, args.output_dir)
    
    # Create visualizations
    visualize_phenotypes(data, phenotype_labels, care_patterns, clinical_factors, args.output_dir)
    
    # Log completion
    logger.info(f"Phenotype creation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Created {args.n_clusters} phenotypes for {len(data)} patients")

if __name__ == "__main__":
    main() 