#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for analyzing care patterns across the created phenotypes.
This script demonstrates how to use the CarePatternAnalyzer to understand
variations in care delivery patterns.
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
import logging
import json
import matplotlib.pyplot as plt
from care_phenotype_analyzer import CarePatternAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/pattern_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze care patterns across phenotypes')
    parser.add_argument('--data_path', type=str, default='../processed_data/cohort_data.csv',
                        help='Path to prepared dataset')
    parser.add_argument('--phenotype_path', type=str, default='../results/phenotype_labels.csv',
                        help='Path to phenotype labels')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--figures_dir', type=str, default='../figures',
                        help='Directory to save figures')
    return parser.parse_args()

def load_data(data_path, phenotype_path):
    """Load dataset and phenotype labels."""
    logger.info(f"Loading data from {data_path} and phenotypes from {phenotype_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(phenotype_path):
        raise FileNotFoundError(f"Phenotype file not found: {phenotype_path}")
        
    data = pd.read_csv(data_path)
    phenotypes = pd.read_csv(phenotype_path, index_col=0)
    
    # Ensure phenotypes are properly formatted
    if isinstance(phenotypes, pd.DataFrame):
        if 'phenotype' in phenotypes.columns:
            phenotypes = phenotypes['phenotype']
        else:
            phenotypes = phenotypes.iloc[:, 0]  # Take first column
    
    if len(data) != len(phenotypes):
        raise ValueError(f"Data length ({len(data)}) does not match phenotype length ({len(phenotypes)})")
    
    logger.info(f"Loaded dataset with {len(data)} rows and {len(phenotypes.unique())} unique phenotypes")
    
    return data, phenotypes

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
        exclude_keywords = ['id', 'age', 'score', 'time', 'date', 'outcome', 'phenotype']
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

def analyze_patterns(data, phenotypes, care_patterns, clinical_factors, output_dir, figures_dir):
    """Analyze care patterns using CarePatternAnalyzer."""
    logger.info("Initializing CarePatternAnalyzer")
    
    # Add phenotypes to data
    data_with_phenotypes = data.copy()
    data_with_phenotypes['phenotype'] = phenotypes
    
    # Initialize pattern analyzer
    analyzer = CarePatternAnalyzer(
        data=data_with_phenotypes,
        clinical_factors=clinical_factors
    )
    
    # Analyze each care pattern
    results = {}
    for pattern in care_patterns:
        logger.info(f"Analyzing pattern: {pattern}")
        
        # Skip if pattern is not a valid column
        if pattern not in data_with_phenotypes.columns:
            logger.warning(f"Pattern {pattern} not found in data. Skipping.")
            continue
            
        # Use first timestamp column as time reference, or create a placeholder
        time_columns = [col for col in data_with_phenotypes.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_columns:
            time_column = time_columns[0]
        else:
            # Create a placeholder timestamp column
            data_with_phenotypes['timestamp'] = pd.date_range(
                start='2020-01-01', 
                periods=len(data_with_phenotypes), 
                freq='D'
            )
            time_column = 'timestamp'
        
        # Analyze frequency
        frequency_analysis = analyzer.analyze_measurement_frequency(
            measurement_column=pattern,
            time_column=time_column,
            clinical_factors=clinical_factors,
            group_by=['phenotype']
        )
        
        results[pattern] = {
            'frequency_analysis': frequency_analysis.to_dict()
        }
        
        # Visualize clinical separation for this pattern
        output_file = os.path.join(figures_dir, f'clinical_separation_{pattern}.png')
        analyzer.visualize_clinical_separation(
            phenotype_labels=phenotypes,
            clinical_factors=clinical_factors,
            output_file=output_file
        )
        
        # Visualize unexplained variation
        output_file = os.path.join(figures_dir, f'unexplained_variation_{pattern}.png')
        analyzer.visualize_unexplained_variation(
            phenotype_labels=phenotypes,
            care_patterns=[pattern],
            clinical_factors=clinical_factors,
            output_file=output_file
        )
    
    # Save analysis results
    with open(os.path.join(output_dir, 'pattern_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)  # Use default=str to handle datetime and other non-serializable types
    
    # Create cross-pattern analysis
    create_cross_pattern_analysis(data_with_phenotypes, care_patterns, phenotypes, figures_dir)
    
    logger.info("Care pattern analysis completed")
    
    return results

def create_cross_pattern_analysis(data, care_patterns, phenotypes, figures_dir):
    """Create visualizations that analyze relationships between multiple care patterns."""
    logger.info("Creating cross-pattern analysis visualizations")
    
    # Create a pair plot of care patterns colored by phenotype
    if len(care_patterns) >= 2:
        try:
            import seaborn as sns
            plt.figure(figsize=(15, 15))
            sns.pairplot(
                data=data, 
                vars=care_patterns[:5],  # Limit to 5 patterns for readability
                hue='phenotype',
                palette='viridis',
                diag_kind='kde',
                plot_kws={'alpha': 0.5}
            )
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'care_pattern_pairplot.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating pair plot: {str(e)}")
    
    # Create radar chart comparing patterns across phenotypes
    if len(care_patterns) >= 3:
        try:
            # Calculate mean values for each pattern by phenotype
            pattern_means = data.groupby('phenotype')[care_patterns].mean()
            
            # Normalize means for radar chart
            normalized_means = pattern_means.copy()
            for pattern in care_patterns:
                max_val = pattern_means[pattern].max()
                min_val = pattern_means[pattern].min()
                if max_val > min_val:
                    normalized_means[pattern] = (pattern_means[pattern] - min_val) / (max_val - min_val)
            
            # Create radar chart
            create_radar_chart(
                normalized_means,
                care_patterns[:6],  # Limit to 6 patterns for readability
                os.path.join(figures_dir, 'care_pattern_radar.png')
            )
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")

def create_radar_chart(data, variables, output_file):
    """Create a radar chart visualizing multiple variables across groups."""
    # Set up the radar chart
    plt.figure(figsize=(10, 8))
    
    # Calculate the angles for each variable
    angles = np.linspace(0, 2*np.pi, len(variables), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up subplot as polar
    ax = plt.subplot(111, polar=True)
    
    # Plot each phenotype
    for i, phenotype in enumerate(data.index):
        values = data.loc[phenotype, variables].tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, label=f'Phenotype {phenotype}')
        ax.fill(angles, values, alpha=0.1)
    
    # Add labels
    plt.xticks(angles[:-1], variables)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.title('Care Patterns by Phenotype')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    """Main function to analyze care patterns."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
    # Load data and phenotypes
    data, phenotypes = load_data(args.data_path, args.phenotype_path)
    
    # Identify care patterns and clinical factors
    care_patterns = identify_care_patterns(data)
    clinical_factors = identify_clinical_factors(data)
    
    # Analyze patterns
    results = analyze_patterns(
        data, phenotypes, care_patterns, clinical_factors, 
        args.output_dir, args.figures_dir
    )
    
    # Log completion
    logger.info(f"Pattern analysis completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Analyzed {len(care_patterns)} care patterns across {len(phenotypes.unique())} phenotypes")

if __name__ == "__main__":
    main() 