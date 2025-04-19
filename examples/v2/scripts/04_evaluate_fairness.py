#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for evaluating fairness using care phenotype labels.
This script demonstrates how to use the FairnessEvaluator to understand
potential biases in healthcare algorithms.
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
import logging
import json
import matplotlib.pyplot as plt
from care_phenotype_analyzer import FairnessEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/fairness_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate algorithmic fairness')
    parser.add_argument('--data_path', type=str, default='../processed_data/cohort_data.csv',
                        help='Path to prepared dataset')
    parser.add_argument('--phenotype_path', type=str, default='../results/phenotype_labels.csv',
                        help='Path to phenotype labels')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--figures_dir', type=str, default='../figures',
                        help='Directory to save figures')
    parser.add_argument('--simulate_model', action='store_true',
                        help='Simulate a predictive model for demonstration purposes')
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

def identify_demographic_factors(data):
    """Identify columns representing demographic factors."""
    # Demographic factors are typically patient demographics
    demographic_keywords = [
        'gender', 'sex', 'ethnicity', 'race', 'age', 'marital', 
        'language', 'income', 'education', 'insurance', 'zip', 'location'
    ]
    
    demographic_factors = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in demographic_keywords):
            demographic_factors.append(col)
            
    if not demographic_factors:
        logger.warning("No demographic factor columns identified by keywords.")
    
    logger.info(f"Identified {len(demographic_factors)} demographic factor columns: {demographic_factors}")
    return demographic_factors

def simulate_predictive_model(data, clinical_factors, random_state=42):
    """
    Simulate a predictive model for demonstration purposes.
    
    For fairness evaluation, we need:
    1. Model predictions
    2. True labels (outcomes)
    
    This function creates a simple model that predicts an outcome
    based on clinical factors and introduces a slight bias based on demographics.
    """
    logger.info("Simulating a predictive model")
    
    np.random.seed(random_state)
    
    # Check if outcome variable exists
    if 'outcome_variable' in data.columns:
        logger.info("Using existing 'outcome_variable' as true labels")
        true_labels = data['outcome_variable'].astype(int)
    else:
        # Create synthetic true labels based on severity score and random noise
        logger.info("Creating synthetic true labels")
        if 'severity_score' in clinical_factors:
            true_labels = (0.6 * data['severity_score'] + 0.1 * np.random.randn(len(data))) > 0.3
        else:
            # Use first clinical factor as proxy
            clinical_factor = clinical_factors[0]
            true_labels = (0.6 * data[clinical_factor] + 0.1 * np.random.randn(len(data))) > 0.3
            
        true_labels = true_labels.astype(int)
    
    # Create predictions based on clinical factors with deliberate bias
    logger.info("Creating biased predictions for demonstration")
    
    # Base predictions on clinical factors
    if clinical_factors:
        # Normalize clinical factors
        clinical_data = data[clinical_factors].copy()
        for factor in clinical_factors:
            clinical_data[factor] = (clinical_data[factor] - clinical_data[factor].mean()) / clinical_data[factor].std()
            
        # Simple linear combination
        base_predictions = clinical_data.mean(axis=1)
        base_predictions = (base_predictions - base_predictions.min()) / (base_predictions.max() - base_predictions.min())
    else:
        # If no clinical factors, use random
        base_predictions = np.random.rand(len(data))
    
    # Add demographic bias (for demonstration)
    if 'gender' in data.columns:
        # Add gender bias (example: slightly lower predictions for certain gender)
        gender_bias = data['gender'].map({'F': -0.05, 'M': 0.05}).fillna(0)
        base_predictions += gender_bias
        
    if 'ethnicity_simplified' in data.columns:
        # Add ethnicity bias (example: slightly lower predictions for certain groups)
        minority_mask = data['ethnicity_simplified'].isin(['Black', 'Hispanic', 'Other'])
        ethnicity_bias = pd.Series(0, index=data.index)
        ethnicity_bias[minority_mask] = -0.1
        base_predictions += ethnicity_bias
    
    # Convert to binary predictions
    predictions = (base_predictions > 0.5).astype(int)
    
    logger.info(f"Created predictions with {predictions.sum()} positive and {len(predictions) - predictions.sum()} negative cases")
    
    return predictions, true_labels

def evaluate_fairness(data, phenotypes, clinical_factors, demographic_factors, predictions, true_labels, output_dir, figures_dir):
    """Evaluate fairness using FairnessEvaluator."""
    logger.info("Initializing FairnessEvaluator")
    
    # Create clinical factors DataFrame
    clinical_data = data[clinical_factors] if clinical_factors else None
    
    # Initialize fairness evaluator
    evaluator = FairnessEvaluator(
        predictions=predictions,
        true_labels=true_labels,
        phenotype_labels=phenotypes,
        clinical_factors=clinical_data,
        demographic_factors=demographic_factors,
        demographic_data=data[demographic_factors]
    )
    
    # Evaluate fairness metrics
    logger.info("Evaluating fairness metrics")
    fairness_metrics = evaluator.evaluate_fairness_metrics(
        metrics=['demographic_parity', 'equal_opportunity', 'predictive_parity', 'treatment_equality'],
        adjust_for_clinical=True
    )
    
    # Save fairness metrics
    with open(os.path.join(output_dir, 'fairness_metrics.json'), 'w') as f:
        json.dump(fairness_metrics, f, indent=2, default=str)
    
    # Visualize fairness metrics
    logger.info("Creating fairness visualizations")
    
    # Visualize fairness metrics
    evaluator.visualize_fairness_metrics(
        metrics=['demographic_parity', 'equal_opportunity'],
        output_file=os.path.join(figures_dir, 'fairness_metrics.png')
    )
    
    # Visualize bias detection
    evaluator.visualize_bias_detection(
        output_file=os.path.join(figures_dir, 'bias_detection.png')
    )
    
    # Visualize bias mitigation strategies
    evaluator.visualize_bias_mitigation(
        strategies=['reweighting', 'threshold_adjustment', 'calibration'],
        output_file=os.path.join(figures_dir, 'bias_mitigation.png')
    )
    
    # Apply bias mitigation and save results
    logger.info("Applying bias mitigation strategies")
    
    mitigation_results = {}
    for strategy in ['reweighting', 'threshold_adjustment', 'calibration']:
        logger.info(f"Applying {strategy} strategy")
        mitigated_predictions = evaluator.mitigate_bias(
            strategy=strategy,
            target_metric='demographic_parity'
        )
        
        # Re-evaluate fairness with mitigated predictions
        original_predictions = evaluator.predictions
        evaluator.predictions = mitigated_predictions
        
        mitigated_metrics = evaluator.evaluate_fairness_metrics(
            metrics=['demographic_parity', 'equal_opportunity'],
            adjust_for_clinical=True
        )
        
        # Restore original predictions
        evaluator.predictions = original_predictions
        
        mitigation_results[strategy] = {
            'strategy': strategy,
            'metrics': mitigated_metrics
        }
    
    # Save mitigation results
    with open(os.path.join(output_dir, 'bias_mitigation_results.json'), 'w') as f:
        json.dump(mitigation_results, f, indent=2, default=str)
    
    logger.info("Fairness evaluation completed")
    
    return fairness_metrics, mitigation_results

def create_fairness_report(fairness_metrics, mitigation_results, output_dir):
    """Create a human-readable fairness report."""
    logger.info("Creating fairness report")
    
    report_path = os.path.join(output_dir, 'fairness_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Fairness Evaluation Report\n\n")
        
        # Write fairness metrics summary
        f.write("## Fairness Metrics\n\n")
        
        for metric, results in fairness_metrics.items():
            f.write(f"### {metric.replace('_', ' ').title()}\n\n")
            
            if isinstance(results, dict):
                for demographic, values in results.items():
                    f.write(f"**{demographic}**:\n\n")
                    
                    if isinstance(values, dict) and 'disparity' in values:
                        f.write(f"* Disparity: {values['disparity']:.4f}\n")
                        
                        if 'group_rates' in values:
                            f.write("* Group rates:\n")
                            for group, rate in values['group_rates'].items():
                                f.write(f"  * {group}: {rate:.4f}\n")
                    
                    f.write("\n")
            
            f.write("\n")
        
        # Write mitigation summary
        f.write("## Bias Mitigation Results\n\n")
        
        for strategy, results in mitigation_results.items():
            f.write(f"### {strategy.replace('_', ' ').title()} Strategy\n\n")
            
            if 'metrics' in results:
                f.write("#### Post-Mitigation Metrics\n\n")
                
                for metric, metric_results in results['metrics'].items():
                    f.write(f"**{metric.replace('_', ' ').title()}**:\n\n")
                    
                    if isinstance(metric_results, dict):
                        for demographic, values in metric_results.items():
                            if isinstance(values, dict) and 'disparity' in values:
                                f.write(f"* {demographic} disparity: {values['disparity']:.4f}\n")
            
            f.write("\n")
        
        # Write recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the fairness evaluation results, consider the following recommendations:\n\n")
        
        # Find the most effective mitigation strategy
        best_strategy = None
        lowest_disparity = float('inf')
        
        for strategy, results in mitigation_results.items():
            if 'metrics' in results and 'demographic_parity' in results['metrics']:
                for demographic, values in results['metrics']['demographic_parity'].items():
                    if isinstance(values, dict) and 'disparity' in values:
                        if values['disparity'] < lowest_disparity:
                            lowest_disparity = values['disparity']
                            best_strategy = strategy
        
        if best_strategy:
            f.write(f"1. **Most effective bias mitigation strategy**: {best_strategy.replace('_', ' ').title()}\n")
        
        f.write("2. **Continuous monitoring**: Regularly monitor fairness metrics in production.\n")
        f.write("3. **Diverse representation**: Ensure diverse representation in training data.\n")
        f.write("4. **Feature selection review**: Review feature selection for potential sources of bias.\n")
        f.write("5. **Clinical context**: Interpret results in clinical context with domain experts.\n")
    
    logger.info(f"Fairness report created at {report_path}")

def main():
    """Main function to evaluate fairness."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
    # Load data and phenotypes
    data, phenotypes = load_data(args.data_path, args.phenotype_path)
    
    # Identify clinical and demographic factors
    clinical_factors = identify_clinical_factors(data)
    demographic_factors = identify_demographic_factors(data)
    
    # Get or simulate predictions and true labels
    if args.simulate_model or 'predictions' not in data.columns:
        predictions, true_labels = simulate_predictive_model(data, clinical_factors)
    else:
        # Use existing predictions and true labels if available
        predictions = data['predictions']
        true_labels = data['true_labels']
    
    # Evaluate fairness
    fairness_metrics, mitigation_results = evaluate_fairness(
        data, phenotypes, clinical_factors, demographic_factors,
        predictions, true_labels, args.output_dir, args.figures_dir
    )
    
    # Create fairness report
    create_fairness_report(fairness_metrics, mitigation_results, args.output_dir)
    
    # Log completion
    logger.info(f"Fairness evaluation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 