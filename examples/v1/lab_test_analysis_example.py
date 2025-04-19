"""
Example script demonstrating the use of the care phenotype analyzer package.
This example analyzes lab test measurement patterns and creates care phenotypes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import sys
from care_phenotype_analyzer import CarePhenotypeCreator, CarePatternAnalyzer, FairnessEvaluator

# Add signal handler for Ctrl+C
def signal_handler(sig, frame):
    print('\nExiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Generate timestamps
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate clinical factors
    sofa_scores = np.random.randint(0, 24, n_samples)
    charlson_scores = np.random.randint(0, 10, n_samples)
    
    # Generate care patterns with some correlation to clinical factors
    lab_test_freq = np.random.normal(2, 0.5, n_samples)
    lab_test_freq += 0.1 * sofa_scores  # Correlation with illness severity
    
    # Add unexplained variation
    unexplained_variation = np.random.normal(0, 0.3, n_samples)
    lab_test_freq += unexplained_variation
    
    # Generate demographic factors for fairness evaluation
    ages = np.random.randint(18, 90, n_samples)
    genders = np.random.choice(['M', 'F'], n_samples)
    ethnicities = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples, 
                                  p=[0.6, 0.15, 0.15, 0.05, 0.05])
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'sofa_score': sofa_scores,
        'charlson_score': charlson_scores,
        'lab_test_frequency': lab_test_freq,
        'routine_care_frequency': np.random.normal(3, 0.5, n_samples),
        'age': ages,
        'gender': genders,
        'ethnicity': ethnicities
    })
    
    return data

def main():
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data()
    
    # Create phenotype labels
    print("\nCreating care phenotype labels...")
    creator = CarePhenotypeCreator(
        data=data,
        clinical_factors=['sofa_score', 'charlson_score'],
        n_clusters=3
    )
    
    # Create phenotype labels - note the function doesn't accept care_patterns parameter
    phenotype_labels = creator.create_phenotype_labels()
    
    # Analyze patterns
    print("\nAnalyzing care patterns...")
    analyzer = CarePatternAnalyzer(data)
    
    # Analyze measurement frequencies
    frequency_results = analyzer.analyze_measurement_frequency(
        measurement_column='lab_test_frequency',
        time_column='timestamp',
        clinical_factors=['sofa_score', 'charlson_score']
    )
    
    # Visualize patterns
    print("\nCreating visualizations...")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Lab test frequency by phenotype
    plt.subplot(1, 3, 1)
    sns.boxplot(x=phenotype_labels, y=data['lab_test_frequency'])
    plt.title('Lab Test Frequency by Phenotype')
    plt.xlabel('Phenotype')
    plt.ylabel('Frequency')
    
    # Plot 2: Lab test frequency vs SOFA score
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=data, x='sofa_score', y='lab_test_frequency', alpha=0.5)
    plt.title('Lab Test Frequency vs SOFA Score')
    
    # Plot 3: Time series of lab test frequency
    plt.subplot(1, 3, 3)
    plt.plot(data['timestamp'], data['lab_test_frequency'])
    plt.title('Lab Test Frequency Over Time')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('lab_test_analysis.png')
    
    # Analyze unexplained variation
    print("\nAnalyzing unexplained variation...")
    unexplained_results = creator.analyze_unexplained_variation(
        care_pattern='lab_test_frequency',
        phenotype_labels=phenotype_labels
    )
    
    # Print results
    print("\nResults Summary:")
    print(f"Number of phenotypes: {len(np.unique(phenotype_labels))}")
    print("\nFrequency Analysis:")
    print(frequency_results)
    print("\nUnexplained Variation Analysis:")
    print(unexplained_results)
    
    # Example of fairness evaluation
    print("\nPerforming fairness evaluation...")
    # Generate sample predictions for demonstration
    sample_predictions = pd.Series(np.random.binomial(1, 0.3, len(data)))
    sample_true_labels = pd.Series(np.random.binomial(1, 0.3, len(data)))
    
    try:
        # Create a proper DataFrame for clinical factors
        clinical_factors_df = data[['sofa_score', 'charlson_score']]
        
        # Create a DataFrame that includes both clinical and demographic factors
        data_for_eval = data.copy()
        
        evaluator = FairnessEvaluator(
            predictions=sample_predictions,
            true_labels=sample_true_labels,
            phenotype_labels=phenotype_labels,
            clinical_factors=clinical_factors_df,
            demographic_factors=['gender', 'ethnicity', 'age']  # Add demographic factors
        )
        
        # Manually add demographic data to the evaluator's data attribute
        for factor in ['gender', 'ethnicity', 'age']:
            evaluator.data[factor] = data[factor]
        
        fairness_results = evaluator.evaluate_fairness_metrics(
            metrics=['demographic_parity', 'equal_opportunity', 'care_pattern_disparity']
        )
        
        print("\nFairness Evaluation Results:")
        print(fairness_results)
    except Exception as e:
        print(f"\nError in fairness evaluation: {str(e)}")
        print("Skipping fairness evaluation due to error.")
    
    print("\nAnalysis completed successfully. Press Ctrl+C to exit.")

if __name__ == "__main__":
    main() 