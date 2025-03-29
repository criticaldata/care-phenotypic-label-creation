"""
Validation tests for phenotype creation and analysis results.

This module contains tests to validate the results of phenotype creation
and analysis, including clinical separation, pattern consistency, and
unexplained variation checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator
from care_phenotype_analyzer.pattern_analyzer import CarePatternAnalyzer
from care_phenotype_analyzer.fairness_evaluator import FairnessEvaluator

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for result validation."""
    generator = SyntheticDataGenerator(n_patients=50, seed=42)
    return generator.generate_all()

@pytest.fixture
def phenotype_creator(synthetic_data):
    """Create a CarePhenotypeCreator instance with synthetic data."""
    # Create sample care pattern data
    care_patterns = pd.DataFrame({
        'subject_id': synthetic_data['patients']['subject_id'],
        'timestamp': [datetime.now() + timedelta(hours=i) for i in range(len(synthetic_data['patients']))],
        'pattern_1': np.random.normal(0, 1, len(synthetic_data['patients'])),
        'pattern_2': np.random.normal(0, 1, len(synthetic_data['patients'])),
        'clinical_factor_1': np.random.normal(0, 1, len(synthetic_data['patients'])),
        'clinical_factor_2': np.random.normal(0, 1, len(synthetic_data['patients']))
    })
    
    return CarePhenotypeCreator(
        data=care_patterns,
        clinical_factors=['clinical_factor_1', 'clinical_factor_2']
    )

@pytest.fixture
def pattern_analyzer(synthetic_data):
    """Create a CarePatternAnalyzer instance with synthetic data."""
    # Create sample care pattern data
    care_patterns = pd.DataFrame({
        'subject_id': synthetic_data['patients']['subject_id'],
        'timestamp': [datetime.now() + timedelta(hours=i) for i in range(len(synthetic_data['patients']))],
        'pattern_1': np.random.normal(0, 1, len(synthetic_data['patients'])),
        'pattern_2': np.random.normal(0, 1, len(synthetic_data['patients']))
    })
    
    return CarePatternAnalyzer(care_patterns)

@pytest.fixture
def fairness_evaluator(synthetic_data):
    """Create a FairnessEvaluator instance with synthetic data."""
    # Create sample phenotype labels
    phenotype_labels = pd.Series(
        np.random.randint(0, 3, len(synthetic_data['patients'])),
        index=synthetic_data['patients']['subject_id']
    )
    
    # Create sample demographic data
    demographic_data = pd.DataFrame({
        'subject_id': synthetic_data['patients']['subject_id'],
        'age': np.random.randint(18, 90, len(synthetic_data['patients'])),
        'gender': np.random.choice(['M', 'F'], len(synthetic_data['patients'])),
        'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], len(synthetic_data['patients']))
    })
    
    return FairnessEvaluator(phenotype_labels, demographic_data)

def test_phenotype_label_validation(phenotype_creator):
    """Test validation of phenotype labels."""
    # Create phenotype labels
    labels = phenotype_creator.create_phenotype_labels()
    
    # Validate label structure
    assert isinstance(labels, pd.Series), "Labels should be a pandas Series"
    assert labels.index.name == 'subject_id', "Labels should be indexed by subject_id"
    assert labels.dtype in ['int32', 'int64'], "Labels should be integers"
    
    # Validate label values
    assert labels.min() >= 0, "Labels should be non-negative"
    assert len(labels.unique()) > 1, "Labels should have multiple unique values"
    
    # Validate label coverage
    assert len(labels) == len(phenotype_creator.data), "Labels should cover all subjects"
    assert not labels.isnull().any(), "Labels should not contain null values"

def test_clinical_separation_validation(phenotype_creator):
    """Test validation of clinical separation in phenotype labels."""
    # Create phenotype labels
    labels = phenotype_creator.create_phenotype_labels()
    
    # Validate clinical separation
    separation_results = phenotype_creator._check_clinical_separation(labels)
    
    # Check separation results structure
    assert isinstance(separation_results, dict), "Separation results should be a dictionary"
    assert all(factor in separation_results for factor in phenotype_creator.clinical_factors), \
        "Separation results should include all clinical factors"
    
    # Check statistical significance
    for factor, results in separation_results.items():
        assert 'f_statistic' in results, f"Missing F-statistic for factor {factor}"
        assert 'p_value' in results, f"Missing p-value for factor {factor}"
        assert results['p_value'] >= 0, f"Invalid p-value for factor {factor}"
        assert results['p_value'] <= 1, f"Invalid p-value for factor {factor}"

def test_pattern_consistency_validation(phenotype_creator):
    """Test validation of pattern consistency in phenotype labels."""
    # Create phenotype labels
    labels = phenotype_creator.create_phenotype_labels()
    
    # Validate pattern consistency
    consistency_results = phenotype_creator._check_pattern_consistency(labels)
    
    # Check consistency results structure
    assert isinstance(consistency_results, dict), "Consistency results should be a dictionary"
    assert 'pattern_1' in consistency_results, "Missing results for pattern_1"
    assert 'pattern_2' in consistency_results, "Missing results for pattern_2"
    
    # Check consistency metrics
    for pattern, results in consistency_results.items():
        assert 'correlation' in results, f"Missing correlation for pattern {pattern}"
        assert 'p_value' in results, f"Missing p-value for pattern {pattern}"
        assert -1 <= results['correlation'] <= 1, f"Invalid correlation for pattern {pattern}"
        assert results['p_value'] >= 0, f"Invalid p-value for pattern {pattern}"
        assert results['p_value'] <= 1, f"Invalid p-value for pattern {pattern}"

def test_unexplained_variation_validation(phenotype_creator):
    """Test validation of unexplained variation in phenotype labels."""
    # Create phenotype labels
    labels = phenotype_creator.create_phenotype_labels()
    
    # Validate unexplained variation
    variation_results = phenotype_creator._check_unexplained_variation(labels)
    
    # Check variation results structure
    assert isinstance(variation_results, dict), "Variation results should be a dictionary"
    assert 'r_squared' in variation_results, "Missing R-squared value"
    assert 'adjusted_r_squared' in variation_results, "Missing adjusted R-squared value"
    
    # Check variation metrics
    assert 0 <= variation_results['r_squared'] <= 1, "Invalid R-squared value"
    assert variation_results['adjusted_r_squared'] <= variation_results['r_squared'], \
        "Adjusted R-squared should not exceed R-squared"

def test_pattern_analysis_validation(pattern_analyzer):
    """Test validation of pattern analysis results."""
    # Analyze patterns
    pattern_results = pattern_analyzer.analyze_patterns()
    
    # Validate pattern results structure
    assert isinstance(pattern_results, dict), "Pattern results should be a dictionary"
    assert 'pattern_1' in pattern_results, "Missing results for pattern_1"
    assert 'pattern_2' in pattern_results, "Missing results for pattern_2"
    
    # Validate pattern metrics
    for pattern, results in pattern_results.items():
        assert 'mean' in results, f"Missing mean for pattern {pattern}"
        assert 'std' in results, f"Missing standard deviation for pattern {pattern}"
        assert 'min' in results, f"Missing minimum for pattern {pattern}"
        assert 'max' in results, f"Missing maximum for pattern {pattern}"
        
        # Check statistical properties
        assert results['std'] >= 0, f"Invalid standard deviation for pattern {pattern}"
        assert results['min'] <= results['mean'] <= results['max'], \
            f"Invalid mean value for pattern {pattern}"

def test_fairness_metrics_validation(fairness_evaluator):
    """Test validation of fairness metrics."""
    # Calculate fairness metrics
    fairness_results = fairness_evaluator.calculate_fairness_metrics()
    
    # Validate fairness results structure
    assert isinstance(fairness_results, dict), "Fairness results should be a dictionary"
    assert 'demographic_parity' in fairness_results, "Missing demographic parity metric"
    assert 'equalized_odds' in fairness_results, "Missing equalized odds metric"
    assert 'disparate_impact' in fairness_results, "Missing disparate impact metric"
    
    # Validate fairness metrics
    for metric, results in fairness_results.items():
        assert isinstance(results, dict), f"Results for {metric} should be a dictionary"
        
        if metric == 'demographic_parity':
            assert all(0 <= value <= 1 for value in results.values()), \
                "Demographic parity values should be between 0 and 1"
        elif metric == 'equalized_odds':
            assert all(0 <= value <= 1 for value in results.values()), \
                "Equalized odds values should be between 0 and 1"
        elif metric == 'disparate_impact':
            assert all(value >= 0 for value in results.values()), \
                "Disparate impact values should be non-negative"

def test_result_consistency_validation(phenotype_creator, pattern_analyzer, fairness_evaluator):
    """Test validation of consistency between different result types."""
    # Create phenotype labels
    labels = phenotype_creator.create_phenotype_labels()
    
    # Analyze patterns
    pattern_results = pattern_analyzer.analyze_patterns()
    
    # Calculate fairness metrics
    fairness_results = fairness_evaluator.calculate_fairness_metrics()
    
    # Validate consistency between results
    assert len(labels) == len(pattern_analyzer.care_patterns), \
        "Number of labels should match number of care patterns"
    
    # Validate that all subjects have both labels and pattern data
    common_subjects = set(labels.index) & set(pattern_analyzer.care_patterns['subject_id'])
    assert len(common_subjects) > 0, "No common subjects between labels and patterns"
    
    # Validate that all subjects in fairness evaluation have labels
    fairness_subjects = set(fairness_evaluator.phenotype_labels.index)
    assert fairness_subjects.issubset(set(labels.index)), \
        "Fairness evaluation subjects should be subset of labeled subjects" 