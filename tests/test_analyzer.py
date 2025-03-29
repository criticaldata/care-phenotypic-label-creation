"""
Tests for the care phenotype analyzer package.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from care_phenotype_analyzer import CarePhenotypeCreator, CarePatternAnalyzer, FairnessEvaluator
import matplotlib.pyplot as plt

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample data with clinical factors and care patterns
    n_samples = 100
    np.random.seed(42)
    
    # Generate timestamps
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate clinical factors
    sofa_scores = np.random.randint(0, 24, n_samples)
    charlson_scores = np.random.randint(0, 10, n_samples)
    
    # Generate care patterns
    lab_test_freq = np.random.normal(2, 0.5, n_samples)  # Base frequency
    lab_test_freq += 0.1 * sofa_scores  # Add some correlation with illness severity
    
    # Add some unexplained variation
    unexplained_variation = np.random.normal(0, 0.3, n_samples)
    lab_test_freq += unexplained_variation
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'sofa_score': sofa_scores,
        'charlson_score': charlson_scores,
        'lab_test_frequency': lab_test_freq,
        'routine_care_frequency': np.random.normal(3, 0.5, n_samples)
    })
    
    return data

@pytest.fixture
def sample_predictions(sample_data):
    """Create sample predictions for fairness evaluation."""
    np.random.seed(42)
    return pd.Series(np.random.binomial(1, 0.3, len(sample_data)))

def test_phenotype_creator_initialization(sample_data):
    """Test that the phenotype creator initializes correctly."""
    creator = CarePhenotypeCreator(
        data=sample_data,
        clinical_factors=['sofa_score', 'charlson_score']
    )
    assert creator.data.equals(sample_data)
    assert creator.clinical_factors == ['sofa_score', 'charlson_score']

def test_phenotype_creation(sample_data):
    """Test creation of phenotype labels."""
    creator = CarePhenotypeCreator(
        data=sample_data,
        clinical_factors=['sofa_score', 'charlson_score']
    )
    
    # Create phenotypes
    labels = creator.create_phenotype_labels(
        care_patterns=['lab_test_frequency', 'routine_care_frequency'],
        n_clusters=3
    )
    
    # Check basic properties
    assert len(labels) == len(sample_data)
    assert len(np.unique(labels)) == 3
    assert all(isinstance(x, (int, np.integer)) for x in labels)

def test_clinical_factor_adjustment(sample_data):
    """Test adjustment for clinical factors."""
    creator = CarePhenotypeCreator(
        data=sample_data,
        clinical_factors=['sofa_score', 'charlson_score']
    )
    
    # Create phenotypes with and without clinical adjustment
    labels_raw = creator.create_phenotype_labels(
        care_patterns=['lab_test_frequency'],
        n_clusters=3,
        adjust_for_clinical=False
    )
    
    labels_adjusted = creator.create_phenotype_labels(
        care_patterns=['lab_test_frequency'],
        n_clusters=3,
        adjust_for_clinical=True
    )
    
    # Labels should be different when adjusting for clinical factors
    assert not np.array_equal(labels_raw, labels_adjusted)

def test_pattern_analyzer_initialization(sample_data):
    """Test that the pattern analyzer initializes correctly."""
    analyzer = CarePatternAnalyzer(sample_data)
    assert analyzer.data.equals(sample_data)

def test_measurement_frequency_analysis(sample_data):
    """Test analysis of measurement frequencies."""
    analyzer = CarePatternAnalyzer(sample_data)
    
    results = analyzer.analyze_measurement_frequency(
        measurement_column='lab_test_frequency',
        time_column='timestamp',
        clinical_factors=['sofa_score', 'charlson_score']
    )
    
    # Check basic metrics
    assert 'count' in results
    assert 'mean' in results
    assert 'std' in results
    assert 'frequency_per_day' in results
    assert 'adjusted_frequency' in results

def test_fairness_evaluator_initialization(sample_data, sample_predictions):
    """Test that the fairness evaluator initializes correctly."""
    evaluator = FairnessEvaluator(
        predictions=sample_predictions,
        true_labels=pd.Series(np.random.binomial(1, 0.3, len(sample_data))),
        phenotype_labels=pd.Series(np.random.randint(0, 3, len(sample_data))),
        clinical_factors=sample_data[['sofa_score', 'charlson_score']]
    )
    assert len(evaluator.predictions) == len(sample_data)

def test_fairness_metrics(sample_data, sample_predictions):
    """Test calculation of fairness metrics."""
    evaluator = FairnessEvaluator(
        predictions=sample_predictions,
        true_labels=pd.Series(np.random.binomial(1, 0.3, len(sample_data))),
        phenotype_labels=pd.Series(np.random.randint(0, 3, len(sample_data))),
        clinical_factors=sample_data[['sofa_score', 'charlson_score']]
    )
    
    results = evaluator.evaluate_fairness_metrics(
        metrics=['demographic_parity', 'equal_opportunity', 'predictive_parity']
    )
    
    # Check that all requested metrics are present
    assert 'demographic_parity' in results
    assert 'equal_opportunity' in results
    assert 'predictive_parity' in results

def test_care_pattern_disparity(sample_data, sample_predictions):
    """Test analysis of care pattern disparities."""
    evaluator = FairnessEvaluator(
        predictions=sample_predictions,
        true_labels=pd.Series(np.random.binomial(1, 0.3, len(sample_data))),
        phenotype_labels=pd.Series(np.random.randint(0, 3, len(sample_data))),
        clinical_factors=sample_data[['sofa_score', 'charlson_score']]
    )
    
    results = evaluator.evaluate_fairness_metrics(
        metrics=['care_pattern_disparity']
    )
    
    # Check that clinical factors are analyzed
    assert 'care_pattern_disparity' in results
    assert 'sofa_score' in results['care_pattern_disparity']
    assert 'charlson_score' in results['care_pattern_disparity']

@pytest.fixture
def phenotype_labels(sample_data):
    """Generate sample phenotype labels."""
    return pd.Series(np.random.randint(0, 3, len(sample_data)), index=sample_data.index)

def test_visualize_clinical_separation(sample_data, phenotype_labels):
    """Test visualization of clinical separation."""
    analyzer = CarePatternAnalyzer(sample_data)
    
    # Test with single clinical factor
    analyzer.visualize_clinical_separation(
        phenotype_labels=phenotype_labels,
        clinical_factors=['sofa_score']
    )
    plt.close()
    
    # Test with multiple clinical factors
    analyzer.visualize_clinical_separation(
        phenotype_labels=phenotype_labels,
        clinical_factors=['sofa_score', 'charlson_score']
    )
    plt.close()
    
    # Test with output file
    analyzer.visualize_clinical_separation(
        phenotype_labels=phenotype_labels,
        clinical_factors=['sofa_score'],
        output_file='test_separation.png'
    )
    plt.close()

def test_visualize_unexplained_variation(sample_data, phenotype_labels):
    """Test visualization of unexplained variation."""
    analyzer = CarePatternAnalyzer(sample_data)
    
    # Test with single care pattern
    analyzer.visualize_unexplained_variation(
        phenotype_labels=phenotype_labels,
        care_patterns=['lab_test_frequency'],
        clinical_factors=['sofa_score', 'charlson_score']
    )
    plt.close()
    
    # Test with multiple care patterns
    analyzer.visualize_unexplained_variation(
        phenotype_labels=phenotype_labels,
        care_patterns=['lab_test_frequency', 'routine_care_frequency'],
        clinical_factors=['sofa_score', 'charlson_score']
    )
    plt.close()
    
    # Test with output file
    analyzer.visualize_unexplained_variation(
        phenotype_labels=phenotype_labels,
        care_patterns=['lab_test_frequency'],
        clinical_factors=['sofa_score', 'charlson_score'],
        output_file='test_variation.png'
    )
    plt.close()

def test_visualize_variation_trends(sample_data, phenotype_labels):
    """Test visualization of variation trends."""
    analyzer = CarePatternAnalyzer(sample_data)
    
    # Test with single care pattern
    analyzer.visualize_variation_trends(
        phenotype_labels=phenotype_labels,
        care_patterns=['lab_test_frequency'],
        clinical_factors=['sofa_score', 'charlson_score'],
        time_column='timestamp'
    )
    plt.close()
    
    # Test with multiple care patterns
    analyzer.visualize_variation_trends(
        phenotype_labels=phenotype_labels,
        care_patterns=['lab_test_frequency', 'routine_care_frequency'],
        clinical_factors=['sofa_score', 'charlson_score'],
        time_column='timestamp'
    )
    plt.close()
    
    # Test with output file
    analyzer.visualize_variation_trends(
        phenotype_labels=phenotype_labels,
        care_patterns=['lab_test_frequency'],
        clinical_factors=['sofa_score', 'charlson_score'],
        time_column='timestamp',
        output_file='test_trends.png'
    )
    plt.close()

def test_fairness_visualization(sample_data, sample_predictions):
    """Test fairness metrics visualization."""
    evaluator = FairnessEvaluator(
        predictions=sample_predictions,
        true_labels=pd.Series(np.random.binomial(1, 0.3, len(sample_data))),
        phenotype_labels=pd.Series(np.random.randint(0, 3, len(sample_data))),
        clinical_factors=sample_data[['sofa_score', 'charlson_score']]
    )
    
    # Test fairness metrics visualization
    evaluator.visualize_fairness_metrics(
        metrics=['demographic_parity', 'equal_opportunity'],
        output_file='test_fairness.png'
    )
    plt.close()
    
    # Test bias detection visualization
    evaluator.visualize_bias_detection(output_file='test_bias.png')
    plt.close()

def test_bias_mitigation(sample_data, sample_predictions):
    """Test bias mitigation strategies."""
    evaluator = FairnessEvaluator(
        predictions=sample_predictions,
        true_labels=pd.Series(np.random.binomial(1, 0.3, len(sample_data))),
        phenotype_labels=pd.Series(np.random.randint(0, 3, len(sample_data))),
        clinical_factors=sample_data[['sofa_score', 'charlson_score']]
    )
    
    # Test reweighting strategy
    reweighted_predictions = evaluator.mitigate_bias(strategy='reweighting')
    assert len(reweighted_predictions) == len(sample_data)
    assert all(0 <= p <= 1 for p in reweighted_predictions)
    
    # Test threshold adjustment strategy
    threshold_predictions = evaluator.mitigate_bias(strategy='threshold_adjustment')
    assert len(threshold_predictions) == len(sample_data)
    assert all(p in [0, 1] for p in threshold_predictions)
    
    # Test calibration strategy
    calibrated_predictions = evaluator.mitigate_bias(strategy='calibration')
    assert len(calibrated_predictions) == len(sample_data)
    assert all(0 <= p <= 1 for p in calibrated_predictions)
    
    # Test bias mitigation visualization
    evaluator.visualize_bias_mitigation(
        strategies=['reweighting', 'threshold_adjustment', 'calibration'],
        output_file='test_mitigation.png'
    )
    plt.close()

def test_bias_mitigation_validation(sample_data, sample_predictions):
    """Test validation of bias mitigation results."""
    evaluator = FairnessEvaluator(
        predictions=sample_predictions,
        true_labels=pd.Series(np.random.binomial(1, 0.3, len(sample_data))),
        phenotype_labels=pd.Series(np.random.randint(0, 3, len(sample_data))),
        clinical_factors=sample_data[['sofa_score', 'charlson_score']]
    )
    
    # Get original fairness metrics
    original_metrics = evaluator.evaluate_fairness_metrics(['demographic_parity'])
    
    # Apply each mitigation strategy
    strategies = ['reweighting', 'threshold_adjustment', 'calibration']
    for strategy in strategies:
        mitigated_predictions = evaluator.mitigate_bias(strategy)
        
        # Temporarily replace predictions to calculate metrics
        original_predictions = evaluator.predictions
        evaluator.predictions = mitigated_predictions
        mitigated_metrics = evaluator.evaluate_fairness_metrics(['demographic_parity'])
        evaluator.predictions = original_predictions
        
        # Validate that mitigation improved fairness
        original_variance = np.var([m['positive_rate'] for m in original_metrics['demographic_parity'].values()])
        mitigated_variance = np.var([m['positive_rate'] for m in mitigated_metrics['demographic_parity'].values()])
        assert mitigated_variance <= original_variance, \
            f"{strategy} mitigation did not improve fairness" 