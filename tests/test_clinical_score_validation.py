"""
Validation tests for clinical score calculations.

This module contains tests to validate the accuracy and correctness of
clinical score calculations, including SOFA, Charlson, and other scores.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.mimic.sofa_calculator import SOFACalculator
from care_phenotype_analyzer.mimic.charlson_calculator import CharlsonCalculator
from care_phenotype_analyzer.mimic.other_scores import APACHEIICalculator, SAPSIICalculator, ElixhauserCalculator

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for clinical score validation."""
    generator = SyntheticDataGenerator(n_patients=50, seed=42)
    return generator.generate_all()

@pytest.fixture
def sofa_calculator(synthetic_data):
    """Create a SOFA calculator instance with synthetic data."""
    return SOFACalculator(
        lab_events=synthetic_data['lab_events'],
        chart_events=synthetic_data['chart_events'],
        patients=synthetic_data['patients'],
        admissions=synthetic_data['admissions'],
        icu_stays=synthetic_data['icu_stays']
    )

@pytest.fixture
def charlson_calculator(synthetic_data):
    """Create a Charlson calculator instance with synthetic data."""
    return CharlsonCalculator(
        lab_events=synthetic_data['lab_events'],
        chart_events=synthetic_data['chart_events'],
        patients=synthetic_data['patients'],
        admissions=synthetic_data['admissions'],
        icu_stays=synthetic_data['icu_stays']
    )

def test_sofa_score_validation(sofa_calculator):
    """Test SOFA score calculation validation."""
    # Calculate SOFA scores
    scores = sofa_calculator.calculate_scores()
    
    # Validate score structure
    assert isinstance(scores, dict)
    assert 'sofa' in scores
    assert isinstance(scores['sofa'], pd.DataFrame)
    
    # Validate required columns
    required_columns = ['subject_id', 'hadm_id', 'score_time', 'score_type', 'score_value', 'components']
    assert all(col in scores['sofa'].columns for col in required_columns)
    
    # Validate score ranges
    assert scores['sofa']['score_value'].min() >= 0
    assert scores['sofa']['score_value'].max() <= 24  # Maximum SOFA score is 24
    
    # Validate component scores
    for _, row in scores['sofa'].iterrows():
        components = row['components']
        total_score = sum(components.values())
        assert abs(total_score - row['score_value']) < 1e-10  # Allow for floating point precision
        
        # Validate individual component ranges
        assert all(0 <= score <= 4 for score in components.values())  # Each component max is 4

def test_charlson_score_validation(charlson_calculator):
    """Test Charlson comorbidity index validation."""
    # Calculate Charlson scores
    scores = charlson_calculator.calculate_scores()
    
    # Validate score structure
    assert isinstance(scores, dict)
    assert 'charlson' in scores
    assert isinstance(scores['charlson'], pd.DataFrame)
    
    # Validate required columns
    required_columns = ['subject_id', 'hadm_id', 'score_time', 'score_type', 'score_value', 'components']
    assert all(col in scores['charlson'].columns for col in required_columns)
    
    # Validate score ranges
    assert scores['charlson']['score_value'].min() >= 0
    assert scores['charlson']['score_value'].max() <= 29  # Maximum Charlson score is 29
    
    # Validate component scores
    for _, row in scores['charlson'].iterrows():
        components = row['components']
        total_score = sum(components.values())
        assert abs(total_score - row['score_value']) < 1e-10  # Allow for floating point precision
        
        # Validate individual component ranges
        assert all(0 <= score <= 6 for score in components.values())  # Each component max is 6

def test_clinical_score_consistency(sofa_calculator, charlson_calculator):
    """Test consistency between different clinical scores."""
    # Calculate both scores
    sofa_scores = sofa_calculator.calculate_scores()
    charlson_scores = charlson_calculator.calculate_scores()
    
    # Get common patients
    sofa_patients = set(sofa_scores['sofa']['subject_id'].unique())
    charlson_patients = set(charlson_scores['charlson']['subject_id'].unique())
    common_patients = sofa_patients.intersection(charlson_patients)
    
    # For each common patient, verify score relationships
    for patient_id in common_patients:
        # Get patient's scores
        patient_sofa = sofa_scores['sofa'][sofa_scores['sofa']['subject_id'] == patient_id]
        patient_charlson = charlson_scores['charlson'][charlson_scores['charlson']['subject_id'] == patient_id]
        
        # Verify that scores are calculated for the same admissions
        sofa_admissions = set(patient_sofa['hadm_id'].unique())
        charlson_admissions = set(patient_charlson['hadm_id'].unique())
        assert sofa_admissions == charlson_admissions, f"Score mismatch for patient {patient_id}"

def test_clinical_score_temporal_consistency(sofa_calculator):
    """Test temporal consistency of clinical scores."""
    # Calculate SOFA scores
    scores = sofa_calculator.calculate_scores()
    
    # For each patient, verify temporal consistency
    for patient_id in scores['sofa']['subject_id'].unique():
        patient_scores = scores['sofa'][scores['sofa']['subject_id'] == patient_id]
        
        # Sort by time
        patient_scores = patient_scores.sort_values('score_time')
        
        # Verify time sequence
        assert all(patient_scores['score_time'].diff().dropna() >= timedelta(hours=0)), \
            f"Invalid time sequence for patient {patient_id}"
        
        # Verify score changes are within reasonable bounds
        score_changes = patient_scores['score_value'].diff().dropna()
        assert all(abs(change) <= 4 for change in score_changes), \
            f"Unreasonable score change for patient {patient_id}"

def test_clinical_score_component_validation(sofa_calculator):
    """Test validation of clinical score components."""
    # Calculate SOFA scores
    scores = sofa_calculator.calculate_scores()
    
    # For each score, validate components
    for _, row in scores['sofa'].iterrows():
        components = row['components']
        
        # Verify all required components are present
        required_components = {'respiratory', 'coagulation', 'liver', 
                             'cardiovascular', 'cns', 'renal'}
        assert all(comp in components for comp in required_components), \
            f"Missing required components for score {row['score_value']}"
        
        # Verify component values are within valid ranges
        for component, value in components.items():
            if component == 'respiratory':
                assert 0 <= value <= 4, f"Invalid respiratory score: {value}"
            elif component == 'coagulation':
                assert 0 <= value <= 4, f"Invalid coagulation score: {value}"
            elif component == 'liver':
                assert 0 <= value <= 4, f"Invalid liver score: {value}"
            elif component == 'cardiovascular':
                assert 0 <= value <= 4, f"Invalid cardiovascular score: {value}"
            elif component == 'cns':
                assert 0 <= value <= 4, f"Invalid CNS score: {value}"
            elif component == 'renal':
                assert 0 <= value <= 4, f"Invalid renal score: {value}"

def test_clinical_score_edge_cases(sofa_calculator):
    """Test clinical score calculation for edge cases."""
    # Create edge case data
    edge_data = {
        'lab_events': pd.DataFrame({
            'subject_id': [1],
            'hadm_id': [1],
            'charttime': [datetime.now()],
            'itemid': [50912],  # Creatinine
            'valuenum': [0.0],  # Minimum value
            'valueuom': ['mg/dL']
        }),
        'chart_events': pd.DataFrame({
            'subject_id': [1],
            'hadm_id': [1],
            'charttime': [datetime.now()],
            'itemid': [220045],  # Heart Rate
            'valuenum': [300.0],  # Maximum value
            'valueuom': ['beats/min']
        }),
        'patients': pd.DataFrame({
            'subject_id': [1],
            'gender': ['F'],
            'anchor_age': [100],
            'anchor_year': [2020]
        }),
        'admissions': pd.DataFrame({
            'subject_id': [1],
            'hadm_id': [1],
            'admittime': [datetime.now()],
            'dischtime': [datetime.now() + timedelta(days=1)]
        }),
        'icu_stays': pd.DataFrame({
            'subject_id': [1],
            'hadm_id': [1],
            'stay_id': [1],
            'intime': [datetime.now()],
            'outtime': [datetime.now() + timedelta(days=1)]
        })
    }
    
    # Create calculator with edge case data
    edge_calculator = SOFACalculator(**edge_data)
    
    # Calculate scores
    scores = edge_calculator.calculate_scores()
    
    # Validate edge case handling
    assert not scores['sofa'].empty, "Edge case scores should not be empty"
    assert scores['sofa']['score_value'].iloc[0] >= 0, "Edge case score should be non-negative"
    assert scores['sofa']['score_value'].iloc[0] <= 24, "Edge case score should not exceed maximum" 