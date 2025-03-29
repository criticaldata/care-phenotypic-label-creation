"""
Integration tests for MIMIC data processing.

This module contains integration tests that verify the interaction between
synthetic data generation and MIMIC data processing components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.mimic.processor import MIMICDataProcessor
from care_phenotype_analyzer.mimic.integrity_checks import DataIntegrityChecker

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for integration testing."""
    generator = SyntheticDataGenerator(n_patients=50, seed=42)
    return generator.generate_all()

@pytest.fixture
def processor(synthetic_data):
    """Create a MIMICDataProcessor instance with synthetic data."""
    return MIMICDataProcessor(
        lab_events=synthetic_data['lab_events'],
        chart_events=synthetic_data['chart_events'],
        patients=synthetic_data['patients'],
        admissions=synthetic_data['admissions'],
        icu_stays=synthetic_data['icu_stays']
    )

@pytest.fixture
def integrity_checker():
    """Create a DataIntegrityChecker instance."""
    return DataIntegrityChecker()

def test_processor_initialization(processor, synthetic_data):
    """Test that the processor initializes correctly with synthetic data."""
    # Check that all data is loaded
    assert processor.lab_events is not None
    assert processor.chart_events is not None
    assert processor.patients is not None
    assert processor.admissions is not None
    assert processor.icu_stays is not None
    
    # Check data shapes
    assert len(processor.lab_events) == len(synthetic_data['lab_events'])
    assert len(processor.chart_events) == len(synthetic_data['chart_events'])
    assert len(processor.patients) == len(synthetic_data['patients'])
    assert len(processor.admissions) == len(synthetic_data['admissions'])
    assert len(processor.icu_stays) == len(synthetic_data['icu_stays'])

def test_data_processing_pipeline(processor):
    """Test the complete data processing pipeline."""
    # Process lab events
    processed_lab = processor.process_lab_events()
    assert isinstance(processed_lab, pd.DataFrame)
    assert len(processed_lab) > 0
    assert all(col in processed_lab.columns for col in [
        'subject_id', 'hadm_id', 'stay_id', 'charttime',
        'specimen_id', 'itemid', 'valuenum', 'valueuom',
        'ref_range_lower', 'ref_range_upper', 'flag'
    ])
    
    # Process chart events
    processed_chart = processor.process_chart_events()
    assert isinstance(processed_chart, pd.DataFrame)
    assert len(processed_chart) > 0
    assert all(col in processed_chart.columns for col in [
        'subject_id', 'hadm_id', 'stay_id', 'charttime',
        'storetime', 'itemid', 'value', 'valuenum',
        'valueuom', 'warning', 'error'
    ])
    
    # Calculate clinical scores
    clinical_scores = processor.calculate_clinical_scores()
    assert isinstance(clinical_scores, dict)
    assert 'sofa' in clinical_scores
    assert 'charlson' in clinical_scores

def test_integrity_checks(integrity_checker, synthetic_data):
    """Test data integrity checks on synthetic data."""
    # Check patient data
    patient_checks = integrity_checker.perform_integrity_checks(
        synthetic_data['patients'],
        'patient'
    )
    assert all(len(issues) == 0 for issues in patient_checks.values())
    
    # Check admission data
    admission_checks = integrity_checker.perform_integrity_checks(
        synthetic_data['admissions'],
        'admission',
        {'patients': synthetic_data['patients']}
    )
    assert all(len(issues) == 0 for issues in admission_checks.values())
    
    # Check ICU stay data
    icu_checks = integrity_checker.perform_integrity_checks(
        synthetic_data['icu_stays'],
        'icu_stay',
        {'admissions': synthetic_data['admissions']}
    )
    assert all(len(issues) == 0 for issues in icu_checks.values())
    
    # Check lab event data
    lab_checks = integrity_checker.perform_integrity_checks(
        synthetic_data['lab_events'],
        'lab_event',
        {'admissions': synthetic_data['admissions']}
    )
    assert all(len(issues) == 0 for issues in lab_checks.values())
    
    # Check chart event data
    chart_checks = integrity_checker.perform_integrity_checks(
        synthetic_data['chart_events'],
        'chart_event',
        {'icu_stays': synthetic_data['icu_stays']}
    )
    assert all(len(issues) == 0 for issues in chart_checks.values())

def test_end_to_end_processing(processor, integrity_checker):
    """Test end-to-end data processing and validation."""
    # Process all data
    processed_lab = processor.process_lab_events()
    processed_chart = processor.process_chart_events()
    clinical_scores = processor.calculate_clinical_scores()
    
    # Validate processed data
    lab_checks = integrity_checker.perform_integrity_checks(
        processed_lab,
        'lab_event',
        {'admissions': processor.admissions}
    )
    assert all(len(issues) == 0 for issues in lab_checks.values())
    
    chart_checks = integrity_checker.perform_integrity_checks(
        processed_chart,
        'chart_event',
        {'icu_stays': processor.icu_stays}
    )
    assert all(len(issues) == 0 for issues in chart_checks.values())
    
    # Validate clinical scores
    for score_type, scores in clinical_scores.items():
        if isinstance(scores, dict):
            for component, component_scores in scores.items():
                score_checks = integrity_checker.perform_integrity_checks(
                    component_scores,
                    'clinical_score',
                    {'icu_stays': processor.icu_stays}
                )
                assert all(len(issues) == 0 for issues in score_checks.values())
        else:
            score_checks = integrity_checker.perform_integrity_checks(
                scores,
                'clinical_score',
                {'icu_stays': processor.icu_stays}
            )
            assert all(len(issues) == 0 for issues in score_checks.values())

def test_data_consistency_after_processing(processor):
    """Test data consistency after processing."""
    # Process data
    processed_lab = processor.process_lab_events()
    processed_chart = processor.process_chart_events()
    clinical_scores = processor.calculate_clinical_scores()
    
    # Check patient coverage
    lab_patients = set(processed_lab['subject_id'].unique())
    chart_patients = set(processed_chart['subject_id'].unique())
    score_patients = set()
    for scores in clinical_scores.values():
        if isinstance(scores, dict):
            for component_scores in scores.values():
                score_patients.update(component_scores['subject_id'].unique())
        else:
            score_patients.update(scores['subject_id'].unique())
    
    # All patients should have some data
    all_patients = set(processor.patients['subject_id'])
    assert lab_patients.issubset(all_patients)
    assert chart_patients.issubset(all_patients)
    assert score_patients.issubset(all_patients)
    
    # Check ICU stay coverage
    lab_stays = set(processed_lab['stay_id'].unique())
    chart_stays = set(processed_chart['stay_id'].unique())
    score_stays = set()
    for scores in clinical_scores.values():
        if isinstance(scores, dict):
            for component_scores in scores.values():
                score_stays.update(component_scores['stay_id'].unique())
        else:
            score_stays.update(scores['stay_id'].unique())
    
    # All ICU stays should have some data
    all_stays = set(processor.icu_stays['stay_id'])
    assert lab_stays.issubset(all_stays)
    assert chart_stays.issubset(all_stays)
    assert score_stays.issubset(all_stays)

def test_error_handling(processor):
    """Test error handling in the processing pipeline."""
    # Test with missing required columns
    invalid_lab = processor.lab_events.copy()
    invalid_lab = invalid_lab.drop('valuenum', axis=1)
    
    with pytest.raises(ValueError):
        processor._validate_lab_events(invalid_lab)
    
    # Test with invalid data types
    invalid_chart = processor.chart_events.copy()
    invalid_chart['valuenum'] = invalid_chart['valuenum'].astype(str)
    
    with pytest.raises(ValueError):
        processor._validate_chart_events(invalid_chart)
    
    # Test with invalid time relationships
    invalid_admissions = processor.admissions.copy()
    invalid_admissions.loc[0, 'dischtime'] = invalid_admissions.loc[0, 'admittime'] - timedelta(days=1)
    
    with pytest.raises(ValueError):
        processor._validate_admissions(invalid_admissions) 