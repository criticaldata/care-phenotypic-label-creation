"""
Validation tests for data processing.

This module contains tests to validate the data processing pipeline,
including data cleaning, transformation, and integrity checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.mimic.processor import MIMICDataProcessor
from care_phenotype_analyzer.mimic.data_formats import (
    STANDARD_COLUMNS, COLUMN_DTYPES, VALUE_CONSTRAINTS,
    REQUIRED_COLUMNS, validate_dataframe, convert_to_standard_format
)

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for processing validation."""
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

def test_data_format_validation(processor):
    """Test validation of data formats and structures."""
    # Validate lab events format
    lab_df = processor.process_lab_events()
    assert validate_dataframe(lab_df, 'lab_event'), "Lab events format validation failed"
    
    # Validate chart events format
    chart_df = processor.process_chart_events()
    assert validate_dataframe(chart_df, 'chart_event'), "Chart events format validation failed"
    
    # Validate data types
    for col, dtype in COLUMN_DTYPES['lab_event'].items():
        if col in lab_df.columns:
            assert pd.api.types.is_dtype_equal(lab_df[col].dtype, dtype), \
                f"Lab events column {col} has incorrect dtype"
    
    for col, dtype in COLUMN_DTYPES['chart_event'].items():
        if col in chart_df.columns:
            assert pd.api.types.is_dtype_equal(chart_df[col].dtype, dtype), \
                f"Chart events column {col} has incorrect dtype"

def test_data_cleaning_validation(processor):
    """Test validation of data cleaning operations."""
    # Process data
    lab_df = processor.process_lab_events()
    chart_df = processor.process_chart_events()
    
    # Validate missing value handling
    assert not lab_df['valuenum'].isnull().any(), "Lab events contain null values after cleaning"
    assert not chart_df['valuenum'].isnull().any(), "Chart events contain null values after cleaning"
    
    # Validate value constraints
    for col, constraints in VALUE_CONSTRAINTS['lab_event'].items():
        if col in lab_df.columns:
            if 'min' in constraints:
                assert (lab_df[col] >= constraints['min']).all(), \
                    f"Lab events column {col} contains values below minimum"
            if 'max' in constraints:
                assert (lab_df[col] <= constraints['max']).all(), \
                    f"Lab events column {col} contains values above maximum"
    
    for col, constraints in VALUE_CONSTRAINTS['chart_event'].items():
        if col in chart_df.columns:
            if 'min' in constraints:
                assert (chart_df[col] >= constraints['min']).all(), \
                    f"Chart events column {col} contains values below minimum"
            if 'max' in constraints:
                assert (chart_df[col] <= constraints['max']).all(), \
                    f"Chart events column {col} contains values above maximum"

def test_data_transformation_validation(processor):
    """Test validation of data transformation operations."""
    # Process data
    lab_df = processor.process_lab_events()
    chart_df = processor.process_chart_events()
    
    # Validate unit conversions
    assert all(lab_df['valueuom'].isin(['mg/dL', 'g/dL', 'mmol/L'])), \
        "Lab events contain invalid units after transformation"
    
    # Validate time transformations
    assert pd.api.types.is_datetime64_any_dtype(lab_df['charttime']), \
        "Lab events charttime is not datetime after transformation"
    assert pd.api.types.is_datetime64_any_dtype(chart_df['charttime']), \
        "Chart events charttime is not datetime after transformation"
    
    # Validate value transformations
    assert (lab_df['valuenum'] >= 0).all(), \
        "Lab events contain negative values after transformation"
    assert (chart_df['valuenum'] >= 0).all(), \
        "Chart events contain negative values after transformation"

def test_data_integrity_validation(processor):
    """Test validation of data integrity and relationships."""
    # Process data
    lab_df = processor.process_lab_events()
    chart_df = processor.process_chart_events()
    
    # Validate patient relationships
    lab_patients = set(lab_df['subject_id'].unique())
    chart_patients = set(chart_df['subject_id'].unique())
    assert lab_patients.issubset(set(processor.patients['subject_id'])), \
        "Lab events contain invalid patient IDs"
    assert chart_patients.issubset(set(processor.patients['subject_id'])), \
        "Chart events contain invalid patient IDs"
    
    # Validate admission relationships
    lab_admissions = set(lab_df['hadm_id'].unique())
    chart_admissions = set(chart_df['hadm_id'].unique())
    assert lab_admissions.issubset(set(processor.admissions['hadm_id'])), \
        "Lab events contain invalid admission IDs"
    assert chart_admissions.issubset(set(processor.admissions['hadm_id'])), \
        "Chart events contain invalid admission IDs"
    
    # Validate ICU stay relationships
    lab_stays = set(lab_df['stay_id'].unique())
    chart_stays = set(chart_df['stay_id'].unique())
    assert lab_stays.issubset(set(processor.icu_stays['stay_id'])), \
        "Lab events contain invalid ICU stay IDs"
    assert chart_stays.issubset(set(processor.icu_stays['stay_id'])), \
        "Chart events contain invalid ICU stay IDs"

def test_temporal_consistency_validation(processor):
    """Test validation of temporal consistency in processed data."""
    # Process data
    lab_df = processor.process_lab_events()
    chart_df = processor.process_chart_events()
    
    # Validate temporal relationships with admissions
    for _, admission in processor.admissions.iterrows():
        patient_lab = lab_df[lab_df['hadm_id'] == admission['hadm_id']]
        patient_chart = chart_df[chart_df['hadm_id'] == admission['hadm_id']]
        
        if not patient_lab.empty:
            assert (patient_lab['charttime'] >= admission['admittime']).all(), \
                f"Lab events before admission for hadm_id {admission['hadm_id']}"
            assert (patient_lab['charttime'] <= admission['dischtime']).all(), \
                f"Lab events after discharge for hadm_id {admission['hadm_id']}"
        
        if not patient_chart.empty:
            assert (patient_chart['charttime'] >= admission['admittime']).all(), \
                f"Chart events before admission for hadm_id {admission['hadm_id']}"
            assert (patient_chart['charttime'] <= admission['dischtime']).all(), \
                f"Chart events after discharge for hadm_id {admission['hadm_id']}"
    
    # Validate temporal relationships with ICU stays
    for _, stay in processor.icu_stays.iterrows():
        patient_lab = lab_df[lab_df['stay_id'] == stay['stay_id']]
        patient_chart = chart_df[chart_df['stay_id'] == stay['stay_id']]
        
        if not patient_lab.empty:
            assert (patient_lab['charttime'] >= stay['intime']).all(), \
                f"Lab events before ICU admission for stay_id {stay['stay_id']}"
            assert (patient_lab['charttime'] <= stay['outtime']).all(), \
                f"Lab events after ICU discharge for stay_id {stay['stay_id']}"
        
        if not patient_chart.empty:
            assert (patient_chart['charttime'] >= stay['intime']).all(), \
                f"Chart events before ICU admission for stay_id {stay['stay_id']}"
            assert (patient_chart['charttime'] <= stay['outtime']).all(), \
                f"Chart events after ICU discharge for stay_id {stay['stay_id']}"

def test_data_processing_edge_cases(processor):
    """Test validation of data processing for edge cases."""
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
    
    # Create processor with edge case data
    edge_processor = MIMICDataProcessor(**edge_data)
    
    # Process data
    lab_df = edge_processor.process_lab_events()
    chart_df = edge_processor.process_chart_events()
    
    # Validate edge case handling
    assert not lab_df.empty, "Edge case lab events processing failed"
    assert not chart_df.empty, "Edge case chart events processing failed"
    assert validate_dataframe(lab_df, 'lab_event'), "Edge case lab events format validation failed"
    assert validate_dataframe(chart_df, 'chart_event'), "Edge case chart events format validation failed" 