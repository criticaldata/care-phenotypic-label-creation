"""
Tests for the synthetic MIMIC data generator.

This module contains test cases for each component of the synthetic data generator,
ensuring that the generated data matches the expected structure and patterns.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.mimic.structures import Gender, AdmissionType, ICUUnit

@pytest.fixture
def generator():
    """Create a synthetic data generator with fixed seed for reproducibility."""
    return SyntheticDataGenerator(n_patients=50, seed=42)

def test_patient_generation(generator):
    """Test patient data generation."""
    patients = generator.generate_patients()
    
    # Check basic structure
    assert isinstance(patients, pd.DataFrame)
    assert len(patients) == 50
    assert all(col in patients.columns for col in [
        'subject_id', 'gender', 'anchor_age', 'anchor_year',
        'anchor_year_group', 'dod'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(patients['subject_id'])
    assert pd.api.types.is_numeric_dtype(patients['anchor_age'])
    assert pd.api.types.is_numeric_dtype(patients['anchor_year'])
    assert patients['gender'].isin([g.value for g in Gender]).all()
    
    # Check value ranges
    assert patients['anchor_age'].between(18, 90).all()
    assert patients['anchor_year'].between(2000, 2020).all()
    
    # Check death dates
    death_dates = patients['dod'].dropna()
    assert len(death_dates) > 0  # Some patients should have death dates
    for _, row in death_dates.iteritems():
        assert isinstance(row, datetime)
        assert row.year >= 2000
        assert row.year <= 2030

def test_admission_generation(generator):
    """Test admission data generation."""
    admissions = generator.generate_admissions()
    
    # Check basic structure
    assert isinstance(admissions, pd.DataFrame)
    assert len(admissions) == 75  # 50 patients * 1.5 average admissions
    assert all(col in admissions.columns for col in [
        'hadm_id', 'subject_id', 'admission_type',
        'admittime', 'dischtime', 'deathtime'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(admissions['hadm_id'])
    assert pd.api.types.is_numeric_dtype(admissions['subject_id'])
    assert admissions['admission_type'].isin([at.value for at in AdmissionType]).all()
    
    # Check time relationships
    for _, row in admissions.iterrows():
        assert isinstance(row['admittime'], datetime)
        assert isinstance(row['dischtime'], datetime)
        assert row['dischtime'] > row['admittime']
        if row['deathtime'] is not None:
            assert row['deathtime'] >= row['admittime']
            assert row['deathtime'] <= row['dischtime']
            
    # Check patient-admission relationships
    assert admissions['subject_id'].isin(generator.patients['subject_id']).all()

def test_icu_stay_generation(generator):
    """Test ICU stay data generation."""
    icu_stays = generator.generate_icu_stays()
    
    # Check basic structure
    assert isinstance(icu_stays, pd.DataFrame)
    assert len(icu_stays) == 60  # 75 admissions * 0.8 ICU stay rate
    assert all(col in icu_stays.columns for col in [
        'stay_id', 'subject_id', 'hadm_id',
        'first_careunit', 'last_careunit',
        'intime', 'outtime', 'los'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(icu_stays['stay_id'])
    assert pd.api.types.is_numeric_dtype(icu_stays['subject_id'])
    assert pd.api.types.is_numeric_dtype(icu_stays['hadm_id'])
    assert icu_stays['first_careunit'].isin([u.value for u in ICUUnit]).all()
    assert icu_stays['last_careunit'].isin([u.value for u in ICUUnit]).all()
    
    # Check time relationships
    for _, row in icu_stays.iterrows():
        assert isinstance(row['intime'], datetime)
        assert isinstance(row['outtime'], datetime)
        assert row['outtime'] > row['intime']
        assert 0 < row['los'] <= 14  # Length of stay should be 1-14 days
        
    # Check relationships with admissions
    assert icu_stays['hadm_id'].isin(generator.admissions['hadm_id']).all()

def test_lab_event_generation(generator):
    """Test lab event data generation."""
    lab_events = generator.generate_lab_events()
    
    # Check basic structure
    assert isinstance(lab_events, pd.DataFrame)
    assert all(col in lab_events.columns for col in [
        'subject_id', 'hadm_id', 'stay_id',
        'charttime', 'specimen_id', 'itemid',
        'valuenum', 'valueuom', 'ref_range_lower',
        'ref_range_upper', 'flag'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(lab_events['valuenum'])
    assert pd.api.types.is_numeric_dtype(lab_events['ref_range_lower'])
    assert pd.api.types.is_numeric_dtype(lab_events['ref_range_upper'])
    
    # Check value ranges and flags
    for _, row in lab_events.iterrows():
        assert isinstance(row['charttime'], datetime)
        assert row['valuenum'] >= row['ref_range_lower']
        assert row['valuenum'] <= row['ref_range_upper']
        if row['flag'] is not None:
            assert row['flag'] in ['LOW', 'HIGH']
            
    # Check relationships with ICU stays
    assert lab_events['stay_id'].isin(generator.icu_stays['stay_id']).all()

def test_chart_event_generation(generator):
    """Test chart event data generation."""
    chart_events = generator.generate_chart_events()
    
    # Check basic structure
    assert isinstance(chart_events, pd.DataFrame)
    assert all(col in chart_events.columns for col in [
        'subject_id', 'hadm_id', 'stay_id',
        'charttime', 'storetime', 'itemid',
        'value', 'valuenum', 'valueuom',
        'warning', 'error'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(chart_events['valuenum'])
    
    # Check time relationships
    for _, row in chart_events.iterrows():
        assert isinstance(row['charttime'], datetime)
        assert isinstance(row['storetime'], datetime)
        assert row['storetime'] >= row['charttime']
        
    # Check warning/error flags
    for _, row in chart_events.iterrows():
        if row['warning'] is not None:
            assert row['warning'] == 'WARNING'
        if row['error'] is not None:
            assert row['error'] == 'CRITICAL'
            
    # Check relationships with ICU stays
    assert chart_events['stay_id'].isin(generator.icu_stays['stay_id']).all()

def test_clinical_score_generation(generator):
    """Test clinical score data generation."""
    clinical_scores = generator.generate_clinical_scores()
    
    # Check basic structure
    assert isinstance(clinical_scores, pd.DataFrame)
    assert all(col in clinical_scores.columns for col in [
        'subject_id', 'hadm_id', 'stay_id',
        'score_time', 'score_type', 'score_value',
        'components'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(clinical_scores['score_value'])
    
    # Check score ranges and components
    for _, row in clinical_scores.iterrows():
        assert isinstance(row['score_time'], datetime)
        assert row['score_type'] in ['SOFA', 'Charlson']
        
        if row['score_type'] == 'SOFA':
            assert 0 <= row['score_value'] <= 24
            assert all(0 <= v <= 4 for v in row['components'].values())
            assert sum(row['components'].values()) == row['score_value']
        else:  # Charlson
            assert 0 <= row['score_value'] <= 33
            assert all(v in [0, 1] for v in row['components'].values())
            
    # Check relationships with ICU stays
    assert clinical_scores['stay_id'].isin(generator.icu_stays['stay_id']).all()

def test_data_relationships(generator):
    """Test relationships between different data components."""
    # Generate all data
    data = generator.generate_all()
    
    # Check patient-admission relationships
    patient_ids = set(data['patients']['subject_id'])
    admission_patient_ids = set(data['admissions']['subject_id'])
    assert admission_patient_ids.issubset(patient_ids)
    
    # Check admission-ICU stay relationships
    admission_ids = set(data['admissions']['hadm_id'])
    icu_admission_ids = set(data['icu_stays']['hadm_id'])
    assert icu_admission_ids.issubset(admission_ids)
    
    # Check ICU stay-event relationships
    stay_ids = set(data['icu_stays']['stay_id'])
    lab_stay_ids = set(data['lab_events']['stay_id'])
    chart_stay_ids = set(data['chart_events']['stay_id'])
    score_stay_ids = set(data['clinical_scores']['stay_id'])
    
    assert lab_stay_ids.issubset(stay_ids)
    assert chart_stay_ids.issubset(stay_ids)
    assert score_stay_ids.issubset(stay_ids)
    
    # Check temporal relationships
    for _, stay in data['icu_stays'].iterrows():
        stay_lab_events = data['lab_events'][data['lab_events']['stay_id'] == stay['stay_id']]
        stay_chart_events = data['chart_events'][data['chart_events']['stay_id'] == stay['stay_id']]
        stay_scores = data['clinical_scores'][data['clinical_scores']['stay_id'] == stay['stay_id']]
        
        if not stay_lab_events.empty:
            assert stay_lab_events['charttime'].min() >= stay['intime']
            assert stay_lab_events['charttime'].max() <= stay['outtime']
            
        if not stay_chart_events.empty:
            assert stay_chart_events['charttime'].min() >= stay['intime']
            assert stay_chart_events['charttime'].max() <= stay['outtime']
            
        if not stay_scores.empty:
            assert stay_scores['score_time'].min() >= stay['intime']
            assert stay_scores['score_time'].max() <= stay['outtime'] 