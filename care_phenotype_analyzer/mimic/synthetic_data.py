"""
Module for generating synthetic MIMIC-like data for testing purposes.

This module provides functionality to generate synthetic data that mimics
the structure and patterns of real MIMIC-IV data, including:
- Patient demographics
- Admission information
- ICU stays
- Lab events
- Chart events
- Clinical scores
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from .structures import (
    Patient, Admission, ICUStay, LabEvent, ChartEvent,
    ClinicalScore, Gender, AdmissionType, ICUUnit
)

class SyntheticDataGenerator:
    """Class for generating synthetic MIMIC-like data."""
    
    def __init__(self, n_patients: int = 100, seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            n_patients: Number of patients to generate
            seed: Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize data storage
        self.patients = None
        self.admissions = None
        self.icu_stays = None
        self.lab_events = None
        self.chart_events = None
        self.clinical_scores = None
        
    def generate_patients(self) -> pd.DataFrame:
        """Generate synthetic patient data."""
        # Generate base patient information
        data = {
            'subject_id': range(1, self.n_patients + 1),
            'gender': np.random.choice([g.value for g in Gender], self.n_patients),
            'anchor_age': np.random.randint(18, 90, self.n_patients),
            'anchor_year': np.random.randint(2000, 2020, self.n_patients),
            'anchor_year_group': np.random.choice(['2000-2005', '2006-2010', '2011-2015', '2016-2020'], 
                                               self.n_patients)
        }
        
        # Add death dates for some patients
        death_prob = 0.2
        death_mask = np.random.binomial(1, death_prob, self.n_patients).astype(bool)
        death_dates = []
        for i in range(self.n_patients):
            if death_mask[i]:
                # Generate death date after anchor year
                anchor_date = datetime(data['anchor_year'][i], 1, 1)
                days_after_anchor = np.random.randint(0, 3650)  # Up to 10 years
                death_date = anchor_date + timedelta(days=days_after_anchor)
                death_dates.append(death_date)
            else:
                death_dates.append(None)
        data['dod'] = death_dates
        
        self.patients = pd.DataFrame(data)
        return self.patients
        
    def generate_admissions(self) -> pd.DataFrame:
        """Generate synthetic admission data."""
        if self.patients is None:
            self.generate_patients()
            
        # Generate multiple admissions per patient
        n_admissions = int(self.n_patients * 1.5)  # Average 1.5 admissions per patient
        data = {
            'hadm_id': range(1, n_admissions + 1),
            'subject_id': np.random.choice(self.patients['subject_id'], n_admissions),
            'admission_type': np.random.choice([at.value for at in AdmissionType], n_admissions),
            'admittime': [],
            'dischtime': [],
            'deathtime': []
        }
        
        # Generate admission times and durations
        for i in range(n_admissions):
            subject_id = data['subject_id'][i]
            patient = self.patients[self.patients['subject_id'] == subject_id].iloc[0]
            
            # Generate admission date after patient's anchor year
            anchor_date = datetime(patient['anchor_year'], 1, 1)
            days_after_anchor = np.random.randint(0, 3650)
            admittime = anchor_date + timedelta(days=days_after_anchor)
            
            # Generate discharge date (1-30 days after admission)
            los = np.random.randint(1, 31)
            dischtime = admittime + timedelta(days=los)
            
            # Generate death time if patient died
            if patient['dod'] is not None and dischtime > patient['dod']:
                dischtime = patient['dod']
                deathtime = patient['dod']
            else:
                deathtime = None
                
            data['admittime'].append(admittime)
            data['dischtime'].append(dischtime)
            data['deathtime'].append(deathtime)
            
        self.admissions = pd.DataFrame(data)
        return self.admissions
        
    def generate_icu_stays(self) -> pd.DataFrame:
        """Generate synthetic ICU stay data."""
        if self.admissions is None:
            self.generate_admissions()
            
        # Generate ICU stays for most admissions
        n_stays = int(len(self.admissions) * 0.8)  # 80% of admissions have ICU stays
        data = {
            'stay_id': range(1, n_stays + 1),
            'subject_id': [],
            'hadm_id': [],
            'first_careunit': [],
            'last_careunit': [],
            'intime': [],
            'outtime': [],
            'los': []
        }
        
        # Generate ICU stays
        for i in range(n_stays):
            # Select random admission
            admission = self.admissions.iloc[np.random.randint(len(self.admissions))]
            
            # Generate ICU stay details
            data['subject_id'].append(admission['subject_id'])
            data['hadm_id'].append(admission['hadm_id'])
            
            # Select ICU unit
            unit = np.random.choice([u.value for u in ICUUnit])
            data['first_careunit'].append(unit)
            data['last_careunit'].append(unit)
            
            # Generate ICU stay times
            intime = admission['admittime'] + timedelta(hours=np.random.randint(0, 24))
            los = np.random.randint(1, 15)  # 1-14 days
            outtime = min(intime + timedelta(days=los), admission['dischtime'])
            
            data['intime'].append(intime)
            data['outtime'].append(outtime)
            data['los'].append((outtime - intime).total_seconds() / (24 * 3600))
            
        self.icu_stays = pd.DataFrame(data)
        return self.icu_stays
        
    def generate_lab_events(self) -> pd.DataFrame:
        """Generate synthetic lab event data."""
        if self.icu_stays is None:
            self.generate_icu_stays()
            
        # Define common lab tests and their reference ranges
        lab_tests = {
            'Sodium': {'min': 135, 'max': 145, 'unit': 'mEq/L'},
            'Potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mEq/L'},
            'Chloride': {'min': 96, 'max': 106, 'unit': 'mEq/L'},
            'Bicarbonate': {'min': 22, 'max': 29, 'unit': 'mEq/L'},
            'BUN': {'min': 7, 'max': 20, 'unit': 'mg/dL'},
            'Creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL'},
            'Glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
            'Calcium': {'min': 8.5, 'max': 10.2, 'unit': 'mg/dL'},
            'Magnesium': {'min': 1.7, 'max': 2.2, 'unit': 'mg/dL'},
            'Phosphate': {'min': 2.5, 'max': 4.5, 'unit': 'mg/dL'}
        }
        
        # Generate lab events
        events = []
        for _, stay in self.icu_stays.iterrows():
            # Generate 2-10 lab events per day
            n_days = int(stay['los'])
            for day in range(n_days):
                n_events = np.random.randint(2, 11)
                for _ in range(n_events):
                    # Generate event time
                    event_time = stay['intime'] + timedelta(
                        days=day,
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    # Select random lab test
                    test_name = np.random.choice(list(lab_tests.keys()))
                    test_info = lab_tests[test_name]
                    
                    # Generate value with some randomness
                    mean_value = (test_info['min'] + test_info['max']) / 2
                    std_value = (test_info['max'] - test_info['min']) / 4
                    value = np.random.normal(mean_value, std_value)
                    
                    # Add some flags
                    flag = None
                    if value < test_info['min']:
                        flag = 'LOW'
                    elif value > test_info['max']:
                        flag = 'HIGH'
                        
                    events.append({
                        'subject_id': stay['subject_id'],
                        'hadm_id': stay['hadm_id'],
                        'stay_id': stay['stay_id'],
                        'charttime': event_time,
                        'specimen_id': len(events) + 1,
                        'itemid': len(events) + 1000,  # Arbitrary itemid
                        'valuenum': value,
                        'valueuom': test_info['unit'],
                        'ref_range_lower': test_info['min'],
                        'ref_range_upper': test_info['max'],
                        'flag': flag
                    })
                    
        self.lab_events = pd.DataFrame(events)
        return self.lab_events
        
    def generate_chart_events(self) -> pd.DataFrame:
        """Generate synthetic chart event data."""
        if self.icu_stays is None:
            self.generate_icu_stays()
            
        # Define common vital signs and their ranges
        vital_signs = {
            'Heart Rate': {'min': 60, 'max': 100, 'unit': 'bpm'},
            'Systolic BP': {'min': 90, 'max': 140, 'unit': 'mmHg'},
            'Diastolic BP': {'min': 60, 'max': 90, 'unit': 'mmHg'},
            'Respiratory Rate': {'min': 12, 'max': 20, 'unit': 'breaths/min'},
            'Temperature': {'min': 36.1, 'max': 37.2, 'unit': '°C'},
            'SpO2': {'min': 95, 'max': 100, 'unit': '%'}
        }
        
        # Generate chart events
        events = []
        for _, stay in self.icu_stays.iterrows():
            # Generate 4-12 vital sign measurements per day
            n_days = int(stay['los'])
            for day in range(n_days):
                n_events = np.random.randint(4, 13)
                for _ in range(n_events):
                    # Generate event time
                    event_time = stay['intime'] + timedelta(
                        days=day,
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    # Select random vital sign
                    vital_name = np.random.choice(list(vital_signs.keys()))
                    vital_info = vital_signs[vital_name]
                    
                    # Generate value with some randomness
                    mean_value = (vital_info['min'] + vital_info['max']) / 2
                    std_value = (vital_info['max'] - vital_info['min']) / 4
                    value = np.random.normal(mean_value, std_value)
                    
                    # Add some warnings/errors
                    warning = None
                    error = None
                    if value < vital_info['min'] * 0.8:
                        warning = 'WARNING'
                    elif value > vital_info['max'] * 1.2:
                        warning = 'WARNING'
                    elif value < vital_info['min'] * 0.6:
                        error = 'CRITICAL'
                    elif value > vital_info['max'] * 1.4:
                        error = 'CRITICAL'
                        
                    events.append({
                        'subject_id': stay['subject_id'],
                        'hadm_id': stay['hadm_id'],
                        'stay_id': stay['stay_id'],
                        'charttime': event_time,
                        'storetime': event_time + timedelta(minutes=np.random.randint(1, 30)),
                        'itemid': len(events) + 2000,  # Arbitrary itemid
                        'value': str(value),
                        'valuenum': value,
                        'valueuom': vital_info['unit'],
                        'warning': warning,
                        'error': error
                    })
                    
        self.chart_events = pd.DataFrame(events)
        return self.chart_events
        
    def generate_clinical_scores(self) -> pd.DataFrame:
        """Generate synthetic clinical score data."""
        if self.icu_stays is None:
            self.generate_icu_stays()
            
        # Generate scores for each ICU stay
        scores = []
        for _, stay in self.icu_stays.iterrows():
            # Generate daily scores
            n_days = int(stay['los'])
            for day in range(n_days):
                # Generate SOFA score (0-24)
                sofa_score = np.random.randint(0, 25)
                sofa_components = {
                    'respiratory': np.random.randint(0, 5),
                    'coagulation': np.random.randint(0, 5),
                    'liver': np.random.randint(0, 5),
                    'cardiovascular': np.random.randint(0, 5),
                    'cns': np.random.randint(0, 5),
                    'renal': np.random.randint(0, 5)
                }
                
                # Generate Charlson score (0-33)
                charlson_score = np.random.randint(0, 34)
                charlson_components = {
                    'myocardial_infarction': np.random.binomial(1, 0.1),
                    'congestive_heart_failure': np.random.binomial(1, 0.1),
                    'peripheral_vascular_disease': np.random.binomial(1, 0.1),
                    'cerebrovascular_disease': np.random.binomial(1, 0.1),
                    'dementia': np.random.binomial(1, 0.1),
                    'chronic_pulmonary_disease': np.random.binomial(1, 0.1),
                    'rheumatic_disease': np.random.binomial(1, 0.1),
                    'peptic_ulcer_disease': np.random.binomial(1, 0.1),
                    'mild_liver_disease': np.random.binomial(1, 0.1),
                    'diabetes_without_complications': np.random.binomial(1, 0.1),
                    'diabetes_with_complications': np.random.binomial(1, 0.1),
                    'hemiplegia_or_paraplegia': np.random.binomial(1, 0.1),
                    'renal_disease': np.random.binomial(1, 0.1),
                    'malignancy': np.random.binomial(1, 0.1),
                    'moderate_or_severe_liver_disease': np.random.binomial(1, 0.1),
                    'metastatic_solid_tumor': np.random.binomial(1, 0.1),
                    'aids': np.random.binomial(1, 0.1)
                }
                
                # Generate score time
                score_time = stay['intime'] + timedelta(days=day)
                
                # Add SOFA score
                scores.append({
                    'subject_id': stay['subject_id'],
                    'hadm_id': stay['hadm_id'],
                    'stay_id': stay['stay_id'],
                    'score_time': score_time,
                    'score_type': 'SOFA',
                    'score_value': sofa_score,
                    'components': sofa_components
                })
                
                # Add Charlson score
                scores.append({
                    'subject_id': stay['subject_id'],
                    'hadm_id': stay['hadm_id'],
                    'stay_id': stay['stay_id'],
                    'score_time': score_time,
                    'score_type': 'Charlson',
                    'score_value': charlson_score,
                    'components': charlson_components
                })
                
        self.clinical_scores = pd.DataFrame(scores)
        return self.clinical_scores
        
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all synthetic data.
        
        Returns:
            Dictionary containing all generated DataFrames
        """
        self.generate_patients()
        self.generate_admissions()
        self.generate_icu_stays()
        self.generate_lab_events()
        self.generate_chart_events()
        self.generate_clinical_scores()
        
        return {
            'patients': self.patients,
            'admissions': self.admissions,
            'icu_stays': self.icu_stays,
            'lab_events': self.lab_events,
            'chart_events': self.chart_events,
            'clinical_scores': self.clinical_scores
        } 