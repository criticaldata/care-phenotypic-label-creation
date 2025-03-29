"""
MIMIC data processing module for handling MIMIC-IV data structures and transformations.

This module provides functionality for processing and analyzing MIMIC-IV data,
including lab events, chart events, and clinical scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

class MIMICDataProcessor:
    """
    A class for processing and analyzing MIMIC-IV data.
    
    This class handles the processing of various MIMIC-IV data types including:
    - Lab events (LABEVENTS)
    - Chart events (CHARTEVENTS)
    - Clinical scores (SOFA, Charlson)
    
    Attributes:
        lab_events (pd.DataFrame): DataFrame containing lab events data
        chart_events (pd.DataFrame): DataFrame containing chart events data
        patients (pd.DataFrame): DataFrame containing patient information
        admissions (pd.DataFrame): DataFrame containing admission information
        icu_stays (pd.DataFrame): DataFrame containing ICU stay information
        clinical_scores (Dict[str, pd.DataFrame]): Dictionary containing various clinical scores
        
    Methods:
        process_lab_events: Process and clean lab events data
        process_chart_events: Process and clean chart events data
        calculate_clinical_scores: Calculate various clinical scores
        validate_data: Validate input data integrity
        clean_data: Clean and preprocess data
    """
    
    # Define required columns for lab events
    REQUIRED_LAB_COLUMNS = [
        'subject_id', 'hadm_id', 'stay_id', 'charttime',
        'specimen_id', 'itemid', 'valuenum', 'valueuom',
        'ref_range_lower', 'ref_range_upper', 'flag'
    ]
    
    # Define required columns for chart events
    REQUIRED_CHART_COLUMNS = [
        'subject_id', 'hadm_id', 'stay_id', 'charttime',
        'storetime', 'itemid', 'value', 'valuenum',
        'valueuom', 'warning', 'error'
    ]
    
    # Define SOFA score components and their itemids
    SOFA_COMPONENTS = {
        'respiratory': {
            'itemids': [220277, 220210, 224690, 224689],  # PaO2/FiO2
            'thresholds': [400, 300, 200, 100],
            'scores': [0, 1, 2, 3, 4]
        },
        'coagulation': {
            'itemids': [51265, 51275, 51277],  # Platelets
            'thresholds': [150, 100, 50, 20],
            'scores': [0, 1, 2, 3, 4]
        },
        'liver': {
            'itemids': [50912, 50913],  # Bilirubin
            'thresholds': [1.2, 2.0, 6.0, 12.0],
            'scores': [0, 1, 2, 3, 4]
        },
        'cardiovascular': {
            'itemids': [220277, 220210],  # MAP and vasopressors
            'thresholds': [70],
            'scores': [0, 1, 2, 3, 4]
        },
        'cns': {
            'itemids': [220739],  # Glasgow Coma Scale
            'thresholds': [15, 13, 10, 6],
            'scores': [0, 1, 2, 3, 4]
        },
        'renal': {
            'itemids': [50912, 50913],  # Creatinine
            'thresholds': [1.2, 2.0, 3.5, 5.0],
            'scores': [0, 1, 2, 3, 4]
        }
    }
    
    # Define Charlson comorbidity components
    CHARLSON_COMPONENTS = {
        'myocardial_infarction': [41071, 41081, 41091],
        'congestive_heart_failure': [42832, 42833, 42840, 42841, 42842],
        'peripheral_vascular_disease': [4439, 4440, 4441],
        'cerebrovascular_disease': [43043, 43100, 43101, 43102, 43103],
        'dementia': [29011, 29012, 29013, 29014, 29015, 29016, 29017, 29018, 29019],
        'chronic_pulmonary_disease': [49121, 49122, 49123, 49124, 49125],
        'rheumatoid_disease': [7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148],
        'peptic_ulcer_disease': [53170, 53171, 53172, 53173, 53174],
        'mild_liver_disease': [5713, 5715, 5716, 5718, 5719],
        'diabetes_without_complications': [25000, 25001, 25002, 25003],
        'diabetes_with_complications': [25010, 25011, 25012, 25013, 25014, 25015, 25016, 25017, 25018, 25019],
        'hemiplegia_or_paraplegia': [34200, 34201, 34202, 34203, 34204, 34205, 34206, 34207, 34208, 34209],
        'renal_disease': [5853, 5854, 5855, 5856, 5857, 5858, 5859],
        'malignancy': [14000, 14001, 14002, 14003, 14004, 14005, 14006, 14007, 14008, 14009],
        'moderate_or_severe_liver_disease': [4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569],
        'metastatic_solid_tumor': [19600, 19601, 19602, 19603, 19604, 19605, 19606, 19607, 19608, 19609],
        'aids': [42000, 42001, 42002, 42003, 42004, 42005, 42006, 42007, 42008, 42009]
    }
    
    def __init__(
        self,
        lab_events: Optional[pd.DataFrame] = None,
        chart_events: Optional[pd.DataFrame] = None,
        patients: Optional[pd.DataFrame] = None,
        admissions: Optional[pd.DataFrame] = None,
        icu_stays: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the MIMICDataProcessor.
        
        Args:
            lab_events (Optional[pd.DataFrame]): DataFrame containing lab events data
            chart_events (Optional[pd.DataFrame]): DataFrame containing chart events data
            patients (Optional[pd.DataFrame]): DataFrame containing patient information
            admissions (Optional[pd.DataFrame]): DataFrame containing admission information
            icu_stays (Optional[pd.DataFrame]): DataFrame containing ICU stay information
        """
        self.lab_events = lab_events
        self.chart_events = chart_events
        self.patients = patients
        self.admissions = admissions
        self.icu_stays = icu_stays
        self.clinical_scores = {}
        
        # Validate input data if provided
        if any([lab_events is not None, chart_events is not None, 
                patients is not None, admissions is not None, 
                icu_stays is not None]):
            self.validate_data()
    
    def _calculate_sofa_component(self, 
                                df: pd.DataFrame, 
                                component: str, 
                                time_window: str = '24H') -> pd.DataFrame:
        """
        Calculate a single SOFA score component.
        
        Args:
            df (pd.DataFrame): DataFrame containing lab/chart events
            component (str): Name of the SOFA component
            time_window (str): Time window for score calculation (default: '24H')
            
        Returns:
            pd.DataFrame: DataFrame containing component scores
        """
        if component not in self.SOFA_COMPONENTS:
            raise ValueError(f"Invalid SOFA component: {component}")
            
        component_info = self.SOFA_COMPONENTS[component]
        itemids = component_info['itemids']
        thresholds = component_info['thresholds']
        scores = component_info['scores']
        
        # Filter for relevant measurements
        mask = df['itemid'].isin(itemids)
        if not mask.any():
            return pd.DataFrame()
            
        component_data = df[mask].copy()
        
        # Calculate scores based on thresholds
        component_data['score'] = 0
        for i, threshold in enumerate(thresholds):
            if component == 'respiratory':
                # Special handling for PaO2/FiO2
                mask = component_data['valuenum'] < threshold
            else:
                mask = component_data['valuenum'] > threshold
            component_data.loc[mask, 'score'] = scores[i + 1]
            
        # Group by patient and time window
        component_data['time_window'] = component_data['charttime'].dt.floor(time_window)
        scores = component_data.groupby(['subject_id', 'hadm_id', 'time_window'])['score'].max().reset_index()
        
        return scores
    
    def _calculate_charlson_component(self, 
                                    df: pd.DataFrame, 
                                    component: str) -> pd.DataFrame:
        """
        Calculate a single Charlson comorbidity component.
        
        Args:
            df (pd.DataFrame): DataFrame containing diagnoses
            component (str): Name of the Charlson component
            
        Returns:
            pd.DataFrame: DataFrame containing component presence
        """
        if component not in self.CHARLSON_COMPONENTS:
            raise ValueError(f"Invalid Charlson component: {component}")
            
        itemids = self.CHARLSON_COMPONENTS[component]
        
        # Filter for relevant diagnoses
        mask = df['itemid'].isin(itemids)
        if not mask.any():
            return pd.DataFrame()
            
        component_data = df[mask].copy()
        
        # Group by patient
        presence = component_data.groupby(['subject_id', 'hadm_id']).size().reset_index(name='count')
        presence['present'] = (presence['count'] > 0).astype(int)
        
        return presence[['subject_id', 'hadm_id', 'present']]
    
    def calculate_clinical_scores(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate various clinical scores from the data.
        
        This method calculates:
        1. SOFA scores (Sequential Organ Failure Assessment)
        2. Charlson comorbidity index
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing calculated clinical scores
        """
        if self.lab_events is None or self.chart_events is None:
            raise ValueError("Lab events and chart events data are required for clinical score calculation")
            
        # Process the data first
        lab_df = self.process_lab_events()
        chart_df = self.process_chart_events()
        
        # Calculate SOFA scores
        sofa_scores = {}
        for component in self.SOFA_COMPONENTS:
            # Determine which dataset to use based on component
            if component in ['respiratory', 'coagulation', 'liver', 'renal']:
                df = lab_df
            else:
                df = chart_df
                
            scores = self._calculate_sofa_component(df, component)
            if not scores.empty:
                sofa_scores[component] = scores
        
        # Calculate total SOFA score
        if sofa_scores:
            total_sofa = pd.concat(sofa_scores.values())
            total_sofa = total_sofa.groupby(['subject_id', 'hadm_id', 'time_window'])['score'].sum().reset_index()
            sofa_scores['total'] = total_sofa
        
        # Calculate Charlson comorbidity index
        charlson_scores = {}
        for component in self.CHARLSON_COMPONENTS:
            scores = self._calculate_charlson_component(chart_df, component)
            if not scores.empty:
                charlson_scores[component] = scores
        
        # Calculate total Charlson score
        if charlson_scores:
            total_charlson = pd.concat(charlson_scores.values())
            total_charlson = total_charlson.groupby(['subject_id', 'hadm_id'])['present'].sum().reset_index()
            charlson_scores['total'] = total_charlson
        
        # Store all scores
        self.clinical_scores = {
            'sofa': sofa_scores,
            'charlson': charlson_scores
        }
        
        return self.clinical_scores
    
    def _validate_chart_events(self) -> None:
        """
        Validate chart events data structure and content.
        
        Raises:
            ValueError: If chart events data is invalid
        """
        if self.chart_events is None:
            raise ValueError("Chart events data is not provided")
            
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_CHART_COLUMNS 
                       if col not in self.chart_events.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in chart events: {missing_cols}")
            
        # Check data types
        if not pd.api.types.is_numeric_dtype(self.chart_events['valuenum']):
            raise ValueError("valuenum column must be numeric")
            
        # Check for critical missing values
        critical_cols = ['subject_id', 'hadm_id', 'charttime', 'itemid']
        for col in critical_cols:
            if self.chart_events[col].isnull().any():
                raise ValueError(f"Critical column {col} contains missing values")
    
    def _clean_chart_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean chart measurement values.
        
        Args:
            df (pd.DataFrame): DataFrame containing chart events
            
        Returns:
            pd.DataFrame: Cleaned chart events DataFrame
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert timestamps to datetime if not already
        for col in ['charttime', 'storetime']:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        # Handle missing values in valuenum
        df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
        
        # Remove rows with missing values in critical columns
        df = df.dropna(subset=['subject_id', 'hadm_id', 'charttime', 'itemid'])
        
        # Remove duplicate measurements within 1 hour
        df = df.sort_values(['subject_id', 'hadm_id', 'itemid', 'charttime'])
        df['time_diff'] = df.groupby(['subject_id', 'hadm_id', 'itemid'])['charttime'].diff()
        df = df[~((df['time_diff'] < pd.Timedelta(hours=1)) & 
                  (df['valuenum'] == df.groupby(['subject_id', 'hadm_id', 'itemid'])['valuenum'].shift(1)))]
        
        # Drop the temporary time_diff column
        df = df.drop('time_diff', axis=1)
        
        return df
    
    def _handle_chart_warnings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle warnings and errors in chart events.
        
        Args:
            df (pd.DataFrame): DataFrame containing chart events
            
        Returns:
            pd.DataFrame: DataFrame with handled warnings and errors
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Combine warning and error information
        df['warning'] = df['warning'].fillna('')
        df['error'] = df['error'].fillna('')
        df['flag'] = df['warning'] + '|' + df['error']
        df['flag'] = df['flag'].str.strip('|')
        
        # Remove rows with critical errors
        df = df[~df['error'].str.contains('CRITICAL', case=False, na=False)]
        
        # Drop original warning and error columns
        df = df.drop(['warning', 'error'], axis=1)
        
        return df
    
    def process_chart_events(self) -> pd.DataFrame:
        """
        Process and clean chart events data.
        
        This method:
        1. Validates the chart events data structure
        2. Cleans measurement values and timestamps
        3. Handles warnings and errors
        4. Removes duplicate measurements
        5. Standardizes units where possible
        
        Returns:
            pd.DataFrame: Processed chart events data
        """
        if self.chart_events is None:
            raise ValueError("Chart events data is not provided")
            
        # Validate data structure
        self._validate_chart_events()
        
        # Clean the data
        df = self._clean_chart_values(self.chart_events)
        
        # Handle warnings and errors
        df = self._handle_chart_warnings(df)
        
        # Sort by time for easier analysis
        df = df.sort_values(['subject_id', 'hadm_id', 'charttime'])
        
        return df
    
    def _validate_lab_events(self) -> None:
        """
        Validate lab events data structure and content.
        
        Raises:
            ValueError: If lab events data is invalid
        """
        if self.lab_events is None:
            raise ValueError("Lab events data is not provided")
            
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_LAB_COLUMNS 
                       if col not in self.lab_events.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in lab events: {missing_cols}")
            
        # Check data types
        if not pd.api.types.is_numeric_dtype(self.lab_events['valuenum']):
            raise ValueError("valuenum column must be numeric")
            
        # Check for critical missing values
        critical_cols = ['subject_id', 'hadm_id', 'charttime', 'itemid']
        for col in critical_cols:
            if self.lab_events[col].isnull().any():
                raise ValueError(f"Critical column {col} contains missing values")
    
    def _clean_lab_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean lab measurement values.
        
        Args:
            df (pd.DataFrame): DataFrame containing lab events
            
        Returns:
            pd.DataFrame: Cleaned lab events DataFrame
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert charttime to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['charttime']):
            df['charttime'] = pd.to_datetime(df['charttime'])
            
        # Handle missing values in valuenum
        df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
        
        # Remove rows with missing values in critical columns
        df = df.dropna(subset=['subject_id', 'hadm_id', 'charttime', 'itemid'])
        
        # Remove duplicate measurements within 1 hour
        df = df.sort_values(['subject_id', 'hadm_id', 'itemid', 'charttime'])
        df['time_diff'] = df.groupby(['subject_id', 'hadm_id', 'itemid'])['charttime'].diff()
        df = df[~((df['time_diff'] < pd.Timedelta(hours=1)) & 
                  (df['valuenum'] == df.groupby(['subject_id', 'hadm_id', 'itemid'])['valuenum'].shift(1)))]
        
        # Drop the temporary time_diff column
        df = df.drop('time_diff', axis=1)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in lab measurements using reference ranges.
        
        Args:
            df (pd.DataFrame): DataFrame containing lab events
            
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert reference ranges to numeric
        df['ref_range_lower'] = pd.to_numeric(df['ref_range_lower'], errors='coerce')
        df['ref_range_upper'] = pd.to_numeric(df['ref_range_upper'], errors='coerce')
        
        # Flag values outside reference ranges
        df['is_outlier'] = (
            (df['valuenum'] < df['ref_range_lower']) | 
            (df['valuenum'] > df['ref_range_upper'])
        )
        
        # Add outlier information to flag column
        df['flag'] = df['flag'].fillna('')
        df['flag'] = df['flag'] + '|OUTLIER' * df['is_outlier']
        df = df.drop('is_outlier', axis=1)
        
        return df
    
    def process_lab_events(self) -> pd.DataFrame:
        """
        Process and clean lab events data.
        
        This method:
        1. Validates the lab events data structure
        2. Cleans measurement values and timestamps
        3. Handles outliers using reference ranges
        4. Removes duplicate measurements
        5. Standardizes units where possible
        
        Returns:
            pd.DataFrame: Processed lab events data
        """
        if self.lab_events is None:
            raise ValueError("Lab events data is not provided")
            
        # Validate data structure
        self._validate_lab_events()
        
        # Clean the data
        df = self._clean_lab_values(self.lab_events)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Sort by time for easier analysis
        df = df.sort_values(['subject_id', 'hadm_id', 'charttime'])
        
        return df
    
    def _validate_patients(self) -> None:
        """
        Validate patients data structure and content.
        
        Raises:
            ValueError: If patients data is invalid
        """
        if self.patients is None:
            return
            
        required_cols = ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']
        missing_cols = [col for col in required_cols if col not in self.patients.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in patients data: {missing_cols}")
            
        # Check data types
        if not pd.api.types.is_numeric_dtype(self.patients['anchor_age']):
            raise ValueError("anchor_age must be numeric")
            
        # Check for critical missing values
        if self.patients['subject_id'].isnull().any():
            raise ValueError("subject_id cannot contain missing values")
            
        # Check for invalid values
        if (self.patients['anchor_age'] < 0).any():
            raise ValueError("anchor_age cannot be negative")
            
        if (self.patients['anchor_year'] < 1900).any():
            raise ValueError("anchor_year contains invalid values")
    
    def _validate_admissions(self) -> None:
        """
        Validate admissions data structure and content.
        
        Raises:
            ValueError: If admissions data is invalid
        """
        if self.admissions is None:
            return
            
        required_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']
        missing_cols = [col for col in required_cols if col not in self.admissions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in admissions data: {missing_cols}")
            
        # Check data types
        for col in ['admittime', 'dischtime', 'deathtime']:
            if not pd.api.types.is_datetime64_any_dtype(self.admissions[col]):
                raise ValueError(f"{col} must be datetime")
                
        # Check for critical missing values
        if self.admissions['subject_id'].isnull().any() or self.admissions['hadm_id'].isnull().any():
            raise ValueError("subject_id and hadm_id cannot contain missing values")
            
        # Check for invalid time relationships
        mask = self.admissions['dischtime'] < self.admissions['admittime']
        if mask.any():
            raise ValueError("dischtime cannot be before admittime")
            
        mask = self.admissions['deathtime'] < self.admissions['admittime']
        if mask.any():
            raise ValueError("deathtime cannot be before admittime")
    
    def _validate_icu_stays(self) -> None:
        """
        Validate ICU stays data structure and content.
        
        Raises:
            ValueError: If ICU stays data is invalid
        """
        if self.icu_stays is None:
            return
            
        required_cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']
        missing_cols = [col for col in required_cols if col not in self.icu_stays.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ICU stays data: {missing_cols}")
            
        # Check data types
        for col in ['intime', 'outtime']:
            if not pd.api.types.is_datetime64_any_dtype(self.icu_stays[col]):
                raise ValueError(f"{col} must be datetime")
                
        # Check for critical missing values
        if self.icu_stays['stay_id'].isnull().any():
            raise ValueError("stay_id cannot contain missing values")
            
        # Check for invalid time relationships
        mask = self.icu_stays['outtime'] < self.icu_stays['intime']
        if mask.any():
            raise ValueError("outtime cannot be before intime")
    
    def _validate_data_consistency(self) -> None:
        """
        Validate consistency across different data tables.
        
        Raises:
            ValueError: If data consistency checks fail
        """
        if self.patients is not None and self.admissions is not None:
            # Check that all admission subject_ids exist in patients
            missing_subjects = set(self.admissions['subject_id']) - set(self.patients['subject_id'])
            if missing_subjects:
                raise ValueError(f"Found admissions for non-existent subjects: {missing_subjects}")
                
        if self.admissions is not None and self.icu_stays is not None:
            # Check that all ICU stay hadm_ids exist in admissions
            missing_admissions = set(self.icu_stays['hadm_id']) - set(self.admissions['hadm_id'])
            if missing_admissions:
                raise ValueError(f"Found ICU stays for non-existent admissions: {missing_admissions}")
                
        if self.lab_events is not None and self.admissions is not None:
            # Check that all lab event hadm_ids exist in admissions
            missing_admissions = set(self.lab_events['hadm_id']) - set(self.admissions['hadm_id'])
            if missing_admissions:
                raise ValueError(f"Found lab events for non-existent admissions: {missing_admissions}")
                
        if self.chart_events is not None and self.icu_stays is not None:
            # Check that all chart event stay_ids exist in ICU stays
            missing_stays = set(self.chart_events['stay_id']) - set(self.icu_stays['stay_id'])
            if missing_stays:
                raise ValueError(f"Found chart events for non-existent ICU stays: {missing_stays}")
    
    def validate_data(self) -> None:
        """
        Validate the input data integrity.
        
        This method checks:
        1. Required columns are present
        2. Data types are correct
        3. No missing values in critical columns
        4. Data consistency across tables
        
        Raises:
            ValueError: If data validation fails
        """
        # Validate each data type
        self._validate_patients()
        self._validate_admissions()
        self._validate_icu_stays()
        self._validate_lab_events()
        self._validate_chart_events()
        
        # Validate consistency across tables
        self._validate_data_consistency()
    
    def _clean_patients(self) -> None:
        """
        Clean patients data.
        """
        if self.patients is None:
            return
            
        # Convert timestamps to datetime
        if 'dod' in self.patients.columns:
            self.patients['dod'] = pd.to_datetime(self.patients['dod'])
            
        # Handle missing values
        self.patients['gender'] = self.patients['gender'].fillna('Unknown')
        self.patients['anchor_age'] = self.patients['anchor_age'].fillna(
            self.patients['anchor_age'].median()
        )
        
        # Remove invalid ages
        self.patients = self.patients[self.patients['anchor_age'] <= 120]
        
        # Sort by subject_id
        self.patients = self.patients.sort_values('subject_id')
    
    def _clean_admissions(self) -> None:
        """
        Clean admissions data.
        """
        if self.admissions is None:
            return
            
        # Convert timestamps to datetime
        for col in ['admittime', 'dischtime', 'deathtime']:
            if col in self.admissions.columns:
                self.admissions[col] = pd.to_datetime(self.admissions[col])
                
        # Handle missing values
        self.admissions['deathtime'] = self.admissions['deathtime'].fillna(pd.NaT)
        
        # Remove invalid time relationships
        self.admissions = self.admissions[
            self.admissions['dischtime'] >= self.admissions['admittime']
        ]
        
        # Sort by subject_id and admittime
        self.admissions = self.admissions.sort_values(['subject_id', 'admittime'])
    
    def _clean_icu_stays(self) -> None:
        """
        Clean ICU stays data.
        """
        if self.icu_stays is None:
            return
            
        # Convert timestamps to datetime
        for col in ['intime', 'outtime']:
            if col in self.icu_stays.columns:
                self.icu_stays[col] = pd.to_datetime(self.icu_stays[col])
                
        # Handle missing values
        self.icu_stays['outtime'] = self.icu_stays['outtime'].fillna(pd.NaT)
        
        # Remove invalid time relationships
        self.icu_stays = self.icu_stays[
            self.icu_stays['outtime'] >= self.icu_stays['intime']
        ]
        
        # Sort by subject_id and intime
        self.icu_stays = self.icu_stays.sort_values(['subject_id', 'intime'])
    
    def clean_data(self) -> None:
        """
        Clean and preprocess the input data.
        
        This method:
        1. Handles missing values
        2. Removes duplicates
        3. Standardizes data formats
        4. Handles outliers
        """
        # Clean each data type
        self._clean_patients()
        self._clean_admissions()
        self._clean_icu_stays()
        
        # Process lab and chart events (already implemented)
        if self.lab_events is not None:
            self.lab_events = self.process_lab_events()
            
        if self.chart_events is not None:
            self.chart_events = self.process_chart_events()
    
    # ... (rest of the class implementation remains unchanged) 