"""
Data integrity checks for MIMIC data processing.

This module provides comprehensive data integrity checks to ensure
the quality and consistency of MIMIC data throughout the processing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from .structures import (
    Patient, Admission, ICUStay, LabEvent, ChartEvent,
    ClinicalScore, Gender, AdmissionType, ICUUnit
)
from .data_formats import (
    STANDARD_COLUMNS, COLUMN_DTYPES, VALUE_CONSTRAINTS,
    REQUIRED_COLUMNS, validate_dataframe
)

class DataIntegrityChecker:
    """Class for performing data integrity checks on MIMIC data."""
    
    def __init__(self):
        """Initialize the DataIntegrityChecker."""
        self.issues = []
        
    def check_data_consistency(self, 
                             df: pd.DataFrame,
                             data_type: str,
                             reference_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Check data consistency within a DataFrame and against reference data.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            reference_data: Dictionary of reference DataFrames for cross-checks
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Basic format validation
        try:
            validate_dataframe(df, data_type)
        except ValueError as e:
            issues.append(f"Format validation failed: {str(e)}")
            
        # Check for duplicate IDs
        id_cols = {
            'patient': ['subject_id'],
            'admission': ['hadm_id'],
            'icu_stay': ['stay_id'],
            'lab_event': ['specimen_id'],
            'chart_event': ['charttime', 'itemid']
        }
        
        if data_type in id_cols:
            for col in id_cols[data_type]:
                duplicates = df[df.duplicated(subset=[col], keep=False)]
                if not duplicates.empty:
                    issues.append(f"Found {len(duplicates)} duplicate {col}s")
                    
        # Cross-reference checks with other tables
        if reference_data:
            if data_type == 'admission' and 'patient' in reference_data:
                # Check that all admission subject_ids exist in patients
                missing_subjects = set(df['subject_id']) - set(reference_data['patient']['subject_id'])
                if missing_subjects:
                    issues.append(f"Found {len(missing_subjects)} admissions for non-existent subjects")
                    
            elif data_type == 'icu_stay' and 'admission' in reference_data:
                # Check that all ICU stay hadm_ids exist in admissions
                missing_admissions = set(df['hadm_id']) - set(reference_data['admission']['hadm_id'])
                if missing_admissions:
                    issues.append(f"Found {len(missing_admissions)} ICU stays for non-existent admissions")
                    
            elif data_type in ['lab_event', 'chart_event'] and 'admission' in reference_data:
                # Check that all event hadm_ids exist in admissions
                missing_admissions = set(df['hadm_id']) - set(reference_data['admission']['hadm_id'])
                if missing_admissions:
                    issues.append(f"Found {len(missing_admissions)} {data_type}s for non-existent admissions")
                    
        return issues
    
    def check_temporal_consistency(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """
        Check temporal consistency of events and measurements.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            
        Returns:
            List of identified issues
        """
        issues = []
        
        if data_type == 'admission':
            # Check admission/discharge time consistency
            invalid_times = df[df['dischtime'] < df['admittime']]
            if not invalid_times.empty:
                issues.append(f"Found {len(invalid_times)} admissions with discharge before admission")
                
            # Check death time consistency
            invalid_deaths = df[df['deathtime'] < df['admittime']]
            if not invalid_deaths.empty:
                issues.append(f"Found {len(invalid_deaths)} admissions with death before admission")
                
        elif data_type == 'icu_stay':
            # Check ICU stay time consistency
            invalid_times = df[df['outtime'] < df['intime']]
            if not invalid_times.empty:
                issues.append(f"Found {len(invalid_times)} ICU stays with out time before in time")
                
            # Check length of stay consistency
            invalid_los = df[df['los'] < 0]
            if not invalid_los.empty:
                issues.append(f"Found {len(invalid_los)} ICU stays with negative length of stay")
                
        elif data_type in ['lab_event', 'chart_event']:
            # Check event time consistency
            invalid_times = df[df['storetime'] < df['charttime']]
            if not invalid_times.empty:
                issues.append(f"Found {len(invalid_times)} events with store time before chart time")
                
        return issues
    
    def check_value_ranges(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """
        Check if values are within expected ranges.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            
        Returns:
            List of identified issues
        """
        issues = []
        
        if data_type == 'patient':
            # Check age range
            invalid_ages = df[~df['anchor_age'].between(0, 120)]
            if not invalid_ages.empty:
                issues.append(f"Found {len(invalid_ages)} patients with invalid age")
                
            # Check year range
            current_year = datetime.now().year
            invalid_years = df[~df['anchor_year'].between(1900, current_year)]
            if not invalid_years.empty:
                issues.append(f"Found {len(invalid_years)} patients with invalid year")
                
        elif data_type == 'lab_event':
            # Check value ranges against reference ranges
            invalid_values = df[
                (df['valuenum'] < df['ref_range_lower']) |
                (df['valuenum'] > df['ref_range_upper'])
            ]
            if not invalid_values.empty:
                issues.append(f"Found {len(invalid_values)} lab values outside reference ranges")
                
        elif data_type == 'clinical_score':
            # Check score ranges
            invalid_scores = df[df['score_value'] < 0]
            if not invalid_scores.empty:
                issues.append(f"Found {len(invalid_scores)} clinical scores with negative values")
                
        return issues
    
    def check_measurement_frequency(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """
        Check for unusual measurement frequencies.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            
        Returns:
            List of identified issues
        """
        issues = []
        
        if data_type in ['lab_event', 'chart_event']:
            # Group by patient and item
            grouped = df.groupby(['subject_id', 'itemid'])
            
            # Check for very high frequencies
            for (subject_id, itemid), group in grouped:
                time_span = (group['charttime'].max() - group['charttime'].min()).total_seconds() / 3600
                if time_span > 0:
                    freq = len(group) / time_span
                    if freq > 24:  # More than once per hour
                        issues.append(
                            f"High measurement frequency for subject {subject_id}, "
                            f"item {itemid}: {freq:.2f} per hour"
                        )
                        
            # Check for very low frequencies
            for (subject_id, itemid), group in grouped:
                time_span = (group['charttime'].max() - group['charttime'].min()).total_seconds() / 3600
                if time_span > 24:  # Only check for stays longer than 24 hours
                    freq = len(group) / time_span
                    if freq < 0.25:  # Less than once per 4 hours
                        issues.append(
                            f"Low measurement frequency for subject {subject_id}, "
                            f"item {itemid}: {freq:.2f} per hour"
                        )
                        
        return issues
    
    def check_data_completeness(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """
        Check for missing or incomplete data.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for missing values in required columns
        required = REQUIRED_COLUMNS[data_type]
        for col in required:
            missing = df[col].isnull().sum()
            if missing > 0:
                issues.append(f"Found {missing} missing values in required column: {col}")
                
        # Check for empty DataFrames
        if df.empty:
            issues.append(f"Empty DataFrame for {data_type}")
            
        # Check for minimum required data points
        if data_type in ['lab_event', 'chart_event']:
            min_events = 10  # Minimum number of events per patient
            patient_counts = df.groupby('subject_id').size()
            low_count_patients = patient_counts[patient_counts < min_events]
            if not low_count_patients.empty:
                issues.append(
                    f"Found {len(low_count_patients)} patients with fewer than "
                    f"{min_events} events"
                )
                
        return issues
    
    def perform_integrity_checks(self,
                               df: pd.DataFrame,
                               data_type: str,
                               reference_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, List[str]]:
        """
        Perform all integrity checks on the data.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            reference_data: Dictionary of reference DataFrames for cross-checks
            
        Returns:
            Dictionary of issues by check type
        """
        self.issues = []
        
        checks = {
            'data_consistency': self.check_data_consistency(df, data_type, reference_data),
            'temporal_consistency': self.check_temporal_consistency(df, data_type),
            'value_ranges': self.check_value_ranges(df, data_type),
            'measurement_frequency': self.check_measurement_frequency(df, data_type),
            'data_completeness': self.check_data_completeness(df, data_type)
        }
        
        # Collect all issues
        for check_type, issues in checks.items():
            self.issues.extend(issues)
            
        return checks
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get a summary of identified issues.
        
        Returns:
            Dictionary with issue counts by type
        """
        return {
            'total_issues': len(self.issues),
            'critical_issues': len([i for i in self.issues if 'critical' in i.lower()]),
            'warning_issues': len([i for i in self.issues if 'warning' in i.lower()])
        }
    
    def has_critical_issues(self) -> bool:
        """
        Check if there are any critical issues.
        
        Returns:
            bool: True if there are critical issues, False otherwise
        """
        return any('critical' in issue.lower() for issue in self.issues) 