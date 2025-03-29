"""
Clinical score calculations for MIMIC data processing.

This module provides base functionality for calculating various clinical scores
from MIMIC data, including SOFA scores and Charlson comorbidity index.
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
    REQUIRED_COLUMNS, validate_dataframe, convert_to_standard_format
)

class ClinicalScoreCalculator:
    """Base class for calculating clinical scores from MIMIC data."""
    
    def __init__(self,
                 lab_events: pd.DataFrame,
                 chart_events: pd.DataFrame,
                 patients: Optional[pd.DataFrame] = None,
                 admissions: Optional[pd.DataFrame] = None,
                 icu_stays: Optional[pd.DataFrame] = None):
        """
        Initialize the clinical score calculator.
        
        Args:
            lab_events: DataFrame containing lab events
            chart_events: DataFrame containing chart events
            patients: Optional DataFrame containing patient information
            admissions: Optional DataFrame containing admission information
            icu_stays: Optional DataFrame containing ICU stay information
        """
        # Convert inputs to standard format
        self.lab_events = convert_to_standard_format(lab_events, 'lab_event')
        self.chart_events = convert_to_standard_format(chart_events, 'chart_event')
        self.patients = convert_to_standard_format(patients, 'patient') if patients is not None else None
        self.admissions = convert_to_standard_format(admissions, 'admission') if admissions is not None else None
        self.icu_stays = convert_to_standard_format(icu_stays, 'icu_stay') if icu_stays is not None else None
        
        # Initialize score storage
        self.scores = {}
        
    def _get_measurements_in_window(self,
                                  df: pd.DataFrame,
                                  subject_id: int,
                                  hadm_id: int,
                                  start_time: datetime,
                                  end_time: datetime,
                                  itemids: List[int]) -> pd.DataFrame:
        """
        Get measurements within a specified time window.
        
        Args:
            df: DataFrame containing measurements
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            start_time: Start of time window
            end_time: End of time window
            itemids: List of item IDs to include
            
        Returns:
            DataFrame containing measurements in the time window
        """
        mask = (
            (df['subject_id'] == subject_id) &
            (df['hadm_id'] == hadm_id) &
            (df['charttime'] >= start_time) &
            (df['charttime'] <= end_time) &
            (df['itemid'].isin(itemids))
        )
        return df[mask].copy()
    
    def _get_latest_measurement(self,
                              df: pd.DataFrame,
                              subject_id: int,
                              hadm_id: int,
                              itemid: int,
                              before_time: datetime) -> Optional[float]:
        """
        Get the latest measurement before a specified time.
        
        Args:
            df: DataFrame containing measurements
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            itemid: Item ID to look for
            before_time: Time to look before
            
        Returns:
            Latest measurement value or None if not found
        """
        mask = (
            (df['subject_id'] == subject_id) &
            (df['hadm_id'] == hadm_id) &
            (df['itemid'] == itemid) &
            (df['charttime'] <= before_time)
        )
        measurements = df[mask]
        if measurements.empty:
            return None
        return measurements.sort_values('charttime').iloc[-1]['valuenum']
    
    def _calculate_component_score(self,
                                 value: float,
                                 thresholds: List[float],
                                 scores: List[int]) -> int:
        """
        Calculate a component score based on value thresholds.
        
        Args:
            value: Measurement value
            thresholds: List of threshold values
            scores: List of corresponding scores
            
        Returns:
            Calculated component score
        """
        for threshold, score in zip(thresholds, scores):
            if value <= threshold:
                return score
        return scores[-1]
    
    def _calculate_reverse_component_score(self,
                                         value: float,
                                         thresholds: List[float],
                                         scores: List[int]) -> int:
        """
        Calculate a component score based on value thresholds (reverse order).
        
        Args:
            value: Measurement value
            thresholds: List of threshold values
            scores: List of corresponding scores
            
        Returns:
            Calculated component score
        """
        for threshold, score in zip(thresholds, scores):
            if value >= threshold:
                return score
        return scores[-1]
    
    def calculate_scores(self,
                        subject_ids: Optional[List[int]] = None,
                        hadm_ids: Optional[List[int]] = None,
                        time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate clinical scores for specified subjects and time windows.
        
        Args:
            subject_ids: Optional list of subject IDs to calculate scores for
            hadm_ids: Optional list of hospital admission IDs to calculate scores for
            time_windows: Optional list of (start_time, end_time) tuples
            
        Returns:
            Dictionary containing calculated scores by type
        """
        raise NotImplementedError("Subclasses must implement calculate_scores")
    
    def get_score_history(self,
                         subject_id: int,
                         hadm_id: int,
                         score_type: str) -> pd.DataFrame:
        """
        Get the history of a specific clinical score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            score_type: Type of clinical score
            
        Returns:
            DataFrame containing score history
        """
        if score_type not in self.scores:
            raise ValueError(f"Score type {score_type} not found")
            
        mask = (
            (self.scores[score_type]['subject_id'] == subject_id) &
            (self.scores[score_type]['hadm_id'] == hadm_id)
        )
        return self.scores[score_type][mask].copy()
    
    def get_score_summary(self,
                         subject_id: int,
                         hadm_id: int,
                         score_type: str) -> Dict[str, float]:
        """
        Get summary statistics for a specific clinical score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            score_type: Type of clinical score
            
        Returns:
            Dictionary containing score summary statistics
        """
        history = self.get_score_history(subject_id, hadm_id, score_type)
        if history.empty:
            return {}
            
        return {
            'mean': history['score_value'].mean(),
            'std': history['score_value'].std(),
            'min': history['score_value'].min(),
            'max': history['score_value'].max(),
            'last': history['score_value'].iloc[-1]
        }
    
    def get_score_trend(self,
                       subject_id: int,
                       hadm_id: int,
                       score_type: str,
                       window_size: str = '24H') -> pd.DataFrame:
        """
        Get the trend of a specific clinical score over time.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            score_type: Type of clinical score
            window_size: Size of rolling window (e.g., '24H', '12H')
            
        Returns:
            DataFrame containing score trend
        """
        history = self.get_score_history(subject_id, hadm_id, score_type)
        if history.empty:
            return pd.DataFrame()
            
        # Set time as index
        history = history.set_index('score_time')
        
        # Calculate rolling statistics
        trend = pd.DataFrame({
            'mean': history['score_value'].rolling(window_size).mean(),
            'std': history['score_value'].rolling(window_size).std(),
            'min': history['score_value'].rolling(window_size).min(),
            'max': history['score_value'].rolling(window_size).max()
        })
        
        return trend.reset_index() 