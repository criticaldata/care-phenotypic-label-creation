"""
Charlson comorbidity index calculation.

This module provides functionality for calculating Charlson comorbidity index
from MIMIC data. The Charlson index predicts mortality by classifying or
weighting comorbid conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from .clinical_scores import ClinicalScoreCalculator

class CharlsonCalculator(ClinicalScoreCalculator):
    """Calculator for Charlson comorbidity index."""
    
    # Define Charlson components and their weights
    CHARLSON_COMPONENTS = {
        'myocardial_infarction': {
            'weight': 1,
            'itemids': [41071, 41081, 41091]  # ICD-9 codes for MI
        },
        'congestive_heart_failure': {
            'weight': 1,
            'itemids': [42832, 42833, 42840, 42841, 42842]
        },
        'peripheral_vascular_disease': {
            'weight': 1,
            'itemids': [4439, 4440, 4441]
        },
        'cerebrovascular_disease': {
            'weight': 1,
            'itemids': [43043, 43100, 43101, 43102, 43103]
        },
        'dementia': {
            'weight': 1,
            'itemids': [29011, 29012, 29013, 29014, 29015, 29016, 29017, 29018, 29019]
        },
        'chronic_pulmonary_disease': {
            'weight': 1,
            'itemids': [49121, 49122, 49123, 49124, 49125]
        },
        'rheumatoid_disease': {
            'weight': 1,
            'itemids': [7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148]
        },
        'peptic_ulcer_disease': {
            'weight': 1,
            'itemids': [53170, 53171, 53172, 53173, 53174]
        },
        'mild_liver_disease': {
            'weight': 1,
            'itemids': [5713, 5715, 5716, 5718, 5719]
        },
        'diabetes_without_complications': {
            'weight': 1,
            'itemids': [25000, 25001, 25002, 25003]
        },
        'diabetes_with_complications': {
            'weight': 2,
            'itemids': [25010, 25011, 25012, 25013, 25014, 25015, 25016, 25017, 25018, 25019]
        },
        'hemiplegia_or_paraplegia': {
            'weight': 2,
            'itemids': [34200, 34201, 34202, 34203, 34204, 34205, 34206, 34207, 34208, 34209]
        },
        'renal_disease': {
            'weight': 2,
            'itemids': [5853, 5854, 5855, 5856, 5857, 5858, 5859]
        },
        'malignancy': {
            'weight': 2,
            'itemids': [14000, 14001, 14002, 14003, 14004, 14005, 14006, 14007, 14008, 14009]
        },
        'moderate_or_severe_liver_disease': {
            'weight': 3,
            'itemids': [4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569]
        },
        'metastatic_solid_tumor': {
            'weight': 6,
            'itemids': [19600, 19601, 19602, 19603, 19604, 19605, 19606, 19607, 19608, 19609]
        },
        'aids': {
            'weight': 6,
            'itemids': [42000, 42001, 42002, 42003, 42004, 42005, 42006, 42007, 42008, 42009]
        }
    }
    
    def __init__(self, *args, **kwargs):
        """Initialize the Charlson calculator."""
        super().__init__(*args, **kwargs)
        self.score_type = 'charlson'
        
    def _check_comorbidity(self,
                          subject_id: int,
                          hadm_id: int,
                          itemids: List[int]) -> bool:
        """
        Check if a patient has a specific comorbidity.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            itemids: List of item IDs to check for
            
        Returns:
            bool: True if comorbidity is present, False otherwise
        """
        # Check diagnoses in chart events
        mask = (
            (self.chart_events['subject_id'] == subject_id) &
            (self.chart_events['hadm_id'] == hadm_id) &
            (self.chart_events['itemid'].isin(itemids))
        )
        return self.chart_events[mask].shape[0] > 0
    
    def _calculate_component_score(self,
                                 subject_id: int,
                                 hadm_id: int,
                                 component: str) -> int:
        """
        Calculate the score for a specific Charlson component.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            component: Name of the component
            
        Returns:
            int: Component score (weight if present, 0 if not)
        """
        if component not in self.CHARLSON_COMPONENTS:
            raise ValueError(f"Invalid component: {component}")
            
        # Check if comorbidity is present
        is_present = self._check_comorbidity(
            subject_id,
            hadm_id,
            self.CHARLSON_COMPONENTS[component]['itemids']
        )
        
        # Return weight if present, 0 if not
        return self.CHARLSON_COMPONENTS[component]['weight'] if is_present else 0
    
    def calculate_scores(self,
                        subject_ids: Optional[List[int]] = None,
                        hadm_ids: Optional[List[int]] = None,
                        time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate Charlson comorbidity index for specified subjects and admissions.
        
        Args:
            subject_ids: Optional list of subject IDs to calculate scores for
            hadm_ids: Optional list of hospital admission IDs to calculate scores for
            time_windows: Optional list of (start_time, end_time) tuples
            
        Returns:
            Dictionary containing calculated Charlson scores
        """
        # Determine subjects and admissions to process
        if subject_ids is None:
            subject_ids = self.chart_events['subject_id'].unique()
        if hadm_ids is None:
            hadm_ids = self.chart_events['hadm_id'].unique()
            
        # Initialize results DataFrame
        results = []
        
        # Process each subject and admission
        for subject_id in subject_ids:
            for hadm_id in hadm_ids:
                # Get admission times
                admission = self.admissions[
                    (self.admissions['subject_id'] == subject_id) &
                    (self.admissions['hadm_id'] == hadm_id)
                ]
                if admission.empty:
                    continue
                    
                # Calculate component scores
                component_scores = {}
                total_score = 0
                
                for component in self.CHARLSON_COMPONENTS:
                    score = self._calculate_component_score(
                        subject_id, hadm_id, component
                    )
                    component_scores[component] = score
                    total_score += score
                    
                # Store results
                results.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'score_time': admission['admittime'].iloc[0],
                    'score_type': self.score_type,
                    'score_value': total_score,
                    'components': component_scores
                })
                    
        # Convert results to DataFrame
        self.scores[self.score_type] = pd.DataFrame(results)
        
        return {self.score_type: self.scores[self.score_type]}
    
    def get_comorbidity_summary(self,
                              subject_id: int,
                              hadm_id: int) -> Dict[str, bool]:
        """
        Get a summary of present comorbidities for a patient.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Dictionary mapping comorbidity names to presence (True/False)
        """
        summary = {}
        for component in self.CHARLSON_COMPONENTS:
            summary[component] = self._check_comorbidity(
                subject_id,
                hadm_id,
                self.CHARLSON_COMPONENTS[component]['itemids']
            )
        return summary
    
    def get_comorbidity_weights(self,
                              subject_id: int,
                              hadm_id: int) -> Dict[str, int]:
        """
        Get the weights of present comorbidities for a patient.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Dictionary mapping comorbidity names to their weights
        """
        weights = {}
        for component in self.CHARLSON_COMPONENTS:
            if self._check_comorbidity(
                subject_id,
                hadm_id,
                self.CHARLSON_COMPONENTS[component]['itemids']
            ):
                weights[component] = self.CHARLSON_COMPONENTS[component]['weight']
        return weights 