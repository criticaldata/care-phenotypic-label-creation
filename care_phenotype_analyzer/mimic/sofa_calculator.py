"""
SOFA (Sequential Organ Failure Assessment) score calculation.

This module provides functionality for calculating SOFA scores from MIMIC data.
The SOFA score assesses organ dysfunction in critically ill patients.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from .clinical_scores import ClinicalScoreCalculator

class SOFACalculator(ClinicalScoreCalculator):
    """Calculator for SOFA (Sequential Organ Failure Assessment) scores."""
    
    # Define SOFA components and their itemids
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
    
    def __init__(self, *args, **kwargs):
        """Initialize the SOFA calculator."""
        super().__init__(*args, **kwargs)
        self.score_type = 'sofa'
        
    def _calculate_respiratory_score(self,
                                   subject_id: int,
                                   hadm_id: int,
                                   time: datetime) -> int:
        """
        Calculate the respiratory component of the SOFA score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time: Time to calculate score at
            
        Returns:
            Respiratory component score (0-4)
        """
        # Get PaO2 and FiO2 measurements
        pao2 = self._get_latest_measurement(
            self.lab_events, subject_id, hadm_id, 220277, time
        )
        fio2 = self._get_latest_measurement(
            self.chart_events, subject_id, hadm_id, 220210, time
        )
        
        if pao2 is None or fio2 is None:
            return 0
            
        # Calculate PaO2/FiO2 ratio
        ratio = pao2 / fio2
        
        # Calculate score based on ratio
        return self._calculate_component_score(
            ratio,
            self.SOFA_COMPONENTS['respiratory']['thresholds'],
            self.SOFA_COMPONENTS['respiratory']['scores']
        )
    
    def _calculate_coagulation_score(self,
                                   subject_id: int,
                                   hadm_id: int,
                                   time: datetime) -> int:
        """
        Calculate the coagulation component of the SOFA score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time: Time to calculate score at
            
        Returns:
            Coagulation component score (0-4)
        """
        # Get platelet count
        platelets = self._get_latest_measurement(
            self.lab_events, subject_id, hadm_id, 51265, time
        )
        
        if platelets is None:
            return 0
            
        # Calculate score based on platelet count
        return self._calculate_component_score(
            platelets,
            self.SOFA_COMPONENTS['coagulation']['thresholds'],
            self.SOFA_COMPONENTS['coagulation']['scores']
        )
    
    def _calculate_liver_score(self,
                             subject_id: int,
                             hadm_id: int,
                             time: datetime) -> int:
        """
        Calculate the liver component of the SOFA score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time: Time to calculate score at
            
        Returns:
            Liver component score (0-4)
        """
        # Get bilirubin level
        bilirubin = self._get_latest_measurement(
            self.lab_events, subject_id, hadm_id, 50912, time
        )
        
        if bilirubin is None:
            return 0
            
        # Calculate score based on bilirubin level
        return self._calculate_component_score(
            bilirubin,
            self.SOFA_COMPONENTS['liver']['thresholds'],
            self.SOFA_COMPONENTS['liver']['scores']
        )
    
    def _calculate_cardiovascular_score(self,
                                      subject_id: int,
                                      hadm_id: int,
                                      time: datetime) -> int:
        """
        Calculate the cardiovascular component of the SOFA score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time: Time to calculate score at
            
        Returns:
            Cardiovascular component score (0-4)
        """
        # Get MAP and vasopressor measurements
        map_value = self._get_latest_measurement(
            self.chart_events, subject_id, hadm_id, 220277, time
        )
        
        # Check for vasopressor use
        vasopressor_items = [221906, 221289, 221662, 221749]  # Norepinephrine, Epinephrine, etc.
        vasopressor_use = False
        for itemid in vasopressor_items:
            value = self._get_latest_measurement(
                self.chart_events, subject_id, hadm_id, itemid, time
            )
            if value is not None and value > 0:
                vasopressor_use = True
                break
                
        if map_value is None and not vasopressor_use:
            return 0
            
        # Calculate score based on MAP and vasopressor use
        if vasopressor_use:
            return 4
        elif map_value < 70:
            return 1
        else:
            return 0
    
    def _calculate_cns_score(self,
                           subject_id: int,
                           hadm_id: int,
                           time: datetime) -> int:
        """
        Calculate the central nervous system component of the SOFA score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time: Time to calculate score at
            
        Returns:
            CNS component score (0-4)
        """
        # Get Glasgow Coma Scale
        gcs = self._get_latest_measurement(
            self.chart_events, subject_id, hadm_id, 220739, time
        )
        
        if gcs is None:
            return 0
            
        # Calculate score based on GCS
        return self._calculate_reverse_component_score(
            gcs,
            self.SOFA_COMPONENTS['cns']['thresholds'],
            self.SOFA_COMPONENTS['cns']['scores']
        )
    
    def _calculate_renal_score(self,
                             subject_id: int,
                             hadm_id: int,
                             time: datetime) -> int:
        """
        Calculate the renal component of the SOFA score.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time: Time to calculate score at
            
        Returns:
            Renal component score (0-4)
        """
        # Get creatinine level
        creatinine = self._get_latest_measurement(
            self.lab_events, subject_id, hadm_id, 50912, time
        )
        
        if creatinine is None:
            return 0
            
        # Calculate score based on creatinine level
        return self._calculate_component_score(
            creatinine,
            self.SOFA_COMPONENTS['renal']['thresholds'],
            self.SOFA_COMPONENTS['renal']['scores']
        )
    
    def calculate_scores(self,
                        subject_ids: Optional[List[int]] = None,
                        hadm_ids: Optional[List[int]] = None,
                        time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate SOFA scores for specified subjects and time windows.
        
        Args:
            subject_ids: Optional list of subject IDs to calculate scores for
            hadm_ids: Optional list of hospital admission IDs to calculate scores for
            time_windows: Optional list of (start_time, end_time) tuples
            
        Returns:
            Dictionary containing calculated SOFA scores
        """
        # Determine subjects and admissions to process
        if subject_ids is None:
            subject_ids = self.lab_events['subject_id'].unique()
        if hadm_ids is None:
            hadm_ids = self.lab_events['hadm_id'].unique()
            
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
                    
                start_time = admission['admittime'].iloc[0]
                end_time = admission['dischtime'].iloc[0]
                
                # Calculate scores at regular intervals
                current_time = start_time
                while current_time <= end_time:
                    # Calculate component scores
                    respiratory = self._calculate_respiratory_score(
                        subject_id, hadm_id, current_time
                    )
                    coagulation = self._calculate_coagulation_score(
                        subject_id, hadm_id, current_time
                    )
                    liver = self._calculate_liver_score(
                        subject_id, hadm_id, current_time
                    )
                    cardiovascular = self._calculate_cardiovascular_score(
                        subject_id, hadm_id, current_time
                    )
                    cns = self._calculate_cns_score(
                        subject_id, hadm_id, current_time
                    )
                    renal = self._calculate_renal_score(
                        subject_id, hadm_id, current_time
                    )
                    
                    # Calculate total score
                    total_score = sum([
                        respiratory, coagulation, liver,
                        cardiovascular, cns, renal
                    ])
                    
                    # Store results
                    results.append({
                        'subject_id': subject_id,
                        'hadm_id': hadm_id,
                        'score_time': current_time,
                        'score_type': self.score_type,
                        'score_value': total_score,
                        'components': {
                            'respiratory': respiratory,
                            'coagulation': coagulation,
                            'liver': liver,
                            'cardiovascular': cardiovascular,
                            'cns': cns,
                            'renal': renal
                        }
                    })
                    
                    # Move to next time point
                    current_time += timedelta(hours=24)
                    
        # Convert results to DataFrame
        self.scores[self.score_type] = pd.DataFrame(results)
        
        return {self.score_type: self.scores[self.score_type]} 