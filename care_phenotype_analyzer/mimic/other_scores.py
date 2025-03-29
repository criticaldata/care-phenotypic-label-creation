"""
Other relevant clinical scores calculation.

This module provides functionality for calculating various clinical scores
commonly used in ICU settings, including APACHE II, SAPS II, and Elixhauser
comorbidity index.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from .clinical_scores import ClinicalScoreCalculator

class APACHEIICalculator(ClinicalScoreCalculator):
    """Calculator for APACHE II (Acute Physiology and Chronic Health Evaluation II) score."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the APACHE II calculator."""
        super().__init__(*args, **kwargs)
        self.score_type = 'apache_ii'
        
    def _calculate_acute_physiology_score(self,
                                        subject_id: int,
                                        hadm_id: int,
                                        time_window: Tuple[datetime, datetime]) -> float:
        """
        Calculate the acute physiology score component.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time_window: (start_time, end_time) tuple
            
        Returns:
            float: Acute physiology score
        """
        # Get measurements within time window
        measurements = self._get_measurements_in_window(
            subject_id, hadm_id, time_window
        )
        
        # Calculate score based on worst values
        score = 0.0
        
        # Temperature
        temp = measurements.get('temperature', [])
        if temp:
            worst_temp = max(temp, key=lambda x: abs(x - 38.5))
            if worst_temp >= 41.0:
                score += 4
            elif worst_temp >= 39.0:
                score += 3
            elif worst_temp >= 38.5:
                score += 2
            elif worst_temp >= 36.0:
                score += 1
            elif worst_temp >= 34.0:
                score += 2
            elif worst_temp >= 32.0:
                score += 3
            else:
                score += 4
                
        # Mean arterial pressure
        map_values = measurements.get('mean_arterial_pressure', [])
        if map_values:
            worst_map = min(map_values)
            if worst_map >= 160:
                score += 4
            elif worst_map >= 130:
                score += 3
            elif worst_map >= 110:
                score += 2
            elif worst_map >= 70:
                score += 1
            elif worst_map >= 50:
                score += 2
            else:
                score += 3
                
        # Heart rate
        hr = measurements.get('heart_rate', [])
        if hr:
            worst_hr = max(hr, key=lambda x: abs(x - 70))
            if worst_hr >= 180:
                score += 4
            elif worst_hr >= 140:
                score += 3
            elif worst_hr >= 110:
                score += 2
            elif worst_hr >= 70:
                score += 1
            elif worst_hr >= 55:
                score += 2
            elif worst_hr >= 40:
                score += 3
            else:
                score += 4
                
        # Respiratory rate
        rr = measurements.get('respiratory_rate', [])
        if rr:
            worst_rr = max(rr, key=lambda x: abs(x - 12))
            if worst_rr >= 50:
                score += 4
            elif worst_rr >= 35:
                score += 3
            elif worst_rr >= 25:
                score += 2
            elif worst_rr >= 12:
                score += 1
            elif worst_rr >= 10:
                score += 2
            elif worst_rr >= 6:
                score += 3
            else:
                score += 4
                
        # Oxygenation
        pao2 = measurements.get('pao2', [])
        fio2 = measurements.get('fio2', [])
        if pao2 and fio2:
            worst_ratio = min(p/f for p, f in zip(pao2, fio2))
            if worst_ratio >= 500:
                score += 0
            elif worst_ratio >= 350:
                score += 1
            elif worst_ratio >= 200:
                score += 2
            else:
                score += 4
                
        # Arterial pH
        ph = measurements.get('arterial_ph', [])
        if ph:
            worst_ph = max(ph, key=lambda x: abs(x - 7.4))
            if worst_ph >= 7.7:
                score += 4
            elif worst_ph >= 7.6:
                score += 3
            elif worst_ph >= 7.5:
                score += 2
            elif worst_ph >= 7.33:
                score += 1
            elif worst_ph >= 7.25:
                score += 2
            elif worst_ph >= 7.15:
                score += 3
            else:
                score += 4
                
        # Serum sodium
        na = measurements.get('sodium', [])
        if na:
            worst_na = max(na, key=lambda x: abs(x - 140))
            if worst_na >= 180:
                score += 4
            elif worst_na >= 160:
                score += 3
            elif worst_na >= 155:
                score += 2
            elif worst_na >= 150:
                score += 1
            elif worst_na >= 130:
                score += 1
            elif worst_na >= 120:
                score += 2
            else:
                score += 3
                
        # Serum potassium
        k = measurements.get('potassium', [])
        if k:
            worst_k = max(k, key=lambda x: abs(x - 4.0))
            if worst_k >= 7.0:
                score += 4
            elif worst_k >= 6.0:
                score += 3
            elif worst_k >= 5.5:
                score += 2
            elif worst_k >= 3.5:
                score += 1
            elif worst_k >= 3.0:
                score += 2
            elif worst_k >= 2.5:
                score += 3
            else:
                score += 4
                
        # Serum creatinine
        cr = measurements.get('creatinine', [])
        if cr:
            worst_cr = max(cr)
            if worst_cr >= 3.5:
                score += 4
            elif worst_cr >= 2.0:
                score += 3
            elif worst_cr >= 1.5:
                score += 2
            elif worst_cr >= 0.6:
                score += 1
            else:
                score += 0
                
        # Hematocrit
        hct = measurements.get('hematocrit', [])
        if hct:
            worst_hct = max(hct, key=lambda x: abs(x - 44))
            if worst_hct >= 60:
                score += 2
            elif worst_hct >= 50:
                score += 1
            elif worst_hct >= 46:
                score += 1
            elif worst_hct >= 30:
                score += 1
            elif worst_hct >= 20:
                score += 2
            else:
                score += 3
                
        # White blood count
        wbc = measurements.get('white_blood_cells', [])
        if wbc:
            worst_wbc = max(wbc, key=lambda x: abs(x - 15))
            if worst_wbc >= 40:
                score += 4
            elif worst_wbc >= 20:
                score += 2
            elif worst_wbc >= 15:
                score += 1
            elif worst_wbc >= 3:
                score += 1
            elif worst_wbc >= 1:
                score += 2
            else:
                score += 4
                
        return score
    
    def _calculate_chronic_health_score(self,
                                      subject_id: int,
                                      hadm_id: int) -> float:
        """
        Calculate the chronic health score component.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            float: Chronic health score
        """
        score = 0.0
        
        # Check for chronic conditions
        conditions = {
            'liver_disease': [4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569],
            'cardiovascular_disease': [41071, 41081, 41091, 42832, 42833, 42840, 42841, 42842],
            'respiratory_disease': [49121, 49122, 49123, 49124, 49125],
            'renal_disease': [5853, 5854, 5855, 5856, 5857, 5858, 5859],
            'immunocompromised': [42000, 42001, 42002, 42003, 42004, 42005, 42006, 42007, 42008, 42009]
        }
        
        for condition, itemids in conditions.items():
            if self._check_comorbidity(subject_id, hadm_id, itemids):
                score += 5
                
        return score
    
    def calculate_scores(self,
                        subject_ids: Optional[List[int]] = None,
                        hadm_ids: Optional[List[int]] = None,
                        time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate APACHE II scores for specified subjects and admissions.
        
        Args:
            subject_ids: Optional list of subject IDs to calculate scores for
            hadm_ids: Optional list of hospital admission IDs to calculate scores for
            time_windows: Optional list of (start_time, end_time) tuples
            
        Returns:
            Dictionary containing calculated APACHE II scores
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
                    
                # Calculate components
                acute_score = self._calculate_acute_physiology_score(
                    subject_id, hadm_id, (admission['admittime'].iloc[0], admission['admittime'].iloc[0] + timedelta(hours=24))
                )
                chronic_score = self._calculate_chronic_health_score(subject_id, hadm_id)
                age_score = self._calculate_age_score(admission['anchor_age'].iloc[0])
                
                # Calculate total score
                total_score = acute_score + chronic_score + age_score
                
                # Store results
                results.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'score_time': admission['admittime'].iloc[0],
                    'score_type': self.score_type,
                    'score_value': total_score,
                    'components': {
                        'acute_physiology': acute_score,
                        'chronic_health': chronic_score,
                        'age': age_score
                    }
                })
                    
        # Convert results to DataFrame
        self.scores[self.score_type] = pd.DataFrame(results)
        
        return {self.score_type: self.scores[self.score_type]}
    
    def _calculate_age_score(self, age: int) -> float:
        """
        Calculate the age score component.
        
        Args:
            age: Patient age
            
        Returns:
            float: Age score
        """
        if age >= 75:
            return 6
        elif age >= 65:
            return 5
        elif age >= 55:
            return 3
        elif age >= 45:
            return 2
        else:
            return 0

class SAPSIICalculator(ClinicalScoreCalculator):
    """Calculator for SAPS II (Simplified Acute Physiology Score II)."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the SAPS II calculator."""
        super().__init__(*args, **kwargs)
        self.score_type = 'saps_ii'
        
    def _calculate_physiology_score(self,
                                  subject_id: int,
                                  hadm_id: int,
                                  time_window: Tuple[datetime, datetime]) -> float:
        """
        Calculate the physiology score component.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            time_window: (start_time, end_time) tuple
            
        Returns:
            float: Physiology score
        """
        # Get measurements within time window
        measurements = self._get_measurements_in_window(
            subject_id, hadm_id, time_window
        )
        
        # Calculate score based on worst values
        score = 0.0
        
        # Heart rate
        hr = measurements.get('heart_rate', [])
        if hr:
            worst_hr = max(hr, key=lambda x: abs(x - 70))
            if worst_hr >= 160:
                score += 7
            elif worst_hr >= 120:
                score += 4
            elif worst_hr >= 70:
                score += 0
            elif worst_hr >= 40:
                score += 2
            else:
                score += 3
                
        # Systolic blood pressure
        sbp = measurements.get('systolic_blood_pressure', [])
        if sbp:
            worst_sbp = max(sbp, key=lambda x: abs(x - 120))
            if worst_sbp >= 200:
                score += 2
            elif worst_sbp >= 100:
                score += 0
            elif worst_sbp >= 70:
                score += 5
            else:
                score += 13
                
        # Temperature
        temp = measurements.get('temperature', [])
        if temp:
            worst_temp = max(temp, key=lambda x: abs(x - 38.5))
            if worst_temp >= 39.0:
                score += 3
            elif worst_temp >= 38.5:
                score += 0
            elif worst_temp >= 36.0:
                score += 0
            elif worst_temp >= 34.0:
                score += 1
            else:
                score += 2
                
        # PaO2/FiO2 ratio
        pao2 = measurements.get('pao2', [])
        fio2 = measurements.get('fio2', [])
        if pao2 and fio2:
            worst_ratio = min(p/f for p, f in zip(pao2, fio2))
            if worst_ratio >= 100:
                score += 0
            elif worst_ratio >= 60:
                score += 6
            else:
                score += 9
                
        # Urine output
        urine = measurements.get('urine_output', [])
        if urine:
            worst_urine = min(urine)
            if worst_urine >= 500:
                score += 0
            elif worst_urine >= 200:
                score += 5
            else:
                score += 11
                
        # Serum urea
        urea = measurements.get('urea', [])
        if urea:
            worst_urea = max(urea)
            if worst_urea >= 28:
                score += 3
            elif worst_urea >= 10:
                score += 0
            else:
                score += 1
                
        # White blood count
        wbc = measurements.get('white_blood_cells', [])
        if wbc:
            worst_wbc = max(wbc, key=lambda x: abs(x - 15))
            if worst_wbc >= 20:
                score += 3
            elif worst_wbc >= 15:
                score += 0
            elif worst_wbc >= 3:
                score += 0
            else:
                score += 12
                
        # Serum potassium
        k = measurements.get('potassium', [])
        if k:
            worst_k = max(k, key=lambda x: abs(x - 4.0))
            if worst_k >= 5.0:
                score += 3
            elif worst_k >= 3.5:
                score += 0
            elif worst_k >= 2.5:
                score += 2
            else:
                score += 3
                
        # Serum sodium
        na = measurements.get('sodium', [])
        if na:
            worst_na = max(na, key=lambda x: abs(x - 140))
            if worst_na >= 145:
                score += 1
            elif worst_na >= 135:
                score += 0
            elif worst_na >= 125:
                score += 2
            else:
                score += 5
                
        # Serum bicarbonate
        hco3 = measurements.get('bicarbonate', [])
        if hco3:
            worst_hco3 = max(hco3, key=lambda x: abs(x - 24))
            if worst_hco3 >= 32:
                score += 5
            elif worst_hco3 >= 20:
                score += 0
            elif worst_hco3 >= 15:
                score += 3
            else:
                score += 5
                
        # Bilirubin
        bilirubin = measurements.get('bilirubin', [])
        if bilirubin:
            worst_bilirubin = max(bilirubin)
            if worst_bilirubin >= 4.0:
                score += 9
            elif worst_bilirubin >= 2.0:
                score += 4
            else:
                score += 0
                
        return score
    
    def calculate_scores(self,
                        subject_ids: Optional[List[int]] = None,
                        hadm_ids: Optional[List[int]] = None,
                        time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate SAPS II scores for specified subjects and admissions.
        
        Args:
            subject_ids: Optional list of subject IDs to calculate scores for
            hadm_ids: Optional list of hospital admission IDs to calculate scores for
            time_windows: Optional list of (start_time, end_time) tuples
            
        Returns:
            Dictionary containing calculated SAPS II scores
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
                    
                # Calculate components
                physiology_score = self._calculate_physiology_score(
                    subject_id, hadm_id, (admission['admittime'].iloc[0], admission['admittime'].iloc[0] + timedelta(hours=24))
                )
                age_score = self._calculate_age_score(admission['anchor_age'].iloc[0])
                admission_type_score = self._calculate_admission_type_score(admission['admission_type'].iloc[0])
                
                # Calculate total score
                total_score = physiology_score + age_score + admission_type_score
                
                # Store results
                results.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'score_time': admission['admittime'].iloc[0],
                    'score_type': self.score_type,
                    'score_value': total_score,
                    'components': {
                        'physiology': physiology_score,
                        'age': age_score,
                        'admission_type': admission_type_score
                    }
                })
                    
        # Convert results to DataFrame
        self.scores[self.score_type] = pd.DataFrame(results)
        
        return {self.score_type: self.scores[self.score_type]}
    
    def _calculate_age_score(self, age: int) -> float:
        """
        Calculate the age score component.
        
        Args:
            age: Patient age
            
        Returns:
            float: Age score
        """
        if age >= 80:
            return 18
        elif age >= 75:
            return 16
        elif age >= 70:
            return 15
        elif age >= 60:
            return 12
        elif age >= 40:
            return 7
        else:
            return 0
            
    def _calculate_admission_type_score(self, admission_type: str) -> float:
        """
        Calculate the admission type score component.
        
        Args:
            admission_type: Type of admission
            
        Returns:
            float: Admission type score
        """
        if admission_type == 'Scheduled surgical':
            return 0
        elif admission_type == 'Medical':
            return 6
        elif admission_type == 'Unscheduled surgical':
            return 8
        else:
            return 0

class ElixhauserCalculator(ClinicalScoreCalculator):
    """Calculator for Elixhauser comorbidity index."""
    
    # Define Elixhauser components and their weights
    ELIXHAUSER_COMPONENTS = {
        'congestive_heart_failure': {
            'weight': 1,
            'itemids': [42832, 42833, 42840, 42841, 42842]
        },
        'cardiac_arrhythmia': {
            'weight': 1,
            'itemids': [42731, 42732, 42733, 42734, 42735, 42736, 42737, 42738, 42739]
        },
        'valvular_disease': {
            'weight': 1,
            'itemids': [3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949]
        },
        'pulmonary_circulation': {
            'weight': 1,
            'itemids': [41511, 41512, 41513, 41514, 41515, 41516, 41517, 41518, 41519]
        },
        'peripheral_vascular': {
            'weight': 1,
            'itemids': [4439, 4440, 4441]
        },
        'hypertension': {
            'weight': 1,
            'itemids': [4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019]
        },
        'paralysis': {
            'weight': 1,
            'itemids': [34200, 34201, 34202, 34203, 34204, 34205, 34206, 34207, 34208, 34209]
        },
        'other_neurological': {
            'weight': 1,
            'itemids': [3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319]
        },
        'chronic_pulmonary': {
            'weight': 1,
            'itemids': [49121, 49122, 49123, 49124, 49125]
        },
        'diabetes_uncomplicated': {
            'weight': 1,
            'itemids': [25000, 25001, 25002, 25003]
        },
        'diabetes_complicated': {
            'weight': 1,
            'itemids': [25010, 25011, 25012, 25013, 25014, 25015, 25016, 25017, 25018, 25019]
        },
        'hypothyroidism': {
            'weight': 1,
            'itemids': [2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449]
        },
        'renal_failure': {
            'weight': 1,
            'itemids': [5853, 5854, 5855, 5856, 5857, 5858, 5859]
        },
        'liver_disease': {
            'weight': 1,
            'itemids': [5713, 5715, 5716, 5718, 5719, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569]
        },
        'peptic_ulcer': {
            'weight': 1,
            'itemids': [53170, 53171, 53172, 53173, 53174]
        },
        'aids': {
            'weight': 1,
            'itemids': [42000, 42001, 42002, 42003, 42004, 42005, 42006, 42007, 42008, 42009]
        },
        'lymphoma': {
            'weight': 1,
            'itemids': [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009]
        },
        'metastatic_cancer': {
            'weight': 1,
            'itemids': [19600, 19601, 19602, 19603, 19604, 19605, 19606, 19607, 19608, 19609]
        },
        'solid_tumor': {
            'weight': 1,
            'itemids': [14000, 14001, 14002, 14003, 14004, 14005, 14006, 14007, 14008, 14009]
        },
        'rheumatoid_arthritis': {
            'weight': 1,
            'itemids': [7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148]
        },
        'coagulopathy': {
            'weight': 1,
            'itemids': [2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869]
        },
        'obesity': {
            'weight': 1,
            'itemids': [27800, 27801, 27802, 27803, 27804, 27805, 27806, 27807, 27808, 27809]
        },
        'weight_loss': {
            'weight': 1,
            'itemids': [26000, 26001, 26002, 26003, 26004, 26005, 26006, 26007, 26008, 26009]
        },
        'fluid_electrolyte': {
            'weight': 1,
            'itemids': [2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769]
        },
        'blood_loss_anemia': {
            'weight': 1,
            'itemids': [2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809]
        },
        'deficiency_anemias': {
            'weight': 1,
            'itemids': [2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809]
        },
        'alcohol_abuse': {
            'weight': 1,
            'itemids': [30300, 30301, 30302, 30303, 30304, 30305, 30306, 30307, 30308, 30309]
        },
        'drug_abuse': {
            'weight': 1,
            'itemids': [30400, 30401, 30402, 30403, 30404, 30405, 30406, 30407, 30408, 30409]
        },
        'psychoses': {
            'weight': 1,
            'itemids': [29500, 29501, 29502, 29503, 29504, 29505, 29506, 29507, 29508, 29509]
        },
        'depression': {
            'weight': 1,
            'itemids': [29620, 29621, 29622, 29623, 29624, 29625, 29626, 29627, 29628, 29629]
        }
    }
    
    def __init__(self, *args, **kwargs):
        """Initialize the Elixhauser calculator."""
        super().__init__(*args, **kwargs)
        self.score_type = 'elixhauser'
        
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
        Calculate the score for a specific Elixhauser component.
        
        Args:
            subject_id: Subject ID
            hadm_id: Hospital admission ID
            component: Name of the component
            
        Returns:
            int: Component score (weight if present, 0 if not)
        """
        if component not in self.ELIXHAUSER_COMPONENTS:
            raise ValueError(f"Invalid component: {component}")
            
        # Check if comorbidity is present
        is_present = self._check_comorbidity(
            subject_id,
            hadm_id,
            self.ELIXHAUSER_COMPONENTS[component]['itemids']
        )
        
        # Return weight if present, 0 if not
        return self.ELIXHAUSER_COMPONENTS[component]['weight'] if is_present else 0
    
    def calculate_scores(self,
                        subject_ids: Optional[List[int]] = None,
                        hadm_ids: Optional[List[int]] = None,
                        time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate Elixhauser comorbidity index for specified subjects and admissions.
        
        Args:
            subject_ids: Optional list of subject IDs to calculate scores for
            hadm_ids: Optional list of hospital admission IDs to calculate scores for
            time_windows: Optional list of (start_time, end_time) tuples
            
        Returns:
            Dictionary containing calculated Elixhauser scores
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
                
                for component in self.ELIXHAUSER_COMPONENTS:
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
        for component in self.ELIXHAUSER_COMPONENTS:
            summary[component] = self._check_comorbidity(
                subject_id,
                hadm_id,
                self.ELIXHAUSER_COMPONENTS[component]['itemids']
            )
        return summary 