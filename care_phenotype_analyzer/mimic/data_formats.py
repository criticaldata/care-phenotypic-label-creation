"""
Standard data formats for MIMIC data processing.

This module defines the standard data formats and schemas used throughout
the MIMIC data processing pipeline. These formats ensure consistency and
interoperability across different components of the system.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from .structures import (
    Patient, Admission, ICUStay, LabEvent, ChartEvent,
    ClinicalScore, Gender, AdmissionType, ICUUnit
)

# Standard column names for different data types
STANDARD_COLUMNS = {
    'patient': [
        'subject_id',
        'gender',
        'anchor_age',
        'anchor_year',
        'anchor_year_group',
        'dod'
    ],
    'admission': [
        'subject_id',
        'hadm_id',
        'admittime',
        'dischtime',
        'deathtime',
        'admission_type',
        'admission_location',
        'discharge_location',
        'insurance',
        'language',
        'marital_status',
        'ethnicity'
    ],
    'icu_stay': [
        'subject_id',
        'hadm_id',
        'stay_id',
        'intime',
        'outtime',
        'first_careunit',
        'last_careunit',
        'los'
    ],
    'lab_event': [
        'subject_id',
        'hadm_id',
        'stay_id',
        'charttime',
        'specimen_id',
        'itemid',
        'valuenum',
        'valueuom',
        'ref_range_lower',
        'ref_range_upper',
        'flag'
    ],
    'chart_event': [
        'subject_id',
        'hadm_id',
        'stay_id',
        'charttime',
        'storetime',
        'itemid',
        'value',
        'valuenum',
        'valueuom',
        'warning',
        'error'
    ],
    'clinical_score': [
        'subject_id',
        'hadm_id',
        'stay_id',
        'score_time',
        'score_type',
        'score_value',
        'components'
    ]
}

# Data type specifications for each column
COLUMN_DTYPES = {
    'patient': {
        'subject_id': 'int64',
        'gender': 'category',
        'anchor_age': 'int64',
        'anchor_year': 'int64',
        'anchor_year_group': 'category',
        'dod': 'datetime64[ns]'
    },
    'admission': {
        'subject_id': 'int64',
        'hadm_id': 'int64',
        'admittime': 'datetime64[ns]',
        'dischtime': 'datetime64[ns]',
        'deathtime': 'datetime64[ns]',
        'admission_type': 'category',
        'admission_location': 'category',
        'discharge_location': 'category',
        'insurance': 'category',
        'language': 'category',
        'marital_status': 'category',
        'ethnicity': 'category'
    },
    'icu_stay': {
        'subject_id': 'int64',
        'hadm_id': 'int64',
        'stay_id': 'int64',
        'intime': 'datetime64[ns]',
        'outtime': 'datetime64[ns]',
        'first_careunit': 'category',
        'last_careunit': 'category',
        'los': 'float64'
    },
    'lab_event': {
        'subject_id': 'int64',
        'hadm_id': 'int64',
        'stay_id': 'int64',
        'charttime': 'datetime64[ns]',
        'specimen_id': 'int64',
        'itemid': 'int64',
        'valuenum': 'float64',
        'valueuom': 'category',
        'ref_range_lower': 'float64',
        'ref_range_upper': 'float64',
        'flag': 'category'
    },
    'chart_event': {
        'subject_id': 'int64',
        'hadm_id': 'int64',
        'stay_id': 'int64',
        'charttime': 'datetime64[ns]',
        'storetime': 'datetime64[ns]',
        'itemid': 'int64',
        'value': 'str',
        'valuenum': 'float64',
        'valueuom': 'category',
        'warning': 'category',
        'error': 'category'
    },
    'clinical_score': {
        'subject_id': 'int64',
        'hadm_id': 'int64',
        'stay_id': 'int64',
        'score_time': 'datetime64[ns]',
        'score_type': 'category',
        'score_value': 'float64',
        'components': 'object'  # Dictionary of component scores
    }
}

# Value constraints and validations
VALUE_CONSTRAINTS = {
    'patient': {
        'anchor_age': {'min': 0, 'max': 120},
        'anchor_year': {'min': 1900, 'max': datetime.now().year},
        'gender': {'valid_values': [g.value for g in Gender]}
    },
    'admission': {
        'admission_type': {'valid_values': [at.value for at in AdmissionType]},
        'dischtime': {'min': 'admittime'},
        'deathtime': {'min': 'admittime'}
    },
    'icu_stay': {
        'first_careunit': {'valid_values': [u.value for u in ICUUnit]},
        'last_careunit': {'valid_values': [u.value for u in ICUUnit]},
        'outtime': {'min': 'intime'},
        'los': {'min': 0}
    },
    'lab_event': {
        'valuenum': {'min': 'ref_range_lower', 'max': 'ref_range_upper'},
        'flag': {'valid_values': ['LOW', 'HIGH', 'LOW|HIGH', None]}
    },
    'chart_event': {
        'storetime': {'min': 'charttime'},
        'warning': {'valid_values': ['WARNING', 'CRITICAL', None]},
        'error': {'valid_values': ['ERROR', 'CRITICAL', None]}
    },
    'clinical_score': {
        'score_value': {'min': 0},
        'components': {'type': 'dict', 'value_type': 'float64'}
    }
}

# Required columns that cannot be null
REQUIRED_COLUMNS = {
    'patient': ['subject_id', 'gender', 'anchor_age', 'anchor_year'],
    'admission': ['subject_id', 'hadm_id', 'admittime', 'dischtime'],
    'icu_stay': ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'],
    'lab_event': ['subject_id', 'hadm_id', 'charttime', 'itemid'],
    'chart_event': ['subject_id', 'hadm_id', 'charttime', 'itemid'],
    'clinical_score': ['subject_id', 'hadm_id', 'score_time', 'score_type', 'score_value']
}

def validate_dataframe(df: pd.DataFrame, data_type: str) -> bool:
    """
    Validate a DataFrame against the standard format for a given data type.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
        
    Returns:
        bool: True if DataFrame is valid, False otherwise
        
    Raises:
        ValueError: If data_type is invalid or DataFrame is invalid
    """
    if data_type not in STANDARD_COLUMNS:
        raise ValueError(f"Invalid data type: {data_type}")
        
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS[data_type] 
                   if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Check data types
    for col, dtype in COLUMN_DTYPES[data_type].items():
        if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
            raise ValueError(f"Invalid dtype for column {col}: expected {dtype}, got {df[col].dtype}")
            
    # Check value constraints
    constraints = VALUE_CONSTRAINTS[data_type]
    for col, constraint in constraints.items():
        if col not in df.columns:
            continue
            
        if 'valid_values' in constraint:
            invalid_values = df[~df[col].isin(constraint['valid_values'])]
            if not invalid_values.empty:
                raise ValueError(f"Invalid values in column {col}: {invalid_values[col].unique()}")
                
        if 'min' in constraint:
            if isinstance(constraint['min'], str):
                min_col = constraint['min']
                invalid_rows = df[df[col] < df[min_col]]
            else:
                invalid_rows = df[df[col] < constraint['min']]
            if not invalid_rows.empty:
                raise ValueError(f"Values below minimum in column {col}")
                
        if 'max' in constraint:
            if isinstance(constraint['max'], str):
                max_col = constraint['max']
                invalid_rows = df[df[col] > df[max_col]]
            else:
                invalid_rows = df[df[col] > constraint['max']]
            if not invalid_rows.empty:
                raise ValueError(f"Values above maximum in column {col}")
                
    # Check for null values in required columns
    for col in REQUIRED_COLUMNS[data_type]:
        if df[col].isnull().any():
            raise ValueError(f"Null values found in required column: {col}")
            
    return True

def convert_to_standard_format(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Convert a DataFrame to the standard format for a given data type.
    
    Args:
        df: DataFrame to convert
        data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
        
    Returns:
        pd.DataFrame: DataFrame in standard format
        
    Raises:
        ValueError: If data_type is invalid or conversion fails
    """
    if data_type not in STANDARD_COLUMNS:
        raise ValueError(f"Invalid data type: {data_type}")
        
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Ensure all required columns exist
    for col in STANDARD_COLUMNS[data_type]:
        if col not in df.columns:
            df[col] = None
            
    # Convert data types
    for col, dtype in COLUMN_DTYPES[data_type].items():
        if col in df.columns:
            try:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                raise ValueError(f"Failed to convert column {col} to {dtype}: {str(e)}")
                
    # Sort columns to match standard order
    df = df[STANDARD_COLUMNS[data_type]]
    
    # Validate the converted DataFrame
    validate_dataframe(df, data_type)
    
    return df 