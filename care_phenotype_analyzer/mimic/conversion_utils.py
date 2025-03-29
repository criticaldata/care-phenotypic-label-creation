"""
Data conversion utilities for MIMIC data processing.

This module provides utilities for converting between different data formats
and handling various data transformations in the MIMIC data processing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from .structures import (
    Patient, Admission, ICUStay, LabEvent, ChartEvent,
    ClinicalScore, Gender, AdmissionType, ICUUnit
)
from .data_formats import (
    STANDARD_COLUMNS, COLUMN_DTYPES, VALUE_CONSTRAINTS,
    REQUIRED_COLUMNS, validate_dataframe, convert_to_standard_format
)

class DataConverter:
    """Utility class for converting between different data formats."""
    
    @staticmethod
    def dataframe_to_objects(df: pd.DataFrame, data_type: str) -> List[Any]:
        """
        Convert a DataFrame to a list of corresponding data objects.
        
        Args:
            df: DataFrame to convert
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            
        Returns:
            List of corresponding data objects
            
        Raises:
            ValueError: If data_type is invalid or conversion fails
        """
        # First convert to standard format
        df = convert_to_standard_format(df, data_type)
        
        # Convert to objects based on data type
        if data_type == 'patient':
            return [Patient(**row) for _, row in df.iterrows()]
        elif data_type == 'admission':
            return [Admission(**row) for _, row in df.iterrows()]
        elif data_type == 'icu_stay':
            return [ICUStay(**row) for _, row in df.iterrows()]
        elif data_type == 'lab_event':
            return [LabEvent(**row) for _, row in df.iterrows()]
        elif data_type == 'chart_event':
            return [ChartEvent(**row) for _, row in df.iterrows()]
        elif data_type == 'clinical_score':
            return [ClinicalScore(**row) for _, row in df.iterrows()]
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    @staticmethod
    def objects_to_dataframe(objects: List[Any], data_type: str) -> pd.DataFrame:
        """
        Convert a list of data objects to a DataFrame.
        
        Args:
            objects: List of data objects to convert
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            
        Returns:
            DataFrame containing the object data
            
        Raises:
            ValueError: If data_type is invalid or conversion fails
        """
        if not objects:
            return pd.DataFrame(columns=STANDARD_COLUMNS[data_type])
            
        # Convert objects to dictionaries
        data = [obj.__dict__ for obj in objects]
        df = pd.DataFrame(data)
        
        # Convert to standard format
        return convert_to_standard_format(df, data_type)
    
    @staticmethod
    def convert_timezone(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
        """
        Convert datetime columns to specified timezone.
        
        Args:
            df: DataFrame with datetime columns
            timezone: Target timezone (e.g., 'UTC', 'America/New_York')
            
        Returns:
            DataFrame with converted datetime columns
        """
        df = df.copy()
        
        # Find all datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        
        # Convert each datetime column
        for col in datetime_cols:
            df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(timezone)
            
        return df
    
    @staticmethod
    def standardize_units(df: pd.DataFrame, 
                         value_col: str,
                         unit_col: str,
                         target_unit: str,
                         conversion_factors: Dict[str, float]) -> pd.DataFrame:
        """
        Standardize measurement units to a target unit.
        
        Args:
            df: DataFrame containing measurements
            value_col: Column containing measurement values
            unit_col: Column containing unit information
            target_unit: Target unit to convert to
            conversion_factors: Dictionary mapping source units to conversion factors
            
        Returns:
            DataFrame with standardized units
        """
        df = df.copy()
        
        # Create mask for rows with units that need conversion
        mask = df[unit_col].isin(conversion_factors.keys())
        
        # Apply conversion factors
        for unit, factor in conversion_factors.items():
            unit_mask = df[unit_col] == unit
            df.loc[unit_mask, value_col] *= factor
            df.loc[unit_mask, unit_col] = target_unit
            
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                            data_type: str,
                            strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame to process
            data_type: Type of data ('patient', 'admission', 'icu_stay', etc.)
            strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Get required columns
        required = REQUIRED_COLUMNS[data_type]
        
        if strategy == 'drop':
            # Drop rows with missing values in required columns
            df = df.dropna(subset=required)
            
        elif strategy == 'fill':
            # Fill missing values based on column type
            for col in df.columns:
                if col in required:
                    continue
                    
                dtype = COLUMN_DTYPES[data_type].get(col)
                if dtype == 'float64':
                    df[col] = df[col].fillna(df[col].mean())
                elif dtype == 'category':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif dtype == 'datetime64[ns]':
                    df[col] = df[col].fillna(pd.NaT)
                    
        elif strategy == 'interpolate':
            # Interpolate missing values for numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            
        return df
    
    @staticmethod
    def aggregate_measurements(df: pd.DataFrame,
                             group_cols: List[str],
                             value_col: str,
                             agg_functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Aggregate measurements by specified columns.
        
        Args:
            df: DataFrame containing measurements
            group_cols: Columns to group by
            value_col: Column containing values to aggregate
            agg_functions: List of aggregation functions to apply
            
        Returns:
            DataFrame with aggregated measurements
        """
        return df.groupby(group_cols)[value_col].agg(agg_functions).reset_index()
    
    @staticmethod
    def resample_time_series(df: pd.DataFrame,
                            time_col: str,
                            value_col: str,
                            freq: str = '1H',
                            method: str = 'linear') -> pd.DataFrame:
        """
        Resample time series data to a specified frequency.
        
        Args:
            df: DataFrame containing time series data
            time_col: Column containing timestamps
            value_col: Column containing values
            freq: Target frequency (e.g., '1H', '4H', '1D')
            method: Interpolation method ('linear', 'ffill', 'bfill')
            
        Returns:
            DataFrame with resampled time series
        """
        df = df.copy()
        
        # Set time column as index
        df = df.set_index(time_col)
        
        # Resample and interpolate
        df = df[value_col].resample(freq).asfreq()
        df = df.interpolate(method=method)
        
        # Reset index
        df = df.reset_index()
        
        return df
    
    @staticmethod
    def calculate_time_differences(df: pd.DataFrame,
                                 start_col: str,
                                 end_col: str,
                                 unit: str = 'hours') -> pd.DataFrame:
        """
        Calculate time differences between two datetime columns.
        
        Args:
            df: DataFrame containing datetime columns
            start_col: Column containing start times
            end_col: Column containing end times
            unit: Unit for time difference ('hours', 'days', 'minutes')
            
        Returns:
            DataFrame with calculated time differences
        """
        df = df.copy()
        
        # Calculate time difference
        df['time_diff'] = (df[end_col] - df[start_col]).dt.total_seconds()
        
        # Convert to specified unit
        if unit == 'hours':
            df['time_diff'] /= 3600
        elif unit == 'days':
            df['time_diff'] /= 86400
        elif unit == 'minutes':
            df['time_diff'] /= 60
            
        return df 