"""
Module for analyzing care patterns and their relationships.
Focuses on understanding variations in lab test measurements and care patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class CarePatternAnalyzer:
    """
    Analyzes care patterns and their relationships to identify meaningful variations.
    Focuses on understanding how lab test measurements and care patterns vary across patients.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the pattern analyzer with care data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing care pattern data
        """
        self.data = data
        
    def analyze_measurement_frequency(self,
                                    measurement_column: str,
                                    time_column: str,
                                    clinical_factors: Optional[List[str]] = None,
                                    group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze the frequency of specific measurements (e.g., lab tests).
        Accounts for clinical factors that may justify variations.
        
        Parameters
        ----------
        measurement_column : str
            Column containing the measurement to analyze
        time_column : str
            Column containing temporal information
        clinical_factors : List[str], optional
            List of clinical factors to consider
        group_by : List[str], optional
            Columns to group the analysis by
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing frequency analysis results
        """
        if group_by:
            grouped = self.data.groupby(group_by)
        else:
            grouped = self.data
            
        # Calculate basic frequency metrics
        results = grouped[measurement_column].agg(['count', 'mean', 'std'])
        
        # Calculate time-based metrics
        time_span = (self.data[time_column].max() - 
                    self.data[time_column].min()).days
        results['frequency_per_day'] = results['count'] / time_span
        
        # Adjust for clinical factors if provided
        if clinical_factors:
            adjusted_frequencies = self._adjust_for_clinical_factors(
                measurement_column,
                clinical_factors,
                group_by
            )
            results['adjusted_frequency'] = adjusted_frequencies
            
        return results
    
    def _adjust_for_clinical_factors(self,
                                   measurement_column: str,
                                   clinical_factors: List[str],
                                   group_by: Optional[List[str]] = None) -> pd.Series:
        """
        Adjust measurement frequencies for clinical factors.
        
        Parameters
        ----------
        measurement_column : str
            Column containing the measurement to analyze
        clinical_factors : List[str]
            List of clinical factors to consider
        group_by : List[str], optional
            Columns to group the analysis by
            
        Returns
        -------
        pd.Series
            Adjusted measurement frequencies
        """
        # Create regression model
        X = self.data[clinical_factors]
        y = self.data[measurement_column]
        
        # Fit linear regression
        model = stats.linregress(X, y)
        
        # Calculate residuals (unexplained variation)
        predicted = model.predict(X)
        residuals = y - predicted
        
        return residuals
    
    def identify_pattern_correlations(self,
                                   pattern_columns: List[str],
                                   clinical_factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Identify correlations between different care patterns and clinical factors.
        Helps understand which variations can be explained by clinical factors.
        
        Parameters
        ----------
        pattern_columns : List[str]
            Columns containing care patterns to analyze
        clinical_factors : List[str], optional
            Columns containing clinical factors to consider
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix between patterns and factors
        """
        columns_to_analyze = pattern_columns.copy()
        if clinical_factors:
            columns_to_analyze.extend(clinical_factors)
            
        return self.data[columns_to_analyze].corr()
    
    def visualize_pattern_distribution(self,
                                     pattern_column: str,
                                     phenotype_column: Optional[str] = None,
                                     clinical_factor: Optional[str] = None,
                                     time_column: Optional[str] = None) -> None:
        """
        Visualize the distribution of care patterns across phenotypes or clinical factors.
        Includes temporal analysis if time information is provided.
        
        Parameters
        ----------
        pattern_column : str
            Column containing the care pattern to visualize
        phenotype_column : str, optional
            Column containing phenotype labels
        clinical_factor : str, optional
            Column containing clinical factor to consider
        time_column : str, optional
            Column containing temporal information
        """
        plt.figure(figsize=(12, 6))
        
        if time_column:
            # Create time series plot
            plt.subplot(1, 2, 1)
            if phenotype_column:
                for phenotype in self.data[phenotype_column].unique():
                    mask = self.data[phenotype_column] == phenotype
                    plt.plot(self.data.loc[mask, time_column],
                            self.data.loc[mask, pattern_column],
                            label=f'Phenotype {phenotype}')
                plt.title(f'Time Series of {pattern_column} by Phenotype')
            else:
                plt.plot(self.data[time_column], self.data[pattern_column])
                plt.title(f'Time Series of {pattern_column}')
                
            plt.xlabel('Time')
            plt.ylabel(pattern_column)
            
            # Create distribution plot
            plt.subplot(1, 2, 2)
            
        if phenotype_column:
            sns.boxplot(x=phenotype_column, y=pattern_column, data=self.data)
            plt.title(f'Distribution of {pattern_column} across Phenotypes')
        elif clinical_factor:
            sns.boxplot(x=clinical_factor, y=pattern_column, data=self.data)
            plt.title(f'Distribution of {pattern_column} across {clinical_factor}')
        else:
            sns.histplot(data=self.data, x=pattern_column)
            plt.title(f'Distribution of {pattern_column}')
            
        plt.tight_layout()
        plt.show()
    
    def analyze_temporal_patterns(self,
                                pattern_column: str,
                                time_column: str,
                                phenotype_column: Optional[str] = None,
                                clinical_factors: Optional[List[str]] = None) -> Dict:
        """
        Analyze temporal patterns in care delivery.
        Accounts for clinical factors that may influence temporal patterns.
        
        Parameters
        ----------
        pattern_column : str
            Column containing the care pattern to analyze
        time_column : str
            Column containing temporal information
        phenotype_column : str, optional
            Column containing phenotype labels
        clinical_factors : List[str], optional
            List of clinical factors to consider
            
        Returns
        -------
        Dict
            Dictionary containing temporal analysis results
        """
        results = {}
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data[time_column]):
            self.data[time_column] = pd.to_datetime(self.data[time_column])
            
        # Calculate time-based metrics
        results['time_span'] = (self.data[time_column].max() - 
                              self.data[time_column].min()).days
        results['pattern_frequency'] = len(self.data) / results['time_span']
        
        if phenotype_column:
            # Analyze patterns by phenotype
            phenotype_patterns = self.data.groupby(phenotype_column).apply(
                lambda x: len(x) / (x[time_column].max() - x[time_column].min()).days
            )
            results['phenotype_patterns'] = phenotype_patterns
            
        if clinical_factors:
            # Analyze relationship with clinical factors
            for factor in clinical_factors:
                correlation = stats.pearsonr(
                    self.data[factor],
                    self.data.groupby(time_column)[pattern_column].count()
                )
                results[f'{factor}_correlation'] = {
                    'correlation': correlation[0],
                    'p_value': correlation[1]
                }
            
        return results 