"""
Module for analyzing care patterns and their relationships.
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
        
    def analyze_pattern_frequency(self,
                                pattern_column: str,
                                time_column: Optional[str] = None,
                                group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze the frequency of specific care patterns.
        
        Parameters
        ----------
        pattern_column : str
            Column containing the care pattern to analyze
        time_column : str, optional
            Column containing temporal information
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
            
        results = grouped[pattern_column].agg(['count', 'mean', 'std'])
        
        if time_column:
            # Add time-based analysis
            time_span = (self.data[time_column].max() - 
                        self.data[time_column].min()).days
            results['frequency_per_day'] = results['count'] / time_span
            
        return results
    
    def identify_pattern_correlations(self,
                                   pattern_columns: List[str],
                                   clinical_factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Identify correlations between different care patterns and clinical factors.
        
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
                                     clinical_factor: Optional[str] = None) -> None:
        """
        Visualize the distribution of care patterns across phenotypes or clinical factors.
        
        Parameters
        ----------
        pattern_column : str
            Column containing the care pattern to visualize
        phenotype_column : str, optional
            Column containing phenotype labels
        clinical_factor : str, optional
            Column containing clinical factor to consider
        """
        plt.figure(figsize=(10, 6))
        
        if phenotype_column:
            sns.boxplot(x=phenotype_column, y=pattern_column, data=self.data)
            plt.title(f'Distribution of {pattern_column} across Phenotypes')
        elif clinical_factor:
            sns.boxplot(x=clinical_factor, y=pattern_column, data=self.data)
            plt.title(f'Distribution of {pattern_column} across {clinical_factor}')
        else:
            sns.histplot(data=self.data, x=pattern_column)
            plt.title(f'Distribution of {pattern_column}')
            
        plt.show()
    
    def analyze_temporal_patterns(self,
                                pattern_column: str,
                                time_column: str,
                                phenotype_column: Optional[str] = None) -> Dict:
        """
        Analyze temporal patterns in care delivery.
        
        Parameters
        ----------
        pattern_column : str
            Column containing the care pattern to analyze
        time_column : str
            Column containing temporal information
        phenotype_column : str, optional
            Column containing phenotype labels
            
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
            
        return results 