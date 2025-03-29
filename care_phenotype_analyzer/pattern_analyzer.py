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
    
    def visualize_clinical_separation(self,
                                    phenotype_labels: pd.Series,
                                    clinical_factors: List[str],
                                    output_file: Optional[str] = None) -> None:
        """
        Visualize clinical separation across phenotypes.
        
        Parameters
        ----------
        phenotype_labels : pd.Series
            Series containing phenotype labels
        clinical_factors : List[str]
            List of clinical factors to visualize
        output_file : str, optional
            Path to save the visualization
        """
        n_factors = len(clinical_factors)
        n_phenotypes = len(phenotype_labels.unique())
        
        # Create subplot grid
        fig, axes = plt.subplots(n_factors, 1, figsize=(12, 4*n_factors))
        if n_factors == 1:
            axes = [axes]
            
        for idx, factor in enumerate(clinical_factors):
            # Create violin plot
            sns.violinplot(x=phenotype_labels, y=self.data[factor], ax=axes[idx])
            axes[idx].set_title(f'Distribution of {factor} across Phenotypes')
            axes[idx].set_xlabel('Phenotype')
            axes[idx].set_ylabel(factor)
            
            # Add statistical significance annotations
            for i in range(n_phenotypes):
                for j in range(i+1, n_phenotypes):
                    # Perform t-test between phenotypes
                    t_stat, p_value = stats.ttest_ind(
                        self.data.loc[phenotype_labels == i, factor],
                        self.data.loc[phenotype_labels == j, factor]
                    )
                    
                    if p_value < 0.05:
                        # Add significance annotation
                        y_max = self.data[factor].max()
                        axes[idx].annotate(
                            f'p={p_value:.3f}',
                            xy=((i+j)/2, y_max),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center',
                            va='bottom'
                        )
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
            
    def visualize_unexplained_variation(self,
                                      phenotype_labels: pd.Series,
                                      care_patterns: List[str],
                                      clinical_factors: List[str],
                                      output_file: Optional[str] = None) -> None:
        """
        Visualize unexplained variation in care patterns across phenotypes.
        
        Parameters
        ----------
        phenotype_labels : pd.Series
            Series containing phenotype labels
        care_patterns : List[str]
            List of care patterns to analyze
        clinical_factors : List[str]
            List of clinical factors to consider
        output_file : str, optional
            Path to save the visualization
        """
        n_patterns = len(care_patterns)
        n_phenotypes = len(phenotype_labels.unique())
        
        # Create subplot grid
        fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 4*n_patterns))
        if n_patterns == 1:
            axes = [axes]
            
        for idx, pattern in enumerate(care_patterns):
            # Calculate explained and unexplained variation
            explained_var = []
            unexplained_var = []
            
            for phenotype in phenotype_labels.unique():
                mask = phenotype_labels == phenotype
                pattern_data = self.data.loc[mask, pattern]
                clinical_data = self.data.loc[mask, clinical_factors]
                
                # Fit regression model
                model = stats.linregress(clinical_data, pattern_data)
                predicted = model.predict(clinical_data)
                
                # Calculate variances
                total_var = np.var(pattern_data)
                explained = np.var(predicted)
                unexplained = total_var - explained
                
                explained_var.append(explained)
                unexplained_var.append(unexplained)
            
            # Create stacked bar plot
            x = np.arange(n_phenotypes)
            width = 0.35
            
            axes[idx].bar(x, explained_var, width, label='Explained', color='lightblue')
            axes[idx].bar(x, unexplained_var, width, bottom=explained_var, label='Unexplained', color='lightcoral')
            
            axes[idx].set_title(f'Variation in {pattern} across Phenotypes')
            axes[idx].set_xlabel('Phenotype')
            axes[idx].set_ylabel('Variance')
            axes[idx].legend()
            
            # Add percentage annotations
            for i in range(n_phenotypes):
                total = explained_var[i] + unexplained_var[i]
                unexplained_pct = unexplained_var[i] / total * 100
                axes[idx].annotate(
                    f'{unexplained_pct:.1f}%',
                    xy=(i, total),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    va='bottom'
                )
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
            
    def visualize_variation_trends(self,
                                 phenotype_labels: pd.Series,
                                 care_patterns: List[str],
                                 clinical_factors: List[str],
                                 time_column: str,
                                 output_file: Optional[str] = None) -> None:
        """
        Visualize trends in explained and unexplained variation over time.
        
        Parameters
        ----------
        phenotype_labels : pd.Series
            Series containing phenotype labels
        care_patterns : List[str]
            List of care patterns to analyze
        clinical_factors : List[str]
            List of clinical factors to consider
        time_column : str
            Column containing temporal information
        output_file : str, optional
            Path to save the visualization
        """
        n_patterns = len(care_patterns)
        
        # Create subplot grid
        fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 4*n_patterns))
        if n_patterns == 1:
            axes = [axes]
            
        for idx, pattern in enumerate(care_patterns):
            # Calculate variation over time
            time_points = sorted(self.data[time_column].unique())
            explained_trend = []
            unexplained_trend = []
            
            for time_point in time_points:
                mask = self.data[time_column] == time_point
                pattern_data = self.data.loc[mask, pattern]
                clinical_data = self.data.loc[mask, clinical_factors]
                
                if len(pattern_data) > 0:
                    # Fit regression model
                    model = stats.linregress(clinical_data, pattern_data)
                    predicted = model.predict(clinical_data)
                    
                    # Calculate variances
                    total_var = np.var(pattern_data)
                    explained = np.var(predicted)
                    unexplained = total_var - explained
                    
                    explained_trend.append(explained)
                    unexplained_trend.append(unexplained)
            
            # Create line plot
            axes[idx].plot(time_points, explained_trend, label='Explained', color='lightblue')
            axes[idx].plot(time_points, unexplained_trend, label='Unexplained', color='lightcoral')
            
            axes[idx].set_title(f'Variation Trends in {pattern} over Time')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Variance')
            axes[idx].legend()
            
            # Add percentage annotations at key points
            for i in [0, len(time_points)-1]:
                total = explained_trend[i] + unexplained_trend[i]
                unexplained_pct = unexplained_trend[i] / total * 100
                axes[idx].annotate(
                    f'{unexplained_pct:.1f}%',
                    xy=(time_points[i], total),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    va='bottom'
                )
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show() 