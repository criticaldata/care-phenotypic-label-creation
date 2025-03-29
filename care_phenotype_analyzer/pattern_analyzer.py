"""
Module for analyzing care patterns and their relationships with clinical factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from .monitoring import SystemMonitor

class CarePatternAnalyzer:
    """Class for analyzing care patterns and their relationships with clinical factors."""
    
    def __init__(self,
                 data: pd.DataFrame,
                 clinical_factors: Optional[List[str]] = None,
                 log_dir: str = "logs"):
        """Initialize the pattern analyzer.
        
        Args:
            data: DataFrame containing care pattern data
            clinical_factors: List of clinical factors to consider
            log_dir: Directory for monitoring logs
        """
        self.data = data
        self.clinical_factors = clinical_factors or []
        
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Log initialization
        self.monitor.logger.info(
            f"Initialized CarePatternAnalyzer with {len(data)} records "
            f"and {len(self.clinical_factors)} clinical factors"
        )
        
    def analyze_measurement_frequency(self,
                                    measurement_column: str,
                                    time_column: str,
                                    clinical_factors: Optional[List[str]] = None,
                                    group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """Analyze the frequency of specific measurements.
        
        Args:
            measurement_column: Column containing the measurement to analyze
            time_column: Column containing temporal information
            clinical_factors: List of clinical factors to consider
            group_by: Columns to group the analysis by
            
        Returns:
            DataFrame containing frequency analysis results
        """
        start_time = time.time()
        
        try:
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
                
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(self.data)
            )
            
            # Log analysis results
            self.monitor.logger.info(
                f"Completed frequency analysis for {measurement_column} "
                f"in {processing_time:.2f} seconds"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error analyzing measurement frequency: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def _adjust_for_clinical_factors(self,
                                   measurement_column: str,
                                   clinical_factors: List[str],
                                   group_by: Optional[List[str]] = None) -> pd.Series:
        """Adjust measurements for clinical factors.
        
        Args:
            measurement_column: Column containing measurements to adjust
            clinical_factors: List of clinical factors to adjust for
            group_by: Columns to group the adjustment by
            
        Returns:
            Series containing adjusted measurements
        """
        try:
            # Prepare data for adjustment
            X = self.data[clinical_factors]
            y = self.data[measurement_column]
            
            # Fit linear model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate residuals (unexplained variation)
            residuals = y - model.predict(X)
            
            # Log adjustment results
            self.monitor.logger.info(
                f"Adjusted measurements for {len(clinical_factors)} clinical factors. "
                f"R-squared: {model.score(X, y):.3f}"
            )
            
            return residuals
            
        except Exception as e:
            error_msg = f"Error adjusting for clinical factors: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def visualize_clinical_separation(self,
                                    phenotype_labels: pd.Series,
                                    clinical_factors: Optional[List[str]] = None,
                                    output_file: Optional[str] = None) -> None:
        """Visualize clinical separation across phenotypes.
        
        Args:
            phenotype_labels: Series containing phenotype labels
            clinical_factors: List of clinical factors to visualize
            output_file: Optional path to save the visualization
        """
        try:
            if clinical_factors is None:
                clinical_factors = self.clinical_factors
                
            if not clinical_factors:
                warning_msg = "No clinical factors provided for visualization"
                self.monitor.record_warning(warning_msg)
                return
                
            # Create subplots for each clinical factor
            n_factors = len(clinical_factors)
            fig, axes = plt.subplots(n_factors, 1, figsize=(10, 4*n_factors))
            
            for i, factor in enumerate(clinical_factors):
                # Create violin plot
                sns.violinplot(
                    data=self.data,
                    x=phenotype_labels,
                    y=factor,
                    ax=axes[i] if n_factors > 1 else axes
                )
                
                # Add statistical significance
                for j in range(len(phenotype_labels.unique())):
                    for k in range(j+1, len(phenotype_labels.unique())):
                        stat, pval = stats.ttest_ind(
                            self.data[phenotype_labels == j][factor],
                            self.data[phenotype_labels == k][factor]
                        )
                        if pval < 0.05:
                            axes[i].text(
                                (j + k) / 2,
                                self.data[factor].max(),
                                f"p={pval:.3f}",
                                ha='center'
                            )
                            
                axes[i].set_title(f"Clinical Separation: {factor}")
                
            plt.tight_layout()
            
            # Save or display the plot
            if output_file:
                plt.savefig(output_file)
                self.monitor.logger.info(f"Saved clinical separation plot to {output_file}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            error_msg = f"Error visualizing clinical separation: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def visualize_unexplained_variation(self,
                                      phenotype_labels: pd.Series,
                                      care_patterns: List[str],
                                      clinical_factors: Optional[List[str]] = None,
                                      output_file: Optional[str] = None) -> None:
        """Visualize unexplained variation in care patterns.
        
        Args:
            phenotype_labels: Series containing phenotype labels
            care_patterns: List of care pattern columns to analyze
            clinical_factors: List of clinical factors to consider
            output_file: Optional path to save the visualization
        """
        try:
            if clinical_factors is None:
                clinical_factors = self.clinical_factors
                
            # Calculate explained and unexplained variation
            total_var = self.data[care_patterns].var()
            
            if clinical_factors:
                clinical_data = self.data[clinical_factors]
                explained_var = clinical_data.var()
                unexplained_var = total_var - explained_var
            else:
                unexplained_var = total_var
                explained_var = pd.Series(0, index=care_patterns)
                
            # Create stacked bar plot
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(care_patterns))
            width = 0.35
            
            plt.bar(x, explained_var, width, label='Explained', color='lightblue')
            plt.bar(x, unexplained_var, width, bottom=explained_var, label='Unexplained', color='lightcoral')
            
            plt.xlabel('Care Patterns')
            plt.ylabel('Variance')
            plt.title('Explained vs. Unexplained Variation in Care Patterns')
            plt.xticks(x, care_patterns, rotation=45)
            plt.legend()
            
            # Add percentage annotations
            for i, pattern in enumerate(care_patterns):
                total = explained_var[pattern] + unexplained_var[pattern]
                explained_pct = (explained_var[pattern] / total) * 100
                unexplained_pct = (unexplained_var[pattern] / total) * 100
                
                plt.text(i, total/2, f'{explained_pct:.1f}%', ha='center')
                plt.text(i, total*0.75, f'{unexplained_pct:.1f}%', ha='center')
                
            plt.tight_layout()
            
            # Save or display the plot
            if output_file:
                plt.savefig(output_file)
                self.monitor.logger.info(f"Saved unexplained variation plot to {output_file}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            error_msg = f"Error visualizing unexplained variation: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def visualize_variation_trends(self,
                                 phenotype_labels: pd.Series,
                                 care_patterns: List[str],
                                 clinical_factors: Optional[List[str]] = None,
                                 time_column: str = 'timestamp',
                                 output_file: Optional[str] = None) -> None:
        """Visualize trends in explained and unexplained variation over time.
        
        Args:
            phenotype_labels: Series containing phenotype labels
            care_patterns: List of care pattern columns to analyze
            clinical_factors: List of clinical factors to consider
            time_column: Column containing temporal information
            output_file: Optional path to save the visualization
        """
        try:
            if clinical_factors is None:
                clinical_factors = self.clinical_factors
                
            # Sort data by time
            data_sorted = self.data.sort_values(time_column)
            
            # Calculate variation over time
            window_size = len(data_sorted) // 10  # Use 10 time windows
            explained_trends = []
            unexplained_trends = []
            time_points = []
            
            for i in range(0, len(data_sorted), window_size):
                window_data = data_sorted.iloc[i:i+window_size]
                
                # Calculate variation for the window
                total_var = window_data[care_patterns].var()
                
                if clinical_factors:
                    clinical_data = window_data[clinical_factors]
                    explained_var = clinical_data.var()
                    unexplained_var = total_var - explained_var
                else:
                    unexplained_var = total_var
                    explained_var = pd.Series(0, index=care_patterns)
                    
                explained_trends.append(explained_var.mean())
                unexplained_trends.append(unexplained_var.mean())
                time_points.append(window_data[time_column].iloc[0])
                
            # Create line plot
            plt.figure(figsize=(12, 6))
            
            plt.plot(time_points, explained_trends, label='Explained', color='lightblue')
            plt.plot(time_points, unexplained_trends, label='Unexplained', color='lightcoral')
            
            plt.xlabel('Time')
            plt.ylabel('Average Variance')
            plt.title('Trends in Explained vs. Unexplained Variation')
            plt.legend()
            
            # Add percentage annotations at key points
            for i in [0, len(time_points)-1]:
                total = explained_trends[i] + unexplained_trends[i]
                explained_pct = (explained_trends[i] / total) * 100
                unexplained_pct = (unexplained_trends[i] / total) * 100
                
                plt.text(time_points[i], total/2, f'{explained_pct:.1f}%', ha='center')
                plt.text(time_points[i], total*0.75, f'{unexplained_pct:.1f}%', ha='center')
                
            plt.tight_layout()
            
            # Save or display the plot
            if output_file:
                plt.savefig(output_file)
                self.monitor.logger.info(f"Saved variation trends plot to {output_file}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            error_msg = f"Error visualizing variation trends: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring() 