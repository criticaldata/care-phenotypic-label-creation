"""
Module for advanced visualization capabilities.

This module provides sophisticated visualization tools for analyzing care patterns,
fairness metrics, and system performance. It includes interactive plots and
advanced analysis visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time
from .monitoring import SystemMonitor

class AdvancedVisualizer:
    """Class for creating advanced visualizations of care patterns and analysis results."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the advanced visualizer.
        
        Args:
            log_dir: Directory for monitoring logs
        """
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Log initialization
        self.monitor.logger.info("Initialized AdvancedVisualizer")
        
    def create_interactive_pattern_plot(self,
                                     data: pd.DataFrame,
                                     phenotype_labels: pd.Series,
                                     clinical_factors: Optional[pd.DataFrame] = None,
                                     output_file: Optional[str] = None) -> go.Figure:
        """Create an interactive visualization of care patterns.
        
        Args:
            data: DataFrame containing care pattern data
            phenotype_labels: Series containing phenotype labels
            clinical_factors: Optional DataFrame containing clinical factors
            output_file: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        start_time = time.time()
        
        try:
            # Create figure
            fig = go.Figure()
            
            # Add traces for each phenotype
            for phenotype in phenotype_labels.unique():
                mask = phenotype_labels == phenotype
                phenotype_data = data[mask]
                
                # Calculate mean pattern
                mean_pattern = phenotype_data.mean()
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=mean_pattern.index,
                    y=mean_pattern.values,
                    name=f'Phenotype {phenotype}',
                    mode='lines+markers'
                ))
            
            # Update layout
            fig.update_layout(
                title='Interactive Care Pattern Analysis',
                xaxis_title='Time',
                yaxis_title='Measurement Value',
                hovermode='x unified',
                showlegend=True
            )
            
            # Add clinical factors if provided
            if clinical_factors is not None:
                for factor in clinical_factors.columns:
                    fig.add_trace(go.Scatter(
                        x=clinical_factors.index,
                        y=clinical_factors[factor],
                        name=factor,
                        mode='lines',
                        line=dict(dash='dash')
                    ))
            
            # Save plot if output file specified
            if output_file:
                fig.write_html(output_file)
                self.monitor.logger.info(f"Saved interactive plot to {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(data)
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating interactive pattern plot: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def create_fairness_heatmap(self,
                              fairness_metrics: Dict[str, Dict[str, float]],
                              output_file: Optional[str] = None) -> go.Figure:
        """Create an interactive heatmap of fairness metrics.
        
        Args:
            fairness_metrics: Dictionary containing fairness metrics
            output_file: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        start_time = time.time()
        
        try:
            # Prepare data for heatmap
            metrics = list(fairness_metrics.keys())
            factors = list(fairness_metrics[metrics[0]].keys())
            
            z_values = []
            for metric in metrics:
                row = [fairness_metrics[metric][factor] for factor in factors]
                z_values.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=factors,
                y=metrics,
                colorscale='RdYlBu_r',
                colorbar=dict(title='Disparity')
            ))
            
            # Update layout
            fig.update_layout(
                title='Fairness Metrics Heatmap',
                xaxis_title='Demographic Factor',
                yaxis_title='Fairness Metric',
                height=600
            )
            
            # Save plot if output file specified
            if output_file:
                fig.write_html(output_file)
                self.monitor.logger.info(f"Saved fairness heatmap to {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(metrics) * len(factors)
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating fairness heatmap: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def create_bias_mitigation_comparison(self,
                                       original_metrics: Dict[str, float],
                                       mitigated_metrics: Dict[str, Dict[str, float]],
                                       output_file: Optional[str] = None) -> go.Figure:
        """Create an interactive comparison of bias mitigation strategies.
        
        Args:
            original_metrics: Dictionary containing original fairness metrics
            mitigated_metrics: Dictionary containing mitigated fairness metrics
            output_file: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        start_time = time.time()
        
        try:
            # Create figure
            fig = go.Figure()
            
            # Add original metrics
            fig.add_trace(go.Bar(
                name='Original',
                x=list(original_metrics.keys()),
                y=list(original_metrics.values()),
                marker_color='blue'
            ))
            
            # Add mitigated metrics
            colors = ['red', 'green', 'purple']
            for i, (strategy, metrics) in enumerate(mitigated_metrics.items()):
                fig.add_trace(go.Bar(
                    name=strategy.capitalize(),
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=colors[i]
                ))
            
            # Update layout
            fig.update_layout(
                title='Bias Mitigation Comparison',
                xaxis_title='Fairness Metric',
                yaxis_title='Value',
                barmode='group',
                height=500
            )
            
            # Save plot if output file specified
            if output_file:
                fig.write_html(output_file)
                self.monitor.logger.info(f"Saved bias mitigation comparison to {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(original_metrics) + sum(len(m) for m in mitigated_metrics.values())
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating bias mitigation comparison: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def create_system_health_dashboard(self,
                                    metrics: Dict[str, List[float]],
                                    health_status: Dict[str, Any],
                                    output_file: Optional[str] = None) -> go.Figure:
        """Create an interactive dashboard of system health metrics.
        
        Args:
            metrics: Dictionary containing system metrics over time
            health_status: Dictionary containing current health status
            output_file: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        start_time = time.time()
        
        try:
            # Create subplots
            fig = go.Figure()
            
            # Add metrics traces
            for metric, values in metrics.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    name=metric,
                    mode='lines+markers'
                ))
            
            # Update layout
            fig.update_layout(
                title='System Health Dashboard',
                xaxis_title='Time',
                yaxis_title='Value',
                height=600,
                showlegend=True
            )
            
            # Add health status annotations
            for component, status in health_status.items():
                fig.add_annotation(
                    text=f"{component}: {status}",
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    showarrow=False
                )
            
            # Save plot if output file specified
            if output_file:
                fig.write_html(output_file)
                self.monitor.logger.info(f"Saved system health dashboard to {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(metrics)
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating system health dashboard: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring() 