"""
Module for creating an interactive dashboard using Dash.

This module provides a web-based interface for exploring care patterns,
phenotypes, and analysis results.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
from datetime import datetime
from .monitoring import SystemMonitor
from .visualization import AdvancedVisualizer
from .export import DataExporter

class CareDashboard:
    """Class for creating and managing the interactive dashboard."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the dashboard.
        
        Args:
            log_dir: Directory for monitoring logs
        """
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Initialize visualizer and exporter
        self.visualizer = AdvancedVisualizer(log_dir=log_dir)
        self.exporter = DataExporter(log_dir=log_dir)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        
        # Set up layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Log initialization
        self.monitor.logger.info("Initialized CareDashboard")
        
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("CARE Phenotype Analysis Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Main content
            html.Div([
                # Left sidebar
                html.Div([
                    html.H3("Analysis Controls"),
                    html.Label("Select Analysis Type:"),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Care Patterns', 'value': 'patterns'},
                            {'label': 'Clinical Separation', 'value': 'separation'},
                            {'label': 'Unexplained Variation', 'value': 'variation'},
                            {'label': 'Fairness Metrics', 'value': 'fairness'}
                        ],
                        value='patterns'
                    ),
                    html.Label("Select Clinical Factors:"),
                    dcc.Dropdown(
                        id='clinical-factors',
                        multi=True,
                        placeholder="Select factors..."
                    ),
                    html.Label("Time Range:"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=datetime(2020, 1, 1),
                        end_date=datetime(2023, 12, 31)
                    ),
                    html.Button(
                        'Update Analysis',
                        id='update-button',
                        n_clicks=0,
                        style={'marginTop': '20px'}
                    ),
                    html.Button(
                        'Export Results',
                        id='export-button',
                        n_clicks=0,
                        style={'marginTop': '10px'}
                    )
                ], style={'width': '25%', 'float': 'left', 'padding': '20px'}),
                
                # Main content area
                html.Div([
                    # Summary statistics
                    html.Div([
                        html.H3("Summary Statistics"),
                        html.Div(id='summary-stats')
                    ], style={'marginBottom': '20px'}),
                    
                    # Main visualization
                    html.Div([
                        html.H3("Analysis Results"),
                        dcc.Graph(id='main-plot')
                    ], style={'marginBottom': '20px'}),
                    
                    # Additional visualizations
                    html.Div([
                        html.H3("Additional Insights"),
                        html.Div(id='additional-plots')
                    ])
                ], style={'width': '75%', 'float': 'right', 'padding': '20px'})
            ]),
            
            # Footer
            html.Div([
                html.P("System Status:", style={'fontWeight': 'bold'}),
                html.Div(id='system-status')
            ], style={'clear': 'both', 'padding': '20px', 'backgroundColor': '#f8f9fa'})
        ])
        
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            Output('clinical-factors', 'options'),
            Input('analysis-type', 'value')
        )
        def update_clinical_factors(analysis_type):
            """Update available clinical factors based on analysis type."""
            # This would be populated with actual clinical factors
            return [
                {'label': 'Age', 'value': 'age'},
                {'label': 'Gender', 'value': 'gender'},
                {'label': 'Race', 'value': 'race'},
                {'label': 'Insurance', 'value': 'insurance'}
            ]
        
        @self.app.callback(
            [Output('main-plot', 'figure'),
             Output('summary-stats', 'children'),
             Output('additional-plots', 'children')],
            [Input('update-button', 'n_clicks')],
            [State('analysis-type', 'value'),
             State('clinical-factors', 'value'),
             State('date-range', 'start_date'),
             State('date-range', 'end_date')]
        )
        def update_analysis(n_clicks, analysis_type, clinical_factors, start_date, end_date):
            """Update the analysis based on user inputs."""
            if n_clicks is None:
                return {}, [], []
            
            start_time = time.time()
            
            try:
                # Generate sample data for demonstration
                # In production, this would use actual data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                data = pd.DataFrame({
                    'date': dates,
                    'value': np.random.randn(len(dates)).cumsum()
                })
                
                # Create main plot
                fig = px.line(data, x='date', y='value',
                            title=f"{analysis_type.title()} Analysis")
                
                # Create summary statistics
                summary_stats = html.Div([
                    html.P(f"Total Records: {len(data)}"),
                    html.P(f"Date Range: {start_date} to {end_date}"),
                    html.P(f"Selected Factors: {', '.join(clinical_factors or [])}")
                ])
                
                # Create additional plots
                additional_plots = html.Div([
                    dcc.Graph(
                        figure=px.box(data, y='value',
                                    title="Value Distribution")
                    ),
                    dcc.Graph(
                        figure=px.scatter(data, x='date', y='value',
                                        title="Value Scatter Plot")
                    )
                ])
                
                # Record processing metrics
                processing_time = time.time() - start_time
                self.monitor.record_processing(
                    processing_time=processing_time,
                    batch_size=len(data)
                )
                
                return fig, summary_stats, additional_plots
                
            except Exception as e:
                error_msg = f"Error updating analysis: {str(e)}"
                self.monitor.record_error(error_msg)
                return {}, html.Div(error_msg), []
        
        @self.app.callback(
            Output('system-status', 'children'),
            Input('update-button', 'n_clicks')
        )
        def update_system_status(n_clicks):
            """Update the system status display."""
            if n_clicks is None:
                return "Ready"
            
            try:
                # Get system metrics
                metrics = self.monitor.get_metrics()
                
                return html.Div([
                    html.P(f"CPU Usage: {metrics['cpu_percent']}%"),
                    html.P(f"Memory Usage: {metrics['memory_percent']}%"),
                    html.P(f"Processing Time: {metrics['processing_time']:.2f}s"),
                    html.P(f"Active Processes: {metrics['active_processes']}")
                ])
                
            except Exception as e:
                error_msg = f"Error updating system status: {str(e)}"
                self.monitor.record_error(error_msg)
                return error_msg
        
        @self.app.callback(
            Output('export-button', 'n_clicks'),
            Input('export-button', 'n_clicks'),
            [State('main-plot', 'figure'),
             State('analysis-type', 'value')]
        )
        def export_results(n_clicks, figure, analysis_type):
            """Export the current analysis results."""
            if n_clicks is None:
                return None
            
            try:
                # Create timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Export visualization
                self.exporter.export_visualization(
                    figure,
                    f"exports/{analysis_type}_{timestamp}.html"
                )
                
                # Export data
                data = pd.DataFrame({
                    'date': pd.date_range(start='2020-01-01', end='2023-12-31', freq='D'),
                    'value': np.random.randn(1461).cumsum()  # Sample data
                })
                
                self.exporter.export_to_csv(
                    data,
                    f"exports/{analysis_type}_data_{timestamp}.csv"
                )
                
                return None
                
            except Exception as e:
                error_msg = f"Error exporting results: {str(e)}"
                self.monitor.record_error(error_msg)
                return None
    
    def run_server(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        try:
            self.app.run_server(debug=debug, port=port)
        except Exception as e:
            error_msg = f"Error running dashboard server: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring() 