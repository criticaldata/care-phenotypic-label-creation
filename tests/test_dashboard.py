"""
Tests for dashboard functionality.
"""

import pytest
import dash
from dash import html, dcc
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from care_phenotype_analyzer.dashboard import CareDashboard

@pytest.fixture
def dashboard():
    """Create a temporary directory for logs and initialize the dashboard."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dashboard = CareDashboard(log_dir=temp_dir)
        yield dashboard
        dashboard.monitor.stop_monitoring()

def test_dashboard_initialization(dashboard):
    """Test proper initialization of the dashboard."""
    assert dashboard.monitor is not None
    assert dashboard.visualizer is not None
    assert dashboard.exporter is not None
    assert isinstance(dashboard.app, dash.Dash)

def test_dashboard_layout(dashboard):
    """Test dashboard layout components."""
    layout = dashboard.app.layout
    
    # Check header
    assert isinstance(layout.children[0], html.H1)
    assert layout.children[0].children == "CARE Phenotype Analysis Dashboard"
    
    # Check main content
    main_content = layout.children[1]
    assert len(main_content.children) == 2  # Sidebar and main area
    
    # Check sidebar
    sidebar = main_content.children[0]
    assert isinstance(sidebar, html.Div)
    assert len(sidebar.children) >= 6  # Controls, dropdowns, buttons
    
    # Check main area
    main_area = main_content.children[1]
    assert isinstance(main_area, html.Div)
    assert len(main_area.children) == 3  # Summary, main plot, additional plots
    
    # Check footer
    footer = layout.children[2]
    assert isinstance(footer, html.Div)
    assert len(footer.children) == 2  # Status label and content

def test_clinical_factors_callback(dashboard):
    """Test the clinical factors dropdown callback."""
    # Simulate callback
    result = dashboard.app.callback_context.triggered[0]['value']
    
    # Check if options are returned
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(opt, dict) for opt in result)
    assert all('label' in opt and 'value' in opt for opt in result)

def test_analysis_update_callback(dashboard):
    """Test the analysis update callback."""
    # Simulate callback with sample inputs
    n_clicks = 1
    analysis_type = 'patterns'
    clinical_factors = ['age', 'gender']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Get callback function
    callback_func = dashboard.app.callback_context.triggered[0]['value']
    
    # Execute callback
    fig, summary_stats, additional_plots = callback_func(
        n_clicks, analysis_type, clinical_factors, start_date, end_date
    )
    
    # Check results
    assert isinstance(fig, dict)  # Plotly figure
    assert isinstance(summary_stats, html.Div)
    assert isinstance(additional_plots, html.Div)
    
    # Check summary stats content
    assert len(summary_stats.children) >= 3
    assert all(isinstance(child, html.P) for child in summary_stats.children)
    
    # Check additional plots
    assert len(additional_plots.children) == 2
    assert all(isinstance(child, dcc.Graph) for child in additional_plots.children)

def test_system_status_callback(dashboard):
    """Test the system status callback."""
    # Simulate callback
    n_clicks = 1
    
    # Get callback function
    callback_func = dashboard.app.callback_context.triggered[0]['value']
    
    # Execute callback
    result = callback_func(n_clicks)
    
    # Check result
    assert isinstance(result, html.Div)
    assert len(result.children) >= 4  # CPU, Memory, Processing Time, Active Processes
    assert all(isinstance(child, html.P) for child in result.children)

def test_export_callback(dashboard):
    """Test the export callback."""
    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate callback with sample inputs
        n_clicks = 1
        figure = {'data': [], 'layout': {}}
        analysis_type = 'patterns'
        
        # Get callback function
        callback_func = dashboard.app.callback_context.triggered[0]['value']
        
        # Execute callback
        result = callback_func(n_clicks, figure, analysis_type)
        
        # Check result
        assert result is None
        
        # Check if export files were created
        export_files = list(Path(temp_dir).glob('*'))
        assert len(export_files) >= 2  # HTML and CSV files

def test_error_handling(dashboard):
    """Test error handling in dashboard callbacks."""
    # Test invalid analysis type
    with pytest.raises(Exception):
        dashboard.app.callback_context.triggered[0]['value']('invalid')
    
    # Test invalid date range
    with pytest.raises(Exception):
        dashboard.app.callback_context.triggered[0]['value'](
            1, 'patterns', [], 'invalid', 'invalid'
        )
    
    # Test invalid clinical factors
    with pytest.raises(Exception):
        dashboard.app.callback_context.triggered[0]['value'](
            1, 'patterns', ['invalid'], datetime.now(), datetime.now()
        )

def test_server_startup(dashboard):
    """Test dashboard server startup."""
    # This is a basic test as we can't actually start the server in tests
    assert hasattr(dashboard, 'run_server')
    assert callable(dashboard.run_server) 