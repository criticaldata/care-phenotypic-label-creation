"""
Tests for advanced visualization functionality.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import shutil
from care_phenotype_analyzer.visualization import AdvancedVisualizer

@pytest.fixture
def visualizer():
    """Create a temporary directory for logs and initialize the visualizer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = AdvancedVisualizer(log_dir=temp_dir)
        yield visualizer
        visualizer.monitor.stop_monitoring()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample care pattern data
    n_samples = 100
    n_features = 10
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create sample phenotype labels
    phenotype_labels = pd.Series(np.random.randint(0, 3, n_samples))
    
    # Create sample clinical factors
    clinical_factors = pd.DataFrame(
        np.random.randn(n_samples, 3),
        columns=['SOFA', 'Charlson', 'APACHE']
    )
    
    return data, phenotype_labels, clinical_factors

@pytest.fixture
def sample_fairness_metrics():
    """Create sample fairness metrics for testing."""
    return {
        'demographic_parity': {
            'age': 0.1,
            'gender': 0.15,
            'race': 0.2
        },
        'equal_opportunity': {
            'age': 0.12,
            'gender': 0.18,
            'race': 0.22
        },
        'predictive_parity': {
            'age': 0.08,
            'gender': 0.14,
            'race': 0.16
        }
    }

@pytest.fixture
def sample_bias_mitigation_metrics():
    """Create sample bias mitigation metrics for testing."""
    original_metrics = {
        'demographic_parity': 0.2,
        'equal_opportunity': 0.25,
        'predictive_parity': 0.18
    }
    
    mitigated_metrics = {
        'reweighting': {
            'demographic_parity': 0.15,
            'equal_opportunity': 0.18,
            'predictive_parity': 0.12
        },
        'threshold_adjustment': {
            'demographic_parity': 0.16,
            'equal_opportunity': 0.19,
            'predictive_parity': 0.13
        },
        'calibration': {
            'demographic_parity': 0.14,
            'equal_opportunity': 0.17,
            'predictive_parity': 0.11
        }
    }
    
    return original_metrics, mitigated_metrics

@pytest.fixture
def sample_system_metrics():
    """Create sample system metrics for testing."""
    metrics = {
        'cpu_usage': np.random.rand(10).tolist(),
        'memory_usage': np.random.rand(10).tolist(),
        'processing_time': np.random.rand(10).tolist()
    }
    
    health_status = {
        'system_status': 'healthy',
        'active_threads': 5,
        'queue_size': 10
    }
    
    return metrics, health_status

def test_visualizer_initialization(visualizer):
    """Test proper initialization of the visualizer."""
    assert visualizer.monitor is not None
    assert visualizer.monitor.logger is not None

def test_create_interactive_pattern_plot(visualizer, sample_data):
    """Test creation of interactive pattern plot."""
    data, phenotype_labels, clinical_factors = sample_data
    
    # Test without clinical factors
    fig = visualizer.create_interactive_pattern_plot(
        data=data,
        phenotype_labels=phenotype_labels
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(phenotype_labels.unique())
    
    # Test with clinical factors
    fig = visualizer.create_interactive_pattern_plot(
        data=data,
        phenotype_labels=phenotype_labels,
        clinical_factors=clinical_factors
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(phenotype_labels.unique()) + len(clinical_factors.columns)
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        fig = visualizer.create_interactive_pattern_plot(
            data=data,
            phenotype_labels=phenotype_labels,
            output_file=temp_file.name
        )
        assert Path(temp_file.name).exists()
        Path(temp_file.name).unlink()

def test_create_fairness_heatmap(visualizer, sample_fairness_metrics):
    """Test creation of fairness heatmap."""
    fig = visualizer.create_fairness_heatmap(sample_fairness_metrics)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # One heatmap trace
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        fig = visualizer.create_fairness_heatmap(
            sample_fairness_metrics,
            output_file=temp_file.name
        )
        assert Path(temp_file.name).exists()
        Path(temp_file.name).unlink()

def test_create_bias_mitigation_comparison(visualizer, sample_bias_mitigation_metrics):
    """Test creation of bias mitigation comparison."""
    original_metrics, mitigated_metrics = sample_bias_mitigation_metrics
    
    fig = visualizer.create_bias_mitigation_comparison(
        original_metrics,
        mitigated_metrics
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 + len(mitigated_metrics)  # Original + each strategy
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        fig = visualizer.create_bias_mitigation_comparison(
            original_metrics,
            mitigated_metrics,
            output_file=temp_file.name
        )
        assert Path(temp_file.name).exists()
        Path(temp_file.name).unlink()

def test_create_system_health_dashboard(visualizer, sample_system_metrics):
    """Test creation of system health dashboard."""
    metrics, health_status = sample_system_metrics
    
    fig = visualizer.create_system_health_dashboard(metrics, health_status)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(metrics)
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        fig = visualizer.create_system_health_dashboard(
            metrics,
            health_status,
            output_file=temp_file.name
        )
        assert Path(temp_file.name).exists()
        Path(temp_file.name).unlink()

def test_error_handling(visualizer):
    """Test error handling in visualization methods."""
    # Test with invalid data
    with pytest.raises(Exception):
        visualizer.create_interactive_pattern_plot(
            data=pd.DataFrame(),
            phenotype_labels=pd.Series()
        )
    
    # Test with invalid fairness metrics
    with pytest.raises(Exception):
        visualizer.create_fairness_heatmap({})
    
    # Test with invalid bias mitigation metrics
    with pytest.raises(Exception):
        visualizer.create_bias_mitigation_comparison({}, {})
    
    # Test with invalid system metrics
    with pytest.raises(Exception):
        visualizer.create_system_health_dashboard({}, {}) 