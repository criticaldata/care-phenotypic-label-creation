"""
Tests for export functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json
import pickle
import h5py
from pathlib import Path
import tempfile
import shutil
import plotly.graph_objects as go
from care_phenotype_analyzer.export import DataExporter

@pytest.fixture
def exporter():
    """Create a temporary directory for logs and initialize the exporter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = DataExporter(log_dir=temp_dir)
        yield exporter
        exporter.monitor.stop_monitoring()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample DataFrame
    df = pd.DataFrame({
        'A': np.random.rand(5),
        'B': np.random.rand(5)
    })
    
    # Create sample Series
    series = pd.Series(np.random.rand(5))
    
    # Create sample dictionary
    data_dict = {
        'array1': np.random.rand(3, 3),
        'array2': np.random.rand(2, 2)
    }
    
    # Create sample visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.random.rand(5)))
    
    return df, series, data_dict, fig

def test_exporter_initialization(exporter):
    """Test proper initialization of the exporter."""
    assert exporter.monitor is not None
    assert exporter.monitor.logger is not None

def test_export_to_csv(exporter, sample_data):
    """Test CSV export functionality."""
    df, _, _, _ = sample_data
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        # Test DataFrame export
        exporter.export_to_csv(df, temp_file.name)
        assert Path(temp_file.name).exists()
        
        # Verify content
        loaded_df = pd.read_csv(temp_file.name)
        pd.testing.assert_frame_equal(df, loaded_df)
        
        Path(temp_file.name).unlink()

def test_export_to_json(exporter, sample_data):
    """Test JSON export functionality."""
    df, series, data_dict, _ = sample_data
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        # Test DataFrame export
        exporter.export_to_json(df, temp_file.name)
        assert Path(temp_file.name).exists()
        
        # Verify content
        with open(temp_file.name, 'r') as f:
            loaded_data = json.load(f)
        assert len(loaded_data) == len(df)
        
        Path(temp_file.name).unlink()

def test_export_to_hdf5(exporter, sample_data):
    """Test HDF5 export functionality."""
    _, _, data_dict, _ = sample_data
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        exporter.export_to_hdf5(data_dict, temp_file.name)
        assert Path(temp_file.name).exists()
        
        # Verify content
        with h5py.File(temp_file.name, 'r') as f:
            assert 'array1' in f
            assert 'array2' in f
            np.testing.assert_array_equal(data_dict['array1'], f['array1'][:])
            np.testing.assert_array_equal(data_dict['array2'], f['array2'][:])
        
        Path(temp_file.name).unlink()

def test_export_to_pickle(exporter, sample_data):
    """Test pickle export functionality."""
    df, _, _, _ = sample_data
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
        exporter.export_to_pickle(df, temp_file.name)
        assert Path(temp_file.name).exists()
        
        # Verify content
        with open(temp_file.name, 'rb') as f:
            loaded_data = pickle.load(f)
        pd.testing.assert_frame_equal(df, loaded_data)
        
        Path(temp_file.name).unlink()

def test_export_visualization(exporter, sample_data):
    """Test visualization export functionality."""
    _, _, _, fig = sample_data
    
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        # Test HTML export
        exporter.export_visualization(fig, temp_file.name, format='html')
        assert Path(temp_file.name).exists()
        Path(temp_file.name).unlink()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        # Test PNG export
        exporter.export_visualization(fig, temp_file.name, format='png')
        assert Path(temp_file.name).exists()
        Path(temp_file.name).unlink()

def test_export_batch(exporter, sample_data):
    """Test batch export functionality."""
    df, series, data_dict, fig = sample_data
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare data dictionary
        batch_data = {
            'dataframe': df,
            'series': series,
            'arrays': data_dict,
            'visualization': fig
        }
        
        # Test batch export
        exporter.export_batch(batch_data, temp_dir)
        
        # Verify files were created
        files = list(Path(temp_dir).glob('*'))
        assert len(files) > 0
        
        # Verify each data type was exported
        file_extensions = {f.suffix for f in files}
        assert '.csv' in file_extensions
        assert '.json' in file_extensions
        assert '.h5' in file_extensions
        assert '.pkl' in file_extensions

def test_error_handling(exporter):
    """Test error handling in export methods."""
    # Test invalid data types
    with pytest.raises(Exception):
        exporter.export_to_csv(None, "test.csv")
    
    with pytest.raises(Exception):
        exporter.export_to_hdf5(None, "test.h5")
    
    # Test invalid file paths
    with pytest.raises(Exception):
        exporter.export_to_json({}, "/invalid/path/test.json")
    
    # Test invalid visualization format
    fig = go.Figure()
    with pytest.raises(ValueError):
        exporter.export_visualization(fig, "test.xyz", format='xyz') 