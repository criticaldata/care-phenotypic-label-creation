"""
Tests for memory optimization functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from care_phenotype_analyzer.memory import MemoryOptimizer

@pytest.fixture
def optimizer():
    """Create a temporary directory for logs and initialize the optimizer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = MemoryOptimizer(log_dir=temp_dir)
        yield optimizer
        optimizer.monitor.stop_monitoring()

def test_optimizer_initialization(optimizer):
    """Test proper initialization of the optimizer."""
    assert optimizer.monitor is not None
    assert optimizer.warning_threshold == 0.8
    assert optimizer.critical_threshold == 0.9

def test_get_memory_usage(optimizer):
    """Test memory usage statistics retrieval."""
    stats = optimizer.get_memory_usage()
    
    assert isinstance(stats, dict)
    assert 'process_rss' in stats
    assert 'process_vms' in stats
    assert 'system_total' in stats
    assert 'system_available' in stats
    assert 'system_percent' in stats
    
    # Verify values are reasonable
    assert stats['process_rss'] > 0
    assert stats['process_vms'] > 0
    assert stats['system_total'] > 0
    assert stats['system_available'] > 0
    assert 0 <= stats['system_percent'] <= 100

def test_optimize_dataframe(optimizer):
    """Test DataFrame memory optimization."""
    # Create test DataFrame with inefficient types
    df = pd.DataFrame({
        'int_col': np.random.randint(-100, 100, 1000, dtype=np.int64),
        'float_col': np.random.rand(1000).astype(np.float64),
        'small_int': np.random.randint(0, 255, 1000, dtype=np.int64),
        'large_int': np.random.randint(0, 65535, 1000, dtype=np.int64)
    })
    
    # Get initial memory usage
    initial_memory = df.memory_usage(deep=True).sum()
    
    # Optimize DataFrame
    optimized_df = optimizer.optimize_dataframe(df)
    
    # Verify optimization
    final_memory = optimized_df.memory_usage(deep=True).sum()
    assert final_memory < initial_memory
    
    # Verify data types
    assert optimized_df['int_col'].dtype in [np.int8, np.int16, np.int32]
    assert optimized_df['float_col'].dtype == np.float32
    assert optimized_df['small_int'].dtype == np.uint8
    assert optimized_df['large_int'].dtype == np.uint16
    
    # Verify data integrity
    pd.testing.assert_frame_equal(df, optimized_df, check_dtype=False)

def test_optimize_array(optimizer):
    """Test array memory optimization."""
    # Create test arrays with inefficient types
    int_arr = np.random.randint(-100, 100, 1000, dtype=np.int64)
    float_arr = np.random.rand(1000).astype(np.float64)
    small_int_arr = np.random.randint(0, 255, 1000, dtype=np.int64)
    
    # Get initial memory usage
    initial_memory = int_arr.nbytes + float_arr.nbytes + small_int_arr.nbytes
    
    # Optimize arrays
    optimized_int = optimizer.optimize_array(int_arr)
    optimized_float = optimizer.optimize_array(float_arr)
    optimized_small = optimizer.optimize_array(small_int_arr)
    
    # Verify optimization
    final_memory = optimized_int.nbytes + optimized_float.nbytes + optimized_small.nbytes
    assert final_memory < initial_memory
    
    # Verify data types
    assert optimized_int.dtype in [np.int8, np.int16, np.int32]
    assert optimized_float.dtype == np.float32
    assert optimized_small.dtype == np.uint8
    
    # Verify data integrity
    np.testing.assert_array_equal(int_arr, optimized_int)
    np.testing.assert_array_equal(float_arr, optimized_float)
    np.testing.assert_array_equal(small_int_arr, optimized_small)

def test_clear_memory(optimizer):
    """Test memory clearing functionality."""
    # Create some large objects to consume memory
    large_arrays = [np.random.rand(1000, 1000) for _ in range(5)]
    
    # Get initial memory usage
    initial_stats = optimizer.get_memory_usage()
    
    # Clear memory
    optimizer.clear_memory()
    
    # Get final memory usage
    final_stats = optimizer.get_memory_usage()
    
    # Verify memory reduction
    assert final_stats['process_rss'] < initial_stats['process_rss']

def test_check_memory_pressure(optimizer):
    """Test memory pressure checking."""
    pressure = optimizer.check_memory_pressure()
    
    assert isinstance(pressure, dict)
    assert 'status' in pressure
    assert 'system_percent' in pressure
    assert 'process_rss' in pressure
    assert 'process_vms' in pressure
    
    # Verify status is one of the expected values
    assert pressure['status'] in ['normal', 'warning', 'critical', 'unknown', 'error']
    
    # Verify percentage is reasonable
    assert 0 <= pressure['system_percent'] <= 100

def test_optimize_batch(optimizer):
    """Test batch processing with memory optimization."""
    # Create test items
    items = [np.random.rand(100, 100) for _ in range(100)]
    
    # Define optimization function
    def optimize_func(arr):
        return optimizer.optimize_array(arr)
    
    # Process items in batches
    results = optimizer.optimize_batch(items, optimize_func, batch_size=10)
    
    # Verify results
    assert len(results) == len(items)
    
    # Verify each result is optimized
    for result in results:
        assert result.dtype == np.float32

def test_error_handling(optimizer):
    """Test error handling in memory optimization."""
    # Test with invalid DataFrame
    with pytest.raises(Exception):
        optimizer.optimize_dataframe(None)
    
    # Test with invalid array
    with pytest.raises(Exception):
        optimizer.optimize_array(None)
    
    # Test with invalid batch processing
    with pytest.raises(Exception):
        optimizer.optimize_batch(None, lambda x: x)

def test_memory_thresholds(optimizer):
    """Test memory threshold handling."""
    # Temporarily modify thresholds for testing
    optimizer.warning_threshold = 0.1
    optimizer.critical_threshold = 0.2
    
    # Create large objects to consume memory
    large_arrays = [np.random.rand(1000, 1000) for _ in range(10)]
    
    # Check memory pressure
    pressure = optimizer.check_memory_pressure()
    
    # Verify pressure status is appropriate
    assert pressure['status'] in ['normal', 'warning', 'critical']

def test_empty_input_handling(optimizer):
    """Test handling of empty inputs."""
    # Test empty DataFrame
    df = pd.DataFrame()
    result = optimizer.optimize_dataframe(df)
    assert len(result) == 0
    
    # Test empty array
    arr = np.array([])
    result = optimizer.optimize_array(arr)
    assert len(result) == 0
    
    # Test empty list
    items = []
    result = optimizer.optimize_batch(items, lambda x: x)
    assert len(result) == 0 