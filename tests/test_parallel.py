"""
Tests for parallel processing functionality.
"""

import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from care_phenotype_analyzer.parallel import ParallelProcessor

@pytest.fixture
def processor():
    """Create a temporary directory for logs and initialize the processor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = ParallelProcessor(max_workers=2, log_dir=temp_dir)
        yield processor
        processor.monitor.stop_monitoring()

def test_processor_initialization(processor):
    """Test proper initialization of the processor."""
    assert processor.monitor is not None
    assert processor.max_workers == 2
    assert processor.process_pool is not None
    assert processor.thread_pool is not None

def test_process_dataframe_parallel(processor):
    """Test parallel DataFrame processing."""
    # Create test DataFrame
    df = pd.DataFrame({
        'A': np.random.rand(1000),
        'B': np.random.rand(1000)
    })
    
    # Define processing function
    def process_chunk(chunk):
        chunk['C'] = chunk['A'] + chunk['B']
        return chunk
    
    # Process DataFrame in parallel
    result = processor.process_dataframe_parallel(
        df, process_chunk, chunk_size=100, use_processes=True
    )
    
    # Verify results
    assert len(result) == len(df)
    assert 'C' in result.columns
    pd.testing.assert_series_equal(
        result['C'],
        df['A'] + df['B']
    )

def test_process_array_parallel(processor):
    """Test parallel array processing."""
    # Create test array
    arr = np.random.rand(1000)
    
    # Define processing function
    def process_chunk(chunk):
        return chunk * 2
    
    # Process array in parallel
    result = processor.process_array_parallel(
        arr, process_chunk, chunk_size=100, use_processes=True
    )
    
    # Verify results
    assert len(result) == len(arr)
    np.testing.assert_array_equal(result, arr * 2)

def test_process_list_parallel(processor):
    """Test parallel list processing."""
    # Create test list
    items = list(range(1000))
    
    # Define processing function
    def process_chunk(chunk):
        return [x * 2 for x in chunk]
    
    # Process list in parallel
    result = processor.process_list_parallel(
        items, process_chunk, chunk_size=100, use_processes=True
    )
    
    # Verify results
    assert len(result) == len(items)
    assert result == [x * 2 for x in items]

def test_process_batch_parallel(processor):
    """Test parallel batch processing."""
    # Create test items
    items = list(range(1000))
    
    # Define processing function
    def process_item(item):
        return item * 2
    
    # Process items in parallel batches
    result = processor.process_batch_parallel(
        items, process_item, batch_size=100, use_processes=True
    )
    
    # Verify results
    assert len(result) == len(items)
    assert result == [x * 2 for x in items]

def test_thread_vs_process_pool(processor):
    """Test thread pool vs process pool performance."""
    # Create large DataFrame
    df = pd.DataFrame({
        'A': np.random.rand(10000),
        'B': np.random.rand(10000)
    })
    
    # Define CPU-intensive function
    def process_chunk(chunk):
        for _ in range(1000):
            chunk['C'] = chunk['A'] * chunk['B']
        return chunk
    
    # Test process pool
    start_time = time.time()
    result_process = processor.process_dataframe_parallel(
        df, process_chunk, chunk_size=1000, use_processes=True
    )
    process_time = time.time() - start_time
    
    # Test thread pool
    start_time = time.time()
    result_thread = processor.process_dataframe_parallel(
        df, process_chunk, chunk_size=1000, use_processes=False
    )
    thread_time = time.time() - start_time
    
    # Verify results
    pd.testing.assert_frame_equal(result_process, result_thread)
    
    # Process pool should be faster for CPU-intensive tasks
    assert process_time < thread_time

def test_worker_stats(processor):
    """Test worker statistics."""
    stats = processor.get_worker_stats()
    
    assert stats['max_workers'] == 2
    assert stats['active_processes'] >= 0
    assert stats['thread_count'] >= 0

def test_error_handling(processor):
    """Test error handling in parallel processing."""
    # Test with invalid function
    def invalid_func(x):
        raise ValueError("Test error")
    
    # Test DataFrame processing
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(Exception):
        processor.process_dataframe_parallel(df, invalid_func)
    
    # Test array processing
    arr = np.array([1, 2, 3])
    with pytest.raises(Exception):
        processor.process_array_parallel(arr, invalid_func)
    
    # Test list processing
    items = [1, 2, 3]
    with pytest.raises(Exception):
        processor.process_list_parallel(items, invalid_func)

def test_chunk_size_handling(processor):
    """Test handling of different chunk sizes."""
    # Create test DataFrame
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100)
    })
    
    # Define processing function
    def process_chunk(chunk):
        chunk['C'] = chunk['A'] + chunk['B']
        return chunk
    
    # Test with different chunk sizes
    chunk_sizes = [10, 25, 50, 100]
    results = []
    
    for chunk_size in chunk_sizes:
        result = processor.process_dataframe_parallel(
            df, process_chunk, chunk_size=chunk_size
        )
        results.append(result)
    
    # Verify all results are identical
    for i in range(1, len(results)):
        pd.testing.assert_frame_equal(results[0], results[i])

def test_empty_input_handling(processor):
    """Test handling of empty inputs."""
    # Test empty DataFrame
    df = pd.DataFrame()
    result = processor.process_dataframe_parallel(df, lambda x: x)
    assert len(result) == 0
    
    # Test empty array
    arr = np.array([])
    result = processor.process_array_parallel(arr, lambda x: x)
    assert len(result) == 0
    
    # Test empty list
    items = []
    result = processor.process_list_parallel(items, lambda x: x)
    assert len(result) == 0 