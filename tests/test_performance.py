"""
Module for performance and stress testing.

This module provides comprehensive tests for evaluating the performance
and stability of the system under various conditions.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import gc

from care_phenotype_analyzer.memory import MemoryOptimizer
from care_phenotype_analyzer.parallel import ParallelProcessor
from care_phenotype_analyzer.monitoring import SystemMonitor

@pytest.fixture
def large_dataset():
    """Create a large test dataset."""
    # Create a large DataFrame
    n_rows = 1_000_000
    n_cols = 100
    
    df = pd.DataFrame({
        f'col_{i}': np.random.randn(n_rows) for i in range(n_cols)
    })
    
    # Add some categorical columns
    df['category'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
    df['status'] = np.random.choice(['active', 'inactive', 'pending'], n_rows)
    
    return df

@pytest.fixture
def performance_monitor():
    """Create a temporary directory for monitoring logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = SystemMonitor(log_dir=temp_dir)
        yield monitor
        monitor.stop_monitoring()

def test_memory_optimization_performance(large_dataset, performance_monitor):
    """Test memory optimization performance with large datasets."""
    optimizer = MemoryOptimizer(log_dir=performance_monitor.log_dir)
    
    # Measure initial memory usage
    initial_memory = large_dataset.memory_usage(deep=True).sum()
    
    # Time the optimization
    start_time = time.time()
    optimized_df = optimizer.optimize_dataframe(large_dataset)
    optimization_time = time.time() - start_time
    
    # Measure final memory usage
    final_memory = optimized_df.memory_usage(deep=True).sum()
    
    # Verify performance metrics
    assert optimization_time < 10.0  # Should complete within 10 seconds
    assert final_memory < initial_memory  # Should reduce memory usage
    
    # Log performance metrics
    performance_monitor.logger.info(
        f"Memory optimization performance: "
        f"time={optimization_time:.2f}s, "
        f"memory_reduction={((initial_memory - final_memory) / initial_memory * 100):.1f}%"
    )

def test_parallel_processing_performance(large_dataset, performance_monitor):
    """Test parallel processing performance."""
    processor = ParallelProcessor(log_dir=performance_monitor.log_dir)
    
    # Create a computationally intensive function
    def process_chunk(chunk):
        return chunk.apply(lambda x: np.exp(x) + np.sin(x))
    
    # Time parallel processing
    start_time = time.time()
    result = processor.process_dataframe_parallel(
        large_dataset,
        process_chunk,
        chunk_size=10000
    )
    processing_time = time.time() - start_time
    
    # Verify performance
    assert processing_time < 30.0  # Should complete within 30 seconds
    assert len(result) == len(large_dataset)
    
    # Log performance metrics
    performance_monitor.logger.info(
        f"Parallel processing performance: "
        f"time={processing_time:.2f}s, "
        f"rows_processed={len(result)}"
    )

def test_stress_memory_usage(performance_monitor):
    """Test system stability under memory stress."""
    optimizer = MemoryOptimizer(log_dir=performance_monitor.log_dir)
    
    # Create a list of large arrays
    large_arrays = []
    for _ in range(100):
        arr = np.random.rand(1000, 1000)
        large_arrays.append(arr)
    
    # Monitor memory pressure
    pressure_history = []
    for _ in range(10):
        pressure = optimizer.check_memory_pressure()
        pressure_history.append(pressure)
        
        # Clear memory periodically
        if pressure['status'] == 'warning':
            optimizer.clear_memory()
    
    # Verify system stability
    assert all(p['status'] != 'critical' for p in pressure_history)
    
    # Log stress test results
    performance_monitor.logger.info(
        f"Memory stress test completed: "
        f"max_pressure={max(p['system_percent'] for p in pressure_history):.1f}%"
    )

def test_stress_parallel_processing(performance_monitor):
    """Test system stability under parallel processing stress."""
    processor = ParallelProcessor(log_dir=performance_monitor.log_dir)
    
    # Create multiple large datasets
    datasets = []
    for _ in range(5):
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(100000) for i in range(50)
        })
        datasets.append(df)
    
    # Process datasets in parallel
    def process_dataset(df):
        return df.apply(lambda x: np.exp(x) + np.sin(x))
    
    start_time = time.time()
    results = []
    
    for df in datasets:
        result = processor.process_dataframe_parallel(
            df,
            process_dataset,
            chunk_size=1000
        )
        results.append(result)
    
    processing_time = time.time() - start_time
    
    # Verify results
    assert all(len(r) == len(d) for r, d in zip(results, datasets))
    assert processing_time < 60.0  # Should complete within 60 seconds
    
    # Log stress test results
    performance_monitor.logger.info(
        f"Parallel processing stress test completed: "
        f"time={processing_time:.2f}s, "
        f"datasets_processed={len(datasets)}"
    )

def test_stress_memory_optimization(performance_monitor):
    """Test system stability under memory optimization stress."""
    optimizer = MemoryOptimizer(log_dir=performance_monitor.log_dir)
    
    # Create a sequence of large datasets
    datasets = []
    for _ in range(10):
        df = pd.DataFrame({
            f'col_{i}': np.random.randint(-1000, 1000, 100000, dtype=np.int64)
            for i in range(20)
        })
        datasets.append(df)
    
    # Optimize datasets sequentially
    start_time = time.time()
    optimized_datasets = []
    
    for df in datasets:
        optimized_df = optimizer.optimize_dataframe(df)
        optimized_datasets.append(optimized_df)
        
        # Check memory pressure
        pressure = optimizer.check_memory_pressure()
        if pressure['status'] == 'warning':
            optimizer.clear_memory()
    
    processing_time = time.time() - start_time
    
    # Verify results
    assert all(len(o) == len(d) for o, d in zip(optimized_datasets, datasets))
    assert processing_time < 30.0  # Should complete within 30 seconds
    
    # Log stress test results
    performance_monitor.logger.info(
        f"Memory optimization stress test completed: "
        f"time={processing_time:.2f}s, "
        f"datasets_optimized={len(datasets)}"
    )

def test_concurrent_operations(performance_monitor):
    """Test system stability under concurrent operations."""
    optimizer = MemoryOptimizer(log_dir=performance_monitor.log_dir)
    processor = ParallelProcessor(log_dir=performance_monitor.log_dir)
    
    # Create test data
    df = pd.DataFrame({
        f'col_{i}': np.random.randn(100000) for i in range(20)
    })
    
    # Define concurrent operations
    def optimize_data():
        return optimizer.optimize_dataframe(df)
    
    def process_data():
        return processor.process_dataframe_parallel(
            df,
            lambda x: np.exp(x) + np.sin(x),
            chunk_size=1000
        )
    
    # Run operations concurrently
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(optimize_data)
        future2 = executor.submit(process_data)
        
        results = [f.result() for f in [future1, future2]]
    
    processing_time = time.time() - start_time
    
    # Verify results
    assert len(results[0]) == len(df)
    assert len(results[1]) == len(df)
    assert processing_time < 20.0  # Should complete within 20 seconds
    
    # Log concurrent operations results
    performance_monitor.logger.info(
        f"Concurrent operations test completed: "
        f"time={processing_time:.2f}s"
    )

def test_long_running_operations(performance_monitor):
    """Test system stability during long-running operations."""
    optimizer = MemoryOptimizer(log_dir=performance_monitor.log_dir)
    
    # Create a large dataset
    df = pd.DataFrame({
        f'col_{i}': np.random.randn(1000000) for i in range(50)
    })
    
    # Perform multiple operations over time
    start_time = time.time()
    operation_times = []
    
    for _ in range(5):
        op_start = time.time()
        
        # Optimize data
        optimized_df = optimizer.optimize_dataframe(df)
        
        # Process data
        result = optimized_df.apply(lambda x: np.exp(x) + np.sin(x))
        
        # Clear memory
        optimizer.clear_memory()
        
        op_time = time.time() - op_start
        operation_times.append(op_time)
        
        # Check memory pressure
        pressure = optimizer.check_memory_pressure()
        if pressure['status'] == 'warning':
            optimizer.clear_memory()
    
    total_time = time.time() - start_time
    
    # Verify stability
    assert total_time < 120.0  # Should complete within 120 seconds
    assert all(t < 30.0 for t in operation_times)  # Each operation should be fast
    
    # Log long-running test results
    performance_monitor.logger.info(
        f"Long-running operations test completed: "
        f"total_time={total_time:.2f}s, "
        f"avg_operation_time={sum(operation_times)/len(operation_times):.2f}s"
    ) 