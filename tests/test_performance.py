"""
Performance tests for the care phenotype analyzer.

This module contains tests to validate the performance characteristics
of the system, including large dataset handling, memory usage, and
processing speed optimization.
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import time
import gc
from datetime import datetime, timedelta
from memory_profiler import profile
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator
from care_phenotype_analyzer.pattern_analyzer import CarePatternAnalyzer
from care_phenotype_analyzer.fairness_evaluator import FairnessEvaluator

@pytest.fixture
def large_synthetic_data():
    """Generate large synthetic dataset for performance testing."""
    generator = SyntheticDataGenerator(n_patients=10000, seed=42)
    return generator.generate_all()

@pytest.fixture
def large_phenotype_creator(large_synthetic_data):
    """Create a CarePhenotypeCreator instance with large synthetic data."""
    # Create sample care pattern data
    care_patterns = pd.DataFrame({
        'subject_id': large_synthetic_data['patients']['subject_id'],
        'timestamp': [datetime.now() + timedelta(hours=i) for i in range(len(large_synthetic_data['patients']))],
        'pattern_1': np.random.normal(0, 1, len(large_synthetic_data['patients'])),
        'pattern_2': np.random.normal(0, 1, len(large_synthetic_data['patients'])),
        'clinical_factor_1': np.random.normal(0, 1, len(large_synthetic_data['patients'])),
        'clinical_factor_2': np.random.normal(0, 1, len(large_synthetic_data['patients']))
    })
    
    return CarePhenotypeCreator(
        data=care_patterns,
        clinical_factors=['clinical_factor_1', 'clinical_factor_2']
    )

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def test_large_dataset_handling(large_phenotype_creator):
    """Test handling of large datasets."""
    # Record initial memory usage
    initial_memory = get_memory_usage()
    
    # Create phenotype labels
    start_time = time.time()
    labels = large_phenotype_creator.create_phenotype_labels()
    end_time = time.time()
    
    # Record final memory usage
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Validate results
    assert len(labels) == len(large_phenotype_creator.data), \
        "Number of labels should match number of subjects"
    assert not labels.isnull().any(), "Labels should not contain null values"
    
    # Log performance metrics
    print(f"\nLarge Dataset Handling Metrics:")
    print(f"Number of subjects: {len(labels)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    # Performance assertions
    assert end_time - start_time < 30, "Processing time should be under 30 seconds"
    assert memory_increase < 1000, "Memory increase should be under 1GB"

def test_memory_usage_optimization(large_phenotype_creator):
    """Test memory usage optimization during processing."""
    # Record initial memory usage
    initial_memory = get_memory_usage()
    
    # Process data in chunks
    chunk_size = 1000
    total_subjects = len(large_phenotype_creator.data)
    memory_usage_history = []
    
    for start_idx in range(0, total_subjects, chunk_size):
        # Record memory before chunk processing
        pre_chunk_memory = get_memory_usage()
        
        # Process chunk
        chunk_data = large_phenotype_creator.data.iloc[start_idx:start_idx + chunk_size]
        chunk_labels = large_phenotype_creator.create_phenotype_labels()
        
        # Record memory after chunk processing
        post_chunk_memory = get_memory_usage()
        memory_usage_history.append(post_chunk_memory - pre_chunk_memory)
        
        # Force garbage collection
        gc.collect()
    
    # Calculate memory metrics
    avg_memory_increase = np.mean(memory_usage_history)
    max_memory_increase = np.max(memory_usage_history)
    
    # Log memory metrics
    print(f"\nMemory Usage Optimization Metrics:")
    print(f"Average memory increase per chunk: {avg_memory_increase:.2f} MB")
    print(f"Maximum memory increase: {max_memory_increase:.2f} MB")
    
    # Performance assertions
    assert avg_memory_increase < 100, "Average memory increase should be under 100MB per chunk"
    assert max_memory_increase < 200, "Maximum memory increase should be under 200MB"

def test_processing_speed_optimization(large_phenotype_creator):
    """Test processing speed optimization."""
    # Test different batch sizes
    batch_sizes = [100, 500, 1000, 2000]
    processing_times = []
    
    for batch_size in batch_sizes:
        # Record start time
        start_time = time.time()
        
        # Process data in batches
        total_subjects = len(large_phenotype_creator.data)
        for start_idx in range(0, total_subjects, batch_size):
            batch_data = large_phenotype_creator.data.iloc[start_idx:start_idx + batch_size]
            batch_labels = large_phenotype_creator.create_phenotype_labels()
        
        # Record end time
        end_time = time.time()
        processing_times.append((batch_size, end_time - start_time))
    
    # Log processing times
    print(f"\nProcessing Speed Optimization Metrics:")
    for batch_size, processing_time in processing_times:
        print(f"Batch size {batch_size}: {processing_time:.2f} seconds")
    
    # Performance assertions
    assert all(time < 60 for _, time in processing_times), \
        "All batch processing times should be under 60 seconds"
    
    # Verify that larger batch sizes are generally faster
    for i in range(len(processing_times) - 1):
        assert processing_times[i][1] >= processing_times[i + 1][1], \
            f"Larger batch size {processing_times[i+1][0]} should not be slower than {processing_times[i][0]}"

def test_concurrent_processing(large_phenotype_creator):
    """Test concurrent processing capabilities."""
    import concurrent.futures
    
    # Split data into chunks
    chunk_size = 1000
    total_subjects = len(large_phenotype_creator.data)
    chunks = []
    
    for start_idx in range(0, total_subjects, chunk_size):
        chunk_data = large_phenotype_creator.data.iloc[start_idx:start_idx + chunk_size]
        chunks.append(chunk_data)
    
    # Process chunks concurrently
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for chunk in chunks:
            future = executor.submit(large_phenotype_creator.create_phenotype_labels)
            futures.append(future)
        
        # Wait for all chunks to complete
        concurrent.futures.wait(futures)
    
    end_time = time.time()
    concurrent_time = end_time - start_time
    
    # Process chunks sequentially for comparison
    start_time = time.time()
    for chunk in chunks:
        large_phenotype_creator.create_phenotype_labels()
    end_time = time.time()
    sequential_time = end_time - start_time
    
    # Log concurrent processing metrics
    print(f"\nConcurrent Processing Metrics:")
    print(f"Sequential processing time: {sequential_time:.2f} seconds")
    print(f"Concurrent processing time: {concurrent_time:.2f} seconds")
    print(f"Speedup factor: {sequential_time / concurrent_time:.2f}x")
    
    # Performance assertions
    assert concurrent_time < sequential_time, "Concurrent processing should be faster than sequential"
    assert concurrent_time < 30, "Concurrent processing should complete under 30 seconds"

def test_memory_cleanup(large_phenotype_creator):
    """Test memory cleanup after processing."""
    # Record initial memory
    initial_memory = get_memory_usage()
    
    # Process data
    labels = large_phenotype_creator.create_phenotype_labels()
    
    # Record memory after processing
    post_processing_memory = get_memory_usage()
    
    # Force garbage collection
    gc.collect()
    
    # Record memory after cleanup
    post_cleanup_memory = get_memory_usage()
    
    # Log memory cleanup metrics
    print(f"\nMemory Cleanup Metrics:")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Post-processing memory: {post_processing_memory:.2f} MB")
    print(f"Post-cleanup memory: {post_cleanup_memory:.2f} MB")
    print(f"Memory cleanup: {post_processing_memory - post_cleanup_memory:.2f} MB")
    
    # Performance assertions
    assert post_cleanup_memory - initial_memory < 100, \
        "Memory usage after cleanup should be close to initial"
    assert post_processing_memory > post_cleanup_memory, \
        "Garbage collection should reduce memory usage" 