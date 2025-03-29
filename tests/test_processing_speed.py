"""
Tests for processing speed optimization in the care phenotype analyzer.

This module contains tests to validate and optimize processing speed during
data processing, including parallel processing, batch processing, and
algorithm efficiency.
"""

import pytest
import pandas as pd
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator
from care_phenotype_analyzer.pattern_analyzer import CarePatternAnalyzer
from care_phenotype_analyzer.fairness_evaluator import FairnessEvaluator

def measure_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@pytest.fixture
def large_dataset():
    """Generate large dataset for processing speed testing."""
    generator = SyntheticDataGenerator(n_patients=10000, seed=42)
    return generator.generate_all()

@pytest.fixture
def phenotype_creator(large_dataset):
    """Create a CarePhenotypeCreator instance with large dataset."""
    # Create sample care pattern data
    care_patterns = pd.DataFrame({
        'subject_id': large_dataset['patients']['subject_id'],
        'timestamp': [datetime.now() + timedelta(hours=i) for i in range(len(large_dataset['patients']))],
        'pattern_1': np.random.normal(0, 1, len(large_dataset['patients'])),
        'pattern_2': np.random.normal(0, 1, len(large_dataset['patients'])),
        'clinical_factor_1': np.random.normal(0, 1, len(large_dataset['patients'])),
        'clinical_factor_2': np.random.normal(0, 1, len(large_dataset['patients']))
    })
    
    return CarePhenotypeCreator(
        data=care_patterns,
        clinical_factors=['clinical_factor_1', 'clinical_factor_2']
    )

def test_processing_speed_scaling(phenotype_creator):
    """Test how processing speed scales with dataset size."""
    # Test with different dataset sizes
    sizes = [1000, 5000, 10000]
    results = []
    
    for size in sizes:
        # Create subset of data
        subset_data = phenotype_creator.data.iloc[:size].copy()
        subset_creator = CarePhenotypeCreator(
            data=subset_data,
            clinical_factors=phenotype_creator.clinical_factors
        )
        
        # Measure processing time
        start_time = time.time()
        subset_creator.create_phenotype_labels()
        end_time = time.time()
        
        processing_time = end_time - start_time
        results.append({
            'size': size,
            'processing_time': processing_time,
            'time_per_record': processing_time / size
        })
    
    # Log scaling metrics
    print("\nProcessing Speed Scaling Metrics:")
    for result in results:
        print(f"\nDataset size: {result['size']}")
        print(f"Total processing time: {result['processing_time']:.2f} seconds")
        print(f"Time per record: {result['time_per_record']*1000:.2f} ms")
    
    # Scaling assertions
    assert all(result['time_per_record'] < 0.1 for result in results), \
        "Processing time per record should be under 100ms"
    assert results[-1]['time_per_record'] < results[0]['time_per_record'], \
        "Processing should show economies of scale"

def test_parallel_processing(phenotype_creator):
    """Test parallel processing capabilities."""
    # Define batch size and number of workers
    batch_size = 1000
    n_workers = 4
    total_subjects = len(phenotype_creator.data)
    
    def process_batch(start_idx):
        """Process a batch of data."""
        end_idx = min(start_idx + batch_size, total_subjects)
        batch_data = phenotype_creator.data.iloc[start_idx:end_idx].copy()
        batch_creator = CarePhenotypeCreator(
            data=batch_data,
            clinical_factors=phenotype_creator.clinical_factors
        )
        return batch_creator.create_phenotype_labels()
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for start_idx in range(0, total_subjects, batch_size):
        sequential_results.append(process_batch(start_idx))
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    parallel_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_batch, start_idx)
            for start_idx in range(0, total_subjects, batch_size)
        ]
        parallel_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    parallel_time = time.time() - start_time
    
    # Log parallel processing metrics
    print("\nParallel Processing Metrics:")
    print(f"Sequential processing time: {sequential_time:.2f} seconds")
    print(f"Parallel processing time: {parallel_time:.2f} seconds")
    print(f"Speedup factor: {sequential_time / parallel_time:.2f}x")
    
    # Parallel processing assertions
    assert parallel_time < sequential_time, "Parallel processing should be faster"
    assert len(parallel_results) == len(sequential_results), "All batches should be processed"

def test_algorithm_efficiency(phenotype_creator):
    """Test efficiency of different algorithms and implementations."""
    # Test different clustering algorithms
    algorithms = {
        'kmeans': lambda: phenotype_creator._cluster_data(algorithm='kmeans'),
        'dbscan': lambda: phenotype_creator._cluster_data(algorithm='dbscan'),
        'hierarchical': lambda: phenotype_creator._cluster_data(algorithm='hierarchical')
    }
    
    results = {}
    for algo_name, algo_func in algorithms.items():
        # Measure execution time
        start_time = time.time()
        result = algo_func()
        end_time = time.time()
        
        results[algo_name] = {
            'execution_time': end_time - start_time,
            'n_clusters': len(np.unique(result))
        }
    
    # Log algorithm efficiency metrics
    print("\nAlgorithm Efficiency Metrics:")
    for algo_name, result in results.items():
        print(f"\nAlgorithm: {algo_name}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        print(f"Number of clusters: {result['n_clusters']}")
    
    # Algorithm efficiency assertions
    assert all(result['execution_time'] < 10 for result in results.values()), \
        "All algorithms should complete within 10 seconds"

def test_batch_processing_optimization(phenotype_creator):
    """Test optimization of batch processing parameters."""
    # Test different batch sizes
    batch_sizes = [500, 1000, 2000, 5000]
    results = []
    
    for batch_size in batch_sizes:
        # Measure processing time
        start_time = time.time()
        total_subjects = len(phenotype_creator.data)
        
        for start_idx in range(0, total_subjects, batch_size):
            end_idx = min(start_idx + batch_size, total_subjects)
            batch_data = phenotype_creator.data.iloc[start_idx:end_idx].copy()
            batch_creator = CarePhenotypeCreator(
                data=batch_data,
                clinical_factors=phenotype_creator.clinical_factors
            )
            batch_creator.create_phenotype_labels()
        
        processing_time = time.time() - start_time
        results.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'time_per_batch': processing_time / (total_subjects / batch_size)
        })
    
    # Log batch processing metrics
    print("\nBatch Processing Optimization Metrics:")
    for result in results:
        print(f"\nBatch size: {result['batch_size']}")
        print(f"Total processing time: {result['processing_time']:.2f} seconds")
        print(f"Time per batch: {result['time_per_batch']:.2f} seconds")
    
    # Batch processing assertions
    assert all(result['time_per_batch'] < 5 for result in results), \
        "Each batch should process within 5 seconds"

def test_processing_pipeline_optimization(phenotype_creator):
    """Test optimization of the entire processing pipeline."""
    # Define pipeline stages
    stages = [
        ('data_loading', lambda: phenotype_creator.data),
        ('preprocessing', lambda: phenotype_creator._preprocess_data()),
        ('clustering', lambda: phenotype_creator._cluster_data()),
        ('label_creation', lambda: phenotype_creator.create_phenotype_labels())
    ]
    
    # Measure time for each stage
    results = []
    for stage_name, stage_func in stages:
        start_time = time.time()
        result = stage_func()
        end_time = time.time()
        
        results.append({
            'stage': stage_name,
            'execution_time': end_time - start_time,
            'result_size': len(result) if hasattr(result, '__len__') else None
        })
    
    # Log pipeline optimization metrics
    print("\nProcessing Pipeline Optimization Metrics:")
    total_time = 0
    for result in results:
        print(f"\nStage: {result['stage']}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        if result['result_size'] is not None:
            print(f"Result size: {result['result_size']}")
        total_time += result['execution_time']
    
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")
    
    # Pipeline optimization assertions
    assert total_time < 30, "Total pipeline should complete within 30 seconds"
    assert all(result['execution_time'] < 10 for result in results), \
        "Each stage should complete within 10 seconds" 