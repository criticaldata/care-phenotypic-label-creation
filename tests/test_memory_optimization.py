"""
Tests for memory usage optimization in the care phenotype analyzer.

This module contains tests to validate and optimize memory usage during
data processing, including memory profiling, garbage collection, and
memory-efficient data structures.
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import time
import gc
import sys
from datetime import datetime, timedelta
from memory_profiler import profile
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator
from care_phenotype_analyzer.pattern_analyzer import CarePatternAnalyzer
from care_phenotype_analyzer.fairness_evaluator import FairnessEvaluator

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_object_size(obj):
    """Get approximate size of an object in memory."""
    return sys.getsizeof(obj) / 1024 / 1024  # Convert to MB

@pytest.fixture
def large_dataset():
    """Generate large dataset for memory optimization testing."""
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

@profile
def test_memory_profiling(phenotype_creator):
    """Test memory profiling during data processing."""
    # Record initial memory
    initial_memory = get_memory_usage()
    
    # Process data in stages
    stages = [
        ('data_loading', lambda: phenotype_creator.data),
        ('preprocessing', lambda: phenotype_creator._preprocess_data()),
        ('clustering', lambda: phenotype_creator._cluster_data()),
        ('label_creation', lambda: phenotype_creator.create_phenotype_labels())
    ]
    
    memory_history = []
    for stage_name, stage_func in stages:
        # Record memory before stage
        pre_stage_memory = get_memory_usage()
        
        # Execute stage
        result = stage_func()
        
        # Record memory after stage
        post_stage_memory = get_memory_usage()
        memory_increase = post_stage_memory - pre_stage_memory
        
        # Store memory history
        memory_history.append({
            'stage': stage_name,
            'memory_increase': memory_increase,
            'total_memory': post_stage_memory - initial_memory
        })
        
        # Force garbage collection
        gc.collect()
    
    # Log memory profiling metrics
    print("\nMemory Profiling Metrics:")
    for record in memory_history:
        print(f"\nStage: {record['stage']}")
        print(f"Memory increase: {record['memory_increase']:.2f} MB")
        print(f"Total memory used: {record['total_memory']:.2f} MB")
    
    # Memory profiling assertions
    assert all(record['memory_increase'] < 500 for record in memory_history), \
        "No stage should increase memory usage by more than 500MB"

def test_memory_efficient_data_structures(phenotype_creator):
    """Test memory efficiency of different data structures."""
    # Test DataFrame memory optimization
    df = phenotype_creator.data.copy()
    original_size = get_object_size(df)
    
    # Optimize DataFrame memory usage
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    optimized_size = get_object_size(df)
    
    # Test Series memory optimization
    series = phenotype_creator.data['pattern_1'].copy()
    series_original_size = get_object_size(series)
    
    # Optimize Series memory usage
    series = series.astype('float32')
    series_optimized_size = get_object_size(series)
    
    # Log memory optimization metrics
    print("\nData Structure Memory Optimization Metrics:")
    print(f"DataFrame original size: {original_size:.2f} MB")
    print(f"DataFrame optimized size: {optimized_size:.2f} MB")
    print(f"DataFrame memory reduction: {(original_size - optimized_size) / original_size * 100:.1f}%")
    print(f"Series original size: {series_original_size:.2f} MB")
    print(f"Series optimized size: {series_optimized_size:.2f} MB")
    print(f"Series memory reduction: {(series_original_size - series_optimized_size) / series_original_size * 100:.1f}%")
    
    # Memory optimization assertions
    assert optimized_size < original_size, "DataFrame optimization should reduce memory usage"
    assert series_optimized_size < series_original_size, "Series optimization should reduce memory usage"

def test_memory_cleanup(phenotype_creator):
    """Test memory cleanup and garbage collection."""
    # Record initial memory
    initial_memory = get_memory_usage()
    
    # Create and process data
    data = phenotype_creator.data.copy()
    processed_data = phenotype_creator._preprocess_data()
    labels = phenotype_creator.create_phenotype_labels()
    
    # Record memory after processing
    post_processing_memory = get_memory_usage()
    
    # Delete intermediate objects
    del processed_data
    gc.collect()
    
    # Record memory after cleanup
    post_cleanup_memory = get_memory_usage()
    
    # Log memory cleanup metrics
    print("\nMemory Cleanup Metrics:")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Post-processing memory: {post_processing_memory:.2f} MB")
    print(f"Post-cleanup memory: {post_cleanup_memory:.2f} MB")
    print(f"Memory cleanup: {post_processing_memory - post_cleanup_memory:.2f} MB")
    
    # Memory cleanup assertions
    assert post_cleanup_memory < post_processing_memory, "Cleanup should reduce memory usage"
    assert post_cleanup_memory - initial_memory < 100, "Final memory usage should be close to initial"

def test_memory_usage_patterns(phenotype_creator):
    """Test memory usage patterns during long-running operations."""
    # Record initial memory
    initial_memory = get_memory_usage()
    memory_samples = []
    
    # Process data in small batches
    batch_size = 1000
    total_subjects = len(phenotype_creator.data)
    
    for start_idx in range(0, total_subjects, batch_size):
        # Record memory before batch
        pre_batch_memory = get_memory_usage()
        
        # Process batch
        batch_data = phenotype_creator.data.iloc[start_idx:start_idx + batch_size]
        batch_creator = CarePhenotypeCreator(
            data=batch_data,
            clinical_factors=phenotype_creator.clinical_factors
        )
        batch_labels = batch_creator.create_phenotype_labels()
        
        # Record memory after batch
        post_batch_memory = get_memory_usage()
        
        # Store memory sample
        memory_samples.append({
            'batch': start_idx // batch_size,
            'memory_usage': post_batch_memory - pre_batch_memory,
            'total_memory': post_batch_memory - initial_memory
        })
        
        # Force garbage collection
        gc.collect()
    
    # Analyze memory usage patterns
    memory_increases = [sample['memory_usage'] for sample in memory_samples]
    avg_increase = np.mean(memory_increases)
    max_increase = np.max(memory_increases)
    
    # Log memory pattern metrics
    print("\nMemory Usage Pattern Metrics:")
    print(f"Average memory increase per batch: {avg_increase:.2f} MB")
    print(f"Maximum memory increase: {max_increase:.2f} MB")
    print(f"Total memory samples: {len(memory_samples)}")
    
    # Memory pattern assertions
    assert avg_increase < 50, "Average memory increase per batch should be under 50MB"
    assert max_increase < 100, "Maximum memory increase should be under 100MB"

def test_memory_optimization_strategies(phenotype_creator):
    """Test different memory optimization strategies."""
    # Test with different optimization strategies
    strategies = {
        'baseline': lambda: phenotype_creator.data.copy(),
        'optimized_dtypes': lambda: phenotype_creator.data.astype({
            'pattern_1': 'float32',
            'pattern_2': 'float32',
            'clinical_factor_1': 'float32',
            'clinical_factor_2': 'float32'
        }),
        'chunked_processing': lambda: phenotype_creator.data.iloc[:1000].copy()
    }
    
    results = {}
    for strategy_name, strategy_func in strategies.items():
        # Record initial memory
        initial_memory = get_memory_usage()
        
        # Apply strategy
        start_time = time.time()
        result = strategy_func()
        end_time = time.time()
        
        # Record final memory
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Store results
        results[strategy_name] = {
            'memory_increase': memory_increase,
            'processing_time': end_time - start_time,
            'object_size': get_object_size(result)
        }
        
        # Force garbage collection
        gc.collect()
    
    # Log optimization strategy metrics
    print("\nMemory Optimization Strategy Metrics:")
    for strategy_name, result in results.items():
        print(f"\nStrategy: {strategy_name}")
        print(f"Memory increase: {result['memory_increase']:.2f} MB")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Object size: {result['object_size']:.2f} MB")
    
    # Strategy comparison assertions
    assert results['optimized_dtypes']['object_size'] < results['baseline']['object_size'], \
        "Optimized dtypes should reduce object size"
    assert results['chunked_processing']['memory_increase'] < results['baseline']['memory_increase'], \
        "Chunked processing should reduce memory increase" 