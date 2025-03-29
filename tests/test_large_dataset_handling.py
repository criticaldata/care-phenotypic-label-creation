"""
Tests for handling large datasets in the care phenotype analyzer.

This module contains tests to validate the system's ability to handle
large datasets efficiently, including various data sizes and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import time
import gc
from datetime import datetime, timedelta
from care_phenotype_analyzer.mimic.synthetic_data import SyntheticDataGenerator
from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator
from care_phenotype_analyzer.pattern_analyzer import CarePatternAnalyzer
from care_phenotype_analyzer.fairness_evaluator import FairnessEvaluator

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

@pytest.fixture
def varying_sized_datasets():
    """Generate datasets of varying sizes for testing."""
    sizes = [1000, 5000, 10000, 50000]
    datasets = {}
    
    for size in sizes:
        generator = SyntheticDataGenerator(n_patients=size, seed=42)
        datasets[size] = generator.generate_all()
    
    return datasets

@pytest.fixture
def varying_sized_creators(varying_sized_datasets):
    """Create CarePhenotypeCreator instances for different dataset sizes."""
    creators = {}
    
    for size, data in varying_sized_datasets.items():
        # Create sample care pattern data
        care_patterns = pd.DataFrame({
            'subject_id': data['patients']['subject_id'],
            'timestamp': [datetime.now() + timedelta(hours=i) for i in range(len(data['patients']))],
            'pattern_1': np.random.normal(0, 1, len(data['patients'])),
            'pattern_2': np.random.normal(0, 1, len(data['patients'])),
            'clinical_factor_1': np.random.normal(0, 1, len(data['patients'])),
            'clinical_factor_2': np.random.normal(0, 1, len(data['patients']))
        })
        
        creators[size] = CarePhenotypeCreator(
            data=care_patterns,
            clinical_factors=['clinical_factor_1', 'clinical_factor_2']
        )
    
    return creators

def test_scaling_performance(varying_sized_creators):
    """Test how performance scales with dataset size."""
    results = []
    
    for size, creator in varying_sized_creators.items():
        # Record initial memory
        initial_memory = get_memory_usage()
        
        # Process data
        start_time = time.time()
        labels = creator.create_phenotype_labels()
        end_time = time.time()
        
        # Record final memory
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Store results
        results.append({
            'size': size,
            'processing_time': end_time - start_time,
            'memory_increase': memory_increase,
            'subjects_per_second': size / (end_time - start_time)
        })
        
        # Force garbage collection
        gc.collect()
    
    # Log scaling metrics
    print("\nScaling Performance Metrics:")
    for result in results:
        print(f"\nDataset size: {result['size']} subjects")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Memory increase: {result['memory_increase']:.2f} MB")
        print(f"Processing rate: {result['subjects_per_second']:.2f} subjects/second")
    
    # Performance assertions
    for i in range(len(results) - 1):
        # Check that processing time scales sub-linearly
        time_ratio = results[i+1]['processing_time'] / results[i]['processing_time']
        size_ratio = results[i+1]['size'] / results[i]['size']
        assert time_ratio < size_ratio, \
            f"Processing time should scale sub-linearly with dataset size"
        
        # Check that memory usage scales sub-linearly
        memory_ratio = results[i+1]['memory_increase'] / results[i]['memory_increase']
        assert memory_ratio < size_ratio, \
            f"Memory usage should scale sub-linearly with dataset size"

def test_large_dataset_edge_cases(varying_sized_creators):
    """Test handling of edge cases in large datasets."""
    # Test with largest dataset
    largest_creator = varying_sized_creators[50000]
    
    # Test with sparse data
    sparse_data = largest_creator.data.copy()
    sparse_data.loc[::2, 'pattern_1'] = np.nan
    sparse_data.loc[::3, 'pattern_2'] = np.nan
    sparse_creator = CarePhenotypeCreator(
        data=sparse_data,
        clinical_factors=largest_creator.clinical_factors
    )
    
    # Test with highly correlated data
    correlated_data = largest_creator.data.copy()
    correlated_data['pattern_2'] = correlated_data['pattern_1'] * 0.9 + np.random.normal(0, 0.1, len(correlated_data))
    correlated_creator = CarePhenotypeCreator(
        data=correlated_data,
        clinical_factors=largest_creator.clinical_factors
    )
    
    # Process each case and measure performance
    cases = {
        'original': largest_creator,
        'sparse': sparse_creator,
        'correlated': correlated_creator
    }
    
    results = {}
    for case_name, creator in cases.items():
        # Record initial memory
        initial_memory = get_memory_usage()
        
        # Process data
        start_time = time.time()
        labels = creator.create_phenotype_labels()
        end_time = time.time()
        
        # Record final memory
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Store results
        results[case_name] = {
            'processing_time': end_time - start_time,
            'memory_increase': memory_increase,
            'label_distribution': labels.value_counts().to_dict()
        }
        
        # Force garbage collection
        gc.collect()
    
    # Log edge case metrics
    print("\nEdge Case Performance Metrics:")
    for case_name, result in results.items():
        print(f"\nCase: {case_name}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Memory increase: {result['memory_increase']:.2f} MB")
        print(f"Label distribution: {result['label_distribution']}")
    
    # Performance assertions
    assert results['sparse']['processing_time'] <= results['original']['processing_time'] * 1.2, \
        "Sparse data processing should not be significantly slower"
    assert results['correlated']['processing_time'] <= results['original']['processing_time'] * 1.2, \
        "Correlated data processing should not be significantly slower"

def test_large_dataset_chunking(varying_sized_creators):
    """Test chunked processing of large datasets."""
    # Test with largest dataset
    creator = varying_sized_creators[50000]
    
    # Test different chunk sizes
    chunk_sizes = [1000, 5000, 10000]
    results = []
    
    for chunk_size in chunk_sizes:
        # Record initial memory
        initial_memory = get_memory_usage()
        
        # Process data in chunks
        start_time = time.time()
        total_subjects = len(creator.data)
        all_labels = []
        
        for start_idx in range(0, total_subjects, chunk_size):
            chunk_data = creator.data.iloc[start_idx:start_idx + chunk_size]
            chunk_creator = CarePhenotypeCreator(
                data=chunk_data,
                clinical_factors=creator.clinical_factors
            )
            chunk_labels = chunk_creator.create_phenotype_labels()
            all_labels.extend(chunk_labels)
        
        end_time = time.time()
        
        # Record final memory
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Store results
        results.append({
            'chunk_size': chunk_size,
            'processing_time': end_time - start_time,
            'memory_increase': memory_increase,
            'subjects_per_second': total_subjects / (end_time - start_time)
        })
        
        # Force garbage collection
        gc.collect()
    
    # Log chunking metrics
    print("\nChunking Performance Metrics:")
    for result in results:
        print(f"\nChunk size: {result['chunk_size']} subjects")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Memory increase: {result['memory_increase']:.2f} MB")
        print(f"Processing rate: {result['subjects_per_second']:.2f} subjects/second")
    
    # Performance assertions
    for i in range(len(results) - 1):
        # Check that larger chunks are generally more efficient
        assert results[i+1]['subjects_per_second'] >= results[i]['subjects_per_second'] * 0.8, \
            f"Larger chunk size should not be significantly less efficient"

def test_large_dataset_memory_management(varying_sized_creators):
    """Test memory management during large dataset processing."""
    # Test with largest dataset
    creator = varying_sized_creators[50000]
    
    # Record initial memory
    initial_memory = get_memory_usage()
    memory_history = []
    
    # Process data in stages
    stages = [
        ('data_loading', lambda: creator.data),
        ('preprocessing', lambda: creator._preprocess_data()),
        ('clustering', lambda: creator._cluster_data()),
        ('label_creation', lambda: creator.create_phenotype_labels())
    ]
    
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
    
    # Log memory management metrics
    print("\nMemory Management Metrics:")
    for record in memory_history:
        print(f"\nStage: {record['stage']}")
        print(f"Memory increase: {record['memory_increase']:.2f} MB")
        print(f"Total memory used: {record['total_memory']:.2f} MB")
    
    # Memory management assertions
    assert all(record['memory_increase'] < 1000 for record in memory_history), \
        "No stage should increase memory usage by more than 1GB"
    assert memory_history[-1]['total_memory'] < 2000, \
        "Total memory usage should not exceed 2GB" 