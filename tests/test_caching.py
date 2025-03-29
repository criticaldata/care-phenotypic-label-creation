"""
Tests for caching functionality.
"""

import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from care_phenotype_analyzer.caching import CacheManager

@pytest.fixture
def cache_manager():
    """Create a temporary directory for cache and initialize the cache manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = CacheManager(
            cache_dir=temp_dir,
            max_size=1024*1024,  # 1MB
            max_age=60  # 1 minute
        )
        yield cache_manager
        cache_manager.monitor.stop_monitoring()

def test_cache_manager_initialization(cache_manager):
    """Test proper initialization of the cache manager."""
    assert cache_manager.monitor is not None
    assert cache_manager.cache_dir.exists()
    assert cache_manager.metadata_file.exists()
    assert isinstance(cache_manager.metadata, dict)

def test_cache_set_get(cache_manager):
    """Test basic cache set and get operations."""
    # Test with DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    key = cache_manager._generate_key(df)
    
    cache_manager.set(key, df)
    cached_df = cache_manager.get(key)
    
    pd.testing.assert_frame_equal(df, cached_df)
    
    # Test with numpy array
    arr = np.array([1, 2, 3])
    key = cache_manager._generate_key(arr)
    
    cache_manager.set(key, arr)
    cached_arr = cache_manager.get(key)
    
    np.testing.assert_array_equal(arr, cached_arr)

def test_cache_expiration(cache_manager):
    """Test cache entry expiration."""
    # Set cache entry
    data = {'test': 'data'}
    key = cache_manager._generate_key(data)
    cache_manager.set(key, data)
    
    # Verify entry exists
    assert cache_manager.get(key) is not None
    
    # Wait for expiration
    time.sleep(61)  # Wait longer than max_age
    
    # Verify entry is expired
    assert cache_manager.get(key) is None

def test_cache_size_limit(cache_manager):
    """Test cache size limit enforcement."""
    # Create large data
    large_data = np.random.rand(1000, 1000)  # ~8MB
    key = cache_manager._generate_key(large_data)
    
    # Set cache entry
    cache_manager.set(key, large_data)
    
    # Verify entry was not cached due to size limit
    assert cache_manager.get(key) is None

def test_cache_cleanup(cache_manager):
    """Test cache cleanup operations."""
    # Set multiple entries
    for i in range(5):
        data = {'test': f'data_{i}'}
        key = cache_manager._generate_key(data)
        cache_manager.set(key, data)
    
    # Verify entries exist
    assert len(cache_manager.metadata) > 0
    
    # Clear cache
    cache_manager.clear()
    
    # Verify cache is empty
    assert len(cache_manager.metadata) == 0
    assert not any(cache_manager.cache_dir.glob("*.cache"))

def test_cache_stats(cache_manager):
    """Test cache statistics."""
    # Set some entries
    for i in range(3):
        data = {'test': f'data_{i}'}
        key = cache_manager._generate_key(data)
        cache_manager.set(key, data)
    
    # Get stats
    stats = cache_manager.get_stats()
    
    # Verify stats
    assert stats['total_entries'] == 3
    assert stats['total_size'] > 0
    assert isinstance(stats['oldest_entry'], str)
    assert stats['max_size'] == 1024*1024
    assert stats['max_age'] == 60

def test_cache_thread_safety(cache_manager):
    """Test thread safety of cache operations."""
    import threading
    
    def worker():
        for i in range(100):
            data = {'test': f'data_{i}'}
            key = cache_manager._generate_key(data)
            cache_manager.set(key, data)
            cache_manager.get(key)
    
    # Create multiple threads
    threads = [threading.Thread(target=worker) for _ in range(5)]
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify no errors occurred
    assert len(cache_manager.monitor.get_errors()) == 0

def test_cache_error_handling(cache_manager):
    """Test error handling in cache operations."""
    # Test invalid key
    assert cache_manager.get('invalid_key') is None
    
    # Test invalid data
    with pytest.raises(Exception):
        cache_manager.set('key', lambda x: x)  # Can't pickle lambda
    
    # Test invalid file operations
    cache_manager.cache_dir.rmdir()
    assert cache_manager.get('any_key') is None
    cache_manager.set('key', 'data')  # Should recreate directory

def test_cache_metadata_persistence(cache_manager):
    """Test cache metadata persistence."""
    # Set some entries
    data = {'test': 'data'}
    key = cache_manager._generate_key(data)
    cache_manager.set(key, data)
    
    # Create new cache manager in same directory
    new_cache_manager = CacheManager(
        cache_dir=cache_manager.cache_dir,
        max_size=cache_manager.max_size,
        max_age=cache_manager.max_age
    )
    
    # Verify metadata is loaded
    assert key in new_cache_manager.metadata
    assert new_cache_manager.get(key) is not None
    
    new_cache_manager.monitor.stop_monitoring() 