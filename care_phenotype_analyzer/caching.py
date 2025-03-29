"""
Module for implementing caching functionality.

This module provides caching mechanisms to store and retrieve frequently
accessed data and computation results, improving overall performance.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
from .monitoring import SystemMonitor

class CacheManager:
    """Class for managing caching operations."""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 1024*1024*1024, 
                 max_age: int = 24*60*60, log_dir: str = "logs"):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory for storing cache files
            max_size: Maximum cache size in bytes (default: 1GB)
            max_age: Maximum age of cache entries in seconds (default: 24 hours)
            log_dir: Directory for monitoring logs
        """
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.max_size = max_size
        self.max_age = max_age
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Log initialization
        self.monitor.logger.info("Initialized CacheManager")
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.monitor.record_error(f"Error loading cache metadata: {str(e)}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            self.monitor.record_error(f"Error saving cache metadata: {str(e)}")
    
    def _generate_key(self, data: Any) -> str:
        """Generate a unique cache key for the given data.
        
        Args:
            data: Data to generate a key for
            
        Returns:
            str: Unique cache key
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # For pandas objects, use a hash of the data
            return hashlib.md5(pickle.dumps(data)).hexdigest()
        elif isinstance(data, np.ndarray):
            # For numpy arrays, use a hash of the data
            return hashlib.md5(data.tobytes()).hexdigest()
        else:
            # For other objects, use a hash of the string representation
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key.
        
        Args:
            key: Cache key
            
        Returns:
            Path: Path to cache file
        """
        return self.cache_dir / f"{key}.cache"
    
    def _cleanup_old_entries(self):
        """Remove old cache entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, metadata in self.metadata.items():
            if current_time - metadata['timestamp'] > self.max_age:
                keys_to_remove.append(key)
                try:
                    self._get_cache_path(key).unlink()
                except Exception as e:
                    self.monitor.record_error(f"Error removing old cache entry: {str(e)}")
        
        for key in keys_to_remove:
            del self.metadata[key]
        
        if keys_to_remove:
            self._save_metadata()
    
    def _cleanup_by_size(self):
        """Remove cache entries to maintain size limit."""
        if not self.metadata:
            return
        
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        current_size = sum(entry['size'] for entry in self.metadata.values())
        keys_to_remove = []
        
        for key, metadata in sorted_entries:
            if current_size <= self.max_size:
                break
                
            try:
                self._get_cache_path(key).unlink()
                current_size -= metadata['size']
                keys_to_remove.append(key)
            except Exception as e:
                self.monitor.record_error(f"Error removing cache entry: {str(e)}")
        
        for key in keys_to_remove:
            del self.metadata[key]
        
        if keys_to_remove:
            self._save_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached data if found and valid, None otherwise
        """
        with self.lock:
            try:
                if key not in self.metadata:
                    return None
                
                metadata = self.metadata[key]
                cache_path = self._get_cache_path(key)
                
                # Check if cache entry is expired
                if time.time() - metadata['timestamp'] > self.max_age:
                    cache_path.unlink()
                    del self.metadata[key]
                    self._save_metadata()
                    return None
                
                # Load cached data
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Record cache hit
                self.monitor.record_processing(
                    processing_time=0.0,  # Cache hits are instant
                    batch_size=metadata['size']
                )
                
                return data
                
            except Exception as e:
                self.monitor.record_error(f"Error retrieving from cache: {str(e)}")
                return None
    
    def set(self, key: str, data: Any):
        """Store data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        with self.lock:
            try:
                start_time = time.time()
                
                # Clean up old entries
                self._cleanup_old_entries()
                
                # Save data to cache file
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Update metadata
                self.metadata[key] = {
                    'timestamp': time.time(),
                    'size': cache_path.stat().st_size
                }
                
                # Clean up by size if needed
                self._cleanup_by_size()
                
                # Save metadata
                self._save_metadata()
                
                # Record processing metrics
                processing_time = time.time() - start_time
                self.monitor.record_processing(
                    processing_time=processing_time,
                    batch_size=self.metadata[key]['size']
                )
                
            except Exception as e:
                self.monitor.record_error(f"Error storing in cache: {str(e)}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        self.monitor.record_error(f"Error removing cache file: {str(e)}")
                
                # Clear metadata
                self.metadata = {}
                self._save_metadata()
                
            except Exception as e:
                self.monitor.record_error(f"Error clearing cache: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        with self.lock:
            try:
                total_size = sum(entry['size'] for entry in self.metadata.values())
                total_entries = len(self.metadata)
                oldest_entry = min(
                    (entry['timestamp'] for entry in self.metadata.values()),
                    default=time.time()
                )
                
                return {
                    'total_size': total_size,
                    'total_entries': total_entries,
                    'oldest_entry': datetime.fromtimestamp(oldest_entry).isoformat(),
                    'max_size': self.max_size,
                    'max_age': self.max_age
                }
                
            except Exception as e:
                self.monitor.record_error(f"Error getting cache stats: {str(e)}")
                return {}
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring() 