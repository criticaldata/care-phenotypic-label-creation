"""
Module for implementing parallel processing functionality.

This module provides parallel processing capabilities to improve performance
by utilizing multiple CPU cores for data processing tasks.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import time
from pathlib import Path
import threading
from .monitoring import SystemMonitor

class ParallelProcessor:
    """Class for managing parallel processing operations."""
    
    def __init__(self, max_workers: Optional[int] = None, log_dir: str = "logs"):
        """Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes/threads
            log_dir: Directory for monitoring logs
        """
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Set up worker pools
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Log initialization
        self.monitor.logger.info(f"Initialized ParallelProcessor with {self.max_workers} workers")
        
    def process_dataframe_parallel(self,
                                 df: pd.DataFrame,
                                 func: Callable,
                                 chunk_size: int = 1000,
                                 use_processes: bool = True) -> pd.DataFrame:
        """Process a DataFrame in parallel by splitting it into chunks.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            use_processes: Whether to use processes (True) or threads (False)
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        start_time = time.time()
        
        try:
            # Split DataFrame into chunks
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            
            # Process chunks in parallel
            pool = self.process_pool if use_processes else self.thread_pool
            results = list(pool.map(func, chunks))
            
            # Combine results
            processed_df = pd.concat(results, ignore_index=True)
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(df)
            )
            
            return processed_df
            
        except Exception as e:
            error_msg = f"Error processing DataFrame in parallel: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def process_array_parallel(self,
                             arr: np.ndarray,
                             func: Callable,
                             chunk_size: int = 1000,
                             use_processes: bool = True) -> np.ndarray:
        """Process a numpy array in parallel by splitting it into chunks.
        
        Args:
            arr: Array to process
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            use_processes: Whether to use processes (True) or threads (False)
            
        Returns:
            np.ndarray: Processed array
        """
        start_time = time.time()
        
        try:
            # Split array into chunks
            chunks = np.array_split(arr, len(arr) // chunk_size + 1)
            
            # Process chunks in parallel
            pool = self.process_pool if use_processes else self.thread_pool
            results = list(pool.map(func, chunks))
            
            # Combine results
            processed_arr = np.concatenate(results)
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(arr)
            )
            
            return processed_arr
            
        except Exception as e:
            error_msg = f"Error processing array in parallel: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def process_list_parallel(self,
                            items: List[Any],
                            func: Callable,
                            chunk_size: int = 100,
                            use_processes: bool = True) -> List[Any]:
        """Process a list of items in parallel by splitting it into chunks.
        
        Args:
            items: List of items to process
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            use_processes: Whether to use processes (True) or threads (False)
            
        Returns:
            List[Any]: Processed list
        """
        start_time = time.time()
        
        try:
            # Split list into chunks
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            
            # Process chunks in parallel
            pool = self.process_pool if use_processes else self.thread_pool
            results = list(pool.map(func, chunks))
            
            # Combine results
            processed_items = [item for chunk in results for item in chunk]
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(items)
            )
            
            return processed_items
            
        except Exception as e:
            error_msg = f"Error processing list in parallel: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def process_batch_parallel(self,
                             items: List[Any],
                             func: Callable,
                             batch_size: int = 100,
                             use_processes: bool = True) -> List[Any]:
        """Process items in parallel batches.
        
        Args:
            items: List of items to process
            func: Function to apply to each item
            batch_size: Size of each batch
            use_processes: Whether to use processes (True) or threads (False)
            
        Returns:
            List[Any]: Processed items
        """
        start_time = time.time()
        
        try:
            # Process items in batches
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # Process batch in parallel
                pool = self.process_pool if use_processes else self.thread_pool
                batch_results = list(pool.map(func, batch))
                results.extend(batch_results)
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(items)
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing batch in parallel: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def get_worker_stats(self) -> Dict:
        """Get worker pool statistics.
        
        Returns:
            Dict: Worker pool statistics
        """
        with self.lock:
            try:
                return {
                    'max_workers': self.max_workers,
                    'active_processes': len(multiprocessing.active_children()),
                    'thread_count': threading.active_count()
                }
                
            except Exception as e:
                self.monitor.record_error(f"Error getting worker stats: {str(e)}")
                return {}
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            # Shutdown worker pools
            self.process_pool.shutdown(wait=True)
            self.thread_pool.shutdown(wait=True)
            
            if hasattr(self, 'monitor'):
                self.monitor.stop_monitoring()
                
        except Exception as e:
            if hasattr(self, 'monitor'):
                self.monitor.record_error(f"Error during cleanup: {str(e)}") 