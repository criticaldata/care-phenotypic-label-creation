"""
Module for implementing memory optimization functionality.

This module provides tools and utilities for optimizing memory usage
in data processing operations.
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import time
from pathlib import Path
import threading
from .monitoring import SystemMonitor

class MemoryOptimizer:
    """Class for managing memory optimization operations."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the memory optimizer.
        
        Args:
            log_dir: Directory for monitoring logs
        """
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Memory thresholds (in bytes)
        self.warning_threshold = 0.8  # 80% of available memory
        self.critical_threshold = 0.9  # 90% of available memory
        
        # Log initialization
        self.monitor.logger.info("Initialized MemoryOptimizer")
        
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics.
        
        Returns:
            Dict: Memory usage statistics
        """
        with self.lock:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                return {
                    'process_rss': memory_info.rss,
                    'process_vms': memory_info.vms,
                    'system_total': system_memory.total,
                    'system_available': system_memory.available,
                    'system_percent': system_memory.percent
                }
                
            except Exception as e:
                self.monitor.record_error(f"Error getting memory usage: {str(e)}")
                return {}
    
    def optimize_dataframe(self,
                          df: pd.DataFrame,
                          optimize_dtypes: bool = True,
                          downcast: bool = True) -> pd.DataFrame:
        """Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            optimize_dtypes: Whether to optimize data types
            downcast: Whether to downcast numeric types
            
        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        start_time = time.time()
        initial_memory = df.memory_usage(deep=True).sum()
        
        try:
            # Create a copy to avoid modifying the original
            optimized_df = df.copy()
            
            if optimize_dtypes:
                # Optimize numeric columns
                for col in optimized_df.select_dtypes(include=['int64', 'float64']).columns:
                    if downcast:
                        # Downcast integers
                        if optimized_df[col].dtype == 'int64':
                            c_min = optimized_df[col].min()
                            c_max = optimized_df[col].max()
                            
                            if c_min >= 0:
                                if c_max < 255:
                                    optimized_df[col] = optimized_df[col].astype(np.uint8)
                                elif c_max < 65535:
                                    optimized_df[col] = optimized_df[col].astype(np.uint16)
                                elif c_max < 4294967295:
                                    optimized_df[col] = optimized_df[col].astype(np.uint32)
                            else:
                                if c_min > -128 and c_max < 127:
                                    optimized_df[col] = optimized_df[col].astype(np.int8)
                                elif c_min > -32768 and c_max < 32767:
                                    optimized_df[col] = optimized_df[col].astype(np.int16)
                                elif c_min > -2147483648 and c_max < 2147483647:
                                    optimized_df[col] = optimized_df[col].astype(np.int32)
                        
                        # Downcast floats
                        elif optimized_df[col].dtype == 'float64':
                            optimized_df[col] = optimized_df[col].astype(np.float32)
            
            # Record memory optimization metrics
            final_memory = optimized_df.memory_usage(deep=True).sum()
            memory_reduction = initial_memory - final_memory
            
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(optimized_df)
            )
            
            # Log optimization results
            self.monitor.logger.info(
                f"DataFrame memory optimization: "
                f"reduced from {initial_memory/1024/1024:.2f}MB to "
                f"{final_memory/1024/1024:.2f}MB "
                f"({memory_reduction/initial_memory*100:.1f}% reduction)"
            )
            
            return optimized_df
            
        except Exception as e:
            error_msg = f"Error optimizing DataFrame: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def optimize_array(self,
                      arr: np.ndarray,
                      downcast: bool = True) -> np.ndarray:
        """Optimize numpy array memory usage.
        
        Args:
            arr: Array to optimize
            downcast: Whether to downcast numeric types
            
        Returns:
            np.ndarray: Optimized array
        """
        start_time = time.time()
        initial_memory = arr.nbytes
        
        try:
            # Create a copy to avoid modifying the original
            optimized_arr = arr.copy()
            
            if downcast:
                # Downcast based on data range
                if optimized_arr.dtype == np.int64:
                    c_min = optimized_arr.min()
                    c_max = optimized_arr.max()
                    
                    if c_min >= 0:
                        if c_max < 255:
                            optimized_arr = optimized_arr.astype(np.uint8)
                        elif c_max < 65535:
                            optimized_arr = optimized_arr.astype(np.uint16)
                        elif c_max < 4294967295:
                            optimized_arr = optimized_arr.astype(np.uint32)
                    else:
                        if c_min > -128 and c_max < 127:
                            optimized_arr = optimized_arr.astype(np.int8)
                        elif c_min > -32768 and c_max < 32767:
                            optimized_arr = optimized_arr.astype(np.int16)
                        elif c_min > -2147483648 and c_max < 2147483647:
                            optimized_arr = optimized_arr.astype(np.int32)
                
                elif optimized_arr.dtype == np.float64:
                    optimized_arr = optimized_arr.astype(np.float32)
            
            # Record memory optimization metrics
            final_memory = optimized_arr.nbytes
            memory_reduction = initial_memory - final_memory
            
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(optimized_arr)
            )
            
            # Log optimization results
            self.monitor.logger.info(
                f"Array memory optimization: "
                f"reduced from {initial_memory/1024/1024:.2f}MB to "
                f"{final_memory/1024/1024:.2f}MB "
                f"({memory_reduction/initial_memory*100:.1f}% reduction)"
            )
            
            return optimized_arr
            
        except Exception as e:
            error_msg = f"Error optimizing array: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def clear_memory(self):
        """Clear unused memory."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Get memory usage before and after
            before = self.get_memory_usage()
            
            # Clear memory
            gc.collect()
            
            after = self.get_memory_usage()
            
            # Log memory reduction
            if before and after:
                reduction = before['process_rss'] - after['process_rss']
                self.monitor.logger.info(
                    f"Memory cleared: reduced by {reduction/1024/1024:.2f}MB"
                )
                
        except Exception as e:
            self.monitor.record_error(f"Error clearing memory: {str(e)}")
    
    def check_memory_pressure(self) -> Dict:
        """Check current memory pressure.
        
        Returns:
            Dict: Memory pressure status
        """
        with self.lock:
            try:
                memory_usage = self.get_memory_usage()
                
                if not memory_usage:
                    return {'status': 'unknown'}
                
                system_percent = memory_usage['system_percent'] / 100
                
                if system_percent >= self.critical_threshold:
                    status = 'critical'
                elif system_percent >= self.warning_threshold:
                    status = 'warning'
                else:
                    status = 'normal'
                
                return {
                    'status': status,
                    'system_percent': system_percent * 100,
                    'process_rss': memory_usage['process_rss'],
                    'process_vms': memory_usage['process_vms']
                }
                
            except Exception as e:
                self.monitor.record_error(f"Error checking memory pressure: {str(e)}")
                return {'status': 'error'}
    
    def optimize_batch(self,
                      items: List[Any],
                      optimize_func: Callable,
                      batch_size: int = 1000) -> List[Any]:
        """Process items in batches to optimize memory usage.
        
        Args:
            items: List of items to process
            optimize_func: Function to optimize each item
            batch_size: Size of each batch
            
        Returns:
            List[Any]: Processed items
        """
        start_time = time.time()
        
        try:
            results = []
            
            # Process items in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # Process batch
                batch_results = [optimize_func(item) for item in batch]
                results.extend(batch_results)
                
                # Clear memory after each batch
                self.clear_memory()
                
                # Check memory pressure
                pressure = self.check_memory_pressure()
                if pressure['status'] == 'critical':
                    self.monitor.logger.warning("Critical memory pressure detected")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(items)
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring() 