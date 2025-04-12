"""
Module for monitoring and logging system performance, metrics, and health.

This module provides comprehensive monitoring capabilities including:
- Performance metrics tracking
- System health monitoring
- Resource usage tracking
- Error tracking and reporting
- Logging with different severity levels
"""

import logging
import time
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import os

# Add a custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    processing_time: float
    batch_size: int
    records_processed: int
    error_count: int
    warning_count: int

@dataclass
class SystemHealth:
    """Data class for storing system health metrics."""
    timestamp: datetime
    status: str
    active_threads: int
    queue_size: int
    last_error: Optional[str]
    last_warning: Optional[str]

class SystemMonitor:
    """Class for monitoring system performance and health."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the system monitor.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize metrics storage
        self.metrics_queue = Queue()
        self.health_queue = Queue()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        # Initialize counters
        self.error_count = 0
        self.warning_count = 0
        self.records_processed = 0
        
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Set up file handler
        file_handler = logging.FileHandler(
            self.log_dir / "system_monitor.log"
        )
        file_handler.setFormatter(file_formatter)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Configure logger
        self.logger = logging.getLogger("SystemMonitor")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _start_monitoring_threads(self):
        """Start background monitoring threads."""
        self.monitoring_active = True
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(
            target=self._collect_metrics,
            daemon=True
        )
        self.metrics_thread.start()
        
        # Start health check thread
        self.health_thread = threading.Thread(
            target=self._check_health,
            daemon=True
        )
        self.health_thread.start()
        
    def _collect_metrics(self):
        """Background thread for collecting system metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                # Safely access records_processed attribute
                try:
                    records_count = self.records_processed
                except AttributeError:
                    # Handle the case where records_processed is not initialized
                    records_count = 0
                    self.records_processed = 0  # Initialize it
                
                # Safely access error_count attribute
                try:
                    error_count = self.error_count
                except AttributeError:
                    # Handle the case where error_count is not initialized
                    error_count = 0
                    self.error_count = 0  # Initialize it
                
                # Safely access warning_count attribute
                try:
                    warning_count = self.warning_count
                except AttributeError:
                    # Handle the case where warning_count is not initialized
                    warning_count = 0
                    self.warning_count = 0  # Initialize it
                
                # Create metrics object
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_usage_percent=disk_usage,
                    processing_time=0.0,  # Will be updated by record_processing
                    batch_size=0,  # Will be updated by record_processing
                    records_processed=records_count,
                    error_count=error_count,
                    warning_count=warning_count
                )
                
                # Add to queue
                self.metrics_queue.put(metrics)
                
                # Log metrics
                self.logger.debug(f"Collected metrics: {asdict(metrics)}")
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                
            time.sleep(60)  # Collect metrics every minute
            
    def _check_health(self):
        """Background thread for checking system health."""
        while self.monitoring_active:
            try:
                # Check system health
                active_threads = threading.active_count()
                queue_size = self.metrics_queue.qsize()
                
                # Create health object
                health = SystemHealth(
                    timestamp=datetime.now(),
                    status="healthy",
                    active_threads=active_threads,
                    queue_size=queue_size,
                    last_error=None,
                    last_warning=None
                )
                
                # Add to queue
                self.health_queue.put(health)
                
                # Log health status
                self.logger.debug(f"Health check: {asdict(health)}")
                
            except Exception as e:
                self.logger.error(f"Error checking health: {str(e)}")
                
            time.sleep(300)  # Check health every 5 minutes
            
    def record_processing(self, processing_time: float, batch_size: int):
        """Record processing metrics.
        
        Args:
            processing_time: Time taken to process the batch
            batch_size: Number of records in the batch
        """
        self.records_processed += batch_size
        
        # Update latest metrics
        if not self.metrics_queue.empty():
            metrics = self.metrics_queue.get()
            metrics.processing_time = processing_time
            metrics.batch_size = batch_size
            self.metrics_queue.put(metrics)
            
    def record_error(self, error: str):
        """Record an error.
        
        Args:
            error: Error message
        """
        self.error_count += 1
        self.logger.error(error)
        
        # Update latest health status
        if not self.health_queue.empty():
            health = self.health_queue.get()
            health.last_error = error
            health.status = "error"
            self.health_queue.put(health)
            
    def record_warning(self, warning: str):
        """Record a warning.
        
        Args:
            warning: Warning message
        """
        self.warning_count += 1
        self.logger.warning(warning)
        
        # Update latest health status
        if not self.health_queue.empty():
            health = self.health_queue.get()
            health.last_warning = warning
            if health.status == "healthy":
                health.status = "warning"
            self.health_queue.put(health)
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        if self.metrics_queue.empty():
            return {}
            
        metrics = self.metrics_queue.get()
        summary = asdict(metrics)
        
        # Add additional metrics
        summary.update({
            'error_rate': self.error_count / max(1, self.records_processed),
            'warning_rate': self.warning_count / max(1, self.records_processed),
            'processing_rate': self.records_processed / max(1, metrics.processing_time)
        })
        
        return summary
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status.
        
        Returns:
            Dictionary containing health status
        """
        if self.health_queue.empty():
            return {}
            
        return asdict(self.health_queue.get())
        
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        # Join threads with timeouts to prevent blocking
        try:
            self.metrics_thread.join(timeout=1.0)
            self.health_thread.join(timeout=1.0)
        except (RuntimeError, AttributeError):
            # Handle cases where threads might not exist
            pass
        
        # Save final metrics
        self._save_final_metrics()
        
    def _save_final_metrics(self):
        """Save final metrics to file."""
        metrics_file = self.log_dir / "final_metrics.json"
        health_file = self.log_dir / "final_health.json"
        
        # Save metrics
        if not self.metrics_queue.empty():
            with open(metrics_file, 'w') as f:
                json.dump(self.get_metrics_summary(), f, indent=2, cls=DateTimeEncoder)
                
        # Save health status
        if not self.health_queue.empty():
            with open(health_file, 'w') as f:
                json.dump(self.get_health_status(), f, indent=2, cls=DateTimeEncoder) 