"""
Tests for the monitoring system.

This module contains tests to validate the monitoring and logging functionality,
including performance metrics collection, system health monitoring, and error tracking.
"""

import pytest
import time
import json
from pathlib import Path
from datetime import datetime
from care_phenotype_analyzer.monitoring import SystemMonitor, PerformanceMetrics, SystemHealth

@pytest.fixture
def monitor(tmp_path):
    """Create a SystemMonitor instance with a temporary log directory."""
    return SystemMonitor(log_dir=str(tmp_path))

def test_monitor_initialization(monitor):
    """Test proper initialization of the monitoring system."""
    # Check log directory creation
    assert monitor.log_dir.exists()
    
    # Check log file creation
    log_file = monitor.log_dir / "system_monitor.log"
    assert log_file.exists()
    
    # Check initial counters
    assert monitor.error_count == 0
    assert monitor.warning_count == 0
    assert monitor.records_processed == 0
    
    # Check monitoring threads
    assert monitor.monitoring_active
    assert monitor.metrics_thread.is_alive()
    assert monitor.health_thread.is_alive()

def test_metrics_collection(monitor):
    """Test collection of system metrics."""
    # Wait for metrics collection
    time.sleep(2)
    
    # Get metrics summary
    metrics = monitor.get_metrics_summary()
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    assert 'timestamp' in metrics
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 'disk_usage_percent' in metrics
    assert 'processing_time' in metrics
    assert 'batch_size' in metrics
    assert 'records_processed' in metrics
    assert 'error_count' in metrics
    assert 'warning_count' in metrics
    
    # Check metric values
    assert 0 <= metrics['cpu_percent'] <= 100
    assert 0 <= metrics['memory_percent'] <= 100
    assert 0 <= metrics['disk_usage_percent'] <= 100
    assert metrics['processing_time'] == 0.0
    assert metrics['batch_size'] == 0
    assert metrics['records_processed'] == 0
    assert metrics['error_count'] == 0
    assert metrics['warning_count'] == 0

def test_health_monitoring(monitor):
    """Test system health monitoring."""
    # Wait for health check
    time.sleep(2)
    
    # Get health status
    health = monitor.get_health_status()
    
    # Check health status structure
    assert isinstance(health, dict)
    assert 'timestamp' in health
    assert 'status' in health
    assert 'active_threads' in health
    assert 'queue_size' in health
    assert 'last_error' in health
    assert 'last_warning' in health
    
    # Check health status values
    assert health['status'] == "healthy"
    assert health['active_threads'] > 0
    assert health['queue_size'] >= 0
    assert health['last_error'] is None
    assert health['last_warning'] is None

def test_processing_recording(monitor):
    """Test recording of processing metrics."""
    # Record processing
    monitor.record_processing(processing_time=1.5, batch_size=100)
    
    # Get metrics summary
    metrics = monitor.get_metrics_summary()
    
    # Check processing metrics
    assert metrics['processing_time'] == 1.5
    assert metrics['batch_size'] == 100
    assert metrics['records_processed'] == 100
    assert metrics['processing_rate'] == 100 / 1.5

def test_error_recording(monitor):
    """Test recording of errors."""
    # Record error
    error_msg = "Test error message"
    monitor.record_error(error_msg)
    
    # Check error count
    assert monitor.error_count == 1
    
    # Get health status
    health = monitor.get_health_status()
    
    # Check health status
    assert health['status'] == "error"
    assert health['last_error'] == error_msg
    
    # Check log file
    log_file = monitor.log_dir / "system_monitor.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert error_msg in log_content

def test_warning_recording(monitor):
    """Test recording of warnings."""
    # Record warning
    warning_msg = "Test warning message"
    monitor.record_warning(warning_msg)
    
    # Check warning count
    assert monitor.warning_count == 1
    
    # Get health status
    health = monitor.get_health_status()
    
    # Check health status
    assert health['status'] == "warning"
    assert health['last_warning'] == warning_msg
    
    # Check log file
    log_file = monitor.log_dir / "system_monitor.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert warning_msg in log_content

def test_metrics_summary(monitor):
    """Test generation of metrics summary."""
    # Record some metrics
    monitor.record_processing(processing_time=2.0, batch_size=200)
    monitor.record_error("Test error")
    monitor.record_warning("Test warning")
    
    # Get metrics summary
    metrics = monitor.get_metrics_summary()
    
    # Check summary metrics
    assert metrics['records_processed'] == 200
    assert metrics['error_count'] == 1
    assert metrics['warning_count'] == 1
    assert metrics['error_rate'] == 1 / 200
    assert metrics['warning_rate'] == 1 / 200
    assert metrics['processing_rate'] == 200 / 2.0

def test_monitor_cleanup(monitor):
    """Test proper cleanup of monitoring system."""
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Check monitoring stopped
    assert not monitor.monitoring_active
    
    # Check final metrics files
    metrics_file = monitor.log_dir / "final_metrics.json"
    health_file = monitor.log_dir / "final_health.json"
    
    assert metrics_file.exists()
    assert health_file.exists()
    
    # Check final metrics content
    with open(metrics_file, 'r') as f:
        final_metrics = json.load(f)
        assert isinstance(final_metrics, dict)
        assert 'timestamp' in final_metrics
    
    # Check final health content
    with open(health_file, 'r') as f:
        final_health = json.load(f)
        assert isinstance(final_health, dict)
        assert 'timestamp' in final_health
        assert 'status' in final_health 