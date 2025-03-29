"""
Module for exporting data and results in various formats.

This module provides functionality to export care patterns, analysis results,
and visualizations in different formats for further analysis or sharing.
"""

import pandas as pd
import numpy as np
import json
import csv
import pickle
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
from datetime import datetime
import plotly.graph_objects as go
from .monitoring import SystemMonitor

class DataExporter:
    """Class for exporting data and results in various formats."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the data exporter.
        
        Args:
            log_dir: Directory for monitoring logs
        """
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Log initialization
        self.monitor.logger.info("Initialized DataExporter")
        
    def export_to_csv(self,
                     data: Union[pd.DataFrame, pd.Series],
                     output_file: str,
                     index: bool = True) -> None:
        """Export data to CSV format.
        
        Args:
            data: DataFrame or Series to export
            output_file: Path to save the CSV file
            index: Whether to include the index
        """
        start_time = time.time()
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to CSV
            data.to_csv(output_file, index=index)
            
            # Log success
            self.monitor.logger.info(f"Exported data to CSV: {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(data)
            )
            
        except Exception as e:
            error_msg = f"Error exporting to CSV: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def export_to_json(self,
                      data: Union[Dict, List, pd.DataFrame, pd.Series],
                      output_file: str,
                      orient: str = 'records') -> None:
        """Export data to JSON format.
        
        Args:
            data: Data to export
            output_file: Path to save the JSON file
            orient: JSON orientation for DataFrame/Series
        """
        start_time = time.time()
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert DataFrame/Series to dict if needed
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.to_dict(orient=orient)
            
            # Export to JSON
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Log success
            self.monitor.logger.info(f"Exported data to JSON: {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(data) if isinstance(data, (list, pd.DataFrame, pd.Series)) else 1
            )
            
        except Exception as e:
            error_msg = f"Error exporting to JSON: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def export_to_hdf5(self,
                      data: Dict[str, Union[pd.DataFrame, np.ndarray]],
                      output_file: str) -> None:
        """Export data to HDF5 format.
        
        Args:
            data: Dictionary of DataFrames or arrays to export
            output_file: Path to save the HDF5 file
        """
        start_time = time.time()
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to HDF5
            with h5py.File(output_file, 'w') as f:
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        # Store DataFrame metadata
                        f.create_dataset(f"{key}/columns", data=list(value.columns))
                        f.create_dataset(f"{key}/index", data=list(value.index))
                        # Store DataFrame values
                        for col in value.columns:
                            f.create_dataset(f"{key}/data/{col}", data=value[col].values)
                    else:
                        f.create_dataset(key, data=value)
            
            # Log success
            self.monitor.logger.info(f"Exported data to HDF5: {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=sum(len(v) for v in data.values() if isinstance(v, (pd.DataFrame, np.ndarray)))
            )
            
        except Exception as e:
            error_msg = f"Error exporting to HDF5: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def export_to_pickle(self,
                        data: Any,
                        output_file: str) -> None:
        """Export data to pickle format.
        
        Args:
            data: Any Python object to export
            output_file: Path to save the pickle file
        """
        start_time = time.time()
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to pickle
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Log success
            self.monitor.logger.info(f"Exported data to pickle: {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=1  # Single object
            )
            
        except Exception as e:
            error_msg = f"Error exporting to pickle: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def export_visualization(self,
                           fig: go.Figure,
                           output_file: str,
                           format: str = 'html') -> None:
        """Export visualization to various formats.
        
        Args:
            fig: Plotly figure object
            output_file: Path to save the visualization
            format: Export format ('html', 'png', 'pdf', 'svg')
        """
        start_time = time.time()
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format == 'html':
                fig.write_html(output_file)
            elif format == 'png':
                fig.write_image(output_file)
            elif format == 'pdf':
                fig.write_image(output_file, format='pdf')
            elif format == 'svg':
                fig.write_image(output_file, format='svg')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Log success
            self.monitor.logger.info(f"Exported visualization to {format}: {output_file}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=1  # Single visualization
            )
            
        except Exception as e:
            error_msg = f"Error exporting visualization: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def export_batch(self,
                    data_dict: Dict[str, Any],
                    output_dir: str,
                    formats: Optional[List[str]] = None) -> None:
        """Export multiple data objects in various formats.
        
        Args:
            data_dict: Dictionary of data objects to export
            output_dir: Directory to save the exports
            formats: List of formats to export (default: all supported)
        """
        start_time = time.time()
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Default formats if none specified
            if formats is None:
                formats = ['csv', 'json', 'hdf5', 'pickle']
            
            # Export each data object in each format
            for name, data in data_dict.items():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for fmt in formats:
                    output_file = output_path / f"{name}_{timestamp}.{fmt}"
                    
                    if fmt == 'csv' and isinstance(data, (pd.DataFrame, pd.Series)):
                        self.export_to_csv(data, str(output_file))
                    elif fmt == 'json':
                        self.export_to_json(data, str(output_file))
                    elif fmt == 'hdf5' and isinstance(data, dict):
                        self.export_to_hdf5(data, str(output_file))
                    elif fmt == 'pickle':
                        self.export_to_pickle(data, str(output_file))
            
            # Log success
            self.monitor.logger.info(f"Exported batch to {output_dir}")
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(data_dict) * len(formats)
            )
            
        except Exception as e:
            error_msg = f"Error exporting batch: {str(e)}"
            self.monitor.record_error(error_msg)
            raise
            
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring() 