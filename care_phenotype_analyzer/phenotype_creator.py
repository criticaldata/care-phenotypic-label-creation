"""
Module for creating care phenotype labels based on observable care patterns.
Focuses on understanding variations in lab test measurements and care patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import logging
from datetime import datetime
import json
from pathlib import Path
import time
from .monitoring import SystemMonitor

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('care_phenotype_creator.log')
    ]
)
logger = logging.getLogger(__name__)

class CarePhenotypeError(Exception):
    """Base exception class for CarePhenotypeCreator errors."""
    pass

class DataValidationError(CarePhenotypeError):
    """Exception raised for data validation errors."""
    pass

class DataTypeError(CarePhenotypeError):
    """Exception raised for data type errors."""
    pass

class CarePhenotypeCreator:
    """
    Creates objective care phenotype labels based on observable care patterns.
    Focuses on understanding variations in lab test measurements and care patterns,
    accounting for legitimate clinical factors while identifying unexplained variations.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 clinical_factors: Optional[List[str]] = None,
                 n_clusters: int = 3,
                 random_state: int = 42,
                 log_dir: str = "logs"):
        """
        Initialize the phenotype creator with care data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing care pattern data
        clinical_factors : List[str], optional
            List of columns containing clinical factors (e.g., SOFA score, Charlson score)
        n_clusters : int
            Number of phenotype groups to create
        random_state : int
            Random state for reproducibility
        log_dir : str
            Directory for monitoring logs
            
        Raises
        ------
        DataValidationError
            If data is empty or clinical factors are not found in data
        DataTypeError
            If data types are incorrect
        """
        self.data = data
        self.clinical_factors = clinical_factors or []
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize monitoring system
        self.monitor = SystemMonitor(log_dir=log_dir)
        
        # Log initialization
        self.monitor.logger.info(
            f"Initialized CarePhenotypeCreator with {len(data)} records "
            f"and {len(self.clinical_factors)} clinical factors"
        )
        
    def create_phenotype_labels(self) -> pd.Series:
        """Create phenotype labels based on care patterns.
        
        Returns:
            Series containing phenotype labels
        """
        start_time = time.time()
        
        try:
            # Preprocess data
            self.monitor.logger.info("Starting data preprocessing")
            preprocessed_data = self._preprocess_data()
            
            # Cluster data
            self.monitor.logger.info("Starting data clustering")
            clusters = self._cluster_data(preprocessed_data)
            
            # Create labels
            self.monitor.logger.info("Creating phenotype labels")
            labels = pd.Series(clusters, index=self.data.index)
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self.monitor.record_processing(
                processing_time=processing_time,
                batch_size=len(self.data)
            )
            
            # Log success
            self.monitor.logger.info(
                f"Successfully created phenotype labels for {len(labels)} records "
                f"in {processing_time:.2f} seconds"
            )
            
            return labels
            
        except Exception as e:
            # Record error
            error_msg = f"Error creating phenotype labels: {str(e)}"
            self.monitor.record_error(error_msg)
            raise CarePhenotypeError(error_msg)
            
    def _preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data for clustering.
        
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Select features for clustering
            features = self.data.select_dtypes(include=[np.number]).columns
            if self.clinical_factors:
                features = [f for f in features if f not in self.clinical_factors]
                
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data[features])
            
            return pd.DataFrame(scaled_data, columns=features)
            
        except Exception as e:
            error_msg = f"Error preprocessing data: {str(e)}"
            self.monitor.record_error(error_msg)
            raise CarePhenotypeError(error_msg)
            
    def _cluster_data(self, data: pd.DataFrame) -> np.ndarray:
        """Cluster the preprocessed data.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            Array of cluster labels
        """
        try:
            # Perform clustering
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )
            clusters = kmeans.fit_predict(data)
            
            # Log clustering results
            cluster_sizes = pd.Series(clusters).value_counts()
            self.monitor.logger.info(
                f"Clustering complete. Cluster sizes: {cluster_sizes.to_dict()}"
            )
            
            return clusters
            
        except Exception as e:
            error_msg = f"Error clustering data: {str(e)}"
            self.monitor.record_error(error_msg)
            raise CarePhenotypeError(error_msg)
            
    def analyze_clinical_separation(self) -> Dict[str, float]:
        """Analyze clinical separation between phenotypes.
        
        Returns:
            Dictionary of separation metrics
        """
        if not self.clinical_factors:
            warning_msg = "No clinical factors provided for separation analysis"
            self.monitor.record_warning(warning_msg)
            return {}
            
        try:
            separation_metrics = {}
            
            for factor in self.clinical_factors:
                # Calculate separation for each clinical factor
                f_stat, p_value = stats.f_oneway(*[
                    self.data[self.data['phenotype'] == i][factor]
                    for i in range(self.n_clusters)
                ])
                
                separation_metrics[factor] = {
                    'f_statistic': f_stat,
                    'p_value': p_value
                }
                
            # Log separation results
            self.monitor.logger.info(
                f"Clinical separation analysis complete. "
                f"Factors analyzed: {list(separation_metrics.keys())}"
            )
            
            return separation_metrics
            
        except Exception as e:
            error_msg = f"Error analyzing clinical separation: {str(e)}"
            self.monitor.record_error(error_msg)
            raise CarePhenotypeError(error_msg)
            
    def analyze_unexplained_variation(self) -> Dict[str, float]:
        """Analyze unexplained variation in care patterns.
        
        Returns:
            Dictionary of variation metrics
        """
        try:
            variation_metrics = {}
            
            # Calculate total variation
            total_var = self.data.var()
            
            # Calculate explained variation (by clinical factors)
            if self.clinical_factors:
                clinical_data = self.data[self.clinical_factors]
                explained_var = clinical_data.var()
                
                # Calculate unexplained variation
                unexplained_var = total_var - explained_var
                
                variation_metrics = {
                    'total_variation': total_var,
                    'explained_variation': explained_var,
                    'unexplained_variation': unexplained_var
                }
                
            # Log variation results
            self.monitor.logger.info(
                f"Variation analysis complete. "
                f"Clinical factors considered: {len(self.clinical_factors)}"
            )
            
            return variation_metrics
            
        except Exception as e:
            error_msg = f"Error analyzing unexplained variation: {str(e)}"
            self.monitor.record_error(error_msg)
            raise CarePhenotypeError(error_msg)
            
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        
    def _validate_input_data(self, 
                           data: pd.DataFrame, 
                           clinical_factors: Optional[List[str]] = None) -> None:
        """
        Validate input data and clinical factors.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing care pattern data
        clinical_factors : List[str], optional
            List of clinical factor column names
            
        Raises
        ------
        DataValidationError
            If data is empty or clinical factors are not found in data
        """
        logger.debug("Starting input data validation")
        
        try:
            # Check if data is empty
            if data.empty:
                raise DataValidationError("Input data cannot be empty")
                
            # Check if clinical factors exist in data
            if clinical_factors:
                missing_factors = [factor for factor in clinical_factors 
                                 if factor not in data.columns]
                if missing_factors:
                    raise DataValidationError(f"Clinical factors not found in data: {missing_factors}")
                    
            # Check for required columns
            required_columns = ['subject_id', 'timestamp']
            missing_columns = [col for col in required_columns 
                             if col not in data.columns]
            if missing_columns:
                raise DataValidationError(f"Required columns missing from data: {missing_columns}")
                
            # Check for non-numeric columns in clinical factors
            if clinical_factors:
                non_numeric = [factor for factor in clinical_factors 
                              if not pd.api.types.is_numeric_dtype(data[factor])]
                if non_numeric:
                    raise DataValidationError(f"Clinical factors must be numeric: {non_numeric}")
                    
            logger.info("Input data validation successful")
            
        except Exception as e:
            self._log_error(e, "input data validation")
            raise
        
    def _validate_numeric_data(self, data: Union[pd.DataFrame, pd.Series], 
                             name: str) -> None:
        """
        Validate numeric data types and values.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series]
            Data to validate
        name : str
            Name of the data for error messages
            
        Raises
        ------
        TypeError
            If data is not numeric
        ValueError
            If data contains invalid values
        """
        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise TypeError(f"Column '{col}' in {name} must be numeric")
                if data[col].isna().any():
                    raise ValueError(f"Column '{col}' in {name} contains NaN values")
        else:
            if not pd.api.types.is_numeric_dtype(data):
                raise TypeError(f"{name} must be numeric")
            if data.isna().any():
                raise ValueError(f"{name} contains NaN values")
                
    def _validate_care_patterns(self, care_patterns: List[str]) -> None:
        """
        Validate care pattern columns.
        
        Parameters
        ----------
        care_patterns : List[str]
            List of care pattern column names
            
        Raises
        ------
        ValueError
            If care patterns are not found in data or are not numeric
        TypeError
            If data types are incorrect
        """
        # Check if care patterns exist in data
        missing_patterns = [pattern for pattern in care_patterns 
                          if pattern not in self.data.columns]
        if missing_patterns:
            raise ValueError(f"Care patterns not found in data: {missing_patterns}")
            
        # Validate numeric data
        self._validate_numeric_data(self.data[care_patterns], "care patterns")
            
        # Check for negative values
        for pattern in care_patterns:
            if (self.data[pattern] < 0).any():
                raise ValueError(f"Care pattern '{pattern}' contains negative values")
                
        logger.info("Care patterns validation successful")
        
    def _validate_phenotype_labels(self, labels: pd.Series) -> None:
        """
        Validate phenotype labels.
        
        Parameters
        ----------
        labels : pd.Series
            Series containing phenotype labels
            
        Raises
        ------
        ValueError
            If labels are invalid or don't match data length
        TypeError
            If data types are incorrect
        """
        # Check if labels match data length
        if len(labels) != len(self.data):
            raise ValueError("Labels length must match data length")
            
        # Validate numeric data
        self._validate_numeric_data(labels, "phenotype labels")
            
        # Check for negative values
        if (labels < 0).any():
            raise ValueError("Labels cannot contain negative values")
            
        # Check for integer values
        if not labels.dtype.kind in 'iu':  # 'i' for signed integer, 'u' for unsigned integer
            raise TypeError("Labels must be integers")
            
        logger.info("Phenotype labels validation successful")
        
    def _adjust_for_clinical_factors(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust care patterns for clinical factors that may justify variations.
        Uses regression analysis to account for legitimate clinical factors.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing care pattern measurements
            
        Returns
        -------
        pd.DataFrame
            Adjusted care pattern measurements (residuals after accounting for clinical factors)
        """
        adjusted_X = X.copy()
        
        for pattern in X.columns:
            # Create regression model for each care pattern
            X_clinical = self.data[self.clinical_factors]
            y = X[pattern]
            
            # Fit linear regression
            model = stats.linregress(X_clinical, y)
            
            # Calculate residuals (unexplained variation)
            predicted = model.predict(X_clinical)
            residuals = y - predicted
            
            adjusted_X[pattern] = residuals
            
        return adjusted_X
    
    def analyze_unexplained_variation(self,
                                    care_pattern: str,
                                    phenotype_labels: pd.Series) -> Dict:
        """
        Analyze unexplained variation in care patterns across phenotypes.
        
        Parameters
        ----------
        care_pattern : str
            Column containing the care pattern to analyze
        phenotype_labels : pd.Series
            Created phenotype labels
            
        Returns
        -------
        Dict
            Dictionary containing analysis of unexplained variation
        """
        results = {}
        
        # Calculate variation within each phenotype
        for phenotype in phenotype_labels.unique():
            mask = phenotype_labels == phenotype
            pattern_data = self.data.loc[mask, care_pattern]
            
            results[phenotype] = {
                'mean': pattern_data.mean(),
                'std': pattern_data.std(),
                'sample_size': len(pattern_data),
                'unexplained_variance': pattern_data.var()
            }
            
        # Calculate statistical significance of variation
        f_stat, p_value = stats.f_oneway(*[
            self.data.loc[phenotype_labels == p, care_pattern]
            for p in phenotype_labels.unique()
        ])
        
        results['f_statistic'] = f_stat
        results['p_value'] = p_value
        
        return results
    
    def validate_phenotypes(self,
                          labels: pd.Series,
                          validation_metrics: List[str]) -> Dict:
        """
        Validate created phenotype labels using various metrics.
        
        Parameters
        ----------
        labels : pd.Series
            Created phenotype labels
        validation_metrics : List[str]
            List of validation metrics to calculate
            
        Returns
        -------
        Dict
            Dictionary containing validation results
        """
        results = {}
        
        # Calculate validation metrics
        for metric in validation_metrics:
            if metric == 'clinical_separation':
                results[metric] = self._check_clinical_separation(labels)
            elif metric == 'pattern_consistency':
                results[metric] = self._check_pattern_consistency(labels)
            elif metric == 'unexplained_variation':
                results[metric] = self._check_unexplained_variation(labels)
            
        return results
    
    def _check_clinical_separation(self, labels: pd.Series) -> Dict:
        """
        Check if phenotypes show meaningful separation in clinical factors.
        """
        results = {}
        
        for factor in self.clinical_factors:
            f_stat, p_value = stats.f_oneway(*[
                self.data.loc[labels == p, factor]
                for p in labels.unique()
            ])
            
            results[factor] = {
                'f_statistic': f_stat,
                'p_value': p_value
            }
            
        return results
    
    def _check_pattern_consistency(self, labels: pd.Series) -> Dict:
        """
        Check if phenotypes show consistent patterns across different care measures.
        
        Parameters
        ----------
        labels : pd.Series
            Created phenotype labels
            
        Returns
        -------
        Dict
            Dictionary containing pattern consistency metrics
        """
        results = {}
        
        # Get all care pattern columns (excluding clinical factors)
        care_patterns = [col for col in self.data.columns 
                        if col not in self.clinical_factors]
        
        # Calculate consistency metrics for each phenotype
        for phenotype in labels.unique():
            mask = labels == phenotype
            phenotype_data = self.data.loc[mask, care_patterns]
            
            # Calculate correlation matrix between patterns
            corr_matrix = phenotype_data.corr()
            
            # Calculate mean absolute correlation
            mean_corr = np.abs(corr_matrix).mean().mean()
            
            # Calculate pattern stability (variance of correlations)
            pattern_stability = corr_matrix.var().mean()
            
            results[phenotype] = {
                'mean_correlation': mean_corr,
                'pattern_stability': pattern_stability,
                'sample_size': np.sum(mask),
                'num_patterns': len(care_patterns)
            }
            
        # Calculate overall consistency metrics
        results['overall'] = {
            'mean_pattern_correlation': np.mean([r['mean_correlation'] 
                                               for r in results.values()]),
            'mean_pattern_stability': np.mean([r['pattern_stability'] 
                                             for r in results.values()]),
            'total_phenotypes': len(results)
        }
        
        return results
    
    def _check_unexplained_variation(self, labels: pd.Series) -> Dict:
        """
        Check the amount of unexplained variation in care patterns.
        
        Parameters
        ----------
        labels : pd.Series
            Created phenotype labels
            
        Returns
        -------
        Dict
            Dictionary containing unexplained variation metrics
        """
        results = {}
        
        # Get all care pattern columns (excluding clinical factors)
        care_patterns = [col for col in self.data.columns 
                        if col not in self.clinical_factors]
        
        # Calculate unexplained variation for each phenotype
        for phenotype in labels.unique():
            mask = labels == phenotype
            phenotype_data = self.data.loc[mask, care_patterns]
            
            # Calculate total variance
            total_variance = phenotype_data.var().mean()
            
            # Calculate explained variance by clinical factors
            explained_variance = 0
            if self.clinical_factors:
                for pattern in care_patterns:
                    X = self.data.loc[mask, self.clinical_factors]
                    y = self.data.loc[mask, pattern]
                    
                    # Fit linear regression
                    model = stats.linregress(X, y)
                    predicted = model.predict(X)
                    
                    # Calculate R-squared
                    r_squared = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
                    explained_variance += r_squared * phenotype_data[pattern].var()
                
                explained_variance /= len(care_patterns)
            
            # Calculate unexplained variance
            unexplained_variance = total_variance - explained_variance
            
            results[phenotype] = {
                'total_variance': total_variance,
                'explained_variance': explained_variance,
                'unexplained_variance': unexplained_variance,
                'unexplained_ratio': unexplained_variance / total_variance if total_variance > 0 else 0,
                'sample_size': np.sum(mask)
            }
        
        # Calculate overall variation metrics
        results['overall'] = {
            'mean_unexplained_ratio': np.mean([r['unexplained_ratio'] 
                                             for r in results.values()]),
            'total_phenotypes': len(results),
            'total_samples': len(labels)
        }
        
        return results
    
    def preprocess_data(self,
                       columns: Optional[List[str]] = None,
                       handle_missing: bool = True,
                       handle_outliers: bool = True,
                       normalize: bool = True,
                       missing_strategy: Literal['drop', 'mean', 'median', 'mode'] = 'mean',
                       outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values, outliers, and normalization.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to preprocess. If None, all numeric columns are processed.
        handle_missing : bool
            Whether to handle missing values
        handle_outliers : bool
            Whether to handle outliers
        normalize : bool
            Whether to normalize the data
        missing_strategy : Literal['drop', 'mean', 'median', 'mode']
            Strategy for handling missing values
        outlier_threshold : float
            Z-score threshold for outlier detection
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data
            
        Raises
        ------
        ValueError
            If invalid strategy is provided or data is empty
        """
        logger.info("Starting data preprocessing")
        
        try:
            # Select columns to process
            if columns is None:
                columns = [col for col in self.data.columns 
                          if pd.api.types.is_numeric_dtype(self.data[col])]
            
            if not columns:
                raise ValueError("No numeric columns found for preprocessing")
                
            data = self.data[columns].copy()
            
            # Handle missing values
            if handle_missing:
                data = self._handle_missing_values(data, strategy=missing_strategy)
                
            # Handle outliers
            if handle_outliers:
                data = self._handle_outliers(data, threshold=outlier_threshold)
                
            # Normalize data
            if normalize:
                data = self._normalize_data(data)
                
            logger.info("Data preprocessing completed successfully")
            return data
            
        except Exception as e:
            self._log_error(e, "data preprocessing")
            raise
            
    def _handle_missing_values(self,
                             data: pd.DataFrame,
                             strategy: Literal['drop', 'mean', 'median', 'mode'] = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing missing values
        strategy : Literal['drop', 'mean', 'median', 'mode']
            Strategy for handling missing values
            
        Returns
        -------
        pd.DataFrame
            Data with handled missing values
        """
        logger.debug(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'drop':
            data = data.dropna()
        elif strategy == 'mean':
            data = data.fillna(data.mean())
        elif strategy == 'median':
            data = data.fillna(data.median())
        elif strategy == 'mode':
            data = data.fillna(data.mode().iloc[0])
        else:
            raise ValueError(f"Invalid missing value strategy: {strategy}")
            
        logger.info(f"Missing values handled using {strategy} strategy")
        return data
        
    def _handle_outliers(self,
                        data: pd.DataFrame,
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the data using z-score method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing potential outliers
        threshold : float
            Z-score threshold for outlier detection
            
        Returns
        -------
        pd.DataFrame
            Data with handled outliers
        """
        logger.debug(f"Handling outliers using z-score threshold: {threshold}")
        
        # Calculate z-scores
        z_scores = np.abs((data - data.mean()) / data.std())
        
        # Replace outliers with mean
        data_clean = data.copy()
        for col in data.columns:
            outliers = z_scores[col] > threshold
            if outliers.any():
                data_clean.loc[outliers, col] = data[col].mean()
                
        logger.info(f"Outliers handled using z-score threshold {threshold}")
        return data_clean
        
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data using standardization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to normalize
            
        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        logger.debug("Normalizing data using standardization")
        
        # Fit and transform the data
        normalized_data = pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        logger.info("Data normalized successfully")
        return normalized_data 