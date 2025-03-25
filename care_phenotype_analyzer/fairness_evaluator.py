"""
Module for evaluating fairness using care phenotype labels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats

class FairnessEvaluator:
    """
    Evaluates fairness of healthcare algorithms using care phenotype labels.
    """
    
    def __init__(self, 
                 predictions: pd.Series,
                 true_labels: pd.Series,
                 phenotype_labels: pd.Series,
                 clinical_factors: Optional[pd.DataFrame] = None):
        """
        Initialize the fairness evaluator.
        
        Parameters
        ----------
        predictions : pd.Series
            Model predictions
        true_labels : pd.Series
            True labels
        phenotype_labels : pd.Series
            Care phenotype labels
        clinical_factors : pd.DataFrame, optional
            DataFrame containing clinical factors
        """
        self.predictions = predictions
        self.true_labels = true_labels
        self.phenotype_labels = phenotype_labels
        self.clinical_factors = clinical_factors
        
    def evaluate_fairness_metrics(self,
                                metrics: List[str]) -> Dict:
        """
        Evaluate fairness metrics across care phenotypes.
        
        Parameters
        ----------
        metrics : List[str]
            List of fairness metrics to calculate
            
        Returns
        -------
        Dict
            Dictionary containing fairness evaluation results
        """
        results = {}
        
        for metric in metrics:
            if metric == 'demographic_parity':
                results[metric] = self._calculate_demographic_parity()
            elif metric == 'equal_opportunity':
                results[metric] = self._calculate_equal_opportunity()
            elif metric == 'predictive_parity':
                results[metric] = self._calculate_predictive_parity()
            elif metric == 'treatment_equality':
                results[metric] = self._calculate_treatment_equality()
                
        return results
    
    def _calculate_demographic_parity(self) -> Dict:
        """
        Calculate demographic parity across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            results[phenotype] = {
                'positive_rate': np.mean(self.predictions[mask]),
                'sample_size': np.sum(mask)
            }
            
        return results
    
    def _calculate_equal_opportunity(self) -> Dict:
        """
        Calculate equal opportunity across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            true_positives = np.sum((self.predictions[mask] == 1) & 
                                  (self.true_labels[mask] == 1))
            total_positives = np.sum(self.true_labels[mask] == 1)
            
            results[phenotype] = {
                'true_positive_rate': true_positives / total_positives if total_positives > 0 else 0,
                'sample_size': np.sum(mask)
            }
            
        return results
    
    def _calculate_predictive_parity(self) -> Dict:
        """
        Calculate predictive parity across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            pred_positives = self.predictions[mask] == 1
            true_positives = self.true_labels[mask] == 1
            
            if np.sum(pred_positives) > 0:
                ppv = np.sum(pred_positives & true_positives) / np.sum(pred_positives)
            else:
                ppv = 0
                
            results[phenotype] = {
                'positive_predictive_value': ppv,
                'sample_size': np.sum(mask)
            }
            
        return results
    
    def _calculate_treatment_equality(self) -> Dict:
        """
        Calculate treatment equality across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            false_positives = np.sum((self.predictions[mask] == 1) & 
                                   (self.true_labels[mask] == 0))
            false_negatives = np.sum((self.predictions[mask] == 0) & 
                                   (self.true_labels[mask] == 1))
            
            results[phenotype] = {
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'ratio': false_positives / false_negatives if false_negatives > 0 else float('inf'),
                'sample_size': np.sum(mask)
            }
            
        return results
    
    def analyze_clinical_factors(self) -> Dict:
        """
        Analyze how clinical factors relate to fairness metrics.
        
        Returns
        -------
        Dict
            Dictionary containing clinical factor analysis results
        """
        if self.clinical_factors is None:
            return {}
            
        results = {}
        
        for factor in self.clinical_factors.columns:
            # Calculate correlation with prediction errors
            errors = (self.predictions != self.true_labels).astype(int)
            correlation = stats.pearsonr(self.clinical_factors[factor], errors)
            
            results[factor] = {
                'correlation': correlation[0],
                'p_value': correlation[1]
            }
            
        return results 