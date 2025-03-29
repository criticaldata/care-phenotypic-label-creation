"""
Module for evaluating fairness using care phenotype labels.
Focuses on understanding how care patterns may reflect disparities in healthcare delivery.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class FairnessEvaluator:
    """
    Evaluates fairness of healthcare algorithms using care phenotype labels.
    Focuses on understanding how care patterns may reflect disparities in healthcare delivery,
    accounting for legitimate clinical factors while identifying potential biases.
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
            DataFrame containing clinical factors (e.g., SOFA score, Charlson score)
        """
        self.predictions = predictions
        self.true_labels = true_labels
        self.phenotype_labels = phenotype_labels
        self.clinical_factors = clinical_factors
        
    def evaluate_fairness_metrics(self,
                                metrics: List[str],
                                adjust_for_clinical: bool = True) -> Dict:
        """
        Evaluate fairness metrics across care phenotypes.
        Optionally adjusts for clinical factors to identify unexplained disparities.
        
        Parameters
        ----------
        metrics : List[str]
            List of fairness metrics to calculate
        adjust_for_clinical : bool
            Whether to adjust for clinical factors before calculating metrics
            
        Returns
        -------
        Dict
            Dictionary containing fairness evaluation results
        """
        results = {}
        
        # Adjust predictions for clinical factors if specified
        if adjust_for_clinical and self.clinical_factors is not None:
            adjusted_predictions = self._adjust_for_clinical_factors()
        else:
            adjusted_predictions = self.predictions
            
        for metric in metrics:
            if metric == 'demographic_parity':
                results[metric] = self._calculate_demographic_parity(adjusted_predictions)
            elif metric == 'equal_opportunity':
                results[metric] = self._calculate_equal_opportunity(adjusted_predictions)
            elif metric == 'predictive_parity':
                results[metric] = self._calculate_predictive_parity(adjusted_predictions)
            elif metric == 'treatment_equality':
                results[metric] = self._calculate_treatment_equality(adjusted_predictions)
            elif metric == 'care_pattern_disparity':
                results[metric] = self._calculate_care_pattern_disparity()
                
        return results
    
    def _adjust_for_clinical_factors(self) -> pd.Series:
        """
        Adjust predictions for clinical factors to identify unexplained disparities.
        
        Returns
        -------
        pd.Series
            Adjusted predictions
        """
        # Create regression model
        X = self.clinical_factors
        y = self.predictions
        
        # Fit linear regression
        model = stats.linregress(X, y)
        
        # Calculate residuals (unexplained variation)
        predicted = model.predict(X)
        residuals = y - predicted
        
        return residuals
    
    def _calculate_demographic_parity(self, predictions: pd.Series) -> Dict:
        """
        Calculate demographic parity across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            results[phenotype] = {
                'positive_rate': np.mean(predictions[mask]),
                'sample_size': np.sum(mask),
                'unexplained_variation': np.var(predictions[mask])
            }
            
        return results
    
    def _calculate_equal_opportunity(self, predictions: pd.Series) -> Dict:
        """
        Calculate equal opportunity across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            true_positives = np.sum((predictions[mask] == 1) & 
                                  (self.true_labels[mask] == 1))
            total_positives = np.sum(self.true_labels[mask] == 1)
            
            results[phenotype] = {
                'true_positive_rate': true_positives / total_positives if total_positives > 0 else 0,
                'sample_size': np.sum(mask),
                'total_positives': total_positives
            }
            
        return results
    
    def _calculate_predictive_parity(self, predictions: pd.Series) -> Dict:
        """
        Calculate predictive parity across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            pred_positives = predictions[mask] == 1
            true_positives = self.true_labels[mask] == 1
            
            if np.sum(pred_positives) > 0:
                ppv = np.sum(pred_positives & true_positives) / np.sum(pred_positives)
            else:
                ppv = 0
                
            results[phenotype] = {
                'positive_predictive_value': ppv,
                'sample_size': np.sum(mask),
                'predicted_positives': np.sum(pred_positives)
            }
            
        return results
    
    def _calculate_treatment_equality(self, predictions: pd.Series) -> Dict:
        """
        Calculate treatment equality across phenotypes.
        """
        results = {}
        
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            false_positives = np.sum((predictions[mask] == 1) & 
                                   (self.true_labels[mask] == 0))
            false_negatives = np.sum((predictions[mask] == 0) & 
                                   (self.true_labels[mask] == 1))
            
            results[phenotype] = {
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'ratio': false_positives / false_negatives if false_negatives > 0 else float('inf'),
                'sample_size': np.sum(mask)
            }
            
        return results
    
    def _calculate_care_pattern_disparity(self) -> Dict:
        """
        Calculate disparities in care patterns across phenotypes.
        """
        if self.clinical_factors is None:
            return {}
            
        results = {}
        
        for factor in self.clinical_factors.columns:
            # Calculate correlation with prediction errors
            errors = (self.predictions != self.true_labels).astype(int)
            correlation = stats.pearsonr(self.clinical_factors[factor], errors)
            
            # Calculate unexplained variation
            model = stats.linregress(self.clinical_factors[factor], errors)
            unexplained = errors - model.predict(self.clinical_factors[factor])
            
            results[factor] = {
                'correlation': correlation[0],
                'p_value': correlation[1],
                'unexplained_variation': np.var(unexplained)
            }
            
        return results
    
    def analyze_clinical_factors(self) -> Dict:
        """
        Analyze how clinical factors relate to fairness metrics.
        """
        if self.clinical_factors is None:
            return {}
            
        results = {}
        
        for factor in self.clinical_factors.columns:
            # Calculate correlation with prediction errors
            errors = (self.predictions != self.true_labels).astype(int)
            correlation = stats.pearsonr(self.clinical_factors[factor], errors)
            
            # Calculate unexplained variation
            model = stats.linregress(self.clinical_factors[factor], errors)
            unexplained = errors - model.predict(self.clinical_factors[factor])
            
            results[factor] = {
                'correlation': correlation[0],
                'p_value': correlation[1],
                'unexplained_variation': np.var(unexplained)
            }
            
        return results
    
    def visualize_fairness_metrics(self,
                                 metrics: List[str],
                                 output_file: Optional[str] = None) -> None:
        """
        Visualize fairness metrics across phenotypes.
        
        Parameters
        ----------
        metrics : List[str]
            List of fairness metrics to visualize
        output_file : str, optional
            Path to save the visualization
        """
        results = self.evaluate_fairness_metrics(metrics)
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
            
        for i, (metric, ax) in enumerate(zip(metrics, axes)):
            if metric == 'demographic_parity':
                self._plot_demographic_parity(results[metric], ax)
            elif metric == 'equal_opportunity':
                self._plot_equal_opportunity(results[metric], ax)
            elif metric == 'predictive_parity':
                self._plot_predictive_parity(results[metric], ax)
            elif metric == 'treatment_equality':
                self._plot_treatment_equality(results[metric], ax)
            elif metric == 'care_pattern_disparity':
                self._plot_care_pattern_disparity(results[metric], ax)
                
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        plt.show()
        
    def _plot_demographic_parity(self, results: Dict, ax: plt.Axes) -> None:
        """Plot demographic parity results."""
        phenotypes = list(results.keys())
        positive_rates = [results[p]['positive_rate'] for p in phenotypes]
        
        ax.bar(phenotypes, positive_rates)
        ax.set_title('Demographic Parity')
        ax.set_xlabel('Phenotype')
        ax.set_ylabel('Positive Rate')
        
        # Add error bars for unexplained variation
        unexplained = [results[p]['unexplained_variation'] for p in phenotypes]
        ax.errorbar(phenotypes, positive_rates, yerr=np.sqrt(unexplained),
                   fmt='none', color='black', capsize=5)
        
    def _plot_equal_opportunity(self, results: Dict, ax: plt.Axes) -> None:
        """Plot equal opportunity results."""
        phenotypes = list(results.keys())
        tprs = [results[p]['true_positive_rate'] for p in phenotypes]
        
        ax.bar(phenotypes, tprs)
        ax.set_title('Equal Opportunity')
        ax.set_xlabel('Phenotype')
        ax.set_ylabel('True Positive Rate')
        
    def _plot_predictive_parity(self, results: Dict, ax: plt.Axes) -> None:
        """Plot predictive parity results."""
        phenotypes = list(results.keys())
        ppvs = [results[p]['positive_predictive_value'] for p in phenotypes]
        
        ax.bar(phenotypes, ppvs)
        ax.set_title('Predictive Parity')
        ax.set_xlabel('Phenotype')
        ax.set_ylabel('Positive Predictive Value')
        
    def _plot_treatment_equality(self, results: Dict, ax: plt.Axes) -> None:
        """Plot treatment equality results."""
        phenotypes = list(results.keys())
        ratios = [results[p]['ratio'] for p in phenotypes]
        
        ax.bar(phenotypes, ratios)
        ax.set_title('Treatment Equality')
        ax.set_xlabel('Phenotype')
        ax.set_ylabel('False Positive/False Negative Ratio')
        
    def _plot_care_pattern_disparity(self, results: Dict, ax: plt.Axes) -> None:
        """Plot care pattern disparity results."""
        factors = list(results.keys())
        correlations = [results[f]['correlation'] for f in factors]
        p_values = [results[f]['p_value'] for f in factors]
        
        # Create bar plot
        bars = ax.bar(factors, correlations)
        ax.set_title('Care Pattern Disparity')
        ax.set_xlabel('Clinical Factor')
        ax.set_ylabel('Correlation with Errors')
        
        # Add significance markers
        for i, p_value in enumerate(p_values):
            if p_value < 0.05:
                bars[i].set_color('red')
                ax.text(i, correlations[i], '*', ha='center', va='bottom')
                
    def visualize_bias_detection(self,
                               output_file: Optional[str] = None) -> None:
        """
        Visualize bias detection results.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the visualization
        """
        # Get bias detection results
        clinical_results = self.analyze_clinical_factors()
        fairness_results = self.evaluate_fairness_metrics(
            ['demographic_parity', 'equal_opportunity', 'care_pattern_disparity']
        )
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot clinical factor correlations
        factors = list(clinical_results.keys())
        correlations = [clinical_results[f]['correlation'] for f in factors]
        p_values = [clinical_results[f]['p_value'] for f in factors]
        
        bars = ax1.bar(factors, correlations)
        ax1.set_title('Clinical Factor Bias')
        ax1.set_xlabel('Clinical Factor')
        ax1.set_ylabel('Correlation with Errors')
        
        # Add significance markers
        for i, p_value in enumerate(p_values):
            if p_value < 0.05:
                bars[i].set_color('red')
                ax1.text(i, correlations[i], '*', ha='center', va='bottom')
                
        # Plot fairness metrics
        phenotypes = list(fairness_results['demographic_parity'].keys())
        demographic_parity = [fairness_results['demographic_parity'][p]['positive_rate'] 
                            for p in phenotypes]
        equal_opportunity = [fairness_results['equal_opportunity'][p]['true_positive_rate'] 
                           for p in phenotypes]
        
        x = np.arange(len(phenotypes))
        width = 0.35
        
        ax2.bar(x - width/2, demographic_parity, width, label='Demographic Parity')
        ax2.bar(x + width/2, equal_opportunity, width, label='Equal Opportunity')
        ax2.set_title('Fairness Metrics by Phenotype')
        ax2.set_xlabel('Phenotype')
        ax2.set_ylabel('Rate')
        ax2.set_xticks(x)
        ax2.set_xticklabels(phenotypes)
        ax2.legend()
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        plt.show()
        
    def mitigate_bias(self,
                     strategy: str = 'reweighting',
                     target_metric: str = 'demographic_parity') -> pd.Series:
        """
        Apply bias mitigation strategies to predictions.
        
        Parameters
        ----------
        strategy : str
            Mitigation strategy to use ('reweighting', 'threshold_adjustment', 'calibration')
        target_metric : str
            Fairness metric to optimize for
            
        Returns
        -------
        pd.Series
            Mitigated predictions
        """
        if strategy == 'reweighting':
            return self._apply_reweighting(target_metric)
        elif strategy == 'threshold_adjustment':
            return self._apply_threshold_adjustment(target_metric)
        elif strategy == 'calibration':
            return self._apply_calibration(target_metric)
        else:
            raise ValueError(f"Unknown mitigation strategy: {strategy}")
            
    def _apply_reweighting(self, target_metric: str) -> pd.Series:
        """Apply reweighting strategy to mitigate bias."""
        # Calculate weights based on phenotype distribution
        phenotype_counts = self.phenotype_labels.value_counts()
        weights = 1 / phenotype_counts[self.phenotype_labels]
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Apply weights to predictions
        mitigated_predictions = self.predictions * weights
        
        return mitigated_predictions
        
    def _apply_threshold_adjustment(self, target_metric: str) -> pd.Series:
        """Apply threshold adjustment strategy to mitigate bias."""
        mitigated_predictions = self.predictions.copy()
        
        # Calculate optimal thresholds for each phenotype
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            if target_metric == 'demographic_parity':
                # Adjust threshold to match overall positive rate
                target_rate = np.mean(self.predictions)
                current_rate = np.mean(self.predictions[mask])
                threshold = np.percentile(self.predictions[mask], 
                                       100 * (1 - target_rate))
                mitigated_predictions[mask] = (self.predictions[mask] > threshold).astype(int)
                
        return mitigated_predictions
        
    def _apply_calibration(self, target_metric: str) -> pd.Series:
        """Apply calibration strategy to mitigate bias."""
        mitigated_predictions = self.predictions.copy()
        
        # Calibrate predictions for each phenotype
        for phenotype in self.phenotype_labels.unique():
            mask = self.phenotype_labels == phenotype
            if target_metric == 'demographic_parity':
                # Calibrate to match overall positive rate
                target_rate = np.mean(self.predictions)
                current_rate = np.mean(self.predictions[mask])
                calibration_factor = target_rate / current_rate
                mitigated_predictions[mask] = np.clip(
                    self.predictions[mask] * calibration_factor, 0, 1
                )
                
        return mitigated_predictions
        
    def visualize_bias_mitigation(self,
                                strategies: List[str] = ['reweighting', 'threshold_adjustment', 'calibration'],
                                output_file: Optional[str] = None) -> None:
        """
        Visualize the effects of different bias mitigation strategies.
        
        Parameters
        ----------
        strategies : List[str]
            List of mitigation strategies to compare
        output_file : str, optional
            Path to save the visualization
        """
        # Calculate original fairness metrics
        original_metrics = self.evaluate_fairness_metrics(['demographic_parity'])
        
        # Calculate metrics after each mitigation strategy
        mitigated_metrics = {}
        for strategy in strategies:
            mitigated_predictions = self.mitigate_bias(strategy)
            # Temporarily replace predictions to calculate metrics
            original_predictions = self.predictions
            self.predictions = mitigated_predictions
            mitigated_metrics[strategy] = self.evaluate_fairness_metrics(['demographic_parity'])
            self.predictions = original_predictions
            
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot original metrics
        phenotypes = list(original_metrics['demographic_parity'].keys())
        original_rates = [original_metrics['demographic_parity'][p]['positive_rate'] 
                         for p in phenotypes]
        plt.bar(np.arange(len(phenotypes)) - 0.3, original_rates, 0.2, 
                label='Original', color='blue')
        
        # Plot mitigated metrics
        colors = ['red', 'green', 'purple']
        for i, (strategy, metrics) in enumerate(mitigated_metrics.items()):
            rates = [metrics['demographic_parity'][p]['positive_rate'] 
                    for p in phenotypes]
            plt.bar(np.arange(len(phenotypes)) + 0.1 + i*0.2, rates, 0.2,
                   label=strategy.capitalize(), color=colors[i])
            
        plt.title('Effect of Bias Mitigation Strategies')
        plt.xlabel('Phenotype')
        plt.ylabel('Positive Rate')
        plt.xticks(np.arange(len(phenotypes)), phenotypes)
        plt.legend()
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        plt.show() 