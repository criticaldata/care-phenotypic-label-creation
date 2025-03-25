"""
Care Phenotype Analyzer

A package for creating objective care phenotype labels based on observable care patterns.
This tool helps researchers identify and use these new labels in their datasets for fairness evaluation.
"""

__version__ = "0.1.0"

from .phenotype_creator import CarePhenotypeCreator
from .pattern_analyzer import CarePatternAnalyzer
from .fairness_evaluator import FairnessEvaluator

__all__ = ['CarePhenotypeCreator', 'CarePatternAnalyzer', 'FairnessEvaluator'] 