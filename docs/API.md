# Care Phenotype Analyzer API Documentation

## Overview

The Care Phenotype Analyzer package provides tools for creating and analyzing care phenotype labels based on observable care patterns. This documentation covers the main classes and their methods.

## Core Classes

### CarePhenotypeCreator

The main class for creating care phenotype labels based on observable care patterns.

```python
from care_phenotype_analyzer import CarePhenotypeCreator

creator = CarePhenotypeCreator(
    data: pd.DataFrame,
    clinical_factors: Optional[List[str]] = None,
    log_file: Optional[str] = None
)
```

#### Methods

##### `create_phenotype_labels(care_patterns: List[str], n_clusters: int = 3, adjust_for_clinical: bool = True) -> pd.Series`

Creates care phenotype labels based on observed care patterns.

**Parameters:**
- `care_patterns`: List of columns containing care pattern measurements
- `n_clusters`: Number of phenotype groups to create (default: 3)
- `adjust_for_clinical`: Whether to adjust for clinical factors (default: True)

**Returns:**
- Series containing phenotype labels for each patient

##### `analyze_unexplained_variation(care_pattern: str, phenotype_labels: pd.Series) -> Dict`

Analyzes unexplained variation in care patterns across phenotypes.

**Parameters:**
- `care_pattern`: Column containing the care pattern to analyze
- `phenotype_labels`: Created phenotype labels

**Returns:**
- Dictionary containing analysis of unexplained variation

##### `validate_phenotypes(labels: pd.Series, validation_metrics: List[str]) -> Dict`

Validates created phenotype labels using various metrics.

**Parameters:**
- `labels`: Created phenotype labels
- `validation_metrics`: List of validation metrics to calculate

**Returns:**
- Dictionary containing validation results

### CarePatternAnalyzer

Class for analyzing care patterns and their relationships.

```python
from care_phenotype_analyzer import CarePatternAnalyzer

analyzer = CarePatternAnalyzer(data: pd.DataFrame)
```

#### Methods

##### `analyze_measurement_frequency(measurement_column: str, time_column: str, clinical_factors: Optional[List[str]] = None, group_by: Optional[List[str]] = None) -> pd.DataFrame`

Analyzes the frequency of specific measurements.

**Parameters:**
- `measurement_column`: Column containing the measurement to analyze
- `time_column`: Column containing temporal information
- `clinical_factors`: List of clinical factors to consider
- `group_by`: Columns to group the analysis by

**Returns:**
- DataFrame containing frequency analysis results

##### `visualize_pattern_distribution(pattern_column: str, phenotype_column: Optional[str] = None, clinical_factor: Optional[str] = None, time_column: Optional[str] = None) -> None`

Visualizes the distribution of care patterns.

**Parameters:**
- `pattern_column`: Column containing the pattern to visualize
- `phenotype_column`: Column containing phenotype labels
- `clinical_factor`: Clinical factor to analyze
- `time_column`: Column containing temporal information

##### `analyze_temporal_patterns(pattern_column: str, time_column: str, phenotype_column: Optional[str] = None, clinical_factors: Optional[List[str]] = None) -> Dict`

Analyzes temporal patterns in care delivery.

**Parameters:**
- `pattern_column`: Column containing the care pattern to analyze
- `time_column`: Column containing temporal information
- `phenotype_column`: Column containing phenotype labels
- `clinical_factors`: List of clinical factors to consider

**Returns:**
- Dictionary containing temporal analysis results

### FairnessEvaluator

Class for evaluating fairness using care phenotype labels.

```python
from care_phenotype_analyzer import FairnessEvaluator

evaluator = FairnessEvaluator(
    predictions: pd.Series,
    true_labels: pd.Series,
    phenotype_labels: pd.Series,
    clinical_factors: Optional[pd.DataFrame] = None
)
```

#### Methods

##### `evaluate_fairness_metrics(metrics: List[str], adjust_for_clinical: bool = True) -> Dict`

Evaluates fairness metrics across care phenotypes.

**Parameters:**
- `metrics`: List of fairness metrics to calculate
- `adjust_for_clinical`: Whether to adjust for clinical factors

**Returns:**
- Dictionary containing fairness evaluation results

##### `analyze_clinical_factors() -> Dict`

Analyzes how clinical factors relate to fairness metrics.

**Returns:**
- Dictionary containing clinical factor analysis results

## Data Requirements

### Input Data Format

The package expects input data in the following format:

```python
data = pd.DataFrame({
    'subject_id': [...],  # Required
    'timestamp': [...],   # Required
    'sofa_score': [...],  # Optional clinical factor
    'charlson_score': [...],  # Optional clinical factor
    'lab_test_frequency': [...],  # Care pattern
    'routine_care_frequency': [...]  # Care pattern
})
```

### Required Columns
- `subject_id`: Unique identifier for each patient
- `timestamp`: Temporal information for measurements

### Optional Columns
- Clinical factors (e.g., SOFA score, Charlson score)
- Care pattern measurements (e.g., lab test frequency, routine care frequency)

## Error Handling

The package includes comprehensive error handling for common scenarios:

- Data validation errors
- Type errors
- Missing required columns
- Invalid clinical factors
- Invalid care patterns

## Logging

The package provides detailed logging capabilities:

```python
creator = CarePhenotypeCreator(
    data=data,
    clinical_factors=['sofa_score'],
    log_file='care_phenotype_creator.log'
)
```

Logs include:
- Initialization information
- Data validation results
- Processing steps
- Error messages with context
- Performance metrics

## Examples

See the `examples/` directory for detailed usage examples:

- `lab_test_analysis_example.py`: Complete workflow for analyzing lab test patterns
- More examples coming soon

## Dependencies

Required packages:
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0 