# Care Phenotype Analyzer

A Python package for creating objective care phenotype labels based on observable care patterns, focusing on easily measurable care metrics like lab test frequency and routine care procedures. This tool helps researchers identify and use these new labels for fairness evaluation in healthcare.

## Overview

This package implements a novel approach to healthcare fairness evaluation by:
- Creating objective care phenotype labels based on observable care patterns
- Moving beyond traditional demographic labels for fairness evaluation
- Focusing on easily measurable care metrics (e.g., lab test frequency, routine care procedures)
- Accounting for clinical factors that may justify care variations

## Features

- **Care Phenotype Creation**: Create objective labels based on care patterns
- **Pattern Analysis**: Analyze care patterns and their relationships
- **Fairness Evaluation**: Evaluate healthcare algorithm fairness using care phenotypes
- **Clinical Factor Integration**: Account for clinical factors in analysis
- **Visualization Tools**: Visualize patterns and distributions

## Installation

```bash
pip install care-phenotype-analyzer
```

## Usage

```python
from care_phenotype_analyzer import CarePhenotypeCreator, CarePatternAnalyzer, FairnessEvaluator
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Create care phenotypes
phenotype_creator = CarePhenotypeCreator(
    data=data,
    clinical_factors=['severity_score', 'chronic_conditions']
)

# Create phenotype labels
phenotype_labels = phenotype_creator.create_phenotype_labels(
    care_patterns=['lab_test_frequency', 'routine_care_frequency'],
    n_clusters=3
)

# Analyze patterns
pattern_analyzer = CarePatternAnalyzer(data)
pattern_results = pattern_analyzer.analyze_pattern_frequency(
    pattern_column='lab_test_frequency',
    time_column='timestamp',
    group_by=['phenotype']
)

# Evaluate fairness
fairness_evaluator = FairnessEvaluator(
    predictions=model_predictions,
    true_labels=true_labels,
    phenotype_labels=phenotype_labels,
    clinical_factors=clinical_data
)

fairness_results = fairness_evaluator.evaluate_fairness_metrics(
    metrics=['demographic_parity', 'equal_opportunity']
)
```

## Examples

### Serum Lactate Measurement Analysis
```python
# Analyze serum lactate measurement frequency
pattern_analyzer.analyze_pattern_frequency(
    pattern_column='lactate_measurement_frequency',
    time_column='timestamp',
    group_by=['phenotype']
)
```

### ICU Care Pattern Analysis
```python
# Analyze ICU care patterns
patterns = ['mouth_care_frequency', 'turning_frequency']
phenotype_labels = phenotype_creator.create_phenotype_labels(
    care_patterns=patterns,
    n_clusters=3
)
```

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests:
   ```bash
   pytest
   ```

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
@software{care_phenotype_analyzer2024,
  title = {Care Phenotype Analyzer: A Tool for Objective Healthcare Fairness Evaluation},
  author = {MIT Critical Data},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MIT-LCP/care-phenotypic-label-creation}
}
```
