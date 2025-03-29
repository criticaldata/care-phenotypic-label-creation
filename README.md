# Care Phenotype Analyzer

A Python package for creating objective care phenotype labels based on observable care patterns, focusing on understanding variations in healthcare data collection that cannot be explained by clinical factors alone.

## Overview

This package implements a novel approach to understanding healthcare disparities by:
- Creating objective care phenotype labels based on observable care patterns
- Moving beyond traditional demographic labels for fairness evaluation
- Focusing on easily measurable care metrics (e.g., lab test frequency, routine care procedures)
- Accounting for legitimate clinical factors while highlighting unexplained variations

## Key Features

- **Care Phenotype Creation**: Create objective labels based on care patterns
- **Pattern Analysis**: Analyze care patterns and their relationships
- **Clinical Factor Integration**: Account for clinical factors (e.g., SOFA score, Charlson score)
- **Fairness Evaluation**: Evaluate healthcare algorithm fairness using care phenotypes
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
    clinical_factors=['sofa_score', 'charlson_score']
)

# Create phenotype labels
phenotype_labels = phenotype_creator.create_phenotype_labels(
    care_patterns=['lab_test_frequency', 'routine_care_frequency'],
    n_clusters=3
)

# Analyze patterns
pattern_analyzer = CarePatternAnalyzer(data)
pattern_results = pattern_analyzer.analyze_measurement_frequency(
    measurement_column='lab_test_frequency',
    time_column='timestamp',
    clinical_factors=['sofa_score', 'charlson_score']
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

Check out the `examples/` directory for detailed examples:
- `lab_test_analysis_example.py`: Complete workflow for analyzing lab test patterns
- More examples coming soon

## Project Structure

```
care-phenotypic-label-creation/
├── care_phenotype_analyzer/     # Main package directory
│   ├── phenotype_creator.py    # Creates care phenotype labels
│   ├── pattern_analyzer.py     # Analyzes care patterns
│   └── fairness_evaluator.py   # Evaluates fairness
├── tests/                      # Test suite
│   └── test_analyzer.py        # Comprehensive tests
├── examples/                   # Example scripts
│   ├── lab_test_analysis_example.py
│   └── README.md
└── docs/                       # Documentation
    └── manuscript/            # Project manuscript
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

## Acknowledgments

This project is developed under the MIT Critical Data organization, with contributions from Pedro, Nebal, and others.
