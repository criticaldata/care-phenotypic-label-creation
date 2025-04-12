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
- **Performance Optimization**: Memory management, caching, and parallel processing capabilities
- **Monitoring**: Comprehensive monitoring and logging of operations
- **Dashboard and Export**: Interactive dashboards and data export functionality

## Installation

The package is available on PyPI:

```bash
pip install care-phenotype-analyzer
```

You can also install the development version directly from the repository:

```bash
git clone https://github.com/MIT-LCP/care-phenotypic-label-creation.git
cd care-phenotypic-label-creation
pip install -e .
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
    clinical_factors=['sofa_score', 'charlson_score'],
    n_clusters=3,
    random_state=42,
    log_dir="logs"
)

# Create phenotype labels
phenotype_labels = phenotype_creator.create_phenotype_labels()

# Analyze patterns
pattern_analyzer = CarePatternAnalyzer(
    data=data,
    clinical_factors=['sofa_score', 'charlson_score'],
    log_dir="logs"
)

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
    clinical_factors=clinical_data,
    demographic_factors=['race', 'gender'],
    log_dir="logs"
)

fairness_results = fairness_evaluator.evaluate_fairness_metrics(
    metrics=['demographic_parity', 'equal_opportunity', 'predictive_parity', 'treatment_equality', 'care_pattern_disparity'],
    adjust_for_clinical=True
)

# Visualize clinical separation
pattern_analyzer.visualize_clinical_separation(
    phenotype_labels=phenotype_labels,
    clinical_factors=['sofa_score', 'charlson_score'],
    output_file='clinical_separation.png'
)
```

## Examples

Check out the `examples/` directory for detailed examples:

- `lab_test_analysis_example.py`: Complete workflow for analyzing lab test patterns
- Documentation in the examples/README.md file

## Project Structure

```
care-phenotypic-label-creation/
├── care_phenotype_analyzer/      # Main package directory
│   ├── __init__.py               # Package initialization and version info
│   ├── phenotype_creator.py      # Creates care phenotype labels
│   ├── pattern_analyzer.py       # Analyzes care patterns
│   ├── fairness_evaluator.py     # Evaluates fairness
│   ├── visualization.py          # Data visualization tools
│   ├── monitoring.py             # Performance monitoring
│   ├── memory.py                 # Memory optimization
│   ├── caching.py                # Caching mechanisms
│   ├── parallel.py               # Parallel processing
│   ├── dashboard.py              # Interactive dashboards
│   ├── export.py                 # Data export functionality
│   └── mimic/                    # MIMIC database specific modules
├── tests/                        # Comprehensive test suite
│   ├── test_analyzer.py          # Core analyzer tests
│   ├── test_caching.py           # Caching tests
│   ├── test_clinical_score_validation.py
│   ├── test_dashboard.py
│   ├── test_data_processing_validation.py
│   ├── test_export.py
│   ├── test_integration.py       # Integration tests
│   ├── test_large_dataset_handling.py
│   ├── test_memory.py            # Memory optimization tests
│   ├── test_memory_optimization.py
│   ├── test_monitoring.py
│   ├── test_parallel.py
│   ├── test_performance.py       # Performance tests
│   ├── test_processing_speed.py
│   ├── test_result_validation.py
│   ├── test_synthetic_data.py
│   └── test_visualization.py
├── examples/                     # Example scripts
│   ├── lab_test_analysis_example.py
│   └── README.md
├── docs/                         # Documentation
│   ├── API.md                    # API documentation
│   ├── ARCHITECTURE.md           # Architecture details
│   ├── DEPLOYMENT.md             # Deployment guide
│   └── manuscript/               # Project manuscript
├── setup.py                      # Package setup configuration
├── requirements.txt              # Development dependencies
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
└── README.md                     # This file
```

## Documentation

The package documentation is available in the `docs/` directory:

- `API.md`: Detailed API documentation for all modules
- `ARCHITECTURE.md`: System architecture and design principles
- `DEPLOYMENT.md`: Deployment and integration guide

## Development

### Prerequisites

- Python 3.8+
- Git

### Setting Up Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/MIT-LCP/care-phenotypic-label-creation.git
   cd care-phenotypic-label-creation
   ```
2. Install development dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Install the package in development mode:

   ```bash
   pip install -e .
   ```

### Running Tests

Run the full test suite:

```bash
pytest
```

Run specific test files:

```bash
pytest tests/test_analyzer.py
```

### Continuous Integration/Deployment

The project uses GitHub Actions for CI/CD:

- CI: Automated testing on pull requests
- CD: Automatic packaging and publishing to PyPI on version tags

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details on:

- Code style and standards
- Pull request process
- Issue reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```
@software{care_phenotype_analyzer2024,
  title = {Care Phenotype Analyzer: A Tool for Objective Healthcare Fairness Evaluation},
  author = {MIT Critical Data},
  year = {2024},
  version = {0.1.0},
  publisher = {GitHub},
  url = {https://github.com/MIT-LCP/care-phenotypic-label-creation}
}
```

## Acknowledgments

This project is developed under the MIT Critical Data organization, with contributions from Pedro, Nebal, and others.
