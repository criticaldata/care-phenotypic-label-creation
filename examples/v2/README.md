# Care Phenotype Analyzer - MIMIC Example Workflow

This example demonstrates how to use the Care Phenotype Analyzer package with MIMIC data to create objective care phenotype labels and evaluate fairness in healthcare algorithms.

## Overview

The workflow shows how to:
1. Obtain and preprocess relevant MIMIC data
2. Create care phenotype labels based on observable care patterns
3. Analyze these patterns to understand variations in care delivery
4. Evaluate algorithmic fairness using the created phenotype labels

## Prerequisites

- **MIMIC Access**: Valid PhysioNet credentials with access to MIMIC-IV
- **Python Environment**: Python 3.7+ with pandas, numpy, scikit-learn, scipy, matplotlib, and seaborn
- **Care Phenotype Analyzer**: Installed package

## Data Acquisition

We'll use a focused subset of MIMIC-IV tables for this example. You have two options:

> **Note**: For a visual representation of these two alternative approaches, see [workflow.md](workflow.md).

### Option 1: Using SQL (Recommended for efficiency)

If you have access to MIMIC-IV in BigQuery or PostgreSQL, you can use the SQL scripts in the `sql_scripts/` directory to extract precisely the data needed:

1. See the [SQL Scripts README](sql_scripts/README.md) for detailed instructions
2. Run the scripts in order (01 through 04) to create the analysis dataset
3. Export the results to CSV files in the `data/` directory

This approach is much more efficient, especially for the large tables like `labevents` and `chartevents`.

### Option 2: Direct CSV Download

Alternatively, you can download the raw CSV files:

1. **Install PhysioNet client** (if not already installed):
   ```bash
   pip install physionet-client
   ```

2. **Download required MIMIC-IV tables**:
   ```bash
   physionet-get -r mimic-iv/2.2/hosp/patients.csv
   physionet-get -r mimic-iv/2.2/hosp/labevents.csv
   physionet-get -r mimic-iv/2.2/hosp/admissions.csv
   physionet-get -r mimic-iv/2.2/icu/chartevents.csv
   ```

3. **Place downloaded files** in a `data/` directory within this folder

**Note:** The `labevents` and `chartevents` tables are very large. Consider extracting only what you need using SQL if possible.

## Example Workflow

### Step 1: Cohort Selection and Data Preparation

We'll focus on ICU patients with multiple lab measurements to observe care patterns.

**Option A: Using the SQL workflow (Recommended)**

1. Follow the instructions in the [SQL Scripts README](sql_scripts/README.md)
2. Run the SQL scripts to generate `cohort_data.csv`
3. Place this file in the `data/` directory
4. Skip directly to Step 2 below

**Option B: Using the Python preprocessing script**

If you've downloaded the raw CSV files instead, use:

```python
# Script: 01_prepare_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load required tables
patients = pd.read_csv('data/patients.csv')
labevents = pd.read_csv('data/labevents.csv')
admissions = pd.read_csv('data/admissions.csv')
chartevents = pd.read_csv('data/chartevents.csv')

# Define cohort: Adult ICU patients with at least 3 days stay
# Code to filter and join relevant data

# Calculate clinical factors
# - Age
# - Length of stay
# - Severity scores (can be approximated from available data)

# Calculate care patterns
# - Lab test frequency (tests per day)
# - Time to specific interventions
# - Vital sign measurement frequency

# Save prepared dataset
cohort_data.to_csv('processed_data/cohort_data.csv', index=False)
```

### Step 2: Create Care Phenotype Labels

```python
# Script: 02_create_phenotypes.py

import pandas as pd
from care_phenotype_analyzer import CarePhenotypeCreator

# Load prepared data
data = pd.read_csv('processed_data/cohort_data.csv')

# Define clinical factors
clinical_factors = ['age', 'los_days', 'severity_score']

# Initialize phenotype creator
creator = CarePhenotypeCreator(
    data=data,
    clinical_factors=clinical_factors,
    n_clusters=3  # Can be adjusted based on dataset characteristics
)

# Create phenotype labels
phenotype_labels = creator.create_phenotype_labels()

# Analyze clinical separation
separation_metrics = creator.analyze_clinical_separation()

# Analyze unexplained variation
variation_metrics = creator.analyze_unexplained_variation()

# Save results
phenotype_labels.to_csv('results/phenotype_labels.csv')
pd.DataFrame(separation_metrics).to_csv('results/clinical_separation.csv')
pd.DataFrame(variation_metrics).to_csv('results/unexplained_variation.csv')
```

### Step 3: Analyze Care Patterns

```python
# Script: 03_analyze_patterns.py

import pandas as pd
from care_phenotype_analyzer import CarePatternAnalyzer

# Load data and phenotype labels
data = pd.read_csv('processed_data/cohort_data.csv')
phenotype_labels = pd.read_csv('results/phenotype_labels.csv', index_col=0)

# Initialize pattern analyzer
analyzer = CarePatternAnalyzer(
    data=data,
    clinical_factors=['age', 'los_days', 'severity_score']
)

# Analyze lab measurement frequency
lab_frequency = analyzer.analyze_measurement_frequency(
    measurement_column='lab_tests_per_day',
    time_column='admission_time'
)

# Visualize clinical separation
analyzer.visualize_clinical_separation(
    phenotype_labels=phenotype_labels,
    output_file='figures/clinical_separation.png'
)

# Visualize unexplained variation
analyzer.visualize_unexplained_variation(
    phenotype_labels=phenotype_labels,
    care_patterns=['lab_tests_per_day', 'vital_signs_per_day'],
    output_file='figures/unexplained_variation.png'
)
```

### Step 4: Evaluate Fairness

```python
# Script: 04_evaluate_fairness.py

import pandas as pd
import numpy as np
from care_phenotype_analyzer import FairnessEvaluator

# Load data and phenotype labels
data = pd.read_csv('processed_data/cohort_data.csv')
phenotype_labels = pd.read_csv('results/phenotype_labels.csv', index_col=0)

# For demonstration, we'll simulate a prediction model
# In a real scenario, you would load actual model predictions
# Here we use a simplified approach where predictions are correlated with a clinical factor
np.random.seed(42)
predictions = (0.7 * data['severity_score'] + 0.3 * np.random.randn(len(data))) > 0
predictions = pd.Series(predictions.astype(int))

# Define "ground truth" for this example (could be mortality, readmission, etc.)
true_labels = data['outcome_variable']

# Initialize fairness evaluator
evaluator = FairnessEvaluator(
    predictions=predictions,
    true_labels=true_labels,
    phenotype_labels=phenotype_labels.iloc[:, 0],
    clinical_factors=data[['age', 'los_days', 'severity_score']],
    demographic_factors=['gender', 'ethnicity'],
    demographic_data=data[['gender', 'ethnicity']]
)

# Evaluate fairness metrics
fairness_metrics = evaluator.evaluate_fairness_metrics(
    metrics=['demographic_parity', 'equal_opportunity', 'predictive_parity'],
    adjust_for_clinical=True
)

# Visualize fairness metrics
evaluator.visualize_fairness_metrics(
    metrics=['demographic_parity', 'equal_opportunity'],
    output_file='figures/fairness_metrics.png'
)

# Try bias mitigation
evaluator.visualize_bias_mitigation(
    strategies=['reweighting', 'threshold_adjustment'],
    output_file='figures/bias_mitigation.png'
)
```

## Interpreting Results

### Phenotype Labels
The created phenotype labels represent distinct care pattern groups. These might reflect:
- Different levels of care intensity
- Different testing/monitoring strategies
- Potentially unexplained variations in care delivery

### Clinical Separation
Strong clinical separation between phenotypes indicates the variations are associated with legitimate clinical factors.

### Unexplained Variation
High unexplained variation may suggest care disparities not justified by clinical need.

### Fairness Metrics
- **Demographic Parity**: Equal prediction rates across demographic groups
- **Equal Opportunity**: Equal true positive rates across demographic groups
- **Predictive Parity**: Equal positive predictive values across demographic groups

## Extending the Example

Consider these extensions to the basic workflow:
- Include more complex clinical factors (comorbidities, procedures)
- Analyze specific care processes (e.g., antibiotic administration timing)
- Evaluate fairness with different prediction models
- Incorporate treatment outcomes to assess impact of care phenotypes

## Directory Structure

```
v2/
├── data/                   # Raw MIMIC data files
├── processed_data/         # Prepared cohort data
├── results/                # Analysis outputs
│   ├── phenotype_labels.csv
│   ├── clinical_separation.csv
│   └── unexplained_variation.csv
├── figures/                # Visualizations
├── scripts/                # Implementation scripts
│   ├── 01_prepare_data.py
│   ├── 02_create_phenotypes.py
│   ├── 03_analyze_patterns.py
│   └── 04_evaluate_fairness.py
├── sql_scripts/            # SQL scripts for data extraction
│   ├── README.md
│   ├── 01_extract_patients.sql
│   ├── 02_extract_lab_events.sql
│   ├── 03_extract_vitals.sql
│   └── 04_create_analysis_table.sql
└── README.md               # This file
```

## Troubleshooting

- **Memory Issues**: If experiencing memory problems with large MIMIC tables, consider:
  - Using a subset of patients
  - Processing data in chunks
  - Filtering to specific lab or chart events of interest

- **PhysioNet Download Issues**: Ensure your PhysioNet credentials are properly configured and that you have the appropriate access level for MIMIC-IV.

- **Interpretation Challenges**: Care phenotypes are exploratory and may require domain expertise to fully interpret. Consider consulting with clinical experts when analyzing results. 