# MIMIC Data Processing Module

This directory contains modules for processing and analyzing MIMIC-IV data (Medical Information Mart for Intensive Care), a large database of de-identified health data from patients in intensive care units.

## Overview

The `mimic` package provides tools for:
- Loading and processing MIMIC-IV data
- Calculating clinical scores (SOFA, Charlson, etc.)
- Validating data integrity
- Converting between data formats
- Generating synthetic data for testing

## Files and Modules

### Core Components

- **processor.py**: Main processing module for MIMIC-IV data. Handles lab events, chart events, and clinical score calculations.
- **structures.py**: Core data structures for MIMIC data (Patient, Admission, ICUStay, LabEvent, etc.).
- **data_formats.py**: Standard column definitions and data format validation utilities.

### Clinical Scores

- **clinical_scores.py**: Base class and utilities for clinical score calculations.
- **sofa_calculator.py**: Sequential Organ Failure Assessment score calculator.
- **charlson_calculator.py**: Charlson Comorbidity Index calculator.
- **other_scores.py**: Additional clinical scoring systems (APACHE, SAPS, MEWS, etc.).

### Utility Functions

- **conversion_utils.py**: Utilities for converting between different data formats.
- **integrity_checks.py**: Functions for validating MIMIC data integrity.
- **synthetic_data.py**: Tools for generating synthetic MIMIC-like data for testing.

## Dependencies

```
structures.py
└── Used by most other modules

data_formats.py
└── Used by processor.py, clinical_scores.py

clinical_scores.py
├── Depends on structures.py, data_formats.py
├── Parent class for sofa_calculator.py, charlson_calculator.py
└── Used by processor.py

processor.py
├── Depends on structures.py, data_formats.py
├── Uses sofa_calculator.py, charlson_calculator.py
└── Main interface for data processing

conversion_utils.py
└── Used by processor.py, integrity_checks.py

integrity_checks.py
├── Depends on structures.py
└── Used by processor.py

synthetic_data.py
└── Depends on structures.py, data_formats.py
```

## Usage Examples

### Basic Data Processing

```python
from care_phenotype_analyzer.mimic.processor import MIMICDataProcessor

# Initialize with data
processor = MIMICDataProcessor(
    lab_events=lab_df,
    chart_events=chart_df,
    patients=patients_df,
    admissions=admissions_df,
    icu_stays=icu_stays_df
)

# Process lab and chart events
processed_labs = processor.process_lab_events()
processed_charts = processor.process_chart_events()

# Calculate clinical scores
scores = processor.calculate_clinical_scores()
```

### Clinical Score Calculation

```python
from care_phenotype_analyzer.mimic.sofa_calculator import SOFACalculator

# Initialize calculator
sofa_calc = SOFACalculator(
    lab_events=lab_df,
    chart_events=chart_df,
    admissions=admissions_df
)

# Calculate SOFA scores
sofa_scores = sofa_calc.calculate_scores()

# Get score history for a patient
patient_scores = sofa_calc.get_score_history(subject_id=123, hadm_id=456, score_type='sofa')
```

### Data Integrity Validation

```python
from care_phenotype_analyzer.mimic.integrity_checks import validate_mimic_data

# Validate data integrity
validation_results = validate_mimic_data(
    lab_events=lab_df,
    chart_events=chart_df,
    patients=patients_df,
    admissions=admissions_df
)
```

### Synthetic Data Generation

```python
from care_phenotype_analyzer.mimic.synthetic_data import generate_synthetic_dataset

# Generate synthetic data for testing
synthetic_data = generate_synthetic_dataset(
    num_patients=100,
    num_admissions_per_patient=1.5,
    num_icu_stays_per_admission=1.2
)
``` 