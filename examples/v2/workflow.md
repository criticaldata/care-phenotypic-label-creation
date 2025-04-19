# Care Phenotype Analysis Workflow

This file explains the two alternative workflows for using the Care Phenotype Analyzer with MIMIC data.

## Workflow Options

```
                                  ┌─────────────────────┐
                                  │     MIMIC Data      │
                                  └──────────┬──────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────┐
              │                   Data Preparation                  │
              └─────────────────────────────────────────────────────┘
                              /                \
                             /                  \
                            /                    \
         ┌────────────────┐                       ┌────────────────┐
         │  SQL Workflow  │                       │ Python Workflow│
         └───────┬────────┘                       └───────┬────────┘
                 │                                        │
                 ▼                                        ▼
    ┌─────────────────────────┐              ┌─────────────────────────┐
    │ 01_extract_patients.sql │              │  01_prepare_data.py     │
    └────────────┬────────────┘              └────────────┬────────────┘
                 │                                        │
                 ▼                                        │
    ┌─────────────────────────┐                          │
    │ 02_extract_lab_events.sql│                         │
    └────────────┬────────────┘                          │
                 │                                        │
                 ▼                                        │
    ┌─────────────────────────┐                          │
    │ 03_extract_vitals.sql   │                          │
    └────────────┬────────────┘                          │
                 │                                        │
                 ▼                                        │
    ┌─────────────────────────┐                          │
    │04_create_analysis_table.sql                        │
    └────────────┬────────────┘                          │
                 │                                        │
                 ▼                                        ▼
              cohort_data.csv                       cohort_data.csv
                 │                                        │
                 │                                        │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │  02_create_phenotypes.py │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │  03_analyze_patterns.py  │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ 04_evaluate_fairness.py  │
                      └─────────────────────────┘
```

## Key Points

1. **Two Alternative Paths for Data Preparation**:
   - **SQL Workflow**: More efficient for large MIMIC tables, performs pre-aggregation in the database
   - **Python Workflow**: Processes raw CSV files, more memory/CPU intensive but doesn't require SQL access

2. **Common Workflow After Data Preparation**:
   - Both paths produce a `cohort_data.csv` file
   - Python scripts 02-04 are used identically regardless of which path you took
   
3. **Choose Based On**:
   - Available resources (SQL access vs. local processing power)
   - Data size constraints
   - Preference for SQL vs. Python

4. **When to Skip Scripts**:
   - If using the SQL workflow, skip `01_prepare_data.py`
   - If using the Python workflow, you don't need the SQL scripts
   
The SQL workflow is generally recommended for efficiency with large MIMIC tables. 