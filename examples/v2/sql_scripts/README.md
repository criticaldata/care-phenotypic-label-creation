# SQL Scripts for Care Phenotype Analysis

This directory contains SQL scripts for extracting analysis-ready datasets from MIMIC-IV. These scripts are designed to work with Google BigQuery or other SQL databases that host MIMIC-IV data.

## Overview

Rather than working with the full MIMIC-IV CSV files (which can be very large), these SQL scripts extract precisely the data needed for care phenotype analysis. This approach:

1. **Reduces data size**: Extracts only the columns and rows needed
2. **Performs aggregation**: Pre-calculates frequencies and statistics
3. **Enforces cohort selection**: Applies inclusion/exclusion criteria consistently

## Workflow

The SQL workflow consists of the following steps:

1. **Extract patient cohort** (`01_extract_patients.sql`):
   - Adult patients (≥18 years)
   - With hospital stays of 3+ days
   - Outputs: `patients_cohort.csv`

2. **Extract lab events** (`02_extract_lab_events.sql`):
   - Lab test frequencies and key values for the cohort
   - Aggregated by patient and admission
   - Outputs: `lab_events_agg.csv`

3. **Extract vital signs** (`03_extract_vitals.sql`):
   - Vital sign frequencies and key measurements
   - Aggregated by patient and admission
   - Outputs: `vitals_agg.csv`

4. **Create analysis dataset** (`04_create_analysis_table.sql`):
   - Joins all extracted data
   - Calculates additional metrics
   - Creates final dataset for Python analysis
   - Outputs: `cohort_data.csv`

## Usage Instructions

### Using Google BigQuery

1. Access MIMIC-IV on BigQuery through your PhysioNet account
2. Create a dataset to store the results (if you don't have one already)
3. Run each SQL script in order, adjusting the output dataset as needed
4. Export the final results to CSV files

```bash
# Example using bq command-line tool (adjust paths as needed)
bq query --use_legacy_sql=false --dataset_id=your_dataset < 01_extract_patients.sql
bq query --use_legacy_sql=false --dataset_id=your_dataset < 02_extract_lab_events.sql
bq query --use_legacy_sql=false --dataset_id=your_dataset < 03_extract_vitals.sql
bq query --use_legacy_sql=false --dataset_id=your_dataset < 04_create_analysis_table.sql

# Export results
bq extract your_dataset.cohort_data '../data/cohort_data.csv'
```

### Using PostgreSQL

If using a PostgreSQL installation of MIMIC-IV:

1. Connect to your PostgreSQL database
2. Run each script in order
3. Export the results to CSV

```bash
# Example using psql (adjust connection details as needed)
psql -h localhost -d mimic -U your_username -f 01_extract_patients.sql
psql -h localhost -d mimic -U your_username -f 02_extract_lab_events.sql
psql -h localhost -d mimic -U your_username -f 03_extract_vitals.sql
psql -h localhost -d mimic -U your_username -f 04_create_analysis_table.sql

# Export results (this command would be run within PostgreSQL)
# COPY cohort_data TO '/path/to/data/cohort_data.csv' DELIMITER ',' CSV HEADER;
```

## Integration with Python Scripts

After generating the CSV files, place them in the `../data/` directory. The Python scripts are designed to work with these files:

- `01_prepare_data.py` expects to find the output from the SQL scripts in the data directory
- If running the full SQL workflow to create `cohort_data.csv`, you can skip directly to `02_create_phenotypes.py`

## Customization

These SQL scripts can be customized to adjust:

- **Cohort definition**: Modify patient inclusion/exclusion criteria
- **Lab tests and vitals**: Add or remove specific lab tests or vital signs of interest
- **Aggregation level**: Change how data is aggregated (by admission, by day, etc.)
- **Output fields**: Add additional clinical or demographic information

Make sure any customizations maintain the expected column names and data structure for the Python scripts, or update the Python scripts accordingly. 