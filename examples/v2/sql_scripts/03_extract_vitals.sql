-- 03_extract_vitals.sql
-- Extracts and aggregates vital sign measurements for the patient cohort
-- Compatible with MIMIC-IV on BigQuery

-- Create or replace a table for the aggregated vital signs
CREATE OR REPLACE TABLE `your_dataset.vitals_agg` AS

-- Common vital sign item IDs in MIMIC-IV
WITH vital_groups AS (
  SELECT * FROM UNNEST([
    -- Heart rate
    STRUCT(220045 as itemid, 'HR' as vital_group, 'Heart Rate' as vital_name),
    
    -- Blood pressure
    STRUCT(220050 as itemid, 'BP' as vital_group, 'Arterial BP Systolic' as vital_name),
    STRUCT(220051 as itemid, 'BP' as vital_group, 'Arterial BP Diastolic' as vital_name),
    STRUCT(220052 as itemid, 'BP' as vital_group, 'Arterial BP Mean' as vital_name),
    STRUCT(220179 as itemid, 'BP' as vital_group, 'Non Invasive BP Systolic' as vital_name),
    STRUCT(220180 as itemid, 'BP' as vital_group, 'Non Invasive BP Diastolic' as vital_name),
    STRUCT(220181 as itemid, 'BP' as vital_group, 'Non Invasive BP Mean' as vital_name),
    
    -- Respiratory
    STRUCT(220210 as itemid, 'RESP' as vital_group, 'Respiratory Rate' as vital_name),
    STRUCT(223835 as itemid, 'RESP' as vital_group, 'O2 Saturation' as vital_name),
    STRUCT(223762 as itemid, 'RESP' as vital_group, 'O2 Delivery Device' as vital_name),
    STRUCT(223834 as itemid, 'RESP' as vital_group, 'O2 Flow' as vital_name),
    
    -- Temperature
    STRUCT(223761 as itemid, 'TEMP' as vital_group, 'Temperature Celsius' as vital_name),
    STRUCT(223762 as itemid, 'TEMP' as vital_group, 'Temperature Fahrenheit' as vital_name),
    
    -- Neurological
    STRUCT(220739 as itemid, 'NEURO' as vital_group, 'GCS Total' as vital_name),
    STRUCT(223900 as itemid, 'NEURO' as vital_group, 'GCS Verbal' as vital_name),
    STRUCT(223901 as itemid, 'NEURO' as vital_group, 'GCS Motor' as vital_name),
    STRUCT(220734 as itemid, 'NEURO' as vital_group, 'GCS Eye Opening' as vital_name),
    
    -- Other
    STRUCT(224639 as itemid, 'OTHER' as vital_group, 'Glucose' as vital_name),
    STRUCT(220624 as itemid, 'OTHER' as vital_group, 'Pain Score' as vital_name)
  ])
)

-- Select vital sign measurements for patients in the cohort
SELECT 
    pc.subject_id,
    pc.hadm_id,
    pc.admittime,
    pc.dischtime,
    pc.los_days,
    
    -- Aggregate vital measurements by admission
    -- Overall measurement frequency metrics
    COUNT(DISTINCT ce.charttime) AS total_vital_timepoints,
    COUNT(DISTINCT ce.charttime) / pc.los_days AS vital_timepoints_per_day,
    
    -- Vital group frequencies (per day)
    COUNT(DISTINCT CASE WHEN vg.vital_group = 'HR' THEN ce.charttime END) / pc.los_days AS hr_checks_per_day,
    COUNT(DISTINCT CASE WHEN vg.vital_group = 'BP' THEN ce.charttime END) / pc.los_days AS bp_checks_per_day,
    COUNT(DISTINCT CASE WHEN vg.vital_group = 'RESP' THEN ce.charttime END) / pc.los_days AS resp_checks_per_day,
    COUNT(DISTINCT CASE WHEN vg.vital_group = 'TEMP' THEN ce.charttime END) / pc.los_days AS temp_checks_per_day,
    COUNT(DISTINCT CASE WHEN vg.vital_group = 'NEURO' THEN ce.charttime END) / pc.los_days AS neuro_checks_per_day,
    
    -- Calculate total vital sign measurements (not just timepoints)
    COUNT(ce.itemid) AS total_vital_measurements,
    COUNT(ce.itemid) / pc.los_days AS vital_measurements_per_day,
    
    -- Key clinical values
    -- Heart rate
    MAX(CASE WHEN vg.vital_name = 'Heart Rate' THEN ce.valuenum ELSE NULL END) AS max_heart_rate,
    MIN(CASE WHEN vg.vital_name = 'Heart Rate' THEN ce.valuenum ELSE NULL END) AS min_heart_rate,
    AVG(CASE WHEN vg.vital_name = 'Heart Rate' THEN ce.valuenum ELSE NULL END) AS avg_heart_rate,
    
    -- Blood pressure (prioritize arterial if available, otherwise non-invasive)
    MAX(CASE WHEN vg.vital_name = 'Arterial BP Systolic' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Systolic' THEN ce.valuenum 
             ELSE NULL END) AS max_sbp,
    MIN(CASE WHEN vg.vital_name = 'Arterial BP Systolic' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Systolic' THEN ce.valuenum 
             ELSE NULL END) AS min_sbp,
    AVG(CASE WHEN vg.vital_name = 'Arterial BP Systolic' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Systolic' THEN ce.valuenum 
             ELSE NULL END) AS avg_sbp,
             
    MAX(CASE WHEN vg.vital_name = 'Arterial BP Diastolic' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Diastolic' THEN ce.valuenum 
             ELSE NULL END) AS max_dbp,
    MIN(CASE WHEN vg.vital_name = 'Arterial BP Diastolic' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Diastolic' THEN ce.valuenum 
             ELSE NULL END) AS min_dbp,
    AVG(CASE WHEN vg.vital_name = 'Arterial BP Diastolic' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Diastolic' THEN ce.valuenum 
             ELSE NULL END) AS avg_dbp,
             
    MAX(CASE WHEN vg.vital_name = 'Arterial BP Mean' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Mean' THEN ce.valuenum 
             ELSE NULL END) AS max_map,
    MIN(CASE WHEN vg.vital_name = 'Arterial BP Mean' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Mean' THEN ce.valuenum 
             ELSE NULL END) AS min_map,
    AVG(CASE WHEN vg.vital_name = 'Arterial BP Mean' THEN ce.valuenum 
             WHEN vg.vital_name = 'Non Invasive BP Mean' THEN ce.valuenum 
             ELSE NULL END) AS avg_map,
    
    -- Respiratory
    MAX(CASE WHEN vg.vital_name = 'Respiratory Rate' THEN ce.valuenum ELSE NULL END) AS max_resp_rate,
    MIN(CASE WHEN vg.vital_name = 'Respiratory Rate' THEN ce.valuenum ELSE NULL END) AS min_resp_rate,
    AVG(CASE WHEN vg.vital_name = 'Respiratory Rate' THEN ce.valuenum ELSE NULL END) AS avg_resp_rate,
    
    MIN(CASE WHEN vg.vital_name = 'O2 Saturation' THEN ce.valuenum ELSE NULL END) AS min_o2_saturation,
    AVG(CASE WHEN vg.vital_name = 'O2 Saturation' THEN ce.valuenum ELSE NULL END) AS avg_o2_saturation,
    
    -- Temperature
    MAX(CASE WHEN vg.vital_name = 'Temperature Celsius' THEN ce.valuenum ELSE NULL END) AS max_temperature,
    MIN(CASE WHEN vg.vital_name = 'Temperature Celsius' THEN ce.valuenum ELSE NULL END) AS min_temperature,
    
    -- Neurological
    MIN(CASE WHEN vg.vital_name = 'GCS Total' THEN ce.valuenum ELSE NULL END) AS min_gcs,
    
    -- Frequency of abnormal vitals (as count per day)
    COUNTIF(vg.vital_name = 'Heart Rate' AND (ce.valuenum < 60 OR ce.valuenum > 100)) / pc.los_days AS abnormal_hr_per_day,
    COUNTIF((vg.vital_name = 'Arterial BP Systolic' OR vg.vital_name = 'Non Invasive BP Systolic') 
            AND (ce.valuenum < 90 OR ce.valuenum > 180)) / pc.los_days AS abnormal_sbp_per_day,
    COUNTIF(vg.vital_name = 'Respiratory Rate' AND (ce.valuenum < 12 OR ce.valuenum > 20)) / pc.los_days AS abnormal_resp_per_day,
    COUNTIF(vg.vital_name = 'O2 Saturation' AND ce.valuenum < 92) / pc.los_days AS abnormal_o2_per_day,
    COUNTIF(vg.vital_name = 'Temperature Celsius' AND (ce.valuenum < 36 OR ce.valuenum > 38)) / pc.los_days AS abnormal_temp_per_day,
    
    -- Stability metrics - standard deviations of key vitals
    STDDEV(CASE WHEN vg.vital_name = 'Heart Rate' THEN ce.valuenum ELSE NULL END) AS hr_stddev,
    STDDEV(CASE WHEN vg.vital_name = 'Arterial BP Mean' OR vg.vital_name = 'Non Invasive BP Mean' 
                THEN ce.valuenum ELSE NULL END) AS map_stddev,
    STDDEV(CASE WHEN vg.vital_name = 'Respiratory Rate' THEN ce.valuenum ELSE NULL END) AS resp_stddev

FROM `your_dataset.patients_cohort` pc
LEFT JOIN `physionet-data.mimic_icu.chartevents` ce
    ON pc.hadm_id = ce.hadm_id
LEFT JOIN vital_groups vg
    ON ce.itemid = vg.itemid

WHERE 
    -- Exclude invalid vital values
    ce.valuenum IS NOT NULL
    AND ce.valuenum > 0
    -- Filter to vitals during this admission
    AND ce.charttime BETWEEN pc.admittime AND pc.dischtime
    -- Filter to known vital groups
    AND vg.vital_group IS NOT NULL

GROUP BY 
    pc.subject_id, pc.hadm_id, pc.admittime, pc.dischtime, pc.los_days;

-- Output summary statistics
SELECT 
    COUNT(*) AS total_admissions,
    AVG(vital_timepoints_per_day) AS avg_vital_checks_per_day,
    AVG(vital_measurements_per_day) AS avg_vital_measurements_per_day,
    AVG(hr_checks_per_day) AS avg_hr_checks_per_day,
    AVG(bp_checks_per_day) AS avg_bp_checks_per_day,
    AVG(min_sbp) AS avg_min_sbp,
    AVG(min_map) AS avg_min_map,
    AVG(min_o2_saturation) AS avg_min_o2_saturation,
    AVG(abnormal_hr_per_day) AS avg_abnormal_hr_per_day,
    AVG(abnormal_sbp_per_day) AS avg_abnormal_sbp_per_day,
    AVG(hr_stddev) AS avg_hr_variability,
    AVG(map_stddev) AS avg_map_variability
FROM `your_dataset.vitals_agg`;

-- Export instructions:
-- In BigQuery UI: Export the table as CSV to 'vitals_agg.csv'
-- Using CLI: bq extract your_dataset.vitals_agg '../data/vitals_agg.csv' 