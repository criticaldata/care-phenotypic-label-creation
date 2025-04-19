-- 04_create_analysis_table.sql
-- Creates the final analysis dataset by joining patient cohort, lab events, and vital signs
-- Compatible with MIMIC-IV on BigQuery

-- Create or replace the final analysis table
CREATE OR REPLACE TABLE `your_dataset.cohort_data` AS

-- First, create a severity score based on clinical values
WITH severity_scores AS (
  SELECT
    va.subject_id,
    va.hadm_id,
    
    -- Create a simplified severity score based on clinical indicators
    -- Each component contributes 0-1 points to the score
    (
      -- Low mean arterial pressure component (0-1 points)
      CASE 
        WHEN va.min_map < 65 THEN 1.0
        WHEN va.min_map < 70 THEN 0.5
        ELSE 0.0
      END +
      
      -- High creatinine component (0-1 points)
      CASE 
        WHEN la.max_creatinine > 2.0 THEN 1.0
        WHEN la.max_creatinine > 1.5 THEN 0.5
        ELSE 0.0
      END +
      
      -- Low O2 saturation component (0-1 points)
      CASE 
        WHEN va.min_o2_saturation < 88 THEN 1.0
        WHEN va.min_o2_saturation < 92 THEN 0.5
        ELSE 0.0
      END +
      
      -- High WBC component (0-1 points)
      CASE 
        WHEN la.max_wbc > 20 THEN 1.0
        WHEN la.max_wbc > 12 THEN 0.5
        ELSE 0.0
      END +
      
      -- High lactate component (0-1 points)
      CASE 
        WHEN la.max_lactate > 4.0 THEN 1.0
        WHEN la.max_lactate > 2.0 THEN 0.5
        WHEN la.max_lactate IS NULL THEN 0.3 -- Mild penalty for missing lactate
        ELSE 0.0
      END +
      
      -- Low GCS component (0-1 points)
      CASE 
        WHEN va.min_gcs < 9 THEN 1.0
        WHEN va.min_gcs < 13 THEN 0.5
        WHEN va.min_gcs IS NULL THEN 0.3 -- Mild penalty for missing GCS
        ELSE 0.0
      END
    ) / 6 AS severity_score  -- Normalize to 0-1 range
  
  FROM `your_dataset.vitals_agg` va
  JOIN `your_dataset.lab_events_agg` la
    ON va.hadm_id = la.hadm_id
)

-- Select and combine all relevant data for the analysis
SELECT
    -- Patient and admission identifiers
    pc.subject_id,
    pc.hadm_id,
    
    -- Demographic information
    pc.gender,
    pc.anchor_age AS age,
    pc.ethnicity_simplified,
    pc.insurance,
    pc.language,
    pc.marital_status,
    
    -- Admission details
    pc.admission_type,
    pc.los_days,
    pc.had_icu_stay,
    pc.num_icu_stays,
    pc.total_icu_los_days,
    
    -- Clinical factors (for explaining variations)
    ss.severity_score,
    la.max_creatinine,
    la.max_wbc,
    la.max_lactate,
    la.max_bilirubin,
    va.min_gcs,
    va.min_map,
    va.min_o2_saturation,
    
    -- Care patterns - Lab tests
    la.lab_tests_per_day,
    la.cbc_tests_per_day,
    la.chem_tests_per_day,
    la.lft_tests_per_day,
    la.coag_tests_per_day,
    la.abg_tests_per_day,
    
    la.creatinine_tests_per_day,
    la.wbc_tests_per_day,
    la.lactate_tests_per_day,
    la.troponin_tests_per_day,
    
    -- Care patterns - Timing of first labs
    la.hours_to_first_creatinine,
    la.hours_to_first_wbc,
    la.hours_to_first_lactate,
    la.hours_to_first_troponin,
    
    -- Care patterns - Vitals monitoring
    va.vital_timepoints_per_day,
    va.hr_checks_per_day,
    va.bp_checks_per_day,
    va.resp_checks_per_day,
    va.temp_checks_per_day,
    va.neuro_checks_per_day,
    
    -- Care patterns - Abnormal values monitoring
    la.pct_abnormal_creatinine,
    la.pct_abnormal_wbc,
    la.pct_abnormal_potassium,
    va.abnormal_hr_per_day,
    va.abnormal_sbp_per_day,
    va.abnormal_resp_per_day,
    va.abnormal_o2_per_day,
    va.abnormal_temp_per_day,
    
    -- Care patterns - Vital sign variability
    va.hr_stddev,
    va.map_stddev,
    va.resp_stddev,
    
    -- Create a simulated outcome variable for demonstration
    -- This is for the FairnessEvaluator demonstration
    -- In a real analysis, you would use actual outcomes like mortality
    CASE
        -- Higher mortality with higher severity, older age
        WHEN ss.severity_score > 0.7 AND pc.anchor_age > 70 THEN 1
        WHEN ss.severity_score > 0.8 THEN 1
        WHEN ss.severity_score > 0.6 AND pc.anchor_age > 80 THEN 1
        WHEN ss.severity_score > 0.5 AND va.min_map < 60 THEN 1
        WHEN ss.severity_score > 0.5 AND la.max_lactate > 4 THEN 1
        -- Add some random variability
        WHEN RAND() < 0.05 THEN 1
        ELSE 0
    END AS outcome_variable

FROM `your_dataset.patients_cohort` pc
JOIN `your_dataset.lab_events_agg` la
  ON pc.hadm_id = la.hadm_id
JOIN `your_dataset.vitals_agg` va
  ON pc.hadm_id = va.hadm_id
JOIN severity_scores ss
  ON pc.hadm_id = ss.hadm_id;

-- Show the balance of demographic variables in the final dataset
SELECT 
    gender,
    COUNT(*) AS count,
    ROUND(AVG(severity_score), 3) AS avg_severity,
    ROUND(AVG(los_days), 1) AS avg_los,
    ROUND(AVG(lab_tests_per_day), 1) AS avg_labs_per_day,
    ROUND(AVG(vital_timepoints_per_day), 1) AS avg_vitals_per_day,
    ROUND(AVG(outcome_variable) * 100, 1) AS outcome_rate
FROM `your_dataset.cohort_data`
GROUP BY gender
ORDER BY count DESC;

SELECT 
    ethnicity_simplified,
    COUNT(*) AS count,
    ROUND(AVG(severity_score), 3) AS avg_severity,
    ROUND(AVG(los_days), 1) AS avg_los,
    ROUND(AVG(lab_tests_per_day), 1) AS avg_labs_per_day,
    ROUND(AVG(vital_timepoints_per_day), 1) AS avg_vitals_per_day,
    ROUND(AVG(outcome_variable) * 100, 1) AS outcome_rate
FROM `your_dataset.cohort_data`
GROUP BY ethnicity_simplified
ORDER BY count DESC;

-- Show correlations between severity and care patterns
SELECT 
    ROUND(CORR(severity_score, lab_tests_per_day), 3) AS corr_severity_labs,
    ROUND(CORR(severity_score, vital_timepoints_per_day), 3) AS corr_severity_vitals,
    ROUND(CORR(severity_score, cbc_tests_per_day), 3) AS corr_severity_cbc,
    ROUND(CORR(severity_score, creatinine_tests_per_day), 3) AS corr_severity_creatinine,
    ROUND(CORR(severity_score, lactate_tests_per_day), 3) AS corr_severity_lactate,
    ROUND(CORR(severity_score, bp_checks_per_day), 3) AS corr_severity_bp
FROM `your_dataset.cohort_data`;

-- Export instructions:
-- In BigQuery UI: Export the table as CSV to 'cohort_data.csv'
-- Using CLI: bq extract your_dataset.cohort_data '../data/cohort_data.csv' 