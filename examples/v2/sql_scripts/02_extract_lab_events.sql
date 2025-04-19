-- 02_extract_lab_events.sql
-- Extracts and aggregates lab measurements for the patient cohort
-- Compatible with MIMIC-IV on BigQuery

-- Create or replace a table for the aggregated lab events
CREATE OR REPLACE TABLE `your_dataset.lab_events_agg` AS

-- Common lab test item IDs in MIMIC-IV
WITH lab_groups AS (
  SELECT * FROM UNNEST([
    -- CBC (Complete Blood Count)
    STRUCT(51301 as itemid, 'WBC' as lab_group, 'WBC' as lab_name),
    STRUCT(51300 as itemid, 'CBC' as lab_group, 'RBC' as lab_name),
    STRUCT(51221 as itemid, 'CBC' as lab_group, 'HGB' as lab_name),
    STRUCT(51222 as itemid, 'CBC' as lab_group, 'HCT' as lab_name),
    STRUCT(51265 as itemid, 'CBC' as lab_group, 'PLT' as lab_name),
    
    -- Basic Metabolic Panel (BMP)
    STRUCT(50912 as itemid, 'BMP' as lab_group, 'CREATININE' as lab_name),
    STRUCT(50806 as itemid, 'BMP' as lab_group, 'BUN' as lab_name),
    STRUCT(50931 as itemid, 'BMP' as lab_group, 'GLUCOSE' as lab_name),
    STRUCT(50971 as itemid, 'BMP' as lab_group, 'POTASSIUM' as lab_name),
    STRUCT(50983 as itemid, 'BMP' as lab_group, 'SODIUM' as lab_name),
    STRUCT(50882 as itemid, 'BMP' as lab_group, 'BICARBONATE' as lab_name),
    STRUCT(50893 as itemid, 'BMP' as lab_group, 'CALCIUM' as lab_name),
    STRUCT(50902 as itemid, 'BMP' as lab_group, 'CHLORIDE' as lab_name),
    
    -- Liver Function Tests (LFT)
    STRUCT(50861 as itemid, 'LFT' as lab_group, 'ALT' as lab_name),
    STRUCT(50863 as itemid, 'LFT' as lab_group, 'AST' as lab_name),
    STRUCT(50878 as itemid, 'LFT' as lab_group, 'BILIRUBIN_TOTAL' as lab_name),
    STRUCT(50927 as itemid, 'LFT' as lab_group, 'GGT' as lab_name),
    STRUCT(50862 as itemid, 'LFT' as lab_group, 'ALBUMIN' as lab_name),
    STRUCT(50976 as itemid, 'LFT' as lab_group, 'PROTEIN_TOTAL' as lab_name),
    STRUCT(50885 as itemid, 'LFT' as lab_group, 'ALP' as lab_name),
    
    -- Coagulation
    STRUCT(51237 as itemid, 'COAG' as lab_group, 'INR' as lab_name),
    STRUCT(51275 as itemid, 'COAG' as lab_group, 'PTT' as lab_name),
    STRUCT(51274 as itemid, 'COAG' as lab_group, 'PT' as lab_name),
    
    -- Inflammatory Markers
    STRUCT(50889 as itemid, 'INFL' as lab_group, 'CRP' as lab_name),
    
    -- Arterial Blood Gas (ABG)
    STRUCT(50820 as itemid, 'ABG' as lab_group, 'PH_ARTERIAL' as lab_name),
    STRUCT(50802 as itemid, 'ABG' as lab_group, 'BASE_EXCESS' as lab_name),
    STRUCT(50804 as itemid, 'ABG' as lab_group, 'BICARB_ARTERIAL' as lab_name),
    STRUCT(50821 as itemid, 'ABG' as lab_group, 'PO2_ARTERIAL' as lab_name),
    STRUCT(50818 as itemid, 'ABG' as lab_group, 'PCO2_ARTERIAL' as lab_name),
    
    -- Other important labs
    STRUCT(51006 as itemid, 'OTHER' as lab_group, 'TROPONIN' as lab_name),
    STRUCT(51144 as itemid, 'OTHER' as lab_group, 'LACTATE' as lab_name)
  ])
)

-- Select lab measurements for patients in the cohort
SELECT 
    pc.subject_id,
    pc.hadm_id,
    pc.admittime,
    pc.dischtime,
    pc.los_days,
    
    -- Aggregate lab measurements by admission
    -- Overall test frequency metrics
    COUNT(DISTINCT le.labevent_id) AS total_lab_measurements,
    COUNT(DISTINCT le.labevent_id) / pc.los_days AS lab_tests_per_day,
    
    -- Lab test group frequencies (per day)
    COUNTIF(lg.lab_group = 'CBC') / pc.los_days AS cbc_tests_per_day,
    COUNTIF(lg.lab_group = 'BMP') / pc.los_days AS chem_tests_per_day,
    COUNTIF(lg.lab_group = 'LFT') / pc.los_days AS lft_tests_per_day,
    COUNTIF(lg.lab_group = 'COAG') / pc.los_days AS coag_tests_per_day,
    COUNTIF(lg.lab_group = 'ABG') / pc.los_days AS abg_tests_per_day,
    
    -- Test frequencies for specific important tests
    COUNTIF(lg.lab_name = 'CREATININE') / pc.los_days AS creatinine_tests_per_day,
    COUNTIF(lg.lab_name = 'WBC') / pc.los_days AS wbc_tests_per_day,
    COUNTIF(lg.lab_name = 'LACTATE') / pc.los_days AS lactate_tests_per_day,
    COUNTIF(lg.lab_name = 'TROPONIN') / pc.los_days AS troponin_tests_per_day,
    
    -- Key clinical values (max, min, or avg depending on clinical relevance)
    MAX(CASE WHEN lg.lab_name = 'CREATININE' THEN le.valuenum ELSE NULL END) AS max_creatinine,
    MAX(CASE WHEN lg.lab_name = 'WBC' THEN le.valuenum ELSE NULL END) AS max_wbc,
    MAX(CASE WHEN lg.lab_name = 'LACTATE' THEN le.valuenum ELSE NULL END) AS max_lactate,
    MIN(CASE WHEN lg.lab_name = 'PH_ARTERIAL' THEN le.valuenum ELSE NULL END) AS min_ph,
    MAX(CASE WHEN lg.lab_name = 'TROPONIN' THEN le.valuenum ELSE NULL END) AS max_troponin,
    MIN(CASE WHEN lg.lab_name = 'HGB' THEN le.valuenum ELSE NULL END) AS min_hgb,
    MAX(CASE WHEN lg.lab_name = 'BILIRUBIN_TOTAL' THEN le.valuenum ELSE NULL END) AS max_bilirubin,
    MAX(CASE WHEN lg.lab_name = 'BUN' THEN le.valuenum ELSE NULL END) AS max_bun,
    MAX(CASE WHEN lg.lab_name = 'CRP' THEN le.valuenum ELSE NULL END) AS max_crp,
    MAX(CASE WHEN lg.lab_name = 'POTASSIUM' THEN le.valuenum ELSE NULL END) AS max_potassium,
    MIN(CASE WHEN lg.lab_name = 'POTASSIUM' THEN le.valuenum ELSE NULL END) AS min_potassium,
    MAX(CASE WHEN lg.lab_name = 'SODIUM' THEN le.valuenum ELSE NULL END) AS max_sodium,
    MIN(CASE WHEN lg.lab_name = 'SODIUM' THEN le.valuenum ELSE NULL END) AS min_sodium,
    
    -- Time to first key lab tests (in hours from admission)
    MIN(CASE WHEN lg.lab_name = 'CREATININE' 
             THEN DATETIME_DIFF(le.charttime, pc.admittime, HOUR) 
             ELSE NULL END) AS hours_to_first_creatinine,
    MIN(CASE WHEN lg.lab_name = 'WBC' 
             THEN DATETIME_DIFF(le.charttime, pc.admittime, HOUR) 
             ELSE NULL END) AS hours_to_first_wbc,
    MIN(CASE WHEN lg.lab_name = 'LACTATE' 
             THEN DATETIME_DIFF(le.charttime, pc.admittime, HOUR) 
             ELSE NULL END) AS hours_to_first_lactate,
    MIN(CASE WHEN lg.lab_name = 'TROPONIN' 
             THEN DATETIME_DIFF(le.charttime, pc.admittime, HOUR) 
             ELSE NULL END) AS hours_to_first_troponin,
             
    -- Frequency of abnormal results (as percentage of total measurements)
    COUNTIF(lg.lab_name = 'CREATININE' AND le.valuenum > 1.3) / 
        NULLIF(COUNTIF(lg.lab_name = 'CREATININE'), 0) * 100 AS pct_abnormal_creatinine,
    COUNTIF(lg.lab_name = 'WBC' AND (le.valuenum < 4.0 OR le.valuenum > 11.0)) / 
        NULLIF(COUNTIF(lg.lab_name = 'WBC'), 0) * 100 AS pct_abnormal_wbc,
    COUNTIF(lg.lab_name = 'POTASSIUM' AND (le.valuenum < 3.5 OR le.valuenum > 5.0)) / 
        NULLIF(COUNTIF(lg.lab_name = 'POTASSIUM'), 0) * 100 AS pct_abnormal_potassium

FROM `your_dataset.patients_cohort` pc
LEFT JOIN `physionet-data.mimic_hosp.labevents` le
    ON pc.hadm_id = le.hadm_id
LEFT JOIN lab_groups lg
    ON le.itemid = lg.itemid

WHERE 
    -- Exclude invalid lab values
    le.valuenum IS NOT NULL
    AND le.valuenum > 0
    -- Filter to labs during this admission (optional, depending on needs)
    AND le.charttime BETWEEN pc.admittime AND pc.dischtime

GROUP BY 
    pc.subject_id, pc.hadm_id, pc.admittime, pc.dischtime, pc.los_days;

-- Output summary statistics
SELECT 
    COUNT(*) AS total_admissions,
    AVG(total_lab_measurements) AS avg_lab_measurements_per_admission,
    AVG(lab_tests_per_day) AS avg_lab_tests_per_day,
    AVG(cbc_tests_per_day) AS avg_cbc_tests_per_day,
    AVG(chem_tests_per_day) AS avg_chem_tests_per_day,
    AVG(max_creatinine) AS avg_max_creatinine,
    AVG(max_wbc) AS avg_max_wbc,
    AVG(max_lactate) AS avg_max_lactate
FROM `your_dataset.lab_events_agg`;

-- Export instructions:
-- In BigQuery UI: Export the table as CSV to 'lab_events_agg.csv'
-- Using CLI: bq extract your_dataset.lab_events_agg '../data/lab_events_agg.csv' 