-- 01_extract_patients.sql
-- Extracts adult patients with hospital stays of 3+ days
-- Compatible with MIMIC-IV on BigQuery

-- Create or replace a table for the patient cohort
CREATE OR REPLACE TABLE `care-phenotypes.mimic_cohorts.patients_cohort` AS

-- Select patients meeting inclusion criteria
SELECT 
    p.subject_id,
    p.anchor_age,  -- Patient age at first admission
    p.anchor_year, -- Year of patient birth
    p.gender,      -- Patient gender
    a.hadm_id,     -- Hospital admission ID
    a.admittime,   -- Admission time
    a.dischtime,   -- Discharge time
    a.admission_type,      -- Type of admission (emergency, elective, etc.)
    a.admission_location,  -- Where patient was admitted from
    a.insurance,           -- Insurance type
    a.language,            -- Patient language
    a.marital_status,      -- Patient marital status
    a.race,                -- Patient race/ethnicity
    
    -- Calculate length of stay in days
    DATETIME_DIFF(a.dischtime, a.admittime, HOUR) / 24.0 AS los_days,
    
    -- Create a simplified ethnicity column
    CASE
        WHEN UPPER(a.race) LIKE '%WHITE%' THEN 'White'
        WHEN UPPER(a.race) LIKE '%BLACK%' THEN 'Black'
        WHEN UPPER(a.race) LIKE '%HISPANIC%' OR UPPER(a.race) LIKE '%LATINO%' THEN 'Hispanic'
        WHEN UPPER(a.race) LIKE '%ASIAN%' THEN 'Asian'
        WHEN UPPER(a.race) LIKE '%PACIFIC ISLANDER%' THEN 'Pacific Islander'
        WHEN UPPER(a.race) LIKE '%NATIVE%' THEN 'Native American'
        ELSE 'Other'
    END AS ethnicity_simplified,
    
    -- Add ICU stay information
    CASE WHEN icu.stay_id IS NOT NULL THEN 1 ELSE 0 END AS had_icu_stay,
    COUNT(DISTINCT icu.stay_id) AS num_icu_stays,
    SUM(DATETIME_DIFF(icu.outtime, icu.intime, HOUR)) / 24.0 AS total_icu_los_days

FROM `physionet-data.mimiciv_3_1_hosp.patients` p
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a 
    ON p.subject_id = a.subject_id
LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` icu 
    ON a.hadm_id = icu.hadm_id

WHERE 
    -- Adult patients only (≥18 years)
    p.anchor_age >= 18
    
    -- Hospital stays of at least 3 days
    AND DATETIME_DIFF(a.dischtime, a.admittime, HOUR) >= 72
    
    -- Exclude patients with missing key demographic data
    AND p.gender IS NOT NULL
    AND a.race IS NOT NULL
    
    -- Optional: Limit to a specific time period
    -- AND EXTRACT(YEAR FROM a.admittime) >= 2008

GROUP BY 
    p.subject_id, p.anchor_age, p.anchor_year, p.gender,
    a.hadm_id, a.admittime, a.dischtime, a.admission_type, 
    a.admission_location, a.insurance, a.language, a.marital_status, a.race,
    ethnicity_simplified, had_icu_stay

-- Optional: Limit the cohort size for testing
-- LIMIT 1000
;

-- Output cohort size and basic demographics
SELECT 
    COUNT(*) AS total_admissions,
    COUNT(DISTINCT subject_id) AS unique_patients,
    AVG(anchor_age) AS avg_age,
    AVG(los_days) AS avg_los_days,
    
    -- Gender distribution
    COUNTIF(gender = 'F') / COUNT(*) * 100 AS pct_female,
    
    -- ICU statistics
    COUNTIF(had_icu_stay = 1) / COUNT(*) * 100 AS pct_with_icu_stay,
    AVG(num_icu_stays) AS avg_icu_stays_per_admission,
    
    -- Race/ethnicity distribution (simplified)
    COUNTIF(ethnicity_simplified = 'White') / COUNT(*) * 100 AS pct_white,
    COUNTIF(ethnicity_simplified = 'Black') / COUNT(*) * 100 AS pct_black,
    COUNTIF(ethnicity_simplified = 'Hispanic') / COUNT(*) * 100 AS pct_hispanic,
    COUNTIF(ethnicity_simplified = 'Asian') / COUNT(*) * 100 AS pct_asian,
    COUNTIF(ethnicity_simplified = 'Other') / COUNT(*) * 100 AS pct_other

FROM `care-phenotypes.mimic_cohorts.patients_cohort`;

-- Export instructions:
-- In BigQuery UI: Export the table as CSV to 'patients_cohort.csv'
-- Using CLI: bq extract care-phenotypes.mimic_cohorts.patients_cohort '../data/patients_cohort.csv' 