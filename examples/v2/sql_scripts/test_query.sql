-- Test query for MIMIC-IV 
-- This is a simplified version to test table access

-- Simple join between patients and admissions
SELECT 
    p.subject_id,
    p.gender,
    a.hadm_id,
    a.admittime
FROM `physionet-data.mimiciv_3_1_hosp.patients` p
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a 
    ON p.subject_id = a.subject_id
LIMIT 10;

-- Test tables individually
-- Uncomment one at a time to test

-- SELECT * FROM `physionet-data.mimiciv_3_1_hosp.patients` LIMIT 5;
-- SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions` LIMIT 5;
-- SELECT * FROM `physionet-data.mimiciv_3_1_icu.icustays` LIMIT 5; 