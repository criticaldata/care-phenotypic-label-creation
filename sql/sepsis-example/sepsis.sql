-- Sepsis Care Phenotype Cohort Extraction from MIMIC-IV
-- Note: This is a skeleton script based on general MIMIC knowledge
-- Table names and column specifics may need adjustment

WITH 
-- Step 1: Identify adult patients in ICU with stays >= 24 hours
adult_icu_patients AS (
    SELECT 
        i.subject_id, 
        i.hadm_id, 
        i.stay_id, 
        i.intime, 
        i.outtime,
        DATETIME_DIFF(i.outtime, i.intime, 'HOUR') AS icu_los_hours,
        p.gender,
        p.anchor_age AS age,
        a.race, 
        a.language, 
        a.insurance,
        CASE 
            WHEN EXTRACT(WEEKDAY FROM i.intime) IN (0, 6) THEN 'weekend' 
            ELSE 'weekday' 
        END AS admission_day_type,
        CASE 
            WHEN EXTRACT(HOUR FROM i.intime) BETWEEN 7 AND 19 THEN 'day' 
            ELSE 'night' 
        END AS admission_time_of_day
    FROM 
        `physionet-data.mimiciv_icu.icustays` i
    INNER JOIN 
        `physionet-data.mimiciv_hosp.patients` p ON i.subject_id = p.subject_id
    INNER JOIN 
        `physionet-data.mimiciv_hosp.admissions` a ON i.hadm_id = a.hadm_id
    WHERE 
        p.anchor_age >= 18
        AND DATETIME_DIFF(i.outtime, i.intime, 'HOUR') >= 24
),

-- Step 2: Calculate SOFA scores and identify sepsis patients (Sepsis-3)
sofa_components AS (
    SELECT 
        i.stay_id,
        i.subject_id,
        i.hadm_id,
        
        -- Respiratory component (PaO2/FiO2 ratio)
        MAX(CASE WHEN c.itemid IN (220224, 220225) THEN c.valuenum ELSE NULL END) AS pao2_fio2_ratio,
        
        -- Cardiovascular component (MAP or vasopressors)
        MAX(CASE WHEN c.itemid IN (220052, 220181, 225312) THEN c.valuenum ELSE NULL END) AS map,
        MAX(CASE WHEN m.itemid IN (221906, 221289, 221662, 221653) THEN 1 ELSE 0 END) AS vasopressor,
        
        -- Hepatic component (Bilirubin)
        MAX(CASE WHEN l.itemid IN (50885, 50927) THEN l.valuenum ELSE NULL END) AS bilirubin,
        
        -- Coagulation component (Platelets)
        MIN(CASE WHEN l.itemid IN (51265) THEN l.valuenum ELSE NULL END) AS platelets,
        
        -- Renal component (Creatinine)
        MAX(CASE WHEN l.itemid IN (50912) THEN l.valuenum ELSE NULL END) AS creatinine,
        
        -- Neurological component (GCS)
        MIN(CASE WHEN c.itemid IN (220739) THEN c.valuenum ELSE NULL END) AS gcs,
        
        -- Additional sepsis markers
        MAX(CASE WHEN l.itemid IN (50813) THEN l.valuenum ELSE NULL END) AS lactate
    FROM 
        adult_icu_patients i
    LEFT JOIN 
        `physionet-data.mimiciv_icu.chartevents` c ON i.stay_id = c.stay_id
    LEFT JOIN 
        `physionet-data.mimiciv_hosp.labevents` l ON i.hadm_id = l.hadm_id
    LEFT JOIN 
        `physionet-data.mimiciv_icu.inputevents` m ON i.stay_id = m.stay_id
    GROUP BY 
        i.stay_id, i.subject_id, i.hadm_id
),

-- Calculate SOFA scores and identify sepsis patients
sepsis_patients AS (
    SELECT 
        a.*,
        s.*,
        -- Calculate SOFA score components (simplified)
        CASE 
            WHEN s.pao2_fio2_ratio < 100 THEN 4
            WHEN s.pao2_fio2_ratio < 200 THEN 3
            WHEN s.pao2_fio2_ratio < 300 THEN 2
            WHEN s.pao2_fio2_ratio < 400 THEN 1
            ELSE 0
        END AS sofa_respiration,
        
        CASE 
            WHEN s.map < 70 OR s.vasopressor = 1 THEN 1
            ELSE 0
        END AS sofa_cardiovascular,
        
        CASE 
            WHEN s.bilirubin > 12 THEN 4
            WHEN s.bilirubin > 6 THEN 3
            WHEN s.bilirubin > 2 THEN 2
            WHEN s.bilirubin > 1.2 THEN 1
            ELSE 0
        END AS sofa_liver,
        
        CASE 
            WHEN s.platelets < 20 THEN 4
            WHEN s.platelets < 50 THEN 3
            WHEN s.platelets < 100 THEN 2
            WHEN s.platelets < 150 THEN 1
            ELSE 0
        END AS sofa_coagulation,
        
        CASE 
            WHEN s.creatinine > 5 THEN 4
            WHEN s.creatinine > 3.5 THEN 3
            WHEN s.creatinine > 2 THEN 2
            WHEN s.creatinine > 1.2 THEN 1
            ELSE 0
        END AS sofa_renal,
        
        CASE 
            WHEN s.gcs < 6 THEN 4
            WHEN s.gcs < 10 THEN 3
            WHEN s.gcs < 13 THEN 2
            WHEN s.gcs < 15 THEN 1
            ELSE 0
        END AS sofa_cns
    FROM 
        adult_icu_patients a
    INNER JOIN 
        sofa_components s ON a.stay_id = s.stay_id
),

sepsis_cohort AS (
    SELECT 
        *,
        (sofa_respiration + sofa_cardiovascular + sofa_liver + sofa_coagulation + sofa_renal + sofa_cns) AS sofa_total
    FROM 
        sepsis_patients
    WHERE 
        -- Increase in SOFA score of 2 or more (would need to calculate baseline - simplified here)
        sofa_total >= 2 
),

-- Step 3: Get comfort care orders to exclude these patients
comfort_care AS (
    SELECT DISTINCT
        subject_id,
        hadm_id
    FROM 
        `physionet-data.mimiciv_hosp.procedures_icd` 
    WHERE 
        icd_code IN ('Z66') -- ICD-10 code for palliative care
),

-- Step 4: Get infection data and culture results
infection_data AS (
    SELECT 
        subject_id, 
        hadm_id,
        CASE
            WHEN icd_code LIKE 'J%' THEN 'pulmonary'
            WHEN icd_code LIKE 'N39.0%' THEN 'urinary'
            WHEN icd_code LIKE 'K%' THEN 'abdominal'
            ELSE 'other'
        END AS infection_site
    FROM 
        `physionet-data.mimiciv_hosp.diagnoses_icd`
    WHERE 
        icd_code IN (
            -- Infection codes (simplified - would need expansion)
            'J18.9', -- Pneumonia
            'N39.0', -- UTI
            'K65.9'  -- Peritonitis
        )
),

culture_results AS (
    SELECT 
        subject_id,
        hadm_id,
        MAX(CASE WHEN org_name IS NOT NULL THEN 1 ELSE 0 END) AS positive_culture
    FROM 
        `physionet-data.mimiciv_hosp.microbiologyevents`
    GROUP BY 
        subject_id, hadm_id
),

-- Step 5: Get lab test frequencies and timing
lab_frequencies AS (
    SELECT 
        l.subject_id,
        l.hadm_id,
        -- CBC frequency
        COUNT(DISTINCT CASE WHEN l.itemid IN (51300, 51301) THEN l.charttime END) / 
            GREATEST(1, DATETIME_DIFF(i.outtime, i.intime, 'DAY')) AS cbc_per_day,
        -- Chemistry frequency
        COUNT(DISTINCT CASE WHEN l.itemid IN (50912, 50902, 50868) THEN l.charttime END) / 
            GREATEST(1, DATETIME_DIFF(i.outtime, i.intime, 'DAY')) AS chem_per_day,
        -- Blood gas frequency
        COUNT(DISTINCT CASE WHEN l.itemid IN (50821, 50818) THEN l.charttime END) / 
            GREATEST(1, DATETIME_DIFF(i.outtime, i.intime, 'DAY')) AS abg_per_day,
        -- Lactate frequency
        COUNT(DISTINCT CASE WHEN l.itemid IN (50813) THEN l.charttime END) / 
            GREATEST(1, DATETIME_DIFF(i.outtime, i.intime, 'DAY')) AS lactate_per_day
    FROM 
        `physionet-data.mimiciv_hosp.labevents` l
    JOIN 
        adult_icu_patients i ON l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id
    GROUP BY 
        l.subject_id, l.hadm_id, i.outtime, i.intime
),

-- Step 6: Get hemodynamic monitoring details
hemodynamic_monitoring AS (
    SELECT 
        ce.subject_id,
        ce.hadm_id,
        ce.stay_id,
        -- Arterial line presence
        MAX(CASE WHEN ce.itemid IN (220050, 220051) THEN 1 ELSE 0 END) AS arterial_line,
        -- Central line presence
        MAX(CASE WHEN ce.itemid IN (220046) THEN 1 ELSE 0 END) AS central_line,
        -- Vital sign frequency
        COUNT(DISTINCT CASE WHEN ce.itemid IN (220045, 220050, 220179) THEN ce.charttime END) / 
            GREATEST(1, DATETIME_DIFF(i.outtime, i.intime, 'DAY')) AS vitals_per_day
    FROM 
        `physionet-data.mimiciv_icu.chartevents` ce
    JOIN 
        adult_icu_patients i ON ce.stay_id = i.stay_id
    GROUP BY 
        ce.subject_id, ce.hadm_id, ce.stay_id, i.outtime, i.intime
),

-- Step 7: Get treatment timing
treatment_timing AS (
    SELECT 
        p.subject_id,
        p.hadm_id,
        -- Time to first antibiotic (simplified)
        MIN(DATETIME_DIFF(m.starttime, a.intime, 'HOUR')) AS hours_to_antibiotic
    FROM 
        `physionet-data.mimiciv_icu.prescriptions` p
    JOIN 
        `physionet-data.mimiciv_hosp.admissions` a ON p.hadm_id = a.hadm_id
    JOIN 
        `physionet-data.mimiciv_icu.inputevents` m ON p.hadm_id = m.hadm_id
    WHERE 
        p.drug_type IN ('ANTIBIOTIC')
    GROUP BY 
        p.subject_id, p.hadm_id
),

-- Step 8: Get comorbidity data for Charlson Index
comorbidity_data AS (
    SELECT 
        d.subject_id,
        d.hadm_id,
        -- Simplified Charlson components
        MAX(CASE WHEN d.icd_code LIKE 'E11%' THEN 1 ELSE 0 END) AS diabetes,
        MAX(CASE WHEN d.icd_code LIKE 'J44%' THEN 1 ELSE 0 END) AS copd,
        MAX(CASE WHEN d.icd_code LIKE 'I50%' THEN 1 ELSE 0 END) AS chf,
        MAX(CASE WHEN d.icd_code LIKE 'C%' THEN 1 ELSE 0 END) AS cancer,
        MAX(CASE WHEN d.icd_code IN ('B20') THEN 1 ELSE 0 END) AS immunosuppression
    FROM 
        `physionet-data.mimiciv_hosp.diagnoses_icd` d
    GROUP BY 
        d.subject_id, d.hadm_id
)

-- Final cohort selection with all required features
SELECT 
    s.subject_id,
    s.hadm_id,
    s.stay_id,
    s.age,
    s.gender,
    s.race,
    s.language,
    s.insurance,
    s.admission_day_type,
    s.admission_time_of_day,
    s.icu_los_hours,
    
    -- Clinical factors
    s.sofa_respiration,
    s.sofa_cardiovascular,
    s.sofa_liver,
    s.sofa_coagulation,
    s.sofa_renal,
    s.sofa_cns,
    s.sofa_total,
    s.lactate,
    
    -- Calculate simplified Charlson (would need refinement)
    (c.diabetes + c.copd + c.chf + c.cancer*2 + c.immunosuppression*6) AS charlson_score,
    c.diabetes,
    c.copd,
    c.chf,
    c.cancer,
    c.immunosuppression,
    
    -- Infection data
    i.infection_site,
    cr.positive_culture,
    
    -- Care patterns - lab frequencies
    l.cbc_per_day,
    l.chem_per_day,
    l.abg_per_day,
    l.lactate_per_day,
    
    -- Care patterns - hemodynamic monitoring
    h.arterial_line,
    h.central_line,
    h.vitals_per_day,
    
    -- Care patterns - treatment timing
    t.hours_to_antibiotic
    
FROM 
    sepsis_cohort s
LEFT JOIN 
    comorbidity_data c ON s.subject_id = c.subject_id AND s.hadm_id = c.hadm_id
LEFT JOIN 
    infection_data i ON s.subject_id = i.subject_id AND s.hadm_id = i.hadm_id
LEFT JOIN 
    culture_results cr ON s.subject_id = cr.subject_id AND s.hadm_id = cr.hadm_id
LEFT JOIN 
    lab_frequencies l ON s.subject_id = l.subject_id AND s.hadm_id = l.hadm_id
LEFT JOIN 
    hemodynamic_monitoring h ON s.stay_id = h.stay_id
LEFT JOIN 
    treatment_timing t ON s.subject_id = t.subject_id AND s.hadm_id = t.hadm_id
WHERE 
    NOT EXISTS (
        SELECT 1 
        FROM comfort_care cc 
        WHERE s.subject_id = cc.subject_id AND s.hadm_id = cc.hadm_id
    )
;