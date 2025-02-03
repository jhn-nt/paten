WITH first_last_measurements AS (
  -- Subquery to get the first and last measurement time for each visit_occurrence_id
  SELECT
    person_id,
    visit_occurrence_id,
    device_exposure_start_date AS intubation,
    device_exposure_start_datetime AS intubation_datetime,
    device_exposure_end_date AS extubation
  FROM amsterdamumcdb.version1_5_0.device_exposure
  WHERE device_concept_id IN (
    4044008, -- Tracheostomy
    4097216  -- Endotracheal Tube
  )
  --GROUP BY visit_occurrence_id
)

SELECT
con.concept_name,
flm.visit_occurrence_id,
flm.intubation_datetime AS intubation,
cond_era.condition_era_start_date,
cond_era.condition_era_end_date
FROM amsterdamumcdb.version1_5_0.condition_era cond_era
inner join amsterdamumcdb.version1_5_0.concept con ON con.concept_id = cond_era.condition_concept_id
-- Only consider measurements within the time window defined by the first and last "Beademings Temperatuur"
JOIN first_last_measurements flm
  ON cond_era.person_id = flm.person_id
  AND cond_era.condition_era_start_date BETWEEN flm.intubation AND flm.extubation
WHERE con.concept_name LIKE '%pneumonia%'
AND con.concept_name != 'Fungal pneumonia'
  --AND cond_era.person_id IN (
  --  SELECT DISTINCT person_id
  --  FROM `amsterdamumcdb.version1_5_0.device_exposure`
  --  WHERE device_concept_id IN (
  --    4044008, -- Tracheostomy
  --    4097216  -- Endotracheal Tube
  --  )
  --)