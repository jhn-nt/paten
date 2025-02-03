SELECT
  obs.measurement_id,
  obs.person_id,
  obs.visit_occurrence_id,
  obs.measurement_datetime,
  obs.measurement_source_value,
  obs.measurement_concept_id,
  obs.value_as_number,
  obs.unit_source_value
FROM `amsterdamumcdb.version1_5_0.measurement` obs
INNER JOIN ({}) cohort ON cohort.visit_occurrence_id=obs.visit_occurrence_id
    AND obs.measurement_datetime BETWEEN cohort.intubation AND cohort.extubation
WHERE obs.measurement_concept_id IN (3027315,3024882, 3027315,3022875)
  AND obs.provider_id IS NOT NULL