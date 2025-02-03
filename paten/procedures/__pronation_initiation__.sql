SELECT
  obs.procedure_occurrence_id,
  obs.person_id,
  obs.visit_occurrence_id,
  obs.procedure_datetime,
  obs.procedure_source_value,
  obs.procedure_concept_id,
  cohort.intubation,
  cohort.extubation
FROM `amsterdamumcdb.version1_5_0.procedure_occurrence` obs
INNER JOIN ({}) cohort ON cohort.visit_occurrence_id=obs.visit_occurrence_id
    AND obs.procedure_datetime BETWEEN cohort.intubation AND cohort.extubation
WHERE obs.procedure_concept_id IN (4196006)
  AND obs.provider_id IS NULL