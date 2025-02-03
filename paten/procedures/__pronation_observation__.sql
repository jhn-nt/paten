SELECT obs.*, cohort.intubation, cohort.extubation
FROM `amsterdamumcdb.version1_5_0.observation` obs
INNER JOIN ({}) cohort ON cohort.person_id=obs.person_id
  AND cohort.visit_occurrence_id=obs.visit_occurrence_id
  AND obs.observation_datetime BETWEEN cohort.intubation AND cohort.extubation
WHERE value_as_concept_id IN (4221822,4050473,4009274,4010960)
  AND provider_id IS NOT NULL