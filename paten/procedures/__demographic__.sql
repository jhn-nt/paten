SELECT DISTINCT
  person.person_id,
  visit.visit_occurrence_id,
  visit.visit_source_value AS unit,
  person.year_of_birth,
  person.gender_source_value AS gender,
  DATETIME_DIFF(visit.visit_end_datetime, visit.visit_start_datetime, HOUR)/24 AS los,
  weight.value_as_number AS weight,
  height.value_as_number AS height,
  weight.value_as_number/POWER(height.value_as_number/100,2) AS bmi,
  apache.value_as_number AS apache
FROM `amsterdamumcdb.version1_5_0.person` person
INNER JOIN ({}) cohort ON cohort.person_id=person.person_id
INNER JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` visit
  ON cohort.visit_occurrence_id=visit.visit_occurrence_id
LEFT JOIN (
  SELECT DISTINCT
  visit_occurrence_id,
  FIRST_VALUE(value_as_number) OVER(PARTITION BY visit_occurrence_id ORDER BY observation_datetime) AS value_as_number
  FROM`amsterdamumcdb.version1_5_0.observation`
  WHERE observation_source_value='A_Apache_Score'
AND provider_id IS NOT NULL) apache
  ON apache.visit_occurrence_id=visit.visit_occurrence_id
LEFT JOIN (
  SELECT DISTINCT
  visit_occurrence_id,
  FIRST_VALUE(value_as_number) OVER(PARTITION BY visit_occurrence_id ORDER BY measurement_datetime) AS value_as_number
  FROM`amsterdamumcdb.version1_5_0.measurement`
  WHERE measurement_concept_id IN (3026600, 3013762, 3023166, 3025315)
    AND provider_id IS NULL) weight
  ON weight.visit_occurrence_id=visit.visit_occurrence_id
LEFT JOIN (
  SELECT DISTINCT
  visit_occurrence_id,
  FIRST_VALUE(value_as_number) OVER(PARTITION BY visit_occurrence_id ORDER BY measurement_datetime) AS value_as_number
  FROM`amsterdamumcdb.version1_5_0.measurement`
  WHERE measurement_concept_id IN (3035463, 3023540, 3019171, 3036277)
    AND provider_id IS NULL) height
  ON height.visit_occurrence_id=visit.visit_occurrence_id