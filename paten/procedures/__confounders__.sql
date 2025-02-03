SELECT 
    measurement.person_id,
    measurement.visit_occurrence_id,
    cohort.intubation,
    concept.concept_name,
    AVG(measurement.value_as_number) AS value_as_number
FROM `amsterdamumcdb.version1_5_0.measurement` measurement 
INNER JOIN ({0:}) cohort ON cohort.visit_occurrence_id=measurement.visit_occurrence_id
    AND measurement.measurement_datetime BETWEEN cohort.intubation AND cohort.extubation
INNER JOIN `amsterdamumcdb.version1_5_0.concept` concept ON concept.concept_id=measurement.measurement_concept_id
    AND concept.concept_id IN ({1:})
WHERE measurement.provider_id IS NOT NULL
GROUP BY measurement.person_id, measurement.visit_occurrence_id, cohort.intubation, cohort.extubation, concept.concept_name
