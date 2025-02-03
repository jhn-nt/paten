SELECT visit.person_id, visit.visit_occurrence_id
FROM `amsterdamumcdb.version1_5_0.visit_occurrence` visit
INNER JOIN ({}) cohort ON cohort.person_id=visit.person_id AND cohort.visit_occurrence_id=visit.visit_occurrence_id
WHERE discharged_to_source_value='Overleden'