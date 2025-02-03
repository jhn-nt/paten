SELECT
    person_id,
    visit_occurrence_id,
    device_exposure_start_datetime AS intubation,
    device_exposure_end_datetime AS extubation,
    ROUND(DATETIME_DIFF(device_exposure_end_datetime, device_exposure_start_datetime, MINUTE)/60,2) AS duration_hours
FROM `amsterdamumcdb.version1_5_0.device_exposure`
WHERE device_concept_id IN (
      4044008, -- Tracheostomy,
      4097216 -- Endotracheal Tube
      )