WITH first_last_measurements AS (
  -- Subquery to get the first and last measurement time for each visit_occurrence_id
  SELECT
    visit_occurrence_id,
    device_exposure_start_datetime AS intubation,
    device_exposure_end_datetime AS extubation
  FROM `amsterdamumcdb.version1_5_0.device_exposure`
  WHERE device_concept_id IN (
    4044008, -- Tracheostomy
    4097216  -- Endotracheal Tube
  )
),
aggregated_measurements AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    flm.intubation,
    m.measurement_source_value,
    -- Adjusted logic for time windows
    CASE
      WHEN m.measurement_source_value IN ({0:})
      THEN TIMESTAMP_TRUNC(m.measurement_datetime, DAY)
      ELSE m.measurement_datetime
    END AS time_window,
    m.value_as_number
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  JOIN first_last_measurements flm
    ON m.visit_occurrence_id = flm.visit_occurrence_id
    AND m.measurement_datetime BETWEEN flm.intubation AND flm.extubation
  WHERE m.measurement_source_value IN ({0:})
    AND m.provider_id IS NOT NULL
),
final_aggregates AS (
  SELECT
    person_id,
    visit_occurrence_id,
    intubation,
    measurement_source_value,
    time_window,
    -- Aggregate temperature values (hourly)
    MAX(CASE WHEN measurement_source_value IN ({0:})
    THEN value_as_number END) AS max_value,
    MIN(CASE WHEN measurement_source_value IN ({0:})
    THEN value_as_number END) AS min_value,
    AVG(CASE WHEN measurement_source_value IN ({0:})
    THEN value_as_number END) AS avg_value
  FROM aggregated_measurements
  GROUP BY person_id, visit_occurrence_id,intubation, measurement_source_value, time_window
)
SELECT * FROM final_aggregates