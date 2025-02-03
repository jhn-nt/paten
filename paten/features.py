from .etl import confounders

CONFOUNDERS=confounders().concept_name.to_list()
INDEX=["person_id","visit_occurrence_id","ventilation_id"]
CENSOR=["intubation","extubation","year_of_birth", "Severe ARDS","average_daily_pronation__hours","LOS","Unit"]
CATEGORICAL=["Gender","unit"]
OUTCOME=["Death"]
TREATMENT=["Pronation"]
