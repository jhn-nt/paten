import pickle
from .utils import TEMP

with open(TEMP / "clinical_variables.pickle","rb") as file:
    CLINICAL_VARIABLES=pickle.load(file)

with open(TEMP / "features.pickle","rb") as file:
    FEATURES=pickle.load(file)

DEMOGRAPHIC_VARIABLES=[
    "Age",
    "Gender",
    "Unit",
    "Severe ARDS",
    "LOS",
    "Death",
    "Height",
    "Weight",
    "BMI",
    "APACHE"
    ]
VAP=["Pneumonia"]
INDEX=["person_id","visit_occurrence_id","ventilation_id"]
CENSOR=["intubation","extubation","year_of_birth", "Severe ARDS","average_daily_pronation__hours","LOS","Unit"]
OUTCOME=["Death"]
TREATMENT=["Pronation"]

PREFILTERED=[
['Age',
 'Gender',
 'APACHE',
 'BMI',
 'Weight',
 'Mean airway pressure',
 'duration_hours',
 'Oxygen gas flow Oxygen delivery system',
 'Base excess in Blood by calculation',
 'Calcium Moles/volume in Serum or Plasma',
 'Oxygen/Inspired gas Respiratory system by O2 Analyzer --on ventilator',
 'Central venous pressure (CVP)',
 'Carbon dioxide Partial pressure in Exhaled gas --at end expiration',
 'Oxygen Partial pressure in Blood',
 'Oxygen content in Blood',
 'Maximum Pressure Respiratory system airway opening --during inspiration on ventilator',
 'Breath rate spontaneous and mechanical --on ventilator',
 'Urine output',
 'Phosphate Moles/volume in Serum or Plasma',
 'Fractional oxyhemoglobin in Blood',
 'Airway pressure delta setting Ventilator',
 'Lactate Moles/volume in Blood',
 'Sodium Moles/volume in Blood',
 'Pulmonary artery Systolic blood pressure']
]

CONFOUNDERS=list(set(PREFILTERED).difference([*INDEX,*CENSOR,*OUTCOME,*TREATMENT]))
CATEGORICAL=list(set(CONFOUNDERS).intersection(["Gender","Unit","Pneumonia","Pronation","Death"]))
CONTINUOUS=list(set(CONFOUNDERS).difference(CATEGORICAL))

