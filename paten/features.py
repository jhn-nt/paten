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
    'Age',
    'Gender',
    'APACHE',
    'BMI',
    'Maximum Pressure Respiratory system airway opening --during inspiration on ventilator',
    'Lactate Moles/volume in Blood',
    'Oxygen gas flow Oxygen delivery system',
    'duration_hours',
    'Urine output',
    'Platelets #/volume in Blood',
    'Invasive Mean blood pressure',
    'Pulmonary artery Systolic blood pressure',
    'Base excess in Blood by calculation',
    'Sodium Moles/volume in Blood',
    'Creatinine Moles/volume in Serum or Plasma',
    'Oxygen Partial pressure in Blood',
    'Tidal volume inspired spontaneous+mechanical Measured --on ventilator',
    'Oxygen/Total gas setting Volume Fraction Ventilator',
    'Fractional oxyhemoglobin in Blood',
    'Potassium Moles/volume in Blood',
    'Airway pressure delta setting Ventilator'
]

CONFOUNDERS=list(set(FEATURES).difference([*INDEX,*CENSOR,*OUTCOME,*TREATMENT]))
CATEGORICAL=list(set(CONFOUNDERS).intersection(["Gender","Unit","Pneumonia","Pronation","Death"]))
CONTINUOUS=list(set(CONFOUNDERS).difference(CATEGORICAL))

