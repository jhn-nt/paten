import pickle
from .utils import TEMP
import pandas as pd
from pathlib import Path

def collect_features():
    try:
        with open(TEMP / "features.pickle","rb") as file:
            features=pickle.load(file)
        return features
    except:
        raise ValueError("Run paten.etl.dataset at least once")

FEATURES=collect_features()
ANNOTATED_FEATURES=pd.read_csv(Path(__file__).parent / "concepts.csv").concept_name.to_list()
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
TABLEONE=["Gender","Age",'APACHE',"BMI","Severe ARDS","Death","LOS","Pronation","duration_hours"]
FIO2=['Oxygen/Inspired gas Respiratory system by O2 Analyzer --on ventilator']

XGB=['Patient Age',
'Patient Gender',
'APACHE Score',
'Patient BMI',
'Patient Weight',
'Mean Airway Pressure',
'IMV Duration in Hours',
'Calculated Blood Base Excess',
'Serum/Plasma Calcium Concentration (Moles per Volume)',
'Oxygen/Inspired Gas Measurement by O2 Analyzer (Ventilator)',
'Serum/Plasma Sodium Concentration (Moles per Volume)',
'Blood Lactate Concentration (Moles per Volume)',
'Ventilator Breath Rate Setting',
'Blood Oxygen Partial Pressure',
'PEEP (Positive End-Expiratory Pressure) on Ventilator',
'Ventilator Airway Pressure Delta Setting',
'Beat-to-Beat Heart Rate (EKG)',
'Blood Platelet Count (Number per Volume)',
'Exhaled Carbon Dioxide Partial Pressure (End Expiration)',
'Dynamic Lung Compliance',
'Central Venous Pressure (CVP)',
'Tidal Volume Inspired (Spontaneous and Mechanical, Ventilator)',
'Blood pH',
'Tidal Volume Expired (Spontaneous and Mechanical, Ventilator)',
'Expired Minute Volume (Mechanical Ventilation)',
'Urine Output',
'Blood Hematocrit (Pure Volume Fraction, Automated Count)',
'Patient Height',
'Respiratory Carbon Dioxide Production (VCO2)',
'Inspiratory Time']

MAPPING={
"Airway plateau pressure":"Airway Plateau Pressure",
"Airway pressure delta setting Ventilator":"Ventilator Airway Pressure Delta Setting",
"Albumin [Mass/volume] in Serum or Plasma":"Serum/Plasma Albumin Concentration (Mass per Volume)",
"Anion gap in Blood":"Blood Anion Gap",
"Base excess in Blood by calculation":"Calculated Blood Base Excess",
"Bicarbonate [Moles/volume] in Blood":"Blood Bicarbonate Concentration (Moles per Volume)",
"Blood temperature":"Blood Temperature (Celsius)",
"Breath rate setting Ventilator":"Ventilator Breath Rate Setting",
"Breath rate spontaneous and mechanical --on ventilator":"Combined Spontaneous and Mechanical Breath Rate (Ventilator)",
"Calcium [Moles/volume] in Serum or Plasma":"Serum/Plasma Calcium Concentration (Moles per Volume)",
"Calcium.ionized [Moles/volume] adjusted to pH 7.4 in Blood":"Ionized Calcium in Blood (Moles per Volume, pH 7.4 Adjusted)",
"Carbon dioxide [Partial pressure] in Blood":"Blood Carbon Dioxide Partial Pressure",
"Carbon dioxide [Partial pressure] in Exhaled gas --at end expiration":"Exhaled Carbon Dioxide Partial Pressure (End Expiration)",
"Carbon dioxide production (VCO2) in Respiratory system":"Respiratory Carbon Dioxide Production (VCO2)",
"Central venous pressure (CVP)":"Central Venous Pressure (CVP)",
"Chloride [Moles/volume] in Blood":"Blood Chloride Concentration (Moles per Volume)",
"Chloride [Moles/volume] in Serum or Plasma":"Serum/Plasma Chloride Concentration (Moles per Volume)",
"Creatinine [Moles/volume] in Serum or Plasma":"Serum/Plasma Creatinine Concentration (Moles per Volume)",
"Dynamic lung compliance":"Dynamic Lung Compliance",
"Expired minute Volume during Mechanical ventilation":"Expired Minute Volume (Mechanical Ventilation)",
"Fractional oxyhemoglobin in Blood":"Blood Fractional Oxyhemoglobin",
"Heart rate.beat-to-beat by EKG":"Beat-to-Beat Heart Rate (EKG)",
"Hematocrit [Pure volume fraction] of Blood by Automated count":"Blood Hematocrit (Pure Volume Fraction, Automated Count)",
"Hemoglobin [Moles/volume] in Blood":"Blood Hemoglobin Concentration (Moles per Volume)",
"INR in Blood by Coagulation assay":"International Normalized Ratio (INR, Coagulation Assay)",
"Inspiratory time":"Inspiratory Time",
"Inspiratory time setting Ventilator":"Ventilator Inspiratory Time Setting",
"Inspired minute Volume during Mechanical ventilation":"Inspired Minute Volume (Mechanical Ventilation)",
"Invasive Mean blood pressure":"Invasive Mean Blood Pressure",
"Lactate [Moles/volume] in Blood":"Blood Lactate Concentration (Moles per Volume)",
"Leukocytes [#/volume] in Blood":"Blood Leukocyte Count (Number per Volume)",
"Maximum [Pressure] Respiratory system airway opening --during inspiration on ventilator":"Maximum Respiratory Airway Pressure During Inspiration (Ventilator)",
"Mean airway pressure":"Mean Airway Pressure",
"Oxygen [Partial pressure] in Blood":"Blood Oxygen Partial Pressure",
"Oxygen content in Blood":"Blood Oxygen Content",
"Oxygen gas flow Oxygen delivery system":"Oxygen Gas Flow (Delivery System)",
"Oxygen saturation [Pure mass fraction] in Blood":"Blood Oxygen Saturation (Pure Mass Fraction)",
"Oxygen/Inspired gas Respiratory system by O2 Analyzer --on ventilator":"Oxygen/Inspired Gas Measurement by O2 Analyzer (Ventilator)",
"Oxygen/Total gas setting [Volume Fraction] Ventilator":"Ventilator Oxygen/Total Gas Setting (Volume Fraction)",
"PEEP Respiratory system --on ventilator":"PEEP (Positive End-Expiratory Pressure) on Ventilator",
"Phosphate [Moles/volume] in Serum or Plasma":"Serum/Plasma Phosphate Concentration (Moles per Volume)",
"Platelets [#/volume] in Blood":"Blood Platelet Count (Number per Volume)",
"Positive end expiratory pressure setting Ventilator":"Ventilator Positive End-Expiratory Pressure Setting",
"Potassium [Moles/volume] in Blood":"Blood Potassium Concentration (Moles per Volume)",
"Potassium [Moles/volume] in Serum or Plasma":"Serum/Plasma Potassium Concentration (Moles per Volume)",
"Pulmonary artery Diastolic blood pressure":"Pulmonary Artery Diastolic Blood Pressure",
"Pulmonary artery Mean blood pressure":"Pulmonary Artery Mean Blood Pressure",
"Pulmonary artery Systolic blood pressure":"Pulmonary Artery Systolic Blood Pressure",
"Respiratory rate":"Respiratory Rate",
"Sodium [Moles/volume] in Blood":"Blood Sodium Concentration (Moles per Volume)",
"Sodium [Moles/volume] in Serum or Plasma":"Serum/Plasma Sodium Concentration (Moles per Volume)",
"Tidal volume expired spontaneous+mechanical Respiratory system airway --on ventilator":"Tidal Volume Expired (Spontaneous and Mechanical, Ventilator)",
"Tidal volume inspired spontaneous+mechanical Measured --on ventilator":"Tidal Volume Inspired (Spontaneous and Mechanical, Ventilator)",
"Urine output":"Urine Output",
"Ventilation cycle time":"Ventilation Cycle Time",
"pH of Blood":"Blood pH",
"average_daily_pronation__hours":"Average Daily Pronation Hours",
"Age":"Patient Age",
"Gender":"Patient Gender",
"Weight":"Patient Weight",
"Height":"Patient Height",
"BMI":"Patient BMI",
"APACHE":"APACHE Score",
"duration_hours":"IMV Duration in Hours",
"Pneumonia":"Pneumonia Diagnosis"
}

REV_MAPPING={item:key for (key,item) in MAPPING.items()}
PREFILTERED=[REV_MAPPING[feat] for feat in XGB]

CONFOUNDERS=list(set(PREFILTERED).difference([*INDEX,*CENSOR,*OUTCOME,*TREATMENT]))
CATEGORICAL=list(set(CONFOUNDERS).intersection(["Gender","Unit","Pneumonia","Pronation","Death"]))
CONTINUOUS=list(set(CONFOUNDERS).difference(CATEGORICAL))

