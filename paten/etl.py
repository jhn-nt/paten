from .utils import read_gbq, read_query, read_procedure
import pandas as pd
import numpy as np
import gspread


def load_cohort():
    return read_gbq(read_query("cohort"))

def load_demographic():
    cohort_q=read_query("cohort")
    demographic_q=read_procedure("demographic").format(cohort_q)
    return read_gbq(demographic_q)

def load_demographic():
    cohort_q=read_query("cohort")
    demographic_q=read_procedure("demographic").format(cohort_q)
    return read_gbq(demographic_q)

def load_ventilatory_params():
    cohort_q=read_query("cohort")
    ventilatory_params_q=read_procedure("ventilatory_params").format(cohort_q)
    return read_gbq(ventilatory_params_q)

def load_pronation_initiation():
    cohort_q=read_query("cohort")
    pronation_initiation_q=read_procedure("pronation_initiation").format(cohort_q)
    return read_gbq(pronation_initiation_q)

def load_pronation_observation():
    cohort_q=read_query("cohort")
    pronation_observation_q=read_procedure("pronation_observation").format(cohort_q)
    return read_gbq(pronation_observation_q)

def load_outcomes():
    cohort_q=read_query("cohort")
    outcomes_q=read_procedure("outcomes").format(cohort_q)
    return read_gbq(outcomes_q)

def load_vap():
    cohort_q=read_query("cohort")
    outcomes_q=read_procedure("vap").format(cohort_q)
    return read_gbq(outcomes_q)

def confounders__legacy():
    gc = gspread.oauth()
    worksheet=gc.open("Variables Datathon")
    data=worksheet.get_worksheet(0).get_all_values()
    headers=data.pop(0)
    important_columns=pd.DataFrame(data,columns=headers)
    important_columns_filt = important_columns.dropna()
    important_columns_filt=important_columns_filt[important_columns_filt["Do we need it? (Y)"].str.lower().str.contains("yes")]
    important_features_sql = ""
    for imp_col in important_columns_filt.measurement_source_value.to_list():
        important_features_sql += "'" + imp_col + "'" + ","
    important_features_sql = important_features_sql[:-1]
    return important_features_sql

def load_confounders_legacy():
    confounders_q=read_procedure("confounders_legacy").format(confounders__legacy())
    confounders_df=read_gbq(confounders_q)
    average_confounders_df=pd.pivot_table(confounders_df,
               index=["person_id","visit_occurrence_id","intubation"],
               columns=["measurement_source_value"],
               values=["avg_value"])
    return average_confounders_df

def cohort_table_from_params(params_df:pd.DataFrame)->pd.DataFrame:
    """
    Returns a DataFrame where each row is a timestamp where at least one new value of pao2__mmhg, fio2__p and peep__cmh2o is updated.
    Each timestamp is measured between intubation and extubation for that given ICU admission.
    """

    params_df["measure"]=params_df.measurement_concept_id.astype(str).replace({
        '3024882':"fio2__p",
        '3022875':"peep__cmh2o",
        '3027315':"pao2__mmhg"})

    pivoted_df=pd.pivot_table(
        params_df,
        index=["person_id","visit_occurrence_id","measurement_datetime"],
        columns=["measure"],
        values="value_as_number")


    # Using data only avilable within the past hour to evaluate ARDS
    grouper=pivoted_df.reset_index().set_index("measurement_datetime").groupby(["person_id","visit_occurrence_id"])
    rolled_df=grouper.rolling("1h").agg(lambda x: x.iloc[-1])



    valid_mask=rolled_df.isna().sum(axis=1)==0
    rolled_df["pf__ratio"]=np.where(valid_mask,100*rolled_df.pao2__mmhg/(rolled_df.fio2__p+1e-9),np.nan)
    # When a patients is missing data, the code assumes is not ARDS
    rolled_df["ards"]=np.where(valid_mask,(rolled_df.peep__cmh2o>=5)&(rolled_df.pf__ratio<=300),False)
    # When a patients is missing data, the code assumes is not severe ARDS
    rolled_df["severe_ards"]=np.where(valid_mask,(rolled_df.peep__cmh2o>=5)&(rolled_df.pf__ratio<=150),False)
    return rolled_df

def intervention_proxy__uniform(pronation_initiaiton_df,pronation_observation_df):

  obse_df=pronation_observation_df[["person_id","visit_occurrence_id","observation_datetime","value_as_string","intubation","extubation","observation_date"]].rename(columns={"observation_datetime":"datetime","value_as_string":"value"}).copy()
  obse_df.value=obse_df.value.replace({"Rugligging":"end",'Re-zijde':"end", 'Li-zijde':"end"})
  init_df=pronation_initiaiton_df[["person_id","visit_occurrence_id","procedure_datetime","intubation","extubation"]].rename(columns={"procedure_datetime":"datetime"}).copy()
  init_df["value"]="init"
  tf=pd.concat([obse_df,init_df],axis=0).copy()

  tf=tf[tf.value.isin(["init","end"])]
  tf["next_datetime"]=tf.groupby(["person_id","visit_occurrence_id","intubation"]).datetime.shift(-1)
  tf["ref_datetime"]=tf[["next_datetime","extubation"]].min(axis=1)


  tf["delta"]=(tf.ref_datetime-tf.datetime)/np.timedelta64(1,"h")
  tf["delta"]=np.where(tf.delta>0,tf.delta,np.nan)
  tf["imv_days"]=(tf.extubation-tf.intubation)/(24*np.timedelta64(1,"h"))


  approx_intervention_df=tf[tf.value=='init'].copy()
  approx_intervention_df["average_daily_pronation__hours"]=approx_intervention_df.delta/approx_intervention_df.imv_days

  return approx_intervention_df[["person_id","visit_occurrence_id","intubation","average_daily_pronation__hours"]]

def intervention_proxy__capped_cumulative(pronation_initiaiton_df,pronation_observation_df):

  obse_df=pronation_observation_df[["person_id","visit_occurrence_id","observation_datetime","value_as_string","intubation","extubation","observation_date"]].rename(columns={"observation_datetime":"datetime","value_as_string":"value"}).copy()
  obse_df.value=obse_df.value.replace({"Rugligging":"end",'Re-zijde':"end", 'Li-zijde':"end"})
  init_df=pronation_initiaiton_df[["person_id","visit_occurrence_id","procedure_datetime","intubation","extubation"]].rename(columns={"procedure_datetime":"datetime"}).copy()
  init_df["value"]="init"
  tf=pd.concat([obse_df,init_df],axis=0).copy()

  tf=tf[tf.value.isin(["init","end"])]
  tf["next"]=tf.groupby(["visit_occurrence_id","intubation"]).value.shift(-1)

  tf["next_datetime"]=tf.groupby(["person_id","visit_occurrence_id","intubation"]).datetime.shift(-1)
  tf["ref_datetime"]=tf[["next_datetime","extubation"]].min(axis=1)


  tf["delta"]=(tf.ref_datetime-tf.datetime)/np.timedelta64(1,"h")
  tf["delta"]=np.where(tf.delta>0,tf.delta,np.nan)


  approx_intervention_df=tf[tf.value=='init'].copy()
  approx_intervention_df["average_daily_pronation__hours"]=np.where(approx_intervention_df.delta>24,24,approx_intervention_df.delta)

  return approx_intervention_df[["person_id","visit_occurrence_id","intubation","average_daily_pronation__hours"]]

def intervention_proxy__duty_cycle(pronation_initiaiton_df,pronation_observation_df)->pd.DataFrame:
    raise NotImplementedError

def legacy_dataset(proxy_f):
    demographic_df=load_demographic()
    ventilation_df=load_cohort()
    vap_df=load_vap()
    params_df=load_ventilatory_params()
    pronation_observation_df=load_pronation_observation()
    pronation_initiaiton_df=load_pronation_initiation()
    average_covariates_df=load_confounders_legacy()
    outcome_df=load_outcomes()

    df=cohort_table_from_params(params_df)
    ards_df=df[df.ards].reset_index().groupby(["person_id","visit_occurrence_id"]).severe_ards.max().reset_index()

    pronation_df=proxy_f(pronation_initiaiton_df,pronation_observation_df)
    temp=average_covariates_df.droplevel(0,1).merge(pronation_df,on=["person_id","visit_occurrence_id","intubation"],how="left")
    outcome_df["death"]=True

    temp=temp.merge(outcome_df.drop_duplicates(),on=["person_id","visit_occurrence_id"],how="left")
    temp["death"]=temp["death"].fillna(False)

    temp=temp.merge(demographic_df,on=["person_id","visit_occurrence_id"])
    temp=temp.merge(ventilation_df,on=["person_id","visit_occurrence_id","intubation"])
    vap_df["pneumonia"]=True
    temp=temp.merge(vap_df,on=["visit_occurrence_id","intubation"],how="left")
    temp["pneumonia"]=temp["pneumonia"].fillna(False)
    temp=temp.drop(["condition_era_start_date","condition_era_end_date"],axis=1)

    temp=temp.drop("concept_name",axis=1)
    dataset=temp.merge(ards_df,on=["person_id","visit_occurrence_id"],how="inner")
    dataset=dataset[dataset['Dynamic lung compliance']<40]
    return dataset