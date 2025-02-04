from .utils import read_gbq, read_query, read_procedure
import pandas as pd
import numpy as np
import gspread
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm


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
    try:
        gc = gspread.oauth()
    except:
        # when in colab
        from google.colab import auth
        from google.auth import default
        auth.authenticate_user()
        credentials,_=default()
        gc=gspread.authorize(credentials)
        
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
               columns=["concept_name"],
               values=["avg_value"])
    return average_confounders_df

def confounders():
    try:
        gc = gspread.oauth()
    except:
        # when in colab
        from google.colab import auth
        from google.auth import default
        auth.authenticate_user()
        credentials,_=default()
        gc=gspread.authorize(credentials)

    worksheet=gc.open("Variables")
    data=worksheet.get_worksheet(0).get_all_values()
    headers=data.pop(0)
    df=pd.DataFrame(data,columns=headers)
    concepts=df[df["During IMV (X)"]=='x'][["concept_id","concept_name"]]
    return concepts

def load_confounders():
    cohort_q=read_query("cohort")
    concepts_str=",".join(confounders().concept_id.to_list())
    confounders_q=read_procedure("confounders").format(cohort_q,concepts_str)
    confounders_df=read_gbq(confounders_q)
    average_confounders_df=pd.pivot_table(confounders_df,
               index=["person_id","visit_occurrence_id","intubation"],
               columns=["concept_name"],
               values=["value_as_number"])
    return average_confounders_df.droplevel(0,1)

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

def intervention_proxy__duty_cycle(pronation_initiation_df: pd.DataFrame,
                                   pronation_observation_df: pd.DataFrame,
                                   plot: bool = False) -> pd.DataFrame:
    """
    Compute pronation session metrics (number of sessions and average pronation hours) per patient-visit.
    
    A pronation session is defined as a contiguous period when a patient is in the 'Prone' position.
    If a session lasts more than 24 hours, it is split into multiple sessions of 24 hours (with a possible remainder).
    Only ventilation periods lasting at least 1 day are processed.
    
    Parameters
    ----------
    pronation_initiation_df : pd.DataFrame
        DataFrame with pronation initiation events (currently unused).
    pronation_observation_df : pd.DataFrame
        DataFrame with pronation observation events. Expected columns:
          'person_id', 'visit_occurrence_id', 'observation_datetime',
          'value_as_string', 'intubation', 'extubation'.
    plot : bool, optional
        If True, generates a step plot of the patient’s position over the ventilation period.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by 'person_id_visit_occurrence_id' containing:
          - person_id
          - visit_occurrence_id
          - intubation (datetime of intubation)
          - pronation_sessions (number of pronation sessions, splitting sessions >24h)
          - average_pronation_hours (average duration per session in hours)
    """
    usecols = ['person_id', 'visit_occurrence_id', 'observation_datetime',
               'value_as_string', 'intubation', 'extubation']
    date_cols = ['observation_datetime', 'intubation', 'extubation']
    pron_obs = pronation_observation_df[usecols].copy()
    pron_obs[date_cols] = pron_obs[date_cols].apply(pd.to_datetime, errors='coerce')
    pron_obs = pron_obs.drop_duplicates(keep='first').reset_index(drop=True)
    
    # Standardize position names
    pos_map = {
        'Rugligging': 'Supine',
        'Buikligging': 'Prone',
        'Re-zijde': 'Right side',
        'Li-zijde': 'Left side'
    }
    pron_obs['value_as_string'] = pron_obs['value_as_string'].map(pos_map)
    pron_obs['person_id_visit_occurrence_id'] = (
        pron_obs['person_id'].astype(str) + '_' + pron_obs['visit_occurrence_id'].astype(str)
    )
    
    # Prepare results DataFrame
    unique_ids = pron_obs['person_id_visit_occurrence_id'].unique()
    res_df = pd.DataFrame(index=pd.Index(unique_ids, name='person_id_visit_occurrence_id'),
                          columns=['person_id', 'visit_occurrence_id', 'intubation', 
                                   'pronation_sessions', 'average_pronation_hours'])
    res_df[['person_id', 'visit_occurrence_id', 'pronation_sessions']] = 0
    res_df['average_pronation_hours'] = 0.0

    pos_numeric_mapping = {'Supine': 0, 'Prone': 1, 'Left side': 2, 'Right side': 3}
    
    groups = pron_obs.groupby('person_id_visit_occurrence_id')
    for pid, group_df in tqdm(groups, total=len(groups)):
        group_df = group_df.sort_values('observation_datetime')\
                           .drop_duplicates(subset='observation_datetime', keep='first')\
                           .reset_index(drop=True)
        res_df.loc[pid, 'person_id'] = group_df['person_id'].iloc[0]
        res_df.loc[pid, 'visit_occurrence_id'] = group_df['visit_occurrence_id'].iloc[0]
        res_df.loc[pid, 'intubation'] = group_df['intubation'].iloc[0]
        
        # Establish ventilation period boundaries (rounded to the hour)
        start = group_df['intubation'].dt.floor('h').iloc[0]
        end = group_df['extubation'].dt.ceil('h').iloc[-1]
        if (end - start) < pd.Timedelta(days=1) or 'Prone' not in group_df['value_as_string'].unique():
            continue
        
        # Create an hourly DataFrame over the ventilation period
        hourly_index = pd.date_range(start, end, freq='h', name='observation_datetime')
        hourly_df = pd.DataFrame(index=hourly_index)
        group_df['observation_datetime'] = group_df['observation_datetime'].dt.round('h')
        group_df = group_df.set_index('observation_datetime')
        hourly_df = pd.merge(hourly_df, group_df, left_index=True, right_index=True, how='left')
        hourly_df.ffill(inplace=True)
        fill_cols = ['person_id', 'visit_occurrence_id', 'intubation', 
                     'extubation', 'person_id_visit_occurrence_id']
        hourly_df[fill_cols] = hourly_df[fill_cols].bfill()
        hourly_df.reset_index(inplace=True)
        
        # Identify pronation periods and map positions to numeric values
        hourly_df['is_prone'] = hourly_df['value_as_string'] == 'Prone'
        hourly_df['position_session'] = (hourly_df['is_prone'] != hourly_df['is_prone'].shift()).cumsum()
        hourly_df['pos_numeric'] = hourly_df['value_as_string'].map(pos_numeric_mapping)
        
        if plot:
            # Compute start and end times for each prone session for shading
            prone_sessions = hourly_df[hourly_df['is_prone']].groupby('position_session')
            start_pron = prone_sessions['observation_datetime'].first().values
            end_pron = (prone_sessions['observation_datetime'].last() + pd.Timedelta(hours=1)).values
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.step(hourly_df['observation_datetime'], hourly_df['pos_numeric'],
                    where='post', linewidth=2, label='Patient Position')
            for sp, ep in zip(start_pron, end_pron):
                ax.axvline(sp, color='k', linewidth=1, linestyle='--')
                ax.axvline(ep, color='k', linewidth=1, linestyle='--')
                ax.axvspan(sp, ep, alpha=0.1, color='red', label='Pronation session')
            # Remove duplicate labels in the legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Patient Position')
            # Use the pos_numeric mapping for yticks
            ax.set_yticks(list(pos_numeric_mapping.values()))
            ax.set_yticklabels(list(pos_numeric_mapping.keys()))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            patient = group_df['person_id'].iloc[0]
            visit = group_df['visit_occurrence_id'].iloc[0]
            ax.set_title(f"Patient Position Over Ventilation Period (Patient {patient}, Visit {visit})")
            plt.tight_layout()
            plt.show()
        
        # Compute durations of each pronation session and split sessions >24 hours
        prone_groups = hourly_df[hourly_df['is_prone']].groupby('position_session')
        session_durations = []
        for _, session in prone_groups:
            duration = session.shape[0]
            full_chunks = duration // 24
            remainder = duration % 24
            session_durations.extend([24] * full_chunks)
            if remainder:
                session_durations.append(remainder)
        if session_durations:
            res_df.loc[pid, 'pronation_sessions'] = len(session_durations)
            res_df.loc[pid, 'average_pronation_hours'] = np.mean(session_durations)
    
    return res_df.sort_values('intubation').reset_index()

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
    dataset=dataset.sort_values(by=["person_id","intubation"]).groupby("person_id").first().reset_index()
    return dataset

def dataset(proxy_f):
    demographic_df=load_demographic()
    ventilation_df=load_cohort()
    ventilation_df=ventilation_df.reset_index().rename(columns={"index":"ventilation_id"})

    vap_df=load_vap()
    params_df=load_ventilatory_params()
    pronation_observation_df=load_pronation_observation()
    pronation_initiaiton_df=load_pronation_initiation()
    average_covariates_df=load_confounders()
    outcome_df=load_outcomes()


    df=cohort_table_from_params(params_df)
    ards_df=df[df.ards].reset_index().groupby(["person_id","visit_occurrence_id"]).severe_ards.max().reset_index()

    pronation_df=proxy_f(pronation_initiaiton_df,pronation_observation_df)
    temp=average_covariates_df.merge(pronation_df,on=["person_id","visit_occurrence_id","intubation"],how="left")
    outcome_df["Death"]=True

    temp=temp.merge(outcome_df.drop_duplicates(),on=["person_id","visit_occurrence_id"],how="left")
    temp["Death"]=temp["Death"].fillna(False)

    temp=temp.merge(demographic_df,on=["person_id","visit_occurrence_id"])
    temp.gender=temp.gender.replace({"Vrouw":"Female","Man":"Male"})
    temp=temp[temp.gender.notna()]
    temp=temp.merge(ventilation_df,on=["person_id","visit_occurrence_id","intubation"])
    vap_df["Pneumonia"]=True
    temp=temp.merge(vap_df,on=["visit_occurrence_id","intubation"],how="left")
    temp["Pneumonia"]=temp["Pneumonia"].fillna(False)
    temp=temp.drop(["condition_era_start_date","condition_era_end_date"],axis=1)

    temp=temp.drop("concept_name",axis=1)
    dataset=temp.merge(ards_df,on=["person_id","visit_occurrence_id"],how="inner")
    dataset=dataset[dataset['Dynamic lung compliance']<40]
    dataset=dataset.sort_values(by=["person_id","intubation"]).groupby("person_id").first().reset_index()
    dataset=dataset.rename(columns={"gender":"Gender","severe_ards":"Severe ARDS","unit":"Unit","los":"LOS"})
    return dataset

