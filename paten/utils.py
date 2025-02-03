from pathlib import Path
import pandas_gbq
import hashlib
import os
import pandas as pd
import json


TEMP=Path("/.cxr")
PROCEDURES=Path(__file__).parent / "procedures"
CREDENTIALS=".config/gcloud/application_default_credentials.json"


def get_credentials():
    with open(Path.home() / CREDENTIALS,"r") as file:
        data=json.load(file)
    return data

def read_gbq(query,*args,**kwargs):
    savepath=TEMP
    filename=hashlib.sha1(query.encode()).hexdigest()

    if not savepath.is_dir():
        _=os.mkdir(savepath)
    
    if (savepath / filename).is_file():
        df=pd.read_pickle(savepath / filename)
    else:
        df=pandas_gbq.read_gbq(query,*args,**kwargs)
        df.to_pickle(savepath / filename)
    return df

def read_procedure(name):
    with open(PROCEDURES / f"__{name}__.sql","r") as file:
        procedure=file.read()
    return procedure

def read_query(name):
    with open(PROCEDURES / f"{name}.sql","r") as file:
        query=file.read()
    return query

def filter_pronation(df:pd.DataFrame,threshold__hours:int)->pd.DataFrame:
  filtered_df=df[(df.average_daily_pronation__hours > threshold__hours)|(df.average_daily_pronation__hours.isna())].copy()
  filtered_df["Pronation"]=filtered_df.average_daily_pronation__hours.notna()
  return filtered_df

