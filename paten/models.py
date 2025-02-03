import warnings
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from .utils import filter_pronation


def formatted_cross_validate(*args,**kwargs)->pd.DataFrame:
  warnings.filterwarnings("ignore")
  scores=sklearn.model_selection.cross_validate(*args,**kwargs)

  train={}
  test={}
  for k,item in scores.items():
    if "test" in k:
      test[k.split("_",1)[1]]=item
    else:
      train[k.split("_",1)[1]]=item

  train_df=pd.DataFrame(train).drop("time",axis=1)
  test_df=pd.DataFrame(test)
  columns=pd.MultiIndex.from_product([["train","test"],train_df.columns])
  scores_df=pd.concat([train_df,test_df],axis=1)
  scores_df.columns=columns

  return scores_df

def make_logistic_regression(confounders,categorical,seed=0,decompose=True,n_components=.95)->sklearn.base.ClassifierMixin:
  cat_ix=[i for (i,k) in enumerate(confounders) if k in categorical]
  cnt_ix=[i for (i,k) in enumerate(confounders) if k not in categorical]

  if decompose:
    cnt_processor=Pipeline([
        ("imputer",SimpleImputer(
            missing_values=1e-6, # Flagging nan with small value to circument EconMl assertion
            strategy="mean",
            keep_empty_features=True)),
        ("normalization",QuantileTransformer(output_distribution="normal")),
        ("pca",PCA(n_components=n_components,random_state=seed))
    ])
  else:
    cnt_processor=Pipeline([
        ("imputer",SimpleImputer(
            strategy="mean",
            keep_empty_features=True,
            missing_values=1e-6)),
        ("normalization",QuantileTransformer(output_distribution="normal"))
    ])

  cat_processor=Pipeline([
      ("encoder",OneHotEncoder(drop="if_binary")),
      ("imputer",SimpleImputer(strategy="most_frequent")),
  ])

  processor=ColumnTransformer([
      ("cnt",cnt_processor,cnt_ix),
      ("cat",cat_processor,cat_ix)
      ] ,
      remainder="drop")

  model=Pipeline([
      ("processor",processor),
      ("classifier",LogisticRegression(random_state=seed,class_weight="balanced"))
      ])

  return model

def propensity_score(df,on,seed=0,threshold__hours=16,include_treatment=False,all_features=False):
  from sklearn.metrics import roc_auc_score, average_precision_score
  warnings.filterwarnings('ignore')

  pre_df=df[(df['Dynamic lung compliance']<40)&(df.Severe_ards==True)].copy()
  filtered_df=filter_pronation(pre_df,threshold__hours=threshold__hours)

  index=["person_id","visit_occurrence_id","ventilation_id"]
  to_drop=["intubation","extubation","year_of_birth", "Severe_ards","average_daily_pronation__hours","unit","los","q"]
  categorical=["Gender"]
  outcome=["Death"]
  treatment=["Pronation"]
  confounders=df.drop([*outcome,*treatment,*index,*to_drop],axis=1).columns.to_list()
  confounders=[conf for conf in confounders if "_2" not in conf]


  if include_treatment:
    confounders=[*confounders,*treatment]
    categorical=[*categorical,*treatment]

  continuous=list(set(confounders).difference(categorical))

  model=make_logistic_regression(confounders,categorical,decompose=False)
  cv=RepeatedStratifiedKFold(n_repeats=10,n_splits=5,random_state=seed)


  X,y=filtered_df[confounders],filtered_df[on]
  X[continuous]=X[continuous].fillna(1e-6)
  odds=[]
  auroc=[]
  auprc=[]
  for train,test in cv.split(X,y):

    X_train,X_test=X.iloc[train,:],X.iloc[test,:]
    y_train, y_test=y.iloc[train],y.iloc[test]


    trained_model=model.fit(X_train,y_train)
    _odds=np.exp(np.squeeze(trained_model[-1].coef_))
    features=trained_model[0].get_feature_names_out()
    features=[feat.split("__",1)[1].split("--")[0] for feat in features]
    odds.append(pd.Series(_odds,index=features))

    y_score=trained_model.predict_proba(X_test)
    auroc.append(roc_auc_score(y_test,y_score[:,1]))
    auprc.append(average_precision_score(y_test,y_score[:,1]))

  odds=pd.concat(odds,axis=1).T.rename(columns={
      "Gender_1.0":"Gender__Male",
      "Pronation_True":"Pronation",
      "Bicarbonate Moles/volume in Blood":"Bicarbonate",
      "Lactate Moles/volume in Blood":"Lactate",
      "Carbon dioxide/Gas.total.at end expiration in Exhaled gas":"etCO2",
      "Oxygen saturation in Arterial blood by Pulse oximetry":"SpO2",
      "Oxygen saturation Pure mass fraction in Blood":"PO2"})

  if not all_features:
    features_of_interest=["Gender__Male","Bicarbonate","bmi","weight","Lactate","apache","SpO2","PO2"]
    if include_treatment:
      features_of_interest+=["Pronation"]
    odds=odds[features_of_interest]

  odds=odds[odds.mean().sort_values().index]
  return odds,pd.Series(auroc),pd.Series(auprc)

def crossval_xlearner(df,outcome,treatment,confounders,categorical,n_splits=5,n_repeats=10,seed=0,name="tau"):
  from econml.metalearners import XLearner
  from sklearn.model_selection import RepeatedStratifiedKFold
  from tqdm import tqdm

  df=df.copy().fillna(1e-6)
  cv=RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=seed)

  learner=XLearner(
      models=make_logistic_regression(confounders,categorical,seed=seed,decompose=True),
      propensity_model=make_logistic_regression(confounders,categorical,seed=seed,decompose=True),
      allow_missing=True)

  pbar=tqdm(cv.split(X=df[confounders],y=df[treatment]))

  tau=[]
  for fold,(train,test) in enumerate(pbar):
    df_train,df_test=df.iloc[train],df.iloc[test]
    fitted_learner=learner.fit(
        df_train[outcome],
        df_train[treatment],
        X=df_train[confounders])

    # We measure the effect of cate on patients that were not pronated
    # tau= E[Y_pronated - Y_supine]
    df_test=df_test.query("Pronation==False")
    tau.append(pd.DataFrame(
        fitted_learner.effect(
            X=df_test[confounders]
            ),
        columns=[fold],
        index=df_test.index))

  return pd.concat(tau,axis=1).mean(axis=0).rename(name)

def crossval_slearner(df,outcome,treatment,confounders,categorical,n_splits=5,n_repeats=10,seed=0,name="tau"):
  # Naive Implementation of a CATE with no major causal assumption
  import numpy as np
  from econml.metalearners import SLearner
  from sklearn.model_selection import RepeatedStratifiedKFold
  from tqdm import tqdm

  df=df.copy().fillna(1e-6)
  cv=RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=seed)

  learner=SLearner(
      overall_model=make_logistic_regression([*confounders,*treatment],[*categorical,*treatment],seed=seed,decompose=True),
      allow_missing=True)

  pbar=tqdm(cv.split(X=df[[*confounders,*treatment]],y=df[treatment]))

  tau=[]
  for fold,(train,test) in enumerate(pbar):
    df_train,df_test=df.iloc[train],df.iloc[test]
    fitted_learner=learner.fit(
        df_train[outcome],
        df_train[treatment],
        X=df_train[confounders])

    # We measure the effect of cate on patients that were not pronated
    # tau= E[Y_pronated - Y_supine]
    df_test=df_test.query("Pronation==False")
    tau.append(pd.DataFrame(
        fitted_learner.effect(
            X=df_test[confounders]
            ),
        columns=[fold],
        index=df_test.index))

  return pd.concat(tau,axis=1).mean(axis=0).rename(name)