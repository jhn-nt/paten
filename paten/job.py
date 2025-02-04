from .etl import dataset, intervention_proxy__capped_cumulative, intervention_proxy__uniform, intervention_proxy__duty_cycle
from .models import emulate_at_different_thresholds, propensity_score, make_lgbm, make_logistic_regression
from .utils import filter_pronation
from .features import CATEGORICAL, CONFOUNDERS, CONTINUOUS, OUTCOME, TREATMENT
import numpy as np
from itertools import product
from functools import partial
import pandas as pd
import pickle
import warnings
from tqdm import tqdm



MODELS=[
    make_lgbm,
    make_logistic_regression
    ]

PROXIES=[
    intervention_proxy__uniform,
    intervention_proxy__capped_cumulative,
    intervention_proxy__duty_cycle
    ]

FILTERS=[
    "Severe ARDS",
    None
]
THRESHOLDS=np.arange(4,24,2)


if __name__=="__main__":
    import argparse
    from pathlib import Path
    warnings.filterwarnings("ignore")

    parser=argparse.ArgumentParser()
    parser.add_argument("-d","--dir")
    parser.add_argument("-r","--repeats",default=10)
    parser.add_argument("-s","--seed",default=0)
    parser.add_argument("-f","--folds",default=5)
    
    args=parser.parse_args()

    SAVEDIR=Path(args.dir)
    SEED=int(args.seed)
    N_REPEATS=int(args.repeats)
    N_SPLITS=int(args.folds)
    DATASETS=list(map(dataset,PROXIES))

    SAVEDIR.mkdir(exist_ok=True)

    output=[]
    pbar=tqdm(enumerate(product(MODELS,zip(PROXIES,DATASETS),FILTERS)))
    for i,(model_f,(proxy_f,df),filter) in pbar:
        pbar.set_description(f"proxy: {proxy_f.__name__}   model: {model_f.__name__} filter: {filter}")
        try:
            model_f_name=model_f.__name__
            model_f=partial(model_f,seed=SEED)
            filtered_df=filter_pronation(df,16) # we store results from vanilla emulation
            if filter:
                filtered_df=filtered_df[filtered_df[filter]]

            py_x=propensity_score(
                filtered_df,
                OUTCOME,
                TREATMENT,
                CONFOUNDERS,
                CATEGORICAL,
                n_repeats=N_REPEATS,
                n_splits=N_SPLITS,
                seed=SEED)
            
            pa_x=propensity_score(
                filtered_df,
                OUTCOME,
                TREATMENT,
                CONFOUNDERS,
                CATEGORICAL,
                on_treatment=True,
                n_repeats=N_REPEATS,
                n_splits=N_SPLITS,
                seed=SEED)

            cates=emulate_at_different_thresholds(
                THRESHOLDS,
                model_f,
                df,
                OUTCOME,
                TREATMENT,
                CONFOUNDERS,
                CATEGORICAL,
                n_repeats=N_REPEATS,
                n_splits=N_SPLITS,
                seed=SEED)

            cates["proxy"]=proxy_f.__name__
            cates["model"]=model_f_name
            cates["filter"]=filter
            cates["checkpoint"]=i
            cates.to_csv(SAVEDIR / f"checkpoint__{i}.csv",index=False)
            output.append(cates)

            with open(SAVEDIR / f"ps__{i}.pickle","wb") as file:
                pickle.dump({"py_x":py_x,"pa_x":pa_x},file)

        except Exception as e:
            print(e)


    pd.concat(output,axis=0).to_csv(SAVEDIR / "results.csv",index=False)

  

