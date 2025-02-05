import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import product


def plot_effect_by_threshold(data,ax,label=None,**kwargs):
    _=ax.plot(data.index,data["mean"],**kwargs,label=label)
    _=ax.fill_between(data.index,data["mean"],data["mean"]+data["std"],**kwargs,alpha=.4)
    _=ax.fill_between(data.index,data["mean"]-data["std"],data["mean"],**kwargs,alpha=.4)
    return ax

def plot_from_grouper(ax,df,grouper,target,rolling=4,with_legend=True):
    grouped_df=100*df.groupby([*grouper,"threshold"])[target].agg(["mean","std"])
    grouped_df=grouped_df.rolling(rolling,min_periods=1).mean()
    ix=grouped_df.index.get_level_values(0).unique()
    

    for n,(i,t) in enumerate(product(ix,target)):
        plot_effect_by_threshold(grouped_df.loc[i][t],ax,label=i,color=cm.tab10(n))

    _=ax.plot([4,22],[0,0],color="k")
    _=ax.plot([4,22],[-16.8,-16.8],color="r")
    _=ax.grid(alpha=.3)
    _=ax.set_ylim(-75,10)
    _=ax.set_xticks(np.arange(4,24,2))
    _=ax.set_xlabel("Pronation Duration [Hour(s)]")
    _=ax.set_ylabel("Est. Mortality Reduction [%]")
    if with_legend:
        _=ax.legend()
    return ax

def plot_pivot_from_grouper(ax,df,grouper,rolling=4):
    grouped_df=100*df.groupby([*grouper,"threshold"])[["Traditional ML","Causal ML"]].agg(["mean","std"])
    grouped_df=grouped_df.rolling(rolling,min_periods=1).mean()
    ix=grouped_df.index.get_level_values(0).unique()
    

    for n,i in enumerate(ix):
        plot_effect_by_threshold(grouped_df.loc[i]["Traditional ML"],ax[n],label="Traditional ML",color=cm.tab10(0)).set_title(i)
        plot_effect_by_threshold(grouped_df.loc[i]["Causal ML"],ax[n],label="Causal ML",color=cm.tab10(1))

        _=ax[n].plot([4,22],[0,0],color="k")
        _=ax[n].plot([4,22],[-16.8,-16.8],color="r")
        _=ax[n].grid(alpha=.3)
        _=ax[n].set_ylim(-75,10)
        _=ax[n].set_xticks(np.arange(4,24,2))
        _=ax[n].set_xlabel("Pronation Duration [Hour(s)]")
        _=ax[n].set_ylabel("Est. Mortality Reduction [%]")
        _=ax[n].legend()
    return ax

def sensitivity_by_causality(df,figsize=(6,10)):
    fig=plt.figure(constrained_layout=True,figsize=figsize)
    sfig=fig.subfigures(3,1)

    ax=sfig[0].subplots(1,2)
    plot_from_grouper(ax[0],df,["Intervention Proxy"],["Traditional ML"]).set_title("Traditional ML")
    plot_from_grouper(ax[1],df,["Intervention Proxy"],["Causal ML"],with_legend=False).set_title("Causal ML")
    sfig[0].suptitle("Different Proxies of Interventions")

    ax=sfig[1].subplots(1,2)
    plot_from_grouper(ax[0],df,["Population"],["Traditional ML"]).set_title("Traditional ML")
    plot_from_grouper(ax[1],df,["Population"],["Causal ML"],with_legend=False).set_title("Causal ML")
    sfig[1].suptitle("Different ARDS Severities")

    ax=sfig[2].subplots(1,2)
    plot_from_grouper(ax[0],df,["Model"],["Traditional ML"]).set_title("Traditional ML")
    plot_from_grouper(ax[1],df,["Model"],["Causal ML"],with_legend=False).set_title("Causal ML")
    sfig[2].suptitle("Different Statistical Estimators")
    return fig

def sensitivity_by_aggregation(df,figsize=(10,10)):
    fig=plt.figure(constrained_layout=True,figsize=figsize)
    sfig=fig.subfigures(3,1)

    ax=sfig[0].subplots(1,3)
    plot_pivot_from_grouper(ax,df,["Intervention Proxy"])
    sfig[0].suptitle("Different Proxies of Interventions")

    ax=sfig[1].subplots(1,2)
    plot_pivot_from_grouper(ax,df,["Population"])
    sfig[1].suptitle("Different ARDS Severities")

    ax=sfig[2].subplots(1,2)
    plot_pivot_from_grouper(ax,df,["Model"])
    sfig[2].suptitle("Different Statistical Estimators")
    return fig


if __name__=="__main__":
    import argparse
    import pandas as pd
    from pathlib import Path

    parser=argparse.ArgumentParser()
    parser.add_argument("-d","--dir")


    args=parser.parse_args()
    SAVEDIR=Path(args.dir)

    try:
        results_df=pd.read_pickle(SAVEDIR / "results.pickle")
    except:
        chk=[]
        for file in SAVEDIR.glob("*.csv"):
            chk.append(pd.read_csv(file))
        results_df=pd.concat(chk,axis=0)

    results_df=results_df.rename(columns={"x":"Causal ML","s":"Traditional ML","proxy":"Intervention Proxy","filter":"Population","model":"Model"})
    results_df=results_df.replace({"intervention_proxy__uniform":"Uniform","intervention_proxy__duty_cycle":"Pseudo Duty Cycle","intervention_proxy__capped_cumulative":"Capped Cumulative","make_lgbm":"LGBM","make_logistic_regression":"Logistic Regression"})
    results_df.threshold=results_df.threshold.astype(int)

    sensitivity_by_aggregation(results_df).savefig(SAVEDIR / "by_aggregation.png",dpi=500)
    sensitivity_by_causality(results_df).savefig(SAVEDIR / "by_causality.png",dpi=500)