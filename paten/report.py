import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from itertools import product
from tableone import TableOne
from .features import TABLEONE,FIO2
from .utils import filter_pronation


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

def make_tableone(df):
    df=filter_pronation(df,16)
    tb=TableOne(
        df[[*TABLEONE,*FIO2]],
        categorical=["Gender","Severe ARDS","Death"],
        missing=False,
        nonnormal=["LOS","duration_hours"],
        groupby="Pronation",
        rename={"duration_hours":"IMV Duration",FIO2[0]:"FiO2"},
        pval=True)
    return tb

def process_results(results_df):
    results_df=results_df.rename(columns={"x":"Causal ML","s":"Traditional ML","proxy":"Intervention Proxy","filter":"Population","model":"Model"})
    results_df=results_df.replace({"intervention_proxy__uniform":"Uniform","intervention_proxy__duty_cycle":"Pseudo Duty Cycle","intervention_proxy__capped_cumulative":"Capped Cumulative","make_lgbm":"LGBM","make_logistic_regression":"Logistic Regression"})
    results_df.Population=results_df.Population.fillna("ARDS")
    results_df.threshold=results_df.threshold.astype(int)
    return results_df

def pivot_results(df,target="Causal ML"):
    pivot={}
    for thr in df["threshold"].unique():
        pivot[thr]=df[df.threshold==thr][target].values

    return pd.DataFrame(pivot).sort_index(axis=1)

def plot_hist(ax,df,title):
    REF=-100*(.328-.16)
    CATE=np.mean(100*df)

    _=(100*df).hist(bins=np.linspace(-75.,75.,30),ax=ax,edgecolor="k")
    _=ax.plot([0,0],[0,13],color="k",label="No Effect")
    _=ax.plot([REF,REF],[0,13],color="r",lw=2,label=f"Reference Trial: {REF:.1f}%")
    _=ax.plot([CATE,CATE],[0,13],color="r",ls="--",lw=2,label=f"Emulated Trial: {CATE:.1f}%")
    _=ax.grid(alpha=.3)
    _=ax.set_ylim([0,13])

    _=ax.set_title(title)
    _=ax.set_ylabel("Crossvalidation Fold [#]")
    _=ax.set_xlabel("Est. Mortality Reduction [%]")
    _=ax.legend()
    return ax

def tte_histogram(df,threshold=16,title="pf<150mmHg , pronation>16h",population="Severe ARDS"):
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    data=df[(df.Population==population)&(df["Intervention Proxy"]=='Pseudo Duty Cycle')&(df["Model"]=='Logistic Regression')]
    plot_hist(ax[0],pivot_results(data)[threshold],("Pseudo Duty Cycle"))
    ax[0].set_title("Pseudo Duty Cycle")

    data=df[(df.Population==population)&(df["Intervention Proxy"]=='Uniform')&(df["Model"]=='Logistic Regression')]
    plot_hist(ax[1],pivot_results(data)[threshold],("Uniform"))
    fig.suptitle(title)
    return fig

def plot_boxplots(ax,data,title):
    REF=-100*(.328-.16)

    _=(100*data).boxplot(showfliers=False,ax=ax)
    _=ax.plot([0.5,10.5],[0,0],color="k",label="No Effect")
    _=ax.plot([0.5,10.5],[REF,REF],color="r",label=f"Reference Trial: {REF:.1f}%")
    _=ax.grid(alpha=.3)
    _=ax.set_xticks(np.arange(1,11,1))
    _=ax.set_xticklabels([2,4,6,8,10,12,14,16,18,20])
    _=ax.set_yticks(np.arange(-75,20,10))
    _=ax.set_ylim([-75,15])
    _=ax.set_ylabel("Est. Mortality Reduction [%]")
    _=ax.set_xlabel("Pronation Duration [Hour(s)]")
    _=ax.legend(loc="lower left")
    _=ax.set_title(title)
    return ax

def tte_boxplots(df,population="Severe ARDS",title="Logistic Regression | pf<150mmHg , pronation>16h",model="Logistic Regression"):
    fig,ax=plt.subplots(1,2,figsize=(14,5))

    data=df[(df.Population==population)&(df["Intervention Proxy"]=='Pseudo Duty Cycle')&(df["Model"]==model)]
    plot_boxplots(ax[0],pivot_results(data),("Pseudo Duty Cycle"))
    ax[0].set_title("Pseudo Duty Cycle")

    data=df[(df.Population==population)&(df["Intervention Proxy"]=='Uniform')&(df["Model"]==model)]
    plot_boxplots(ax[1],pivot_results(data),("Uniform"))
    fig.suptitle(title)
    return fig

if __name__=="__main__":
    import argparse
    import pandas as pd
    from pathlib import Path
    from .etl import dataset, intervention_proxy__duty_cycle

    parser=argparse.ArgumentParser()
    parser.add_argument("-d","--dir")


    args=parser.parse_args()
    SAVEDIR=Path(args.dir)

    try:
        results_df=pd.read_pickle(SAVEDIR / "results.pickle")
    except:
        print("Invalid result, trying to recover checkpoints")
        chk=[]
        for file in SAVEDIR.glob("*.csv"):
            chk.append(pd.read_csv(file))
        results_df=pd.concat(chk,axis=0)


    results_df=process_results(results_df)
    sensitivity_by_aggregation(results_df).savefig(SAVEDIR / "by_aggregation.png",dpi=500)
    sensitivity_by_causality(results_df).savefig(SAVEDIR / "by_causality.png",dpi=500)
    tte_histogram(results_df).savefig(SAVEDIR / "severeards_histograms.png",dpi=500)
    tte_histogram(results_df,population="ARDS",title="pronation>16h").savefig(SAVEDIR / "ards_histograms.png",dpi=500)
    tte_boxplots(results_df).savefig(SAVEDIR / "log_regr_severeards_boxplots.png",dpi=500)
    tte_boxplots(results_df,population="ARDS",title="pronation>16h").savefig(SAVEDIR / "log_regr_ards_boxplots.png",dpi=500)
    tte_boxplots(results_df,model="LGBM",title="LGBM | pf<150mmHg , pronation>16h").savefig(SAVEDIR / "lgbm_severeards_boxplots.png",dpi=500)
    tte_boxplots(results_df,model="LGBM",population="ARDS",title="LGBM | pronation>16h").savefig(SAVEDIR / "lgbm_ards_boxplots.png",dpi=500)
    
    tb=make_tableone(dataset(intervention_proxy__duty_cycle))
    with open(SAVEDIR / "tableone.txt","w") as file:
        file.write(tb.to_latex().replace("%","\%").replace("<","$<$"))