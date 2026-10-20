"""Recreate English Figures 3--6 from the supplied existing-result snapshots.

Requires numpy and matplotlib. Run: python plot_figures.py
No training or attacks are run. Outputs go to the parent directory.
All standard deviations use ddof=1. Curves are not smoothed.
"""
from pathlib import Path
import csv, os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
SEEDS=(42,43,44)
METHODS=['fedavg','random_k','top_k','powersgd','flocora','dense_rp','random_k_clip_noise','pri_prune_magnitude','drpe']
ATTACK_METHODS=[m for m in METHODS if m!='random_k_clip_noise']
NAMES=dict(zip(METHODS,['FedAvg','Random-k','Top-k','PowerSGD','FLoCoRA','Dense-RP','Random-k + clipping + noise','PriPrune-Magnitude','DRPE']))
NAMES.update(static='Fixed random states + clipping + noise',clip_only='Dynamic projection + clipping, no noise')
COLORS=dict(zip(METHODS,['#636363','#2878B5','#28917E','#8E6BA8','#BD982D','#648594','#E68A2E','#8B564B','#C84C3A']))
COLORS.update(static='#817093',clip_only='#2878B5')
STYLES=dict(zip(METHODS,['--','-.',':',':','-.','--',':','-.','-']))
MARKERS=dict(zip(METHODS,['o','s','^','D','v','P','X','h','*']))
def read(name):
    with (HERE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
CURVES=read('training_curves.csv')
SUMMARY=read('training_summary.csv')
MAIN=read('main_attacks.csv')
EXTRA=read('label_noise_attacks.csv')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.labelsize':11,
    'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
    'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white'})
def clean(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E6E8',lw=.6)
def vals(rows,key):return np.array([float(r[key]) for r in rows])
def train(method):
    curves=np.array([[float(r['val_accuracy']) for r in CURVES if r['method']==method and int(r['seed'])==s] for s in SEEDS])
    rr=[r for r in SUMMARY if r['method']==method]
    return curves,vals(rr,'test_accuracy'),vals(rr,'uplink_mib')
def attack(method,kind,extra=False):
    rr=[r for r in (EXTRA if extra else MAIN) if r['method']==method and r['attack']==kind]
    assert len(rr)==60,(method,kind,len(rr))
    return rr
def caption(ax,text,y=-.26):ax.text(.5,y,text,ha='center',va='top',transform=ax.transAxes,fontsize=11)
def save(fig,name):
    for ax in fig.axes:clean(ax)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
    print(name)

def plot3():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.3,5.5))
    fig.subplots_adjust(left=.075,right=.99,top=.97,bottom=.43,wspace=.28)
    x=np.arange(1,201)
    for m in ['fedavg','random_k','random_k_clip_noise','pri_prune_magnitude','drpe']:
        curves,_,_=train(m);mean=curves.mean(0);sd=curves.std(0,ddof=1)
        ax.plot(x,mean,color=COLORS[m],ls=STYLES[m],lw=1.4,label=NAMES[m])
        ax.fill_between(x,mean-sd,mean+sd,color=COLORS[m],alpha=.10,lw=0)
    ax.set(xlabel='Communication round',ylabel='Validation accuracy (%)',xlim=(1,200),ylim=(10,95))
    caption(ax,'(a) Validation accuracy across three seeds')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.39),ncol=1,frameon=False,fontsize=9.5)
    for m in METHODS:
        _,accuracy,uplink=train(m)
        bx.errorbar(uplink.mean(),accuracy.mean(),yerr=accuracy.std(ddof=1),fmt=MARKERS[m],
            color=COLORS[m],ms=9 if m=='drpe' else 6,capsize=3,elinewidth=1,label=NAMES[m])
    bx.set(xlabel='Uplink per round (MiB)',ylabel='Test accuracy (%)',xlim=(20,91),ylim=(78,90))
    caption(bx,'(b) Uplink payload and test accuracy')
    bx.legend(loc='upper center',bbox_to_anchor=(.5,-.39),ncol=2,frameon=False,fontsize=9,
              columnspacing=.8,handletextpad=.5)
    save(fig,'fig3_utility_communication')

def plot4():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.3,5.2))
    fig.subplots_adjust(left=.075,right=.99,top=.96,bottom=.36,wspace=.30)
    methods=['static','clip_only','drpe']
    for m,ls in zip(methods,['--','-.','-']):
        curves,acc,_=train(m);mean=curves.mean(0);sd=curves.std(0,ddof=1)
        ax.plot(np.arange(1,201),mean,color=COLORS[m],ls=ls,lw=1.4,label=NAMES[m])
        ax.fill_between(np.arange(1,201),mean-sd,mean+sd,color=COLORS[m],alpha=.11,lw=0)
    ax.set(xlabel='Communication round',ylabel='Validation accuracy (%)',xlim=(1,200),ylim=(10,95))
    caption(ax,'(a) Training trajectories',y=-.27)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.39),frameon=False,fontsize=9)
    for i,m in enumerate(methods):
        _,acc,_=train(m)
        bx.errorbar(i,acc.mean(),yerr=acc.std(ddof=1),fmt='_',markersize=19,color=COLORS[m],capsize=5,lw=1.2)
        for offset,v,mark in zip([-.105,0,.105],acc,['o','s','^']):
            bx.scatter(i+offset,v,marker=mark,s=43,facecolors='white',edgecolors=COLORS[m],zorder=3)
        bx.text(i,acc.max()+.75,f'{acc.mean():.2f} $\\pm$ {acc.std(ddof=1):.2f}',ha='center',fontsize=10)
    bx.set(xticks=range(3),xticklabels=['Fixed random states\n+ clipping + noise','Dynamic projection\n+ clipping, no noise','Full DRPE'],
           ylabel='Test accuracy (%)',ylim=(75,89),xlim=(-.5,2.5))
    bx.tick_params(axis='x',labelsize=8.8)
    caption(bx,'(b) Test accuracy for each training seed',y=-.27)
    bx.legend(handles=[Line2D([],[],color='#555555',marker=m,ls='',markerfacecolor='white',label=f'Seed {s}')
                       for s,m in zip(SEEDS,['o','s','^'])],loc='upper center',bbox_to_anchor=(.5,-.39),frameon=False,ncol=3,fontsize=9)
    save(fig,'fig4_component_training')

def plot5():
    fig,(ax,bx,cx)=plt.subplots(1,3,figsize=(10.1,4.8))
    fig.subplots_adjust(left=.065,right=.99,top=.96,bottom=.43,wspace=.38)
    rows=[attack(m,'idlg') for m in ATTACK_METHODS]
    rates=[100*vals(r,'label_correct').mean() for r in rows]
    ax.bar(range(8),rates,color=[COLORS[m] for m in ATTACK_METHODS],width=.65)
    for i,v in enumerate(rates):ax.text(i,v+1.5,f'{v:.1f}'.rstrip('0').rstrip('.'),ha='center',fontsize=9)
    ax.set(ylabel='Label recovery rate (%)',ylim=(0,112))
    boxes=bx.boxplot([vals(r,'lpips') for r in rows],positions=range(8),widths=.6,patch_artist=True,
        medianprops={'color':'#303030','linewidth':1},flierprops={'marker':'.','markersize':2.5,'markeredgecolor':'#777777'})
    for patch,m in zip(boxes['boxes'],ATTACK_METHODS):patch.set(facecolor=COLORS[m],alpha=.5,linewidth=.7)
    bx.set(ylabel='LPIPS',ylim=(0,.52))
    ticks=[NAMES[m] if m!='pri_prune_magnitude' else 'PriPrune-M' for m in ATTACK_METHODS]
    for a in [ax,bx]:
        a.set_xticks(range(8));a.set_xticklabels(ticks,rotation=47,ha='right',fontsize=9)
    for rr,m in zip(rows,ATTACK_METHODS):
        psnr=np.sort(vals(rr,'psnr'))
        cx.step(psnr,np.arange(1,61)/60,where='post',color=COLORS[m],ls=STYLES[m],lw=1.4,label=NAMES[m])
    cx.set(xlabel='PSNR (dB)',ylabel='Cumulative fraction of runs',ylim=(0,1.025))
    caption(ax,'(a) Label recovery',y=-.46)
    caption(bx,'(b) Perceptual distance',y=-.46)
    caption(cx,'(c) PSNR empirical CDF',y=-.46)
    handles,labels=cx.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False,fontsize=9.5,columnspacing=1.6)
    save(fig,'fig5_idlg_comparison')

def plot6():
    fig,axes=plt.subplots(2,2,figsize=(9.4,6.9))
    fig.subplots_adjust(left=.09,right=.99,top=.96,bottom=.21,hspace=.72,wspace=.29)
    oracle=[attack(m,'oracle',True) for m in ['fedavg_raw','drpe_encoded','drpe_decoded']]
    for j,field in enumerate(['psnr','ssim']):
        ax=axes[0,j]
        for i,rr in enumerate(oracle):
            values=vals(rr,field);mean=values.mean();sd=values.std(ddof=1)
            col=[COLORS['fedavg'],COLORS['drpe'],COLORS['dense_rp']][i]
            ax.bar(i,mean,.55,color=col,alpha=.8)
            ax.errorbar(i,mean,yerr=sd,fmt='none',color='#333333',capsize=4,lw=.9)
            ax.text(i,mean+sd+(.6 if field=='psnr' else .022),f'{mean:.2f}' if field=='psnr' else f'{mean:.3f}',ha='center',fontsize=10)
        ax.set(xticks=range(3),xticklabels=['FedAvg\nRaw gradients','DRPE\nEncoded','DRPE\nDecoded'],
               ylabel='PSNR (dB)' if field=='psnr' else 'SSIM',ylim=(0,27) if field=='psnr' else (0,.9))
        caption(ax,f'({"ab"[j]}) Known true labels: {field.upper()}',y=-.35)
    for j,field in enumerate(['psnr','ssim']):
        ax=axes[1,j]
        full=[attack('drpe','dlg'),attack('drpe','idlg'),oracle[1]]
        noise_free=[attack('clip_only',a,True) for a in ['dlg','idlg','oracle']]
        for offset,data,color,label,hatch in [(-.17,noise_free,COLORS['clip_only'],'Dynamic projection + clipping, no noise',''),
                                             (.17,full,COLORS['drpe'],'Full DRPE','//')]:
            means=[vals(rr,field).mean() for rr in data];sds=[vals(rr,field).std(ddof=1) for rr in data]
            ax.bar(np.arange(3)+offset,means,.30,color=color,alpha=.8,label=label,hatch=hatch,edgecolor='white',linewidth=.5)
            ax.errorbar(np.arange(3)+offset,means,yerr=sds,fmt='none',color='#333333',capsize=3,lw=.9)
        ax.set(xticks=range(3),xticklabels=['DLG','iDLG','Known labels\nEncoded space'],
               ylabel='PSNR (dB)' if field=='psnr' else 'SSIM',ylim=(0,30) if field=='psnr' else (0,1.12))
        caption(ax,f'({"cd"[j]}) Component comparison: {field.upper()}',y=-.35)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,ncol=2,fontsize=10)
    save(fig,'fig6_label_noise_comparison')

if __name__=='__main__':
    plot3();plot4();plot5();plot6()
