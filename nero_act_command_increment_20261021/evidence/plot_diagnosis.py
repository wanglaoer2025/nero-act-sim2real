from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'nero_act_grouped_gripper_20261021'
fig,axes=plt.subplots(3,1,figsize=(9,8),constrained_layout=True)
for ax,name,title in zip(axes[:2],['nominal_pre_close','heldout_support_release'],['Closing stalls at chunk boundaries','Opening stalls at chunk boundaries']):
    with np.load(OLD/'results/mechanism_original'/name/'rollout.npz',allow_pickle=False) as trace:width=trace['sent_targets'][-24:,7]*1000
    ax.plot(np.arange(len(width)),width,'o-',markersize=3)
    ax.set(title=title,xlabel='Issued control point (last24)',ylabel='Command width (mm)');ax.grid(alpha=.2)
absolute=json.loads((ROOT/'evidence/nero_head_capacity_probe.json').read_text())['results'];increment=json.loads((ROOT/'evidence/nero_increment_capacity_probe.json').read_text())['results'][0]
x=np.arange(3);width=.24
for offset,result,label in [(-1,absolute[0],'Absolute ACT features'),(0,absolute[1],'Absolute features + current15'),(1,increment,'Previous-command increment')]:
    errors=[result['critical'][j]['error_mm'] for j in range(3)]
    axes[2].bar(x+offset*width,errors,width,label=label)
axes[2].axhline(.1,color='black',linestyle='--',linewidth=1,label='Original0.1mm input gate')
axes[2].set(xticks=x,xticklabels=['Holding2017:5','Pre-close2018:5','Boundary2018:10'],ylabel='Max first2 error (mm)',title='Same cached-feature development probe:1000 steps per temporary head')
axes[2].legend(fontsize=8,ncol=2);axes[2].grid(axis='y',alpha=.2)
fig.savefig(ROOT/'evidence/diagnostic_figure.png',dpi=160)
