"""Export measured replay drift and optimizer update comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
proof=json.loads((OUT/'pilot_verification.json').read_text());reports=proof['reports']
old=json.loads((OUT.parents[1]/'nero_act_approach_preservation_20261021/training/pilot_A/artifacts/approach_after.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
names=['Previous preservation','Fixed 1e-6','Linear decay']
values=[old]+[reports[a]['approach_after'] for a in ['fixed','decay']]
x=np.arange(3)
axes[0].bar(x-.18,[r['rms_arm_prefix_change_rad']*1000 for r in values],.36,label='RMS')
axes[0].bar(x+.18,[r['maximum_arm_prefix_change_rad']*1000 for r in values],.36,label='Maximum')
axes[0].axhline(.1,color='C0',ls='--',lw=1,label='RMS gate 0.1')
axes[0].axhline(.3,color='C1',ls='--',lw=1,label='Max gate 0.3')
axes[0].set(yscale='log',ylabel='Arm prefix drift (mrad)',title='423 training replay queries')
axes[0].set_xticks(x,names,rotation=15,ha='right');axes[0].legend(fontsize=7)
for arm,label in [('fixed','Fixed'),('decay','Linear decay')]:
    rows=reports[arm]['measured_updates']
    axes[1].plot([r['update'] for r in rows],[r['maximum_arm_update_rad']*1000 for r in rows],marker='o',label=label)
    axes[2].plot([1,1000],[reports[arm]['first_arm_lr'],reports[arm]['last_arm_lr']],label=label)
axes[1].set(xlabel='Optimizer update',ylabel='Max arm output change (mrad)',title='Measured step on 8 cyclic queries',yscale='log');axes[1].legend(fontsize=8)
axes[2].set(xlabel='Optimizer update',ylabel='Arm learning rate',title='Actual rates bound to 1000-row traces');axes[2].legend(fontsize=8)
fig.savefig(OUT/'optimizer_comparison.png',dpi=180)
