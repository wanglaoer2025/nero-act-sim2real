"""Export the observed action-horizon mismatch and frozen-input command fixed point."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
h=json.loads((OUT/'horizon_probe.json').read_text())['records'][0]
p=json.loads((OUT/'policy_probe.json').read_text())
s=next(r for r in p['previous_command_sensitivity'] if r['policy']=='fixed' and 'pre_close' in r['query'])
x=np.arange(1,21);fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].plot(x,h['teacher_width_mm'],marker='.',label='Teacher (0.2 mm per point)')
axes[0].plot(x,h['prediction_width_mm'],marker='.',label='Fixed policy (diagnostic batch of 5)')
axes[0].axvspan(.7,2.3,color='grey',alpha=.17,label='Only first 2 are executed')
axes[0].set(xlabel='Point in predicted action chunk',ylabel='Gripper target width (mm)',title='Same nominal training recovery input')
axes[0].legend(fontsize=8)
u=np.array([r['previous_mm'] for r in s['values']]);d=np.array([r['next_minus_previous_mm'] for r in s['values']])
mask=(u>70)&(u<74.6);axes[1].plot(u[mask],d[mask],label='Next second-point target minus previous')
axes[1].axhline(0,color='black',lw=1)
axes[1].axvline(s['measured_width_mm']-3,color='C1',ls='--',label='3 mm closing-margin boundary')
axes[1].axvspan(72.55265522,72.65265522,color='C3',alpha=.25,label='Stable zero bracket (frozen RGB/state)')
axes[1].set(xlabel='Previous gripper command (mm)',ylabel='Next command change (mm)',title='Learned local command map stops closing')
axes[1].legend(fontsize=8)
fig.savefig(OUT/'root_cause.png',dpi=180)
