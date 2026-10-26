"""Plot recorded simulation evidence, explicitly a failed full-task attempt."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];p=root/'evidence/run10_supervised_development_v6/nominal'
m=json.loads((p/'measurements.json').read_text());r=json.loads((p/'result.json').read_text());trace=np.load(p/'rollout.npz')
t=np.arange(1,len(m)+1)*.05;tcp=np.array([x['tcp_xyz'] for x in m]);cup=np.array([x['cup_xyz'] for x in m])
fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
ax[0,0].plot(t,tcp[:,2]*1000,label='Measured TCP');ax[0,0].plot(t,(cup[:,2]-.0475)*1000,label='Cup bottom (nominal geometry)');ax[0,0].set(ylabel='Height over table (mm)',xlabel='Physical simulation time (s)');ax[0,0].legend()
ax[0,1].plot(cup[:,0]*1000,cup[:,1]*1000,label='Actual cup path');ax[0,1].scatter(*((np.array([-.5247430758,-.1620556790])*1000)),color='blue',label='Target');ax[0,1].set(xlabel='Base X (mm)',ylabel='Base Y (mm)');ax[0,1].legend();ax[0,1].axis('equal')
width=np.array([x['gripper_width'] for x in m])*1000;cmd=trace['sent_targets'][:,7]*1000
ax[1,0].plot(t,width,label='Measured width');ax[1,0].plot(t,cmd,label='Sent ACT/protected reference');ax[1,0].set(xlabel='Physical simulation time (s)',ylabel='Total opening (mm)');ax[1,0].legend()
qt=trace['query_steps']*.05;raw=trace['raw_chunks'][:,:10,7]*1000
ax[1,1].plot(qt,raw[:,-1],label='Raw ACT last prefix width');ax[1,1].plot(t,width,label='Measured width');ax[1,1].set(xlim=(95,t[-1]),ylim=(63,80),xlabel='Physical simulation time (s)',ylabel='Release detail (mm)');ax[1,1].legend()
for event in r['simulated_events']:
 for a in [ax[0,0],ax[1,0]]:a.axvline(event['physics_step']*.05,color='grey',alpha=.4,linestyle=':')
for a in ax.flat:a.grid(alpha=.2)
fig.suptitle('ACT simulation diagnostic: lift/carry completed, release stalled; full task FAILED')
fig.savefig(root/'evidence/run10_v6_motion_diagnostic.png',dpi=150);plt.close(fig)
