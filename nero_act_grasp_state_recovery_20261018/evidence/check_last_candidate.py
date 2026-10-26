from pathlib import Path
import subprocess,sys
r=Path.cwd()
for name,args in [('export',['-m','training.export','--checkpoint',str(r/'training/run20_grasp_state_recovery/checkpoints/policy_last.ckpt'),'--output',str(r/'model/run20_last_development_candidate')]),('nominal',['-m','simulation.evaluate_generalized','--model',str(r/'model/run20_last_development_candidate'),'--output',str(r/'evaluation/run20_last_nominal'),'--nominal-only','--development-regression'])]:
 with (r/'evidence'/('run20_last_'+name+'.log')).open('x') as stream:subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)
