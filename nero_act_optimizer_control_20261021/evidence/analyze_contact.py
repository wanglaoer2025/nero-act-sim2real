"""Measure learned closing margins in the four failed nominal recovery traces."""
from pathlib import Path
import json,hashlib
import numpy as np

BASE=Path(__file__).resolve().parents[2]
rows=[];sources=[Path(__file__)]
for arm in ['fixed','decay']:
    root=BASE/f'nero_act_optimizer_{arm}_20261021'
    sources += [root/'executor/phases.py',root/'simulation/supervised_rollout.py']
    for name in ['nominal_pre_close','nominal_hold_to_lift']:
        folder=root/'results/mechanism'/name
        sources += [folder/'result.json',folder/'rollout.npz']
        result=json.loads((folder/'result.json').read_text())
        with np.load(folder/'rollout.npz',allow_pickle=False) as f:
            raw=f['raw_chunks'][:,:2,7];feedback=f['feedback_at_query'][:,7]
            gap=feedback[:,None]-raw;executed_gap=f['states'][:,7]-f['sent_targets'][:,7]
            rows.append(dict(arm=arm,case=name,contact_state=result['contact_state'],new_points=result['new_active_points'],error=result['error'],
                raw_prefix_width_mm=[float(raw.min()*1000),float(raw.max()*1000)],
                raw_query_closing_margin_mm=[float(gap.min()*1000),float(gap.max()*1000)],
                executed_closing_margin_mm=[float(executed_gap.min()*1000),float(executed_gap.max()*1000)],
                final_executed_closing_margin_mm=float(executed_gap[-1]*1000),final_measured_width_mm=float(f['states'][-1,7]*1000),
                last_raw_prefix_mm=(raw[-1]*1000).tolist(),last_feedback_width_mm=float(feedback[-1]*1000)))
proof=dict(complete=True,unchanged_holding_margin_mm=[3,12],required_stable_seconds=.5,rows=rows,
    gap_is_contact_proxy_not_TCP_error_or_force_calibration=True,hardware_commands_sent=False,
    source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(Path(__file__).parent/'nominal_contact_diagnostics.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(complete=True,records=len(rows))))
