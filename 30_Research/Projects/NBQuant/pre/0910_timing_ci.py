import json
from pathlib import Path
import numpy as np
root=Path('/home/byxin/Documents/academic/QuantData/e2e_share/results/han_full_v7_solver_carry')
rows=[]
for p in root.glob('w*/*/seed_*/summary.json'):
 d=json.loads(p.read_text())
 if d['window_idx']==8 and d['seed']==3: continue
 t=json.loads((p.parent/'timing_summary.json').read_text())['phases']
 phase=lambda k:t.get(k,{}).get('work_seconds',0)/60
 vals=[phase('training_data_prepare'),phase('training_period_prepare'),phase('training_predictor_forward'),phase('training_backward'),phase('training_oracle_solver'),sum(phase(k) for k in ['training_zero_grad','training_gradient_clip','training_optimizer_step','training_scheduler_step']),d['validation_wall_seconds']/60]
 vals += [d['fit_wall_seconds']/60-sum(vals), d['fit_wall_seconds']/60]
 assert vals[-2]>=0
 rows.append((d['method'],d['window_idx'],vals))
rng=np.random.default_rng(20260911)
ix=rng.integers(0,9,(20000,9))
out={}
for m in ['pto_pearsonIC','spo_plus']:
 a=[r for r in rows if r[0]==m];assert len(a)==26
 sums=np.array([np.sum([r[2] for r in a if r[1]==w],axis=0) for w in range(9)])
 counts=np.array([sum(r[1]==w for r in a) for w in range(9)])
 boots=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)[:,None]
 mean=np.mean([r[2] for r in a],axis=0)
 low,high=np.quantile(boots,[.025,.975],axis=0)
 out[m]=dict(mean=mean.tolist(),low=low.tolist(),high=high.tolist())
print(json.dumps(out,indent=2))
Path(__file__).with_name('0910_timing_ci.json').write_text(json.dumps(out))
