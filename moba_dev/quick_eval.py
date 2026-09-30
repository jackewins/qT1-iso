import sys, glob
from pathlib import Path
import numpy as np
REPO = Path('/Users/jackewins/Documents/MRI_PhD/Projects/Blood-and-bleeds/AGLOW_scans/qT1-iso/.claude/worktrees/agent-a675baaf449620218')
sys.path.insert(0, str(REPO / 'synth'))
import pipeline_utils as pu
P = Path.home() / 'work/moba/phantom'
plane = sys.argv[1]
truth = dict(np.load(P / f'truth_{plane}.npz'))
head = truth['label'] >= 3
for f in sorted(glob.glob(str(P / f'moba/moba_{plane}_e*.npz'))):
    r = np.load(f)
    m = r['maps']
    R1 = np.real(m[..., 2])
    T1 = np.where(head & (R1 > 0), 1 / np.maximum(R1, 1e-6), np.nan)
    print(f, float(r['runtime_s']) / 60, 'min')
    print(str(r.get('log', '')))
    for l, n in ((5, 'WM'), (4, 'GM'), (3, 'CSF')):
        print(n, 'median |Mss|', np.median(np.abs(m[..., 0])[truth['label'] == l]), 'R1', np.median(R1[truth['label'] == l]))
    pu.print_eval(pu.evaluate_t1(T1, truth), f)
