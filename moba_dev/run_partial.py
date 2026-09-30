"""Execute the first N cells of Recon_MOBA_V1.ipynb into a scratch copy (for testing)."""
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
nb = nbf.read(src, as_version=4)
nb.cells = nb.cells[:n]
ExecutePreprocessor(timeout=-1, kernel_name='python3').preprocess(nb, {'metadata': {'path': sys.argv[4]}})
for c in nb.cells:
    if c.cell_type == 'code':
        for o in c.get('outputs', []):
            if 'text' in o:
                print(o['text'][-3000:])
            elif o.get('output_type') == 'error':
                print('ERROR', o['ename'], o['evalue'])
nbf.write(nb, out)
