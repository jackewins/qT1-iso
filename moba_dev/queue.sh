#!/bin/bash
# Sequential job queue: moba COR echo 1, LLR reference COR, then moba AX and SAG.
export QT1_PHANTOM_DIR=${QT1_PHANTOM_DIR:-$HOME/work/phantom}
export BART_TOOLBOX_PATH=${BART_TOOLBOX_PATH:-$HOME/bart}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=$REPO/moba_dev; L=$QT1_PHANTOM_DIR/moba
cd $REPO
while pgrep -f "moba_core.py COR 0" > /dev/null; do sleep 20; done
echo "COR e1 start $(date)"
python -u $W/moba_core.py COR 1 > $L/run_COR_e1.log 2>&1
echo "LLR start $(date)"
( time python -u synth/run_llr_reference.py $QT1_PHANTOM_DIR COR ) > $L/run_llr_COR.log 2>&1
echo "AX start $(date)"
python -u $W/moba_core.py AX 0 1 > $L/run_AX.log 2>&1
echo "SAG start $(date)"
python -u $W/moba_core.py SAG 0 1 > $L/run_SAG.log 2>&1
echo "done $(date)"
