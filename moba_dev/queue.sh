#!/bin/zsh
# Sequential job queue: moba COR echo 1, LLR reference COR, then moba AX and SAG.
export QT1_PHANTOM_DIR=$HOME/work/moba/phantom
export BART_TOOLBOX_PATH=/Users/jackewins/bart
export OMP_NUM_THREADS=10
REPO=/Users/jackewins/Documents/MRI_PhD/Projects/Blood-and-bleeds/AGLOW_scans/qT1-iso/.claude/worktrees/agent-a675baaf449620218
W=$HOME/work/moba
cd $REPO
while pgrep -f "moba_core.py COR 0" > /dev/null; do sleep 20; done
echo "COR e1 start $(date)"
python -u $W/moba_core.py COR 1 > $W/run_COR_e1.log 2>&1
echo "LLR start $(date)"
( time python -u synth/run_llr_reference.py $QT1_PHANTOM_DIR COR ) > $W/run_llr_COR.log 2>&1
echo "AX start $(date)"
python -u $W/moba_core.py AX 0 1 > $W/run_AX.log 2>&1
echo "SAG start $(date)"
python -u $W/moba_core.py SAG 0 1 > $W/run_SAG.log 2>&1
echo "done $(date)"
