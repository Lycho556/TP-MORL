#!/bin/bash
cd /Users/user/Desktop/TP-MORL
PY=/Users/user/.claude-science/conda/envs/tpmorl/bin/python
VF="201 202 203 204 205 206 207 208 209 210"
run(){ arm=$1; s=$2; nt=$3; ep=$4; extra=$5
  mkdir -p results_p2_val/$arm
  PYTHONPATH=src TPMORL_NJOBS=1 nohup $PY scripts/p2_generalize.py --seed $s --n-train $nt --ep-each $ep \
    --history 1 --test-fields $VF $extra --out results_p2_val/$arm > results_p2_val/$arm/log_s$s.txt 2>&1 &
}
run A 0 20 2 --baselines
run A 1 20 2
run B 0 50 2
run B 1 50 2
run C 0 20 5
run C 1 20 5
run D 0 100 2
wait
