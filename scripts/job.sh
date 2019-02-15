#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
ulimit -S -s unlimited
cd "$PBS_O_WORKDIR"

~/scripts/vespa.o  << EOF
../../15-50
input.dat
5001
-1.4 0 0.01 1
EOF

awk 'NR>1 {printf "%.1f %.2f %.3f\n", $1-400, $3, $2}' vespa.txt > test1
mv -f test1 vespa.txt

