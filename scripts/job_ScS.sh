#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
cd "$PBS_O_WORKDIR"
~/scripts/vespa.o << EOF
data/
input.dat
41001
-20 5 0.01 4
EOF

