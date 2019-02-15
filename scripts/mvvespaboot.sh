#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
cd "$PBS_O_WORKDIR"
awk 'NR>1 {printf "%.1f %.2f %.3f %.3f \n", $1-400, $3, $2, $4}' boot_vespa.txt > test1
mv -f test1 boot_vespa.txt
