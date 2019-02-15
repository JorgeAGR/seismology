#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
cd "$PBS_O_WORKDIR"
awk '{print $1, $2, $4, $3}' boot_vespa.txt > test
mv -f test boot_vespa.txt
awk '{print $1, $2, $4, $3}' boot_wiggle.dat > test
mv -f test boot_wiggle.dat

