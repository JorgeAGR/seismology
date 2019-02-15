#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00

cd "$PBS_O_WORKDIR"

for dir1 in 0* 1* n0* n1*
do
cd $dir1
~/scripts/plotvespa_sdfill.sh
cd ../
done
