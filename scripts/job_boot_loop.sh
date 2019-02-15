#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
cd "$PBS_O_WORKDIR"

for dir in *_*
do
cd $dir
if [ ! -f vespa_boot.txt ]
then

~/scripts/stack_boot.sh
fi
cd ../
done
