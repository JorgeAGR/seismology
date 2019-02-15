#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
ulimit -S -s unlimited
cd "$PBS_O_WORKDIR"

for dir in *_*
do
cd $dir
if [ ! -f vespa.txt ]
then

~/scripts/stack.sh

fi

cd ../
done
