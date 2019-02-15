#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00

cd "$PBS_O_WORKDIR"

~/scripts/plotvespa_sdfill.sh
