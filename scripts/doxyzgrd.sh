#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
cd "$PBS_O_WORKDIR"

gmt xyz2grd vespa.txt -Gdata.grd -I1/0.1 -R-399/99/-1.4/0
