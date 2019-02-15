#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
cd "$PBS_O_WORKDIR"
while read line 
do
time=$(awk '{print $1}' <<<"$line")
slow=$(awk '{print $2}' <<<"$line")
awk -v t=$time -v s=$slow '{if ($1==t && $2==s) print $0}' boot_vespa.txt >> boot_wiggle.dat
done < ~/scripts/SSinfo_0.1s.dat

