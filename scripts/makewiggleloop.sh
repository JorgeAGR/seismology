#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
cd "$PBS_O_WORKDIR"

for dir in *_*
do
cd $dir
if [ ! -f wiggle_orig.dat ]
then

while read line 
do
time=$(awk '{print $1}' <<<"$line")
slow=$(awk '{print $2}' <<<"$line")
awk -v t=$time -v s=$slow '{if ($1==t && $2==s) print $0}' vespa.txt >> wiggle_orig.dat
done < ~/scripts/SSinfo_0.1s.dat

fi
cd ../
done
