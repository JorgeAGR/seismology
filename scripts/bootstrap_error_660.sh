#! /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=100:00:00
cd "$PBS_O_WORKDIR"

for dir in 0* 1* n0* n1*
do
cd $dir/

taup_time -h 0 -ph SS,S^660S -mod ~/TauP-2.1.2/StdModels/SdS_0.0_2.5.2.taup -deg 125 > output 
slowSdS=`awk 'NR==6 {print $5}' output`
timeSdS=`awk 'NR==6 {print $4}' output`
slowSS=`awk 'NR==7 {print $5}' output`
timeSS=`awk 'NR==7 {print $4}' output`
time1=`echo $timeSdS-$timeSS-15 | bc -l`
time2=`echo $timeSdS-$timeSS+15 | bc -l`
slow=`echo $slowSdS-$slowSS | bc -l`

~/scripts/bootstrap_errors.o << EOF
../../660
input.dat
5001
$slow $slow 0.01 1
$time1 $time2
EOF

mv SdStimes.dat S660Stimes.dat
rm -fr output
cd ../
done
