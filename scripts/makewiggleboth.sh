#! /bin/bash

rm -fr *wig*
while read line 
do echo $line > test3
time=`awk '{print $1}' test3`
slow=`awk '{print $2}' test3`
rm -fr test3
awk -v t=$time -v s=$slow '{if ($1==t && $2==s) print $0}' boot_vespa.txt >> wig
awk -v t=$time -v s=$slow '{if ($1==t && $2==s) print $0}' vespa.txt >> wig1
done < ~/scripts/SSinfo_0.1s.dat
rm -fr test test2
mv wig boot_wiggle.dat
mv wig1 wiggle_orig.dat
awk '{print $1, $2, $3}' wiggle_orig.dat | sample1d -I0.01 > wiggle_0.01s.dat
awk '{print $1, $2, $3, $4}' boot_wiggle.dat | sample1d -I0.01 > boot_wiggle_0.01s.dat
