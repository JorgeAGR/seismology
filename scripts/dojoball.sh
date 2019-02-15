for dir in 0* 1* n*
do
cd $dir
../job.sh
~/scripts/mvvespadata.sh
cd ../
done
