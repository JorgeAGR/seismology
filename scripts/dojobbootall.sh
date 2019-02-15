for dir in 0* 1* n*
do
cd $dir
if [ ! -f boot_vespa.txt ]
then
qsub -q default ../job_boot.sh
fi
cd ../
done
