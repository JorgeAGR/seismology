for dir in 0* 1* n*; do cd $dir; if [ -f boot_vespa.txt ]; then if [ ! -f boot_wiggle.dat ]; then qsub -q default ~/scripts/makewiggleboot.sh ; fi; fi; cd ../; done
