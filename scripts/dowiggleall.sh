for dir in 0* 1* n*; do cd $dir; if [ -f vespa.txt ]; then if [ ! -f wiggle_orig.dat ]; then qsub -q default ~/scripts/makewiggle.sh ; fi; fi; cd ../; done
