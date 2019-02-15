#!/bin/bash
ulimit –s unlimited
~/scripts/vespa.o  << EOF
../../15-50
input.dat
5001
-1.4 0 0.01 1
EOF
