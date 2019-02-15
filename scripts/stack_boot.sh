
~/scripts/vespa_boot.o  << EOF
../../15-50
input.dat
5001
-1.4 0 0.01 1
EOF

awk 'NR>1 {printf "%.1f %.2f %.3f %.3f \n", $1-400, $3, $2, $4}' boot_vespa.txt > test1
mv -f test1 boot_vespa.txt

