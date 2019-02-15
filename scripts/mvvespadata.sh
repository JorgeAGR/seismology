awk 'NR>1 {printf "%.1f %.2f %.3f\n", $1-400, $3, $2}' vespa.txt > test
mv -f test vespa.txt
