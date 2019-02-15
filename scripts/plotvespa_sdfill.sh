#! /bin/bash
gmt gmtset ANOT_FONT_SIZE 11
gmt gmtset LABEL_FONT_SIZE 11
gmt gmtset HEADER_FONT_SIZE 13

pwd > test
loc=`awk -F'/' '{print $(NF)}' test`
rm -fr test
echo $loc > test
loc1=`awk -F'_' '{print $1}' test`
loc2=`awk -F'_' '{print $2}' test`
rm -fr test

# Plots vespagrams after running dopwsall. Need data.dat from dopwsall.sh. Need dopwsall.sh in folder you are running this in
gmt grd2cpt data.grd -Z -C../../../scripts/bluered.cpt -T= > scale.cpt 
set orientation = ''

mulfactor=5
# write PS header and time stamp
echo 0 0  | gmt psxy -R1/2/1/2 -JX7.5/11 -Sp -K  $orientation > vespa_sd_fill.ps

# plot vespagram
awk 'NR==4 {print $1}' scale.cpt > test1
echo $mulfactor > test2
paste test1 test2 > test3
newlim=`awk '{print -1*$1/$2}' test3`
newlim2=`awk '{print $1/$2}' test3`
scalelim=`awk '{print -1*$1/($2*10)}' test3`
rm -fr test1 test2 test3
gmt makecpt -Cscale -T$newlim2/$newlim/$scalelim -D > scale2.cpt
gmt grdimage data.grd -R-400/50/-1.4/0 -Cscale2.cpt -JX12/-8 $o -E500 -Bf20a40:"":/f0.1a0.2:"Slowness (s/deg)"::."":Wsne -O -K -Y3 -X2 >> vespa_sd_fill.ps
gmt pstext -R -JX -O -K -N >> vespa_sd_fill.ps <<EOF
-175 -1.6 11 0 0 CM (${loc1}, ${loc2}), bootstrap
#-460 -1.4 11 0 0 CM a
#95 -1.25 11 0 0 CM Max
#95 -0.15 11 0 0 CM Min
EOF

awk '{print $2, $3}' ~/scripts/SSpred.dat | gmt psxy -R -JX -O -K -W1p,-- >> vespa_sd_fill.ps

awk '{if ($1>-350.01) print $0}' wiggle_orig.dat > wig1
awk '{if ($1>-350.01) print $0}' boot_wiggle.dat > wig2
paste wig1 wig2 > wiggle
rm -fr wig1 wig2

awk '{if ($1<-75) print $1, $2, ($3*$3)^0.5, $7}' wiggle > wigprec
awk '{if ($1>-75) print $1, $2, ($3*$3)^0.5, $7}' wiggle > wigphase

awk 'BEGIN{first=1;} {if (first) { max = min = $3; first = 0; next;} if (max < $3) max=$3; } END { print max }' wigprec > test
norm=`awk '{print $1}' test`
awk -v norm=$norm '{if ($1<-75) print $1, $3/norm, ($3/norm)-($7/norm), ($3/norm)+($7/norm)}' wiggle > wiggle.dat
rm -fr wig test wigprec

awk 'BEGIN{first=1;} {if (first) { max = min = $3; first = 0; next;} if (max < $3) max=$3; } END { print max }' wigphase > test
norm=`awk '{print $1}' test`
awk -v norm=$norm '{if ($1>-75) print $1, $3/norm, ($3/norm)-($7/norm), ($3/norm)+($7/norm)}' wiggle >> wiggle.dat
rm -fr test wigphase wiggle


awk '{print $1, $2}' wiggle.dat | gmt psxy -R-400/50/-1.5/1.5 -JX12/2 -W1 -Y-2.4 -Bf20a40:"Travel time (s)":/f2a0:"":WSne -O -K -N >> vespa_sd_fill.ps

awk '{if ($3>0) print $3, $1}' wiggle.dat > wig1.dat
awk '{if ($3<=0) print "NEW", $1}' wiggle.dat >> wig1.dat
sort -k 2 -n wig1.dat > test
mv -f test wig1.dat

echo "NEW 360" > test
awk '$1==last{next} {last=$1} 1' wig1.dat >> test
awk '/NEW/{x="F"++i;}{print > x;}' test
rm -fr test
for file in F*
do grep -Ev NEW $file > test
   mv -f test $file
done
find ./ -type f -empty -delete
for file in F*
	    do
awk 'NR==1 {print $0}' $file > test2
cat $file test2 > test3
mv -f test3 $file
rm -fr test2
done

for file in F*
	    do
awk '{print $2, $1}' $file | gmt psxy -R -J -O -K -N -L -W0.01p,255/0/0 -G255/0/0 >> vespa_sd_fill.ps
done
rm -fr F*

awk '{if ($4<0) print $4, $1}' wiggle.dat > wig2.dat
awk '{if (0<=$4) print "NEW", $1}' wiggle.dat >> wig2.dat
sort -k 2 -n wig2.dat > test
mv -f test wig2.dat

echo "NEW 360" > test
awk '$1==last{next} {last=$1} 1' wig2.dat >> test
awk '/NEW/{x="F"++i;}{print > x;}' test
rm -fr test
for file in F*
do grep -Ev NEW $file > test
   mv -f test $file
done
find ./ -type f -empty -delete
for file in F*
	    do
awk 'NR==1 {print $0}' $file > test2
cat $file test2 > test3
mv -f test3 $file
rm -fr test2
done

for file in F*
	    do
awk '{print $2, $1}' $file | gmt psxy -R -J -O -K -N -L -W0.01p,255/0/0 -G255/0/0 >> vespa_sd_fill.ps
done
rm -fr F*

#awk '{print $1, $3}' wiggle.dat | gmt psxy -R -JX -W0.5,-- -O -K -N >> vespa_sd_fill.ps
#awk '{print $1, $4}' wiggle.dat | gmt psxy -R -JX -W0.5,-- -O -K -N >> vespa_sd_fill.ps
awk '{print $1, $2}' wiggle.dat | gmt psxy -R -JX -W1 -O -K -N >> vespa_sd_fill.ps
     
gmt psxy -R -J -W1p,-- -O -K -N >> vespa_sd_fill.ps <<EOF
-75 -1.5
-75 1.5
EOF

#pstext -R -J -O -K -N >> vespa_sd_fill.ps << EOF
#0 -1.1 11 0 0 CM SS
#-155 -1.1 11 0 0 CM S410S
#-224 -1.1 11 0 0 CM S660S
#-302 -1.1 11 0 0 CM S1000S
#EOF

#make colour scale
#awk 'NR<=3 {print $0}' scale2.cpt > test
#awk '4<=NR && NR<=13 {printf "%.7f %.0f %.0f %.0f %.7f %.0f %.0f %.0f\n", $1, $2, $3, $4, $5, $6, $7, $8}' scale2.cpt >> test
#awk '14<=NR {print $0}' scale2.cpt >> test
#mv -f test scale2.cpt
#gmt psscale -Cscale2.cpt -D4/0.875/5/0.4 -X9 -Y5.5 -O -K -N -B100000:."": >> vespa_sd_fill.ps
#
# write PS trailer 
echo 0 0  | gmt psxy -R1/2/1/2 -JX -Sp -O >> vespa_sd_fill.ps
#
#
#gv boot_sd_fill.ps 
rm -fr scale.cpt *tmp scale2.cpt wiggle.dat wig1.dat wig2.dat
#
#
