#!/bin/bash
#A script to download and decompress the netlib mps files
#download netlib if we havent
wget --no-clobber --recursive --no-parent https://netlib.org/lp/data/
nld=netlib.org/lp/data
nl=netlib.org/lp
#create directory for not problem data
mkdir $nl/not_problem_data
nlnd=$nl/not_problem_data
#mv robots.txt
mv netlib.org/robots.txt $nlnd
#remove stochfor and truss problems since it requires some Fortran mumbo jumbo
rm -rf $nld/stocfor3 $nld/stocfor*
rm -rf $nld/truss
#move non-problem files to not_problem_data
mv --backup=numbered $nld/mpc.src $nld/nams.ps.gz $nld/minos $nld/index.html $nld/ascii $nld/changes $nld/emps.* $nld/readme $nld/kennington/readme $nld/kennington/index.html $nlnd
#compile the C file for decompression
$CC $nlnd/emps.c -o $nlnd/decompress.out

mkdir mps_problems
#uncompress the files
echo "uncompressing regular files"
#this does all the regular files
for f in $(find $nld -maxdepth 1 -type f)
    do echo $f;
    $nlnd/decompress.out $f > mps_problems/$(basename $f);
done
echo "ungzipping kennington files" 
#and now let's do the kennington files too 
for f in $(find $nld/kennington/ -maxdepth 1 -type f)
    do echo $f;
    gzip -d --force $f;
done

echo "uncompressing kennington files" 
#and now let's do the kennington files too 
for f in $(find $nld/kennington/ -maxdepth 1 -type f)
    do echo $f;
    $nlnd/decompress.out $f > mps_problems/$(basename $f);
done


