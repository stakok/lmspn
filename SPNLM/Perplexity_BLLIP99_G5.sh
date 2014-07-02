#!/bin/ksh

i=40

head -n 8 configs/BLLIP99_test.ini > configs/test5.ini
echo "weights_file	=	savedWeights/BLLIP99/BLLIP99.weights."$i  >> configs/test5.ini
head -n 13 configs/BLLIP99_test.ini | tail -n 4 >> configs/test5.ini
echo "iterations	=	0" >> configs/test5.ini
tail -n 42 configs/BLLIP99_test.ini >> configs/test5.ini

cp codes/large/* codes
./compile.sh

./spn_train configs/test5.ini > ppl_G5.txt
./printPPL.sh ppl_G5.txt

rm configs/test5.ini
rm ppl_G5.txt
cp codes/normal/* codes
./compile.sh



