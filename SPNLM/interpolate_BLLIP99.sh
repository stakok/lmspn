#!/bin/ksh

i=40

cat configs/BLLIP99_test.ini | head -n8 > configs/test5.ini
echo "weights_file	=	savedWeights/BLLIP99/BLLIP99.weights.$i" >> configs/test5.ini
cat configs/BLLIP99_test.ini | head -n21 | tail -n12 >> configs/test5.ini
echo "OUT_PROB	=	1" >> configs/test5.ini
cat configs/BLLIP99_test.ini | tail -n35 >> configs/test5.ini

cp codes/large/* codes
./compile.sh

./spn_train configs/test5.ini > Interpolation/SPN.prob
cd Interpolation
java Mixture
cd ..

rm configs/test5.ini
cp codes/normal/* codes
./compile.sh
