#!/bin/ksh

i=3

cat configs/Treebank_test_G5.ini | head -n7 > configs/test5.ini
echo "weights_file	=	savedWeights/G5_iG4/Treebank.weights.$i" >> configs/test5.ini
cat configs/Treebank_test_G5.ini | head -n21 | tail -n13 >> configs/test5.ini
echo "OUT_PROB	=	1" >> configs/test5.ini
cat configs/Treebank_test_G5.ini | tail -n35 >> configs/test5.ini
./spn_train configs/test5.ini > Interpolation/SPN.prob
cd Interpolation
java Mixture
cd ..

