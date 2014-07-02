#!/bin/ksh

i=40

head -n 7 configs/Treebank_test_G5.ini > configs/test5.ini
echo "weights_file	=	savedWeights/5Gram/Treebank.weights."$i  >> configs/test5.ini
tail -n 48 configs/Treebank_test_G5.ini >> configs/test5.ini

./spn_train configs/test5.ini > ppl_G5.txt
./printPPL.sh ppl_G5.txt

rm configs/test5.ini
rm ppl_G5.txt




