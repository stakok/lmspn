#!/bin/ksh

i=40

head -n 7 configs/Treebank_test_G4.ini > configs/test4.ini
echo "weights_file	=	savedWeights/4Gram/Treebank.weights."$i  >> configs/test4.ini
tail -n 48 configs/Treebank_test_G4.ini >> configs/test4.ini

./spn_train configs/test4.ini > ppl_G4.txt
./printPPL.sh ppl_G4.txt

rm configs/test4.ini
rm ppl_G4.txt




