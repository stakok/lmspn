#!/bin/bash
t1File=configs/test4.ini
t2File=configs/test5.ini
prototypeFile=configs/Treebank_test_G4.ini
weightFile=savedWeights2/4Gram/Treebank.weights

for (( i=1; i<=40; i++ ))
do
  echo $i
  head -n 7 $prototypeFile > $t1File
  echo "weights_file      =       $weightFile.$i"  >> $t1File
  tail -n 48 $prototypeFile >> $t1File

  ./spn_train $t1File > ppl_G4.txt
  ./printPPL.sh ppl_G4.txt

  rm $t1File
  rm ppl_G4.txt

  cat $prototypeFile | head -n7 > $t2File
  echo "weights_file      =       $weightFile.$i" >> $t2File
  cat $prototypeFile | head -n21 | tail -n13 >> $t2File
  echo "OUT_PROB  =       1" >> $t2File
  cat $prototypeFile | tail -n35 >> $t2File
  ./spn_train $t2File > Interpolation/SPN.prob
  cd Interpolation
  java Mixture
  cd ..
done
