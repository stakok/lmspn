#!/bin/ksh

file=$1
# while loop
while read line
do
        # display line or do somthing on $line
	if [[ "$line" =~ '^[MSG] The perplexity of training dataset is' ]] then
		train_ppl=`echo $line | cut -f8 -d' '`
		#echo "$train_ppl"
	fi

	if [[ "$line" =~ '^[MSG] The perplexity of validation dataset is' ]] then
		valid_ppl=`echo $line | cut -f8 -d' '`
		#echo "$valid_ppl"
	fi

	if [[ "$line" =~ '^[MSG] The perplexity of testing dataset is' ]] then
		test_ppl=`echo $line | cut -f8 -d' '`
		#echo "$test_ppl"
	fi

done <"$file"

echo -e "\t\ttrain set\tvalidation set\ttesting set"
echo -e "\t\t$train_ppl\t\t$valid_ppl\t\t$test_ppl"
