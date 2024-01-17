#!/bin/bash  
nn=16


for num in {1..12}  
do  
let "nn=nn*2"
# echo ./figure1 $nn $nn $nn
# ./figure1 $nn $nn $nn
# echo ./figure1 65536 65536 $nn
# ./figure1 65536 65536 $nn
echo ./figure1 32768 32768 $nn
./figure1 32768 32768 $nn
# echo ./figure1 $nn $nn 32768
# ./figure1 $nn $nn 32768


done 
