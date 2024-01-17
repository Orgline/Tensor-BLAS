#!/bin/bash  
nn=$2
ldn=$3
kk=$4
ldk=$5

for num in {1..10}  
do  
let "nn=nn+ldn"
let "kk=kk+ldk"
echo ./$1 $nn $kk 256 1 0 0
./$1 $nn $kk 256 1 0 0
# echo ./$1 $nn $kk 256 1 1
# ./$1 $nn $kk 256 1 1

done 
