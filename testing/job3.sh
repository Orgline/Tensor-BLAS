#!/bin/bash  
nn=512
kk=512
for num in {1..6}  
do  
echo ./$1 $nn $kk 256 1 0 0
./$1 $nn $kk 256 1 0 0
let "nn=nn*2"
let "kk=kk*2"
done 
