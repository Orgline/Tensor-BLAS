#!/bin/bash
#YBATCH -r a100_1
#SBATCH -N 1
#SBATCH -o /home/szhang94/TensorBLAS/testing/syrk%j.out
#SBATCH --time=10:00:00
#SBATCH -J tc_syrk
#SBATCH --error /home/szhang94/TensorBLAS/testing/syrk_b%j.err

n_values=(24576 32768 40960 49152 57344 65536)
k_values=(16384 24576 32768 40960 49152 57344 65536 73728 81920 90112 98304 106496 114688 122880 131072)

for n in "${n_values[@]}"; do
	for k in "${k_values[@]}"; do
		echo "Processing with n=$n and k=$k"
		./testing_syrk "$n" "$k" 256 1 0 0 
	done
done
