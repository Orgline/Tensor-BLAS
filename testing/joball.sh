# ./job.sh testing_cumpsgemm_syrk 0 4000 0 4000 >cumpsgemm_syrk.out

# ./job.sh testing_cumpsgemm_syrk 0 4000 40000 8000 >> cumpsgemm_syrk.out

# ./job.sh testing_cumpsgemm_syrk 40000 4000 40000 4000 >> cumpsgemm_syrk.out

# ./job.sh testing_syrk 0 4000 0 4000 > syrk.out

# ./job.sh testing_syrk 0 4000 40000 8000 >> syrk.out

# ./job.sh testing_syrk 40000 4000 40000 4000 >> syrk.out

# ./job.sh testing_cumpsgemm_syr2k 0 4000 0 4000 >cumpsgemm_syr2k.out

# ./job.sh testing_cumpsgemm_syr2k 0 4000 40000 8000 >> cumpsgemm_syr2k.out

# ./job.sh testing_cumpsgemm_syr2k 40000 4000 40000 4000 >> cumpsgemm_syr2k.out

# ./job.sh testing_syr2k 0 4000 0 4000 > syr2k.out

# ./job.sh testing_syr2k 0 4000 40000 8000 >> syr2k.out

# ./job.sh testing_syr2k 40000 4000 40000 4000 >> syr2k.out

# ./job2.sh testing_cumpsgemm_trmm 0 4000 0 4000 >cumpsgemm_trmm.out

# ./job2.sh testing_cumpsgemm_trmm 0 4000 40000 8000 >> cumpsgemm_trmm.out

# ./job2.sh testing_cumpsgemm_trmm 40000 4000 40000 4000 >> cumpsgemm_trmm.out

# ./job2.sh testing_trmm 0 4000 0 4000 > trmm.out

# ./job2.sh testing_trmm 0 4000 40000 8000 >> trmm.out

# ./job2.sh testing_trmm 40000 4000 40000 4000 >> trmm.out

# ./job2.sh testing_trsm 0 4000 0 4000 > trsm.out

# ./job2.sh testing_trsm 0 4000 40000 8000 >> trsm.out

# ./job2.sh testing_trsm 40000 4000 40000 4000 >> trsm.out


./job.sh testing_cumpsgemm_syrk 1 4000 1 4000 >> cumpsgemm_syrk.out

./job.sh testing_cumpsgemm_syrk 1 4000 40001 8000 >> cumpsgemm_syrk.out

./job.sh testing_cumpsgemm_syrk 40001 4000 40001 4000 >> cumpsgemm_syrk.out

./job.sh testing_syrk 1 4000 1 4000 >> syrk.out

./job.sh testing_syrk 1 4000 40001 8000 >> syrk.out

./job.sh testing_syrk 40001 4000 40001 4000 >> syrk.out

./job.sh testing_cumpsgemm_syr2k 1 4000 1 4000 >> cumpsgemm_syr2k.out

./job.sh testing_cumpsgemm_syr2k 1 4000 40001 8000 >> cumpsgemm_syr2k.out

./job.sh testing_cumpsgemm_syr2k 40001 4000 40001 4000 >> cumpsgemm_syr2k.out

./job.sh testing_syr2k 1 4000 1 4000 >> syr2k.out

./job.sh testing_syr2k 1 4000 40001 8000 >> syr2k.out

./job.sh testing_syr2k 40001 4000 40001 4000 >> syr2k.out

./job2.sh testing_cumpsgemm_trmm 1 4000 1 4000 >> cumpsgemm_trmm.out

./job2.sh testing_cumpsgemm_trmm 1 4000 40001 8000 >> cumpsgemm_trmm.out

./job2.sh testing_cumpsgemm_trmm 40001 4000 40001 4000 >> cumpsgemm_trmm.out

./job2.sh testing_trmm 1 4000 1 4000 >> trmm.out

./job2.sh testing_trmm 1 4000 40001 8000 >> trmm.out

./job2.sh testing_trmm 40001 4000 40001 4000 >> trmm.out

./job2.sh testing_trsm 1 4000 1 4000 >> trsm.out

./job2.sh testing_trsm 1 4000 40001 8000 >> trsm.out

./job2.sh testing_trsm 40001 4000 40001 4000 >> trsm.out