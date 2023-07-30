#include "../include/TensorBLAS.h"



int64_t* find_mat_size_syrk(int64_t n, int *length) 
{
    int64_t powers[] = {8192, 16384, 16384, 16384, 16384, 16384, 16384,16384, 16384, 16384, 16384, 16384};

    int64_t* result = (int64_t*) malloc((sizeof(powers) / sizeof(int64_t) + 1) * sizeof(int64_t)+1);
    int result_index = 0;

    for (int i = sizeof(powers) / sizeof(int64_t)-1 ; i >= -1; i--) 
    {
        if (n < 8192) {
            if(n == 0)
                break;
            result[result_index++] = n;
            break;
        }
        if (n >= powers[i]) {
            result[result_index++] = powers[i];
            n -= powers[i];
        }
    }

    result[result_index] = -1;
    *length = result_index-1;
    return result;
}

int64_t* find_mat_size_trsm(int64_t n, int *length) 
{
    
    int64_t powers[] = {2048, 4096, 8192, 16384,  16384, 16384, 16384,16384, 16384, 16384, 16384, 16384};
    int64_t* result = (int64_t*) malloc((sizeof(powers) / sizeof(int64_t) + 1) * sizeof(int64_t)+1);
    int result_index = 0;

    for (int i = sizeof(powers) / sizeof(int64_t)-1 ; i >= -1; i--) 
    {
        if (n < 2048) {
            if(n == 0)
                break;
            result[result_index++] = n;
            break;
        }
        if (n >= powers[i]) {
            result[result_index++] = powers[i];
            n -= powers[i];
        }
    }

    result[result_index] = -1;
    *length = result_index-1;
    return result;
}