#include "../include/TensorBLAS.h" 

void tc_cumpsgemm_syr2k_p2(cumpsgemm::handle_t cumpsgemm_handle, long int n, long int k, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				nb, nb, k,
				&alpha,
				A, lda, nb,
				B, ldb, nb,
				&beta,
				C, ldc, nb+nb*lda,
				n/nb,
				CUMPSGEMM_AUTO
				);
    cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				nb, nb, k,
				&alpha,
				B, ldb, nb,
				A, lda, nb,
				&sone,
				C, ldc, nb+nb*lda,
				n/nb,
				CUMPSGEMM_AUTO
				);

    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				i*nb, i*nb, k,
				&alpha,
				A+i*nb, lda, 2*i*nb,
				B, ldb, 2*i*nb,
				&beta,
				C+i*nb, ldc, 2*(i*nb+i*nb*lda),
				n/nb/i/2,
				CUMPSGEMM_AUTO
				);
        cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				i*nb, i*nb, k,
				&alpha,
				B+i*nb, ldb, 2*i*nb,
				A, lda, 2*i*nb,
				&sone,
				C+i*nb, ldc, 2*(i*nb+i*nb*lda),
				n/nb/i/2,
				CUMPSGEMM_AUTO
				);
    }
}

void tc_cumpsgemm_syr2k(cumpsgemm::handle_t cumpsgemm_handle, long int n, long int k,  float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    
    int length;
    int64_t* matSize = find_mat_size_syrk(n, &length);
    int offset;
    int rest_n = n;

    for(int i = length; i>=0; i--)
    {

        int nn = matSize[i];
        
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;

        if(nn % 8192 ==0 )
        {
            tc_cumpsgemm_syr2k_p2(cumpsgemm_handle, nn, k, alpha, A+offset, lda, B+offset, ldb, beta, C+offset+offset*ldc, ldc, nb);
        }
        else
        {
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    nn, nn, k,
                    &alpha,
                    A+offset, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset+offset*ldc, ldc,
                    CUMPSGEMM_AUTO
                    );
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    nn, nn, k,
                    &alpha,
                    B+offset, ldb,
                    A+offset, lda,
                    &sone,
                    C+offset+offset*ldc, ldc,
                    CUMPSGEMM_AUTO
                    );
        }
        if(i != 0)
        {
            rest_n -=  nn;
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    rest_n, nn, k,
                    &alpha,
                    A+offset+nn, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset+offset*ldc+nn, ldc,
                    CUMPSGEMM_AUTO
                    );
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    rest_n, nn, k,
                    &alpha,
                    B+offset+nn, ldb,
                    A+offset, lda,
                    &sone,
                    C+offset+offset*ldc+nn, ldc,
                    CUMPSGEMM_AUTO
                    );
        }
        else
            return;
        
    }
    return;
}