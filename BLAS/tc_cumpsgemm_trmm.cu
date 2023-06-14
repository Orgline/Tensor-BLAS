#include "../include/TensorBLAS.h" 


void tc_cumpsgemm_trmm_p2(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    // cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    //                             nb, n, nb, &alpha,
    //                             Ah, CUDA_R_16F, lda, nb+nb*lda,
    //                             Bh, CUDA_R_16F, ldb, nb,
    //                             &beta, C, CUDA_R_32F, ldc, nb,
    //                             m/nb, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				nb, n, nb,
				&alpha,
				A, lda, nb+nb*lda,
				B, ldb, nb,
				&beta,
				C, ldc, nb,
				m/nb,
				CUMPSGEMM_AUTO
				);
    for(long int i = 1; m / nb / i / 2 >= 1; i*=2)
    {
        //beginTimer();
        // cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        //                            i*nb, n, i*nb, &alpha,
        //                            Ah+i*nb, CUDA_R_16F, lda, 2*(i*nb+i*nb*lda),
        //                            Bh, CUDA_R_16F, ldb, 2*i*nb,
        //                            &sone, C+i*nb, CUDA_R_32F, ldc, 2*i*nb,
        //                            m/nb/i/2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				i*nb, n, i*nb,
				&alpha,
				A+i*nb, lda, 2*(i*nb+i*nb*lda),
				B, ldb, 2*i*nb,
				&sone,
				C+i*nb, ldc, 2*i*nb,
				m/nb/i/2,
				CUMPSGEMM_AUTO
				);
    }
}

void tc_cumpsgemm_trmm(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, long int nb)
{
    int length;
    int64_t* matSize = find_mat_size_syrk(m, &length);
    int offset = 0;
    int rest_m = m;

    for(int i = length; i>=0; i--)
    {
        int mm = matSize[i];
        float beta;
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        if(i == length)
            beta = szero;
        else
            beta = sone;
        if (mm % 8192 == 0)
        {
            tc_cumpsgemm_trmm_p2(cumpsgemm_handle, mm, n, alpha, A+offset+offset*lda, lda, B+offset, ldb, beta, C+offset, ldc, nb);
        }
        else
        {
            //printf("offset = %d\n", offset);
            // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, mm,
            //     &alpha, Ah+offset+offset*lda, CUDA_R_16F, lda, Bh+offset, CUDA_R_16F, ldb,
            //     &beta, C+offset, CUDA_R_32F, ldc, CUDA_R_32F,
            //     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    mm, n, mm,
                    &alpha,
                    A+offset+offset*lda, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset, ldc,
                    CUMPSGEMM_AUTO
                    );
        }
        if(i != 0)
        {
            rest_m -= mm;         
            // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rest_m, n, mm,
            //     &alpha, Ah+offset+mm+offset*lda, CUDA_R_16F, lda, Bh+offset, CUDA_R_16F, ldb,
            //     &beta, C+offset+mm, CUDA_R_32F, ldc, CUDA_R_32F,
            //     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    rest_m, n, mm,
                    &alpha,
                    A+offset+mm+offset*lda, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset+mm, ldc,
                    CUMPSGEMM_AUTO
                    );
        }
    }
}