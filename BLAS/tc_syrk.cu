#include "../include/TensorBLAS.h" 
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row)* (ld) + (col) )
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

bool syrk_python_flag = false;
int tc_syrk_wrapper(long int n, long int k, float* A, float* C, long int nb)
{
    syrk_python_flag =true;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*n*k);

    float alpha = sone;
    float beta = szero;

    // float *tmp;
    // cudaMalloc(&tmp, sizeof(float)*n*k);
    // dim3 grida((k+31)/32, (n+31)/32);
    // dim3 blocka(32,32);
    // transpose<<<grida, blocka>>>(k, n ,A, tmp);
    // printMatrixDeviceBlock("A.csv",n, k, A, n);
    // cudaFree(tmp);

    tc_syrk(cublas_handle, n, k, alpha, A, n, beta, C, n, hwork, nb);
    dim3 gridc((n+31)/32, (n+31)/32);
    dim3 blockc(32,32);
    copy_lower_to_upper<<<gridc, blockc>>>(n, C, n);

    cudaFree(hwork);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

__global__ void myHGEMMAlignedV1(
    __half * __restrict__ a, __half * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    // printf("a[0] = %lf\n", (float)a[1]);
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    // __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_a[BM][BK];
    // __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    //wrap 64x32x64
    //256thread per block
    int load_a_smem_m = (tid >> 2) << 1; //(tid/4)* 2  warp row id  4 per row
    int load_a_smem_k = (tid &  3) << 3; //(tid%4)* 8  warp col id  one element offset is 8 
    // int load_b_smem_k = (tid >> 5) << 2; //(tid/32)*4  8 per row 
    int load_b_smem_k = (tid >> 5) << 3; //(tid/32)*8  4 per row (64) 
    int load_b_smem_n = (tid & 31) << 3; //tid%32*8        

    int load_a_gmem_m = by * BM + load_a_smem_m; //use block id to locate global row
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    // int load_a_gmem_addr = load_a_gmem_m + load_a_smem_k*N;
    // int load_b_gmem_addr = load_b_smem_k + load_b_smem_k*K;
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;
    // printf("tid:%d, sam:%d, sak:%d, sbm:%d, sbk:%d, load_a_gmem_addr%d, load_b_gmem_addr:%d\n", tid, load_a_smem_m, load_a_smem_k, load_b_smem_k, load_b_smem_n, load_a_gmem_addr, load_b_gmem_addr);
    for (int bk = 0; bk < K / BK; bk++) {
        FLOAT4(s_a[load_a_smem_m    ][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr        ]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr +     K]);
        
        // printf("%d %d %lf\n", load_a_gmem_m, load_a_gmem_addr, (float)a[load_a_gmem_addr        ]);
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
        
        __syncthreads();
        
        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK);
printf("%lf %lf %lf\n", (float)s_a[comp_c_frag_m * 64][ 16], (float)s_a[comp_c_frag_m * 64+16][ 16], (float)s_a[comp_c_frag_m * 64+32][ 16]);
        // wmma::load_matrix_sync(frag_b[0][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        // wmma::load_matrix_sync(frag_b[0][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        // wmma::load_matrix_sync(frag_b[0][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        // wmma::load_matrix_sync(frag_b[0][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);
        // wmma::load_matrix_sync(frag_b[1][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        // wmma::load_matrix_sync(frag_b[1][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        // wmma::load_matrix_sync(frag_b[1][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        // wmma::load_matrix_sync(frag_b[1][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_b[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK);
        wmma::load_matrix_sync(frag_b[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK);
        wmma::load_matrix_sync(frag_b[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK);
        wmma::load_matrix_sync(frag_b[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK);
        wmma::load_matrix_sync(frag_b[1][0], &s_a[comp_c_frag_m * 64     ][16], BK);
        wmma::load_matrix_sync(frag_b[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK);
        wmma::load_matrix_sync(frag_b[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK);
        wmma::load_matrix_sync(frag_b[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_col_major);
            // printf("%f ", c[store_c_gmem_addr + i * 16 * N + j * 16]);
        }
    }
}

void tc_syrk_p4(cublasHandle_t handle, long int n, long int k, float alpha, __half* Ah, long int lda, __half* Bh, long int ldb, float beta, float* C, long int ldc, long int nb)
{

        const int BM = 128, BN = 128;
        dim3 blockDim(256);
        int BX = (n + BN - 1) / BN;
        int BY = (n + BM - 1) / BM;
        dim3 gridDim(BX, BY);
        // printf("inside %d %d\n", n, k);
        printMatrixDeviceBlock("Ah.csv", n, k, Ah, lda);
        myHGEMMAlignedV1<<<gridDim, blockDim>>>(Ah, Bh, C, n, n, k);
        printMatrixDeviceBlock("Ah2.csv", n, k, Ah, lda);
        printMatrixDeviceBlock("C.csv", n, n, C, n);

                // printMatrixDeviceBlock("C.csv", N, N, C, N);
    // printf("finish my kernel\n");
}

void tc_syrk_p2(cublasHandle_t handle, long int n, long int k, float alpha, __half* Ah, long int lda, float beta, float* C, long int ldc, long int nb)
{

    for (int batch_id = 0; batch_id < n/nb; batch_id++)
        tc_syrk_p4(handle, nb, k, alpha, 
                Ah+nb*batch_id, lda, 
                Ah+nb*batch_id, lda, beta, 
                C+(nb+nb*lda)*batch_id, ldc, nb);
    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        for (int batch_id = 0; batch_id < n/nb/i/2; batch_id++)
            tc_syrk_p4(handle, i*nb, k, alpha, 
                    Ah+i*nb+2*i*nb*batch_id, lda, 
                    Ah+2*i*nb*batch_id, lda, beta, 
                    C+i*nb+(2*(i*nb+i*nb*lda))*batch_id, ldc, nb);
    }
}

void tc_syrk(cublasHandle_t handle, long int n, long int k,  float alpha, float* A, long int lda, float beta, float* C, long int ldc, __half* Ah, long int nb)
{
    
    int length;
    int64_t* matSize = find_mat_size_syrk(n, &length);
    int offset;
    int rest_n = n;

    if(!syrk_python_flag)
    {
        constexpr auto block_size = 256;
        constexpr auto smem_len = block_size * 16;
        const auto grid_size = k;
        s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_size, block_size>>>(
                    n, k,
                    A, lda,
                    Ah, lda
                    );
    }
    else
    {
        dim3 grid((k+31)/32, (n+31)/32);
        dim3 block(32,32);
        s2hTranspose<<<grid, block>>>(k, n, A, Ah);
    }

    // dim3 grida((k+31)/32, (n+31)/32);
    // dim3 blocka(32,32);
    // s2hTranspose<<<grida, blocka>>>(k, n, A, AhT);
    constexpr auto block_size = 256;
    constexpr auto smem_len = block_size * 16;
    const auto grid_size = k;
    s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_size, block_size>>>(
                n, k,
                A, lda,
                Ah, lda
                );
    printMatrixDeviceBlock("A.csv", n, k, A, lda);
    printMatrixDeviceBlock("Ah1.csv", n, k, Ah, lda);
    // printf("outside %d %d\n", n, k);
    for(int i = length; i>=0; i--)
    {

        int nn = matSize[i];
        

        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        //printf("n = %ld, offset = %d\n", nn ,offset);   

        if(nn % 8192 ==0 )
        {
            tc_syrk_p2(handle, nn, k, alpha, Ah+offset, lda, beta, C+offset+offset*ldc, ldc, nb);
        }
        else
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
                    &alpha, Ah+offset, CUDA_R_16F, lda, Ah+offset, CUDA_R_16F, lda,
                    &beta, C+offset+offset*ldc, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            printMatrixDeviceBlock("C2.csv", n, n, C, n);
            tc_syrk_p4(handle, nn, k, alpha, Ah+offset, lda, Ah+offset, lda, beta, C+offset+offset*ldc, ldc, nb);
        }
        // tc_syrk_p4(handle, nn, k, alpha, Ah+offset, lda, beta, C+offset+offset*ldc, ldc, nb);
        if(i != 0)
        {
            rest_n -=  nn;
            //printf("rest_n = %d, nn = %d, offset = %d\n", rest_n, nn, offset);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
                    &alpha, Ah+offset+nn, CUDA_R_16F, lda, Ah+offset, CUDA_R_16F, lda,
                    &beta, C+offset+offset*ldc+nn, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else
            return;
        
    }
    return;
}