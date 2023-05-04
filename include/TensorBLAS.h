#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cusolverDn.h>


__global__
void s2h(long int m, long int n, float *as, long int ldas, __half *ah, long int ldah);

__global__
void clearTri(char uplo, long int m, long int n, float *a, long int lda);

__global__
void setEyePlus( long int m, long int n, float *a, long int lda);

__global__
void setEye( long int m, long int n, float *a, long int lda);

__global__
void setInitialValue( long int m, long int n, float *a, long int lda, float val);

__global__ 
void copy_lower_to_upper(long int n, float *A, long int lda);

void beginTimer();

float endTimer();

void startTimer();

float stopTimer();

void generateNormalMatrix(float *dA,long int m,long int n);

void generateUniformMatrix(float *dA,long int m, long int n);


float snorm(long int m, long int n, float *d_A, long int lda);

size_t free_mem();


// float snorm(long int m, long int n, float *d_A);

int64_t* find_mat_size_syrk(int64_t n, int *length);

int64_t* find_mat_size_trsm(int64_t n, int *length);


template<typename T>
void printMatrixDeviceBlock(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

void print_env();

void tc_syrk(cublasHandle_t handle, long int n,long int k,  float alpha, float* A, long int lda, float beta, float* C, long int ldc, __half* Ah, long int nb);

void tc_trsm(cublasHandle_t handle, long int m, long int n, float* A, long int lda, float* B, long int ldb, __half* hwork, long int nb);

void tc_trmm(cublasHandle_t handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, __half* hwork, long int nb);

void tc_syr2k(cublasHandle_t handle, long int n, long int k, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, __half* Ah, long int nb);

const float sone = 1.0;
const float snegone = -1.0;
const float szero = 0.0;