CXX = nvcc
CFLAGS=-O2 -arch sm_80 -I $(CUDA_PATH)/..\include/
LFLAGS=-L $(CUDA_PATH)/lib64 -lcublas -lcublasLt -lcurand -lcudart -lcuda
NVCCFLAGS=-I/home/z52527/cuMpSGEMM/include -L/home/z52527/cuMpSGEMM/build -lcumpsgemm
CC = gcc

# all: env syrk trsm trmm syr2k gemm cumpsgemm_syrk
all: cumpsgemm_syrk
util.o: ../util/util.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)

mat_size.o: ../util/mat_size.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)

env: testing_env

testing_env: testing_env.o util.o
	$(CXX) $(LFLAGS) $^ -o $@ $(NVCCFLAGS)
 
testing_env.o: testing_env.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)

syrk: testing_syrk

testing_syrk: testing_syrk.o tc_syrk.o util.o mat_size.o
	$(CXX) $(LFLAGS) $^ -o $@ $(NVCCFLAGS)

testing_syrk.o: testing_syrk.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)

tc_syrk.o: ../BLAS/tc_syrk.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)

cumpsgemm_syrk: testing_cumpsgemm_syrk

testing_cumpsgemm_syrk: testing_cumpsgemm_syrk.o tc_cumpsgemm_syrk.o util.o mat_size.o
	$(CXX) $(LFLAGS) $^ -o $@ $(NVCCFLAGS)

testing_cumpsgemm_syrk.o: testing_cumpsgemm_syrk.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)

tc_cumpsgemm_syrk.o: ../BLAS/tc_cumpsgemm_syrk.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@ $(NVCCFLAGS)


trsm: testing_trsm

testing_trsm: testing_trsm.o tc_trsm.o util.o mat_size.o
	$(CXX) $(LFLAGS) $^ -o $@

testing_trsm.o: testing_trsm.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

tc_trsm.o: ../BLAS/tc_trsm.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

trmm: testing_trmm

testing_trmm: testing_trmm.o tc_trmm.o util.o mat_size.o
	$(CXX) $(LFLAGS) $^ -o $@

testing_trmm.o: testing_trmm.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

tc_trmm.o: ../BLAS/tc_trmm.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

syr2k: testing_syr2k

testing_syr2k: testing_syr2k.o tc_syr2k.o util.o mat_size.o
	$(CXX) $(LFLAGS) $^ -o $@

testing_syr2k.o: testing_syr2k.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

tc_syr2k.o: ../BLAS/tc_syr2k.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

gemm: testing_gemm

testing_gemm: testing_gemm.o util.o
	$(CXX) $(LFLAGS) $^ -o $@

testing_gemm.o: testing_gemm.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

clean:
	rm -f *.o testing_env testing_syrk  testing_trsm testing_trmm testing_syr2k testing_gemm
