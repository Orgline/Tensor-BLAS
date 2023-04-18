CXX = nvcc
CFLAGS=-O2 -arch sm_80 -I $(CUDA_PATH)/..\include/
LFLAGS=-L $(CUDA_PATH)/lib64 -lcublas -lcublasLt -lcurand -lcudart -lcuda
CC = gcc

all: env syrk

util.o: ../util/util.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

mat_size.o: ../util/mat_size.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

env: testing_env

testing_env: testing_env.o util.o
	$(CXX) $(LFLAGS) $^ -o $@

testing_env.o: testing_env.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

syrk: testing_syrk

testing_syrk: testing_syrk.o tc_syrk.o util.o mat_size.o
	$(CXX) $(LFLAGS) $^ -o $@

testing_syrk.o: testing_syrk.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

tc_syrk.o: ../BLAS/tc_syrk.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

clean:
	rm -f *.o testing_env testing_syrk
