NVCC = nvcc

NVCCFLAGS = -O2 -arch sm_80 -Xcompiler '-fPIC' -I $(CUDA_PATH)/..\include/ -I\../cuMpSGEMM/include/

LDFLAGS = -shared

LIBS = -L$(CUDA_PATH)/lib64 -L\../cuMpSGEMM/build/ -lcublas -lcublasLt -lcurand -lcudart -lcuda -lcumpsgemm

SOURCES = $(wildcard ../BLAS/*.cu) $(wildcard ../util/*.cu)

OBJECTS = $(SOURCES:.cu=.o)

LIBNAME = TensorBLAS.so

all: $(LIBNAME)

$(LIBNAME): $(OBJECTS)
	$(NVCC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(LIBNAME)
