#------------------------------------------------------------------------------

USE_GPU = YES

ifeq ($(USE_GPU),YES)

  CUDA_ROOT = /usr/local/cuda-10.1
  CUDA_TARGET = x86_64-linux
  MPI_ROOT = $(shell spack location --install-dir openmpi 2>/dev/null)

  CC = nvcc
  CFLAGS = -I$(MPI_ROOT)/include -DUSE_GPU -std=c++11
  SRC = main.cu
  OBJ = main.o

  LD = $(MPI_ROOT)/bin/mpiCC
  LDFLAGS = -L$(CUDA_ROOT)/targets/$(CUDA_TARGET)/lib \
            -Wl,-rpath=$(CUDA_ROOT)/targets/$(CUDA_TARGET)/lib \
            -Wl,-rpath=$(CUDA_ROOT)/lib64 \
            -lcublas -lcudart
  EXEC = exec.gpu

else

  MPI_ROOT = $(shell spack location --install-dir openmpi 2>/dev/null)
  OPENBLAS_ROOT = $(shell spack location --install-dir openblas 2>/dev/null)

  CC = $(MPI_ROOT)/bin/mpiCC
  CFLAGS = -std=c++11 -I$(OPENBLAS_ROOT)/include -include cblas.h
  SRC = main.cc
  OBJ = main.o

  LD = $(MPI_ROOT)/bin/mpiCC

  LDFLAGS = -L$(OPENBLAS_ROOT)/lib -Wl,-rpath,$(OPENBLAS_ROOT)/lib -lopenblas
  EXEC = exec.cpu

endif

all:
	$(CC) -o $(OBJ) -c $(SRC) $(CFLAGS)
	$(LD) -o $(EXEC) $(OBJ) $(LDFLAGS)

distclean:
	rm -f exec.gpu exec.cpu $(OBJ)

#------------------------------------------------------------------------------
