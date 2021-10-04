
# This clang++ is assumed to be the Intel DPC++ compiler.
CXX = clang++

CXXFLAGS = -fsycl -fPIC -g -O0 -DMKL_ILP64 -I../include
#LDFLAGS = -fsycl -fPIC -g -O0 -shared -Wl,-rpath=${MKLROOT}/lib/intel64 -Wl,-rpath=${CMPLR_ROOT}/linux/lib
LDFLAGS = -fsycl -fPIC -g -O0 -shared 

TARGET = libgemm_mkl.so
SRCS = gemm_mkl.cpp
OBJS = ${SRCS:.cpp=.o}
LIBS = -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

all: ${TARGET}

${TARGET}: ${OBJS}
	${CXX} -o $@ ${LDFLAGS} ${OBJS} ${LIBS}

%.o: %.cpp
	${CXX} -c ${CXXFLAGS} $^

install: all
	cp ${TARGET} ../lib/

clean:
	${RM} ${OBJS}

distclean: clean
	${RM} ${TARGET}
