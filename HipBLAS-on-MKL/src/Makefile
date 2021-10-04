
GEMMLIB_ROOT = ${PWD}/../GemmLib

# Build the stub library itself using the HIPLZ compiler.
CXX = clang++
LD = clang++-link

CXXFLAGS = -g -O0 -fPIC -I../include -I${GEMMLIB_ROOT}/include
LDFLAGS = -g -O0 -fPIC -shared -Wl,-rpath=${GEMMLIB_ROOT}/lib -L${GEMMLIB_ROOT}/lib

TARGET = libhipblas_mkl.so
SRCS = hipblas-stub.cpp
OBJS = ${SRCS:.cpp=.o}
LIBS = -lgemm_mkl


all: ${TARGET}

${TARGET}: ${OBJS}
	${LD} ${LDFLAGS} -o $@ ${OBJS} ${LIBS}

%.o: %.cpp
	$(CXX) ${CXXFLAGS} -c $^ 

install: ${TARGET}
	cp ${TARGET} ../lib/

clean:
	${RM} ${OBJS}

distclean:	clean
	${RM} ${TARGET}

#------------------------------------------------------------------------------
