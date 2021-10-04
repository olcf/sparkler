#------------------------------------------------------------------------------

# Use MPI compiler driver.  We assume it is using the HIPLZ compiler
# as its underlying compiler.
CXX	= mpicxx
LD = OMPI_CXX=clang++-link mpicxx

CXXFLAGS = -g -O0 -DUSE_GPU -IHipBLAS-on-MKL/include
LDFLAGS = -g -O0 -LHipBLAS-on-MKL/lib

TARGET = sparkler.gpu
SRCS = main.cpp
OBJS = ${SRCS:.cpp=.o}
LIBS = -lhipblas_mkl -lhiplz -lOpenCL -lze_loader -lmpi


all: ${TARGET}

${TARGET}: ${OBJS}
	${LD} ${LDFLAGS} -o $@ ${OBJS} ${LIBS}

%.o: %.cpp
	$(CXX) ${CXXFLAGS} -c $^ 

clean:
	${RM} ${OBJS}

distclean:	clean
	${RM} ${TARGET}

#------------------------------------------------------------------------------
