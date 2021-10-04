#------------------------------------------------------------------------------

# We assume these are the HIPLZ compiler.
CXX	= clang++
LD = clang++-link

CXXFLAGS = -g -O0 -I../../include -I${BOOST_ROOT}/include
LDFLAGS = -g -O0 -L../../lib -L${BOOST_ROOT}/lib

TARGET = sgemm_test
SRCS = main.cpp
OBJS = main.o
LIBS = -lboost_program_options -lhipblas_mkl -lhiplz -lOpenCL -lze_loader


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
