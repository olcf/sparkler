
# We use the Hipblas-MKL library.
find_package(hipblas_mkl)

# The NVIDIA version of Sparkler uses the major and minor 
# compute capability values obtained via cudaGetDeviceProperties() 
# to determine if the GPU supports half precision.  HIP 
# has an analogous hipGetDeviceProperties() function but the
# compute capability values provided in the result are *not*
# comparable across different types of GPUs.  So using
# those numbers to identify half precision support is not
# correct.
# The HIP device properties structure does have an 'arch' member
# that indicates specific capabilities of the underlying GPU.
# Unfortunately, there is not a 'arch' flag indicating 
# half precision support.
# It *may* be possible to do an empirical test to determine
# half precision support, but for now we're going to let
# the user indicate whether to use floating point or
# mixed half-single precision GEMM.
option(USE_MIXED_PRECISION "Whether to use mixed half-single precision GEMMs" ON)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/SparklerConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/SparklerConfig.h)

# Specify how to build the executable.
# We only support building the GPU version.
add_executable(sparkler
    main.cpp)
target_include_directories(sparkler PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(sparkler PUBLIC USE_GPU)
target_link_libraries(sparkler
    PUBLIC hipblas_mkl)

install(TARGETS sparkler
    TYPE RUNTIME)

