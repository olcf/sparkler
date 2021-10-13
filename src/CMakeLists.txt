
# We use the Hipblas-MKL library.
find_package(hipblas_mkl)

# We support a few options for the type of 16-bit floating point values.
# If this option is ON, we will use __half.
# If OFF, we will use _Float16.
option(SPARKLER_USE_HALF_TYPE "Whether to use __half for 16-bit floating point values" ON)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/SparklerConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/SparklerConfig.h)

# Specify how to build the executable.
# We only support building the GPU version.
add_executable(sparkler
    main.cpp)
target_compile_definitions(sparkler PUBLIC USE_GPU)
target_include_directories(sparkler PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(sparkler
    PUBLIC hipblas_mkl)

install(TARGETS sparkler
    TYPE RUNTIME)

