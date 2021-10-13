
# We use the Hipblas-MKL library.
find_package(hipblas_mkl)

# Specify how to build the executable.
# We only support building the GPU version.
add_executable(sparkler
    main.cpp)
target_compile_definitions(sparkler PUBLIC USE_GPU)
target_link_libraries(sparkler
    PUBLIC hipblas_mkl)

install(TARGETS sparkler
    TYPE RUNTIME)

