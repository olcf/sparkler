# Overview

This branch contains a port of Sparkler to HIPLZ, 
a HIP implementation that uses Intel's Level Zero runtime
to run on Intel GPUs.  Like the original HIP version, this version uses
hipBLAS to provide its SGEMM and GemmEx functionality.  Because
the stock hipBLAS implementation does not support Intel GPUs, 
as a proof-of-concept we rely on the HipBLAS-MKL project
(https://code.ornl.gov/6rp/hipblas-mkl) that implements
only as much of the hipBLAS interface as is needed by Sparkler,
and uses Intel's oneMKL library to make use of Intel GPUs.

The version in this branch does not support running Sparkler
on the CPU only.

# Prerequisites

To configure, build, and run this version requires:
* An installation of HIPLZ.
* An installation of the HipBLAS-MKL library.
* CMake version 3.20 or newer.
* An MPI implementation like OpenMPI.

# Configure

Ensure that your environment is updated to use HIPLZ, CMake, MKL, and OpenMPI.
For instance, on the Argonne National Laboratory JLSE Iris systems, do 
something like this:

```bash
$ module load intel_compute_runtime
$ module load hiplz
$ module use /home/ac.jyoung/gpfs_share/compilers/modulefiles/oneapi/2021.3.0
$ module load mkl
$ module load cmake
$ module load openmpi
$ module unload -f cuda llvm
```

Because it can be challenging to use a custom linker with CMake that has
to be invoked as a separate step, use the HipBLAS-MKL project's `hiplz-clang++`
wrapper that determines if one is compiling or linking, and invokes the
appropriate HIPLZ script.  Make sure `hiplz-clang++` is in your `PATH`.
Assuming HipBLAS-MKL is installed at `${HBM_ROOT}`:

```bash
$ cd ${HBM_ROOT}/bin
$ export PATH=$PWD:$PATH
$ cd -
```

We will be using the OpenMPI compiler driver to build the software, but we need
to make sure that `mpicxx` uses the HIPLZ compiler wrapper as an underlying compiler.

```bash
$ export OMPI_CXX=hiplz-clang++
```

Then, in whichever build directory you choose, and with the HIPLZ brancnh of the Sparkler
sources checked out at `${SPARKLER_SRC_DIR}` and installation directory `${SPARKLER_INST_DIR}`,
configure the build with something like:

```bash
$ CXX=mpicxx \
cmake \
-DCMAKE_INSTALL_PREFIX=${SPARKLER_INST_DIR} \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_PREFIX_PATH=${HBM_ROOT}/lib/cmake \
-DSPARKLER_USE_HALF_TYPE=ON \
${SPARKLER_SRC_DIR}
```

The `SPARKLER_USE_HALF_TYPE` option indicates whether to use the `__half` type for
16-bit floating point variables, or the `_Float16` type.  Set it `ON` to use `__half`.

# Build and Install
Once configured, from the build directory:
```bash
$ make
$ make install
```

# Run a Test
Once built and installed into `${SPARKLER_INST_DIR}`, run a test with something like:
```bash
$ mpirun -np 2 ${SPARKLER_INST_DIR}/bin/sparkler --num_vector 4000 --num_field 90000 --num_iterations 1
Using mixed precision
num_vector 4000 num_field 90000 num_iterations 1 num_proc 2
Using mixed precision
Iteration 1 of 1, step 1 of 2, elapsed sec 1.451: setup... GEMM... check...
Iteration 1 of 1, step 2 of 2, elapsed sed 8.886: setup... GEMM... check...
TF 8.640 GEMM sec 9.995 GEMM TF/sec 0.864 total sec 11.806 hash 1089999826774532
```

