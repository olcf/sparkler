#include <iostream>
#include <string>
#include <tuple>
#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "CommandLine.h"
#include "HipStream.h"
#include "HipblasContext.h"
#include "Matrix.h"
#include "CommandLine.h"

// This is an SGEMM test.
// The data type has to be 'float'.
using DataType = float;


int
main(int argc, char* argv[])
{
    int ret = 0;

    try
    {
        bool shouldRun = true;

        // Variables that specify matrix sizes.
        // A: m x k
        // B: k x n
        // C: m x n
        int m = -1;
        int k = -1;
        int n = -1;

        // Scaling factors for A*B and for C as input.
        DataType alpha;
        DataType beta;

        // Parse the command line.
        std::tie(shouldRun,
                    ret,
                    m,
                    k,
                    n,
                    alpha,
                    beta) = ParseCommandLine<DataType>(argc, argv);

        if(shouldRun)
        {
            // Initialize HIP and hipBLAS.
            // The hipBLAS context needs to be in an inner scope so its
            // destructor is called before the hipStream destructor.
            HipStream hipStream;
            HipblasContext hipblasContext(hipStream);

            // We also need to provide the 'leading dimension' for the matrices.
            // We assume hipblasSgemm and MKL underneath expect column major order.
            // So leading dimension would be number of rows?
            int ldA = m;
            int ldB = k;
            int ldC = m;

            // Create the input matrices with known values.
            // Current test is:
            // * Items in col 0 of A are all 1.  Otherwise 0.
            // * Items in logical row 0 of B are all 1.  Otherwise 0.
            // * Storage for B in memory is transposed.
            // * C[r, c] = r*c.
            // After the SGEMM, C[r,c] should be alpha + beta * r * c
            Matrix<DataType> A(m, k);
            for(auto r = 0; r < m; ++r)
            {
                A.El(r, 0) = 1;
            }
            A.CopyHostToDeviceAsync(hipStream);

            Matrix<DataType> B(n, k);
            for(auto c = 0; c < n; ++c)
            {
                B.El(c, 0) = 1;
            }
            B.CopyHostToDeviceAsync(hipStream);

            Matrix<DataType> C(m, n);
            for(auto c = 0; c < n; ++c)
            {
                for(auto r = 0; r < m; ++r)
                {
                    C.El(r, c) = r*c;
                }
            }
            C.CopyHostToDeviceAsync(hipStream);

            // Wait for matrices to be copied to GPU.
            hipStream.Synchronize();

            // Do the GEMM on the GPU.
            CHECK(hipblasSgemm(hipblasContext.GetHandle(),
                                HIPBLAS_OP_N,
                                HIPBLAS_OP_T,
                                m,
                                n,
                                k,
                                &alpha,
                                A.GetDeviceData(),
                                ldA, 
                                B.GetDeviceData(),
                                ldB, 
                                &beta,
                                C.GetDeviceData(),
                                ldC)); 
            hipStream.Synchronize();    // necessary?

            // Read C from device to host.
            C.CopyDeviceToHostAsync(hipStream);
            hipStream.Synchronize();

            // Verify the GPU-computed results match the expected results.
            // Assumes column major ordering.
            uint32_t nMismatches = 0;
            for(auto c = 0; c < n; c++)
            {
                for(auto r = 0; r < m; r++)
                {
                    if(C.El(r, c) != (alpha + beta * r * c))
                    {
                        ++nMismatches;
                        std::cout << "mismatch at: (" << r << ", " << c << ")"
                            << " expected " << 1
                            << ", got " << C.El(r,c)
                            << std::endl;
                    }
                }
            }
            std::cout << "Total mismatches: " << nMismatches << std::endl;
        }
    }
    catch(const HipException& e)
    {
        std::cerr << "In HipException catch block" << std::endl;
        std::cerr << "HIP Exception: " << e.GetCode() << ": " << e.what() << std::endl;
        ret = 1;
    }
    catch(const HipblasException& e)
    {
        std::cerr << "hipBLAS Exception: " << e.GetCode() << ": " << e.what() << std::endl;
        ret = 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
        ret = 1;
    }
    catch(...)
    {
        std::cerr << "unrecognized exception caught" << std::endl;
        ret = 1;
    }

    return ret;
}

