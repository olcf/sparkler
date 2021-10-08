#include <iostream>
#include <iomanip>
#include <string>
#include <tuple>
#include <cstring>  // for memset
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "hiplz_h2f.h"
#include "hipblas.h"
#include "HipStream.h"
#include "HipblasContext.h"
#include "HipstarException.h"
#include "Matrix.h"
#include "CommandLine.h"

// Types for the scalars, A, B, and C matrix elements.
using InType = _Float16;
hipblasDatatype_t hipblasInType = HIPBLAS_R_16F;

using OutType = float;
hipblasDatatype_t hipblasOutType = HIPBLAS_R_32F;


inline
std::ostream&
operator<<(std::ostream& os, _Float16 f16)
{
    os << (float)f16;
    return os;
}


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
        OutType alpha;
        OutType beta;

        // Parse the command line.
        std::tie(shouldRun,
                    ret,
                    m,
                    k,
                    n,
                    alpha,
                    beta) = ParseCommandLine<OutType>(argc, argv);

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
            int ldB = n;    // B is transposed.
            int ldC = m;

            // Create the input matrices with known values.
            // Current test is:
            // * Items in col 0 of A are all 1.  Otherwise 0.
            // * Items in logical row 0 of B are all 1.  Otherwise 0.
            // * Storage for B in memory is transposed.
            // * C[r, c] = r*c.
            // After the SGEMM, C[r,c] should be alpha + beta * r * c
            Matrix<InType> A(m, k);
            for(auto r = 0; r < m; ++r)
            {
                A.El(r, 0) = 1;
            }
            A.CopyHostToDeviceAsync(hipStream);

            // B is tranposed.
            Matrix<InType> B(n, k);
            for(auto c = 0; c < n; ++c)
            {
                B.El(c, 0) = 1;
            }
            B.CopyHostToDeviceAsync(hipStream);

            Matrix<OutType> C(m, n);
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

#if READY
#else
            std::cout << "alpha: " << alpha 
                    << "beta: " << beta
                    << std::endl;
            // Dump input matrices.
            std::vector<InType> hostA(A.nItems());
            auto Asize = A.size();
            CHECK(hipMemcpy(hostA.data(), A.GetDeviceData(), Asize, hipMemcpyDeviceToHost));
            std::cout << "A.nItems: " << A.nItems()
                << ", A.size: " << Asize
                << ", vals: ";
            for(auto i = 0; i < A.nItems(); ++i)
            {
                std::cout << hostA[i] << ' ';
            }
            std::cout << std::endl;

            std::vector<InType> hostB(B.nItems());
            auto Bsize = B.size();
            CHECK(hipMemcpy(hostB.data(), B.GetDeviceData(), Bsize, hipMemcpyDeviceToHost));
            std::cout << "B.nItems: " << B.nItems()
                << ", B.size: " << Bsize
                << ", vals: ";
            for(auto i = 0; i < B.nItems(); ++i)
            {
                std::cout << hostB[i] << ' ';
            }
            std::cout << std::endl;

            std::vector<OutType> hostC(C.nItems());
            auto Csize = C.size();
            CHECK(hipMemcpy(hostC.data(), C.GetDeviceData(), Csize, hipMemcpyDeviceToHost));
            std::cout << "C.nItems: " << C.nItems()
                << ", C.size: " << Csize
                << ", vals: ";
            for(auto i = 0; i < C.nItems(); ++i)
            {
                std::cout << hostC[i] << ' ';
            }
            std::cout << std::endl;

#endif // READY

            // Do the GEMM on the GPU.
            CHECK(hipblasGemmEx(hipblasContext.GetHandle(),
                                HIPBLAS_OP_N,
                                HIPBLAS_OP_T,
                                m,
                                n,
                                k,
                                &alpha,
                                A.GetDeviceData(),
                                hipblasInType,
                                ldA, 
                                B.GetDeviceData(),
                                hipblasInType,
                                ldB, 
                                &beta,
                                C.GetDeviceData(),
                                hipblasOutType,
                                ldC,
                                hipblasOutType,
                                HIPBLAS_GEMM_DEFAULT)); 
            hipStream.Synchronize();    // necessary?

            // Read C from device to host.
            C.CopyDeviceToHostAsync(hipStream);
            hipStream.Synchronize();

#if READY
#else
            {
                std::vector<OutType> hostC(C.nItems());
                auto Csize = C.size();
                CHECK(hipMemcpy(hostC.data(), C.GetDeviceData(), Csize, hipMemcpyDeviceToHost));
                std::cout << "C.nItems: " << C.nItems()
                    << ", C.size: " << Csize
                    << ", vals: ";
                for(auto i = 0; i < C.nItems(); ++i)
                {
                    std::cout << hostC[i] << ' ';
                }
                std::cout << std::endl;
            }
#endif // READY

            // Verify the GPU-computed results match the expected results.
            // Assumes column major ordering.
            uint32_t nMismatches = 0;
            for(auto c = 0; c < n; c++)
            {
                for(auto r = 0; r < m; r++)
                {
                    auto expVal = (alpha * 1 + beta * r * c);
                    if(C.El(r, c) != expVal)
                    {
                        ++nMismatches;
                        std::cout << "mismatch at: (" << r << ", " << c << ")"
                            << " expected " << expVal
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

