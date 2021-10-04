#include <iostream>
#include <string>
#include <tuple>
#include <cstring>  // for memset
#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "boost/program_options.hpp"

namespace bpo = boost::program_options;

template<typename ECodeType>
class HipstarException : public std::runtime_error
{
private:
    ECodeType code;

public:
    HipstarException(ECodeType _code, const char* _msg)
      : std::runtime_error(_msg),
        code(_code)
    { }

    HipstarException(ECodeType _code, const std::string& _msg)
      : std::runtime_error(_msg),
        code(_code)
    { }

    ECodeType GetCode(void) const { return code; }
};

using HipException = HipstarException<hipError_t>;
using HipblasException = HipstarException<hipblasStatus_t>;



inline
void
CHECK(hipError_t code)
{
    if(code != hipSuccess)
    {
        throw HipException(code, "HIP call failed");
    }
}

inline
void
CHECK(hipblasStatus_t code)
{
    if(code != HIPBLAS_STATUS_SUCCESS)
    {
        throw HipblasException(code, "hipBLAS call failed");
    }
}


class HipStream
{
private:
    hipStream_t handle;

public:
    HipStream(void)
    {
        hipStreamCreate(&handle);        
    }

    ~HipStream(void)
    {
        // std::cerr << "In ~HipStream" << std::endl;
        // TODO this causes an exception to be thrown, that
        // seems not to be caught.  The HIPLZ implementation
        // of this seems to be expecting to run on top of OpenCL - 
        // not yet ported to hiplz?
        // Re-enable this once know for sure (and if fixed).
        // CHECK(hipStreamDestroy(handle));
    }

    hipStream_t GetHandle(void) const   { return handle; }

    void Synchronize(void) const  { CHECK(hipStreamSynchronize(handle)); }
};


class HipblasContext
{
private:
    hipblasHandle_t handle;

public:
    HipblasContext(HipStream& stream)
    {
        CHECK(hipblasCreate(&handle));
        CHECK(hipblasSetStream(handle, stream.GetHandle()));
    }

    ~HipblasContext(void)
    {
        // std::cerr << "In ~HipblasContext" << std::endl;
        CHECK(hipblasDestroy(handle));
    }

    hipblasHandle_t GetHandle(void) const   { return handle; }
};



// A Matrix in CPU and GPU memory.
// The matrix elements are stored in column major order
// to be easier to pass to SGEMM.
template<typename T>
class Matrix
{
private:
    int nRows;
    int nCols;

    T* hostData;
    T* devData;

public:
    Matrix(int _nRows, int _nCols)
      : nRows(_nRows),
        nCols(_nCols),
        hostData(nullptr),
        devData(nullptr)
    {
        CHECK(hipHostMalloc(&hostData, size()));
        memset(hostData, 0, size());
        CHECK(hipMalloc(&devData, size()));
        CHECK(hipMemset(devData, 0, size()));
    }

    ~Matrix(void)
    {
        if(hostData != nullptr)
        {
            CHECK(hipHostFree(hostData));
            hostData = nullptr;
        }
        if(devData != nullptr)
        {
            CHECK(hipFree(devData));
            devData = nullptr;
        }
    }

    int nItems(void) const  { return nRows * nCols; }
    int size(void) const    { return nItems() * sizeof(T); }

    T* GetDeviceData(void) const  { return devData; }
    T* GetHostData(void) const { return hostData; }

    // Access element from host storage.
    T& El(int r, int c)
    {
        return hostData[c*nRows + r];
    }

    const T& El(int r, int c) const
    {
        return hostData[c*nRows + r];
    }

    void CopyHostToDevice(void)
    {
        CHECK(hipMemcpy(devData, hostData, size(), hipMemcpyHostToDevice));
    }

    void CopyHostToDeviceAsync(const HipStream& stream)
    {
        CHECK(hipMemcpyAsync(devData, hostData, size(), hipMemcpyHostToDevice, stream.GetHandle()));
    }

    void CopyDeviceToHost(void)
    {
        CHECK(hipMemcpy(hostData, devData, size(), hipMemcpyDeviceToHost));
    }

    void CopyDeviceToHostAsync(const HipStream& stream)
    {
        CHECK(hipMemcpyAsync(hostData, devData, size(), hipMemcpyDeviceToHost, stream.GetHandle()));
    }
};


std::tuple<bool, int, int, int, int, float, float>
ParseCommandLine(int argc, char* argv[])
{
    int ret = 0;
    bool shouldRun = true;

    bpo::options_description desc("SGEMM using hipBLAS over HIPLZ.\nSupported options");
    desc.add_options()
        ("help,h", "show this help message")
        ("nRowsA,m", bpo::value<int>()->default_value(8), "Number of rows in A")
        ("nColsA,k", bpo::value<int>()->default_value(4), "Number of columns in A")
        ("nColsC,n", bpo::value<int>()->default_value(12), "Number of columns in C")
        ("alpha,a", bpo::value<float>()->default_value(0.5), "Scale for A*B")
        ("beta,b", bpo::value<float>()->default_value(0.25), "Scale for C input")
    ;

    bpo::variables_map opts;
    bpo::store(bpo::parse_command_line(argc, argv, desc), opts);
    bpo::notify(opts);

    if(opts.count("help") > 0)
    {
        std::cout << desc << std::endl;
        shouldRun = false;
    }

    auto m = opts["nRowsA"].as<int>();
    auto k = opts["nColsA"].as<int>();
    auto n = opts["nColsC"].as<int>();

    if( (m <= 0) or (k <= 0) or (n <= 0) )
    {
        std::cerr << "m, n, and k must each be >=1" << std::endl;
        shouldRun = false;
        ret = 1;
    }

    auto alpha = opts["alpha"].as<float>();
    auto beta = opts["beta"].as<float>();

    return std::make_tuple(shouldRun, ret, m, k, n, alpha, beta);
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
        float alpha;
        float beta;

        // Parse the command line.
        std::tie(shouldRun,
                    ret,
                    m,
                    k,
                    n,
                    alpha,
                    beta) = ParseCommandLine(argc, argv);

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

            // We also have some scalars to define.
            float alpha = 0.5;
            float beta = 0.25;

            // Create the input matrices with known values.
            // Current test is:
            // * Items in col 0 of A are all 1.  Otherwise 0.
            // * Items in logical row 0 of B are all 1.  Otherwise 0.
            // * Storage for B in memory is transposed.
            // * C[r, c] = r*c.
            // After the SGEMM, C[r,c] should be alpha + beta * r * c
            Matrix<float> A(m, k);
            for(auto r = 0; r < m; ++r)
            {
                A.El(r, 0) = 1;
            }
            A.CopyHostToDeviceAsync(hipStream);

            Matrix<float> B(n, k);
            for(auto c = 0; c < n; ++c)
            {
                B.El(c, 0) = 1;
            }
            B.CopyHostToDeviceAsync(hipStream);

            Matrix<float> C(m, n);
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

