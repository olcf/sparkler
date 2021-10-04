#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "gemmlib.h"


inline
GEMM::Operation
ToGEMMOp(hipblasOperation_t hop)
{
    std::unordered_map<hipblasOperation_t, GEMM::Operation> map =
    {
        {HIPBLAS_OP_N, GEMM::N},
        {HIPBLAS_OP_T, GEMM::T},
        {HIPBLAS_OP_C, GEMM::C}
    };

    return map[hop];
}


hipblasStatus_t
hipblasCreate(hipblasHandle_t* handle)
{
    if(handle != nullptr)
    {
        *handle = GEMM::Create();
    }
    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}


hipblasStatus_t
hipblasDestroy(hipblasHandle_t handle)
{
    if(handle != nullptr)
    {
        GEMM::Context* ctxt = static_cast<GEMM::Context*>(handle);
        GEMM::Destroy(ctxt);
    }

    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}


hipblasStatus_t
hipblasSetStream(hipblasHandle_t handle, hipStream_t stream)
{
    if(handle != nullptr)
    {
        GEMM::Context* ctxt = static_cast<GEMM::Context*>(handle);

        // Obtain the handles to the LZ handlers.
        unsigned long lzHandles[4];
        int nHandles = 0;
        hiplzStreamNativeInfo(stream, lzHandles, &nHandles);

        GEMM::SetStream(ctxt, lzHandles, nHandles);
    }

    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}


hipblasStatus_t
hipblasSgemm(hipblasHandle_t    handle,
                hipblasOperation_t transa,
                hipblasOperation_t transb,
                int                m,
                int                n,
                int                k,
                const float*       alpha,
                const float*       A,
                int                ldA,
                const float*       B,
                int                ldB,
                const float*       beta,
                float*             C,
                int                ldC)
{
    hipblasStatus_t ret = HIPBLAS_STATUS_SUCCESS;
    if(handle != nullptr)
    {
        GEMM::Context* ctxt = static_cast<GEMM::Context*>(handle);

        try
        {
            GEMM::SGEMM(ctxt,
                            ToGEMMOp(transa),
                            ToGEMMOp(transb),
                            m,
                            n,
                            k,
                            alpha,
                            A,
                            ldA,
                            B,
                            ldB,
                            beta,
                            C,
                            ldC);
        }
        catch(std::exception const& e)
        {
            std::cerr << "SGEMM exception: " << e.what() << std::endl;
            ret = HIPBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    else
    {
        ret = HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    return ret;
}



hipblasStatus_t
hipblasGemmEx(hipblasHandle_t    handle,
                     hipblasOperation_t transa,
                     hipblasOperation_t transb,
                     int                m,
                     int                n,
                     int                k,
                     const void*        alpha,
                     const void*        a,
                     hipblasDatatype_t  a_type,
                     int                lda,
                     const void*        b,
                     hipblasDatatype_t  b_type,
                     int                ldb,
                     const void*        beta,
                     void*              c,
                     hipblasDatatype_t  c_type,
                     int                ldc,
                     hipblasDatatype_t  compute_type,
                     hipblasGemmAlgo_t  algo)
{
    hipblasStatus_t ret = HIPBLAS_STATUS_SUCCESS;
    if(handle != nullptr)
    {
#if READY
        GEMM::Context* ctxt = static_cast<GEMM::Context*>(handle);

        GEMM::GEMMEx(ctxt,
                        ToGEMMOp(transa),
                        ToGEMMOp(transb),
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        ldA,
                        B,
                        ldB,
                        beta,
                        C,
                        ldC);
        catch(std::exception const& e)
        {
            std::cerr << "SGEMM exception: " << e.what() << std::endl;
            ret = HIPBLAS_STATUS_EXECUTION_FAILED;
        }
#else
        ret = HIPBLAS_STATUS_EXECUTION_FAILED;
#endif // READY
    }
    else
    {
        ret = HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    return ret;
}

