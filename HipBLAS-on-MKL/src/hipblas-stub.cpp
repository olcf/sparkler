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


inline
GEMM::Datatype
ToGEMMType(hipblasDatatype_t dt)
{
    std::unordered_map<hipblasDatatype_t, GEMM::Datatype> map =
    {
        {HIPBLAS_R_8I, GEMM::Real8I},
        {HIPBLAS_R_32I, GEMM::Real32I},
        {HIPBLAS_R_16F, GEMM::Real16F},
        {HIPBLAS_R_32F, GEMM::Real32F}
    };

    return map[dt];
}


inline
GEMM::GemmAlgorithm
ToGEMMAlg(hipblasGemmAlgo_t alg)
{
    std::unordered_map<hipblasGemmAlgo_t, GEMM::GemmAlgorithm> map =
    {
        {HIPBLAS_GEMM_DEFAULT, GEMM::Default}
    };

    return map[alg];
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
                     const void*        A,
                     hipblasDatatype_t  AType,
                     int                ldA,
                     const void*        B,
                     hipblasDatatype_t  BType,
                     int                ldB,
                     const void*        beta,
                     void*              C,
                     hipblasDatatype_t  CType,
                     int                ldC,
                     hipblasDatatype_t  ComputeType,
                     hipblasGemmAlgo_t  algo)
{
    hipblasStatus_t ret = HIPBLAS_STATUS_SUCCESS;
    if(handle != nullptr)
    {
        GEMM::Context* ctxt = static_cast<GEMM::Context*>(handle);

        try
        {
            GEMM::GEMMEx(ctxt,
                            ToGEMMOp(transa),
                            ToGEMMOp(transb),
                            m,
                            n,
                            k,
                            alpha,
                            A,
                            ToGEMMType(AType),
                            ldA,
                            B,
                            ToGEMMType(BType),
                            ldB,
                            beta,
                            C,
                            ToGEMMType(CType),
                            ldC,
                            ToGEMMType(ComputeType),
                            ToGEMMAlg(algo));
        }
        catch(std::exception const& e)
        {
            std::cerr << "GemmEx exception: " << e.what() << std::endl;
            ret = HIPBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    else
    {
        ret = HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    return ret;
}


#if READY
    hipblasStatus_t ret = HIPBLAS_STATUS_SUCCESS;
    if(handle != nullptr)
    {
        // Verify we were given the configuration we support.
        if( (a_type == HIPBLAS_R_16F) and
            (b_type == HIPBLAS_R_16F) and
            (c_type == HIPBLAS_R_32F) and
            (compute_type == HIPBLAS_R_32F) and
            (algo == HIPBLAS_GEMM_DEFAULT) )
        {
            // We were given the configuration we support.
            // Do the GEMM using our Gemm library.
        }
        else
        {
            ret = invalid type;
        }
    }
    else
    {
        ret = HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    return ret;
}

#endif // READY


#if RREADY

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

#endif //  RREADY
