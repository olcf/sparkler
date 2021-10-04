#include <unordered_map>
#include "gemmlib.h"
#include "level_zero/ze_api.h"
#include "CL/sycl/backend/level_zero.hpp"
#include "oneapi/mkl.hpp"

namespace GEMM
{

struct Context
{
    sycl::platform platform;
    sycl::device device;
    sycl::context context;
    sycl::queue queue;

    // Create default SYCL context and queue.
    // TODO Does this create a platform/context/device/queue for running SYCL
    // code on the host CPU cores?
    Context(void)
      : platform(),
        device(),
        context(),
        queue()
    {
        // Nothing else to do.
    }
};

Context*
Create(void)
{
    return new Context();
}

void
Destroy(Context* ctxt)
{
    delete ctxt;
}


inline
oneapi::mkl::transpose
ToMKLOp(Operation op)
{
    std::unordered_map<Operation, oneapi::mkl::transpose> map =
    {
        {GEMM::N, oneapi::mkl::transpose::N},
        {GEMM::T, oneapi::mkl::transpose::T},
        {GEMM::C, oneapi::mkl::transpose::C}
    };

    return map[op];
}


void
SetStream(Context* ctxt, unsigned long const* lzHandles, int nHandles)
{
    if(ctxt != nullptr)
    {
        // Obtain the handles to the LZ constructs.
        assert(nHandles == 4);
        auto hDriver = (ze_driver_handle_t)lzHandles[0];
        auto hDevice = (ze_device_handle_t)lzHandles[1];
        auto hContext = (ze_context_handle_t)lzHandles[2];
        auto hQueue = (ze_command_queue_handle_t)lzHandles[3];

        // Build SYCL platform/device/queue from the LZ handles.
        ctxt->platform = sycl::level_zero::make<sycl::platform>(hDriver);
        ctxt->device = sycl::level_zero::make<sycl::device>(ctxt->platform, hDevice);
        std::vector<sycl::device> devs;
        devs.push_back(ctxt->device);
        ctxt->context = sycl::level_zero::make<sycl::context>(devs, hContext);

        auto asyncExceptionHandler = [](sycl::exception_list exceptions) {

            // Report all asynchronous exceptions that occurred.
            for(std::exception_ptr const& e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch(std::exception& e)
                {
                    std::cerr << "Async exception: " << e.what() << std::endl;
                }
            }

            // Rethrow the first asynchronous exception.
            for(std::exception_ptr const& e : exceptions)
            {
                std::rethrow_exception(e);
            }
        };

        ctxt->queue = sycl::level_zero::make<sycl::queue>(ctxt->context,
                                                                    hQueue);
//                                                                    asyncExceptionHandler);
    }
}


void
SGEMM(Context* ctxt,
                Operation transa,
                Operation transb,
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
    if(ctxt != nullptr)
    {
        // Do the SGEMM via MKL.
        try
        {
            oneapi::mkl::blas::gemm(ctxt->queue,
                                    ToMKLOp(transa),
                                    ToMKLOp(transb),
                                    m,
                                    n,
                                    k,
                                    *alpha,
                                    A,
                                    ldA,
                                    B,
                                    ldB,
                                    *beta,
                                    C,
                                    ldC);
        }
        catch(sycl::exception const& e)
        {
            std::cerr << "SGEMM SYCL exception: " << e.what() << std::endl;
            throw;
        }
        catch(std::exception const& e)
        {
            std::cerr << "SGEMM exception: " << e.what() << std::endl;
            throw;
        }

        // Catch any asynchronous exceptions before continuing.
        ctxt->queue.wait_and_throw();
    }
}

} // namespace GEMM

