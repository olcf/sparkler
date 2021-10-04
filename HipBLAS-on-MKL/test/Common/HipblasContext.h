#ifndef TEST_HIPBLAS_CONTEXT_H
#define TEST_HIPBLAS_CONTEXT_H

#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "HipstarException.h"

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

#endif // TEST_HIPBLAS_CONTEXT_H
