#ifndef TEST_HIPSTREAM_H
#define TEST_HIPSTREAM_H

#include "hip/hip_runtime.h"
#include "HipstarException.h"

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

#endif // TEST_HIPSTREAM_H
