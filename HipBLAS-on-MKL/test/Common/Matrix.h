#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include "hip/hip_runtime.h"
#include "HipstarException.h"


// A Matrix in CPU and GPU memory.
// The matrix elements are stored in column major order
// to be easier to pass to traditional BLAS library
// implementations that were originally designed for
// Fortran applications.
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

#endif // TEST_MATRIX_H
