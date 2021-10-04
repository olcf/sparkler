#ifndef HIPSTAR_EXCEPTION_H
#define HIPSTAR_EXCEPTION_H

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

#endif // HIPSTAR_EXCEPTION_H
