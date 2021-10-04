#ifndef GEMMLIB_H
#define GEMMLIB_H

namespace GEMM
{

struct Context;

enum Operation
{
    N = 0,
    T = 1,
    C = 2
};

Context* Create(void);

void Destroy(Context* context);

void SetStream(Context* context, unsigned long const* handles, int nHandles);

void SGEMM(Context* context,
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
                int                ldC);

} // namespace GEMM

#endif // GEMMLIB_H
