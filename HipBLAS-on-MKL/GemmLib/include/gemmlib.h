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

enum Datatype
{
    Real8I = 0,
    Real32I = 1,
    Real16F = 2,
    Real32F = 3
};

enum GemmAlgorithm
{
    Default = 0    
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

void GEMMEx(Context* context,
                Operation transa,
                Operation transb,
                int         m,
                int         n,
                int         k,
                const void* alpha,
                const void* a,
                Datatype    a_type,
                int         lda,
                const void* b,
                Datatype    b_type,
                int         ldb,
                const void* beta,
                void*       c,
                Datatype    c_type,
                int         ldc,
                Datatype    compute_type,
                GemmAlgorithm alg);

} // namespace GEMM

#endif // GEMMLIB_H
