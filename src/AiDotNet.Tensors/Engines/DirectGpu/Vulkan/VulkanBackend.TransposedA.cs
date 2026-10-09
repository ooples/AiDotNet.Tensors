namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Row-major C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C: the Vulkan port of CudaBackend.MatMulTransposedA.
public sealed partial class VulkanBackend : ITransposedAGemm
{
    private const string GemmFp32TransposedAGlsl = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Ab { float A[]; };
layout(set=0,binding=1) readonly buffer Bb { float B[]; };
layout(set=0,binding=2) buffer Cb { float C[]; };
layout(push_constant) uniform PC { uint M; uint N; uint K; float alpha; float beta; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= M * N) return;
    uint row = gid / N;
    uint col = gid % N;
    float acc = 0.0;
    for (uint kk = 0u; kk < K; ++kk)
        acc += A[kk * M + row] * B[kk * N + col];
    C[gid] = (beta != 0.0) ? alpha * acc + beta * C[gid] : alpha * acc;
}";

    public void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        TransposedAGemmArgs.Validate(A, B, C, M, N, K);
        var push = new uint[]
        {
            (uint)M, (uint)N, (uint)K,
            unchecked((uint)SingleToInt32BitsCompat(alpha)),
            unchecked((uint)SingleToInt32BitsCompat(beta)),
        };
        GlslBinaryOp(GemmFp32TransposedAGlsl, A, B, C, checked(M * N), push, (uint)(push.Length * sizeof(uint)));
    }
}