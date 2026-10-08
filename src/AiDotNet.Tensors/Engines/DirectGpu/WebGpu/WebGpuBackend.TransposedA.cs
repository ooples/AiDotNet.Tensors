#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Row-major C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C: the WebGPU port of CudaBackend.MatMulTransposedA.
public sealed partial class WebGpuBackend : ITransposedAGemm
{
    private const string GemmTransposedASource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
struct GemmParams { M: u32, N: u32, K: u32, alpha: f32, beta: f32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(3) var<uniform> params: GemmParams;

@compute @workgroup_size(256)
fn gemm_transposed_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.M * params.N) { return; }
    let row = idx / params.N;
    let col = idx % params.N;
    var acc: f32 = 0.0;
    for (var kk: u32 = 0u; kk < params.K; kk = kk + 1u) {
        acc = acc + A[kk * params.M + row] * B[kk * params.N + col];
    }
    if (params.beta != 0.0) {
        C[idx] = params.alpha * acc + params.beta * C[idx];
    } else {
        C[idx] = params.alpha * acc;
    }
}
";

    public void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        TransposedAGemmArgs.Validate(A, B, C, M, N, K);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(N),
            BitConverter.Int32BitsToSingle(K),
            alpha,
            beta, 0, 0, 0,
        };
        Dispatch3BufferAsync("GemmTransposedA", GemmTransposedASource, "gemm_transposed_a", A, B, C, uniforms, checked(M * N))
            .GetAwaiter().GetResult();
    }
}
#endif