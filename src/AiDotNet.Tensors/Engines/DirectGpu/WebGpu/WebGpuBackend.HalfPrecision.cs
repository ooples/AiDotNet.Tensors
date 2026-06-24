// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the WebGPU backend (issue #560):
// FP16 GEMM so MatrixMultiply<Half> runs on the GPU instead of dropping to a
// scalar CPU fallback.


// WebGPU is only available on .NET 7+ (Blazor WebAssembly / Dawn); the rest of
// the backend is compiled out on net471, so this partial must be too.
#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Half-precision GEMM dispatch for the <see cref="IGpuHalfPrecisionBackend"/>
/// capability interface, so the backend-agnostic
/// <c>DirectGpuTensorEngine.MatrixMultiply</c> FP16 path (issue #560) runs on
/// WebGPU devices instead of falling back to the scalar CPU loop.
/// <para>
/// WebGPU has no native 16-bit storage by default — the optional
/// <c>shader-f16</c> feature is not used here, and all WebGPU buffers are f32.
/// This backend therefore models FP16 as f32 with a truncated 10-bit mantissa
/// (see <see cref="ConvertToFp16"/>). Under that model an FP16 GEMM is exactly
/// the existing real WGSL f32 GEMM run on the truncated-precision inputs: the
/// multiply sees FP16-precision operands, the accumulation is full FP32, and the
/// f32 output flows on to downstream ops — which is precisely the
/// <see cref="GemmFp16In32fOut"/> (mixed-precision) contract. No separate FP16
/// kernel is needed because the data is already f32.
/// </para>
/// </summary>
public sealed partial class WebGpuBackend : IGpuHalfPrecisionBackend
{
    /// <inheritdoc/>
    /// <remarks>
    /// True whenever the backend is initialized: the real WGSL GEMM that backs
    /// the FP16 path is always present (WebGPU has no CPU-fallback GEMM), so the
    /// only gate is device availability.
    /// </remarks>
    public bool SupportsHgemm => IsAvailable;

    /// <inheritdoc/>
    /// <remarks>The fused backward routes through the real WGSL transposed GEMMs (always present on WebGPU,
    /// which has no CPU-fallback GEMM), so it tracks device availability like <see cref="SupportsHgemm"/>.</remarks>
    public bool SupportsFp16FusedBackward => IsAvailable;

    /// <inheritdoc/>
    /// <remarks>
    /// WebGPU models FP16 as f32 (truncated mantissa — see class remarks), so the inputs already carry FP16
    /// precision and the backward is two real f32 transposed GEMMs with no materialized transpose:
    /// gradA[M,K] = gradC·Bᵀ via the rhs-transposed kernel (<see cref="MatMulTransposed"/>) and gradB[K,N] = Aᵀ·gradC
    /// via the lhs-transposed kernel (<see cref="MatMulLhsTransposedAsync"/>). FP16-output grads run into an f32
    /// scratch then <c>ConvertToFp16</c> (mirrors <see cref="Hgemm"/>); the accumulate is f32 throughout.
    /// </remarks>
    public void MatMulBackwardFp16Fused(
        IGpuBuffer gradCFp16, IGpuBuffer aFp16, IGpuBuffer bFp16,
        IGpuBuffer gradAOut, IGpuBuffer gradBOut,
        int m, int n, int k, bool gradOutHalf)
    {
        if (gradCFp16 is null) throw new ArgumentNullException(nameof(gradCFp16));
        if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
        if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
        if (gradAOut is null) throw new ArgumentNullException(nameof(gradAOut));
        if (gradBOut is null) throw new ArgumentNullException(nameof(gradBOut));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

        // gradA[M,K] = gradC[M,N] · Bᵀ  (b stored [K,N] → read transposed as [N,K]).
        DispatchBackward(scratchElems: m * k, gradOutHalf, gradAOut, dest =>
            MatMulTransposed(gradCFp16, bFp16, dest, m, k, n));
        // gradB[K,N] = Aᵀ · gradC  (a stored [M,K] → contraction over leading dim M).
        DispatchBackward(scratchElems: k * n, gradOutHalf, gradBOut, dest =>
            MatMulLhsTransposedAsync(aFp16, gradCFp16, dest, k, n, m).GetAwaiter().GetResult());
    }

    /// <summary>Runs one backward GEMM into <paramref name="gradOut"/> (FP32) or, when
    /// <paramref name="gradOutHalf"/>, into an f32 scratch then packs to FP16 via <c>ConvertToFp16</c>.</summary>
    private void DispatchBackward(int scratchElems, bool gradOutHalf, IGpuBuffer gradOut, Action<IGpuBuffer> gemm)
    {
        if (!gradOutHalf)
        {
            gemm(gradOut);
            return;
        }
        using var scratch = AllocateBuffer(scratchElems);
        gemm(scratch);
        ConvertToFp16(scratch, gradOut, scratchElems);
    }

    /// <inheritdoc/>
    public void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp32, m, n, k);
        // Inputs already carry FP16 precision (f32 storage, truncated mantissa);
        // the WGSL GEMM accumulates in f32 and writes an f32 result.
        Gemm(aFp16, bFp16, cFp32, m, n, k);
    }

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);
        // Accumulate in f32 into a scratch buffer, then round the result down to
        // FP16 precision so the output buffer carries half-precision values
        // (matches cublasHgemm's FP16 output). A separate scratch avoids binding
        // the same buffer as both GEMM output and convert input.
        using var scratch = AllocateBuffer(m * n);
        Gemm(aFp16, bFp16, scratch, m, n, k);
        ConvertToFp16(scratch, cFp16, m * n);
    }

    private static void ValidateHalfGemmArgs(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (c is null) throw new ArgumentNullException(nameof(c));
        // Report the offending dimension by name (mirrors the CUDA backend).
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");
    }
    // #1650/#638 FP16 conv im2col port (WebGpu): fused im2col + FP32→FP16 in the TRANSPOSED [K,N] layout
    // so conv becomes out[outC,N] = weights[outC,K] · col[K,N] via GemmFp16In32fOut (the industry path).
    // WebGPU models FP16 as f32 with a truncated 10-bit mantissa (no native 16-bit storage / shader-f16
    // feature; see class remarks), so outputHalf is an ordinary f32 buffer of K*N elements whose values are
    // the same truncated-f32 that ConvertToFp16 produces — exactly what GemmFp16In32fOut consumes. The WGSL
    // kernel's fused f16_trunc helper is a byte-for-byte copy of fp16_convert's rounding, so the result is
    // identical to a separate gather + ConvertToFp16.

    private readonly object _im2colFp16Lock = new();
    private bool _im2colFp16Tried;
    private bool _im2colFp16Available;

    /// <inheritdoc/>
    /// <remarks>
    /// Gated on the <c>im2col_kn_fp16hw</c> WGSL shader actually compiling and the compute pipeline being
    /// created (lazily, on first query, cached). Unlike <see cref="SupportsHgemm"/> (which only needs device
    /// availability because the FP16 GEMM reuses the always-present f32 GEMM), this kernel is its own shader:
    /// gating on a successful pipeline build means a device/driver that rejects it degrades gracefully to the
    /// FP32 conv instead of throwing. The FP16 model is truncated-f32 (no <c>shader-f16</c> feature), so this
    /// is expected to compile on every WebGPU device.
    /// </remarks>
    public bool Fp16Im2colAvailable => EnsureIm2colFp16Pipeline();

    /// <summary>Lazily builds and caches the <c>im2col_kn_fp16hw</c> pipeline. Non-fatal on failure
    /// (returns false ⇒ the engine keeps the FP32 conv on WebGpu); a broken/unsupported shader never
    /// crashes the conv path.</summary>
    private bool EnsureIm2colFp16Pipeline()
    {
        if (_im2colFp16Tried)
            return _im2colFp16Available;

        lock (_im2colFp16Lock)
        {
            if (_im2colFp16Tried)
                return _im2colFp16Available;

            if (!IsAvailable)
            {
                // #671 review: the backend isn't initialized yet — do NOT latch this miss (_im2colFp16Tried
                // stays false) so a later query retries the pipeline build once IsAvailable becomes true.
                return false;
            }

            _im2colFp16Tried = true;

            try
            {
                // Build (and cache in _pipelineCache) the compute pipeline. Throws if the WGSL shader
                // fails to compile or the pipeline cannot be created on this device.
                GetOrCreatePipelineAsync("Im2colKnFp16", WebGpuKernels.Im2colKnFp16Source, "im2col_kn_fp16hw")
                    .GetAwaiter().GetResult();
                _im2colFp16Available = true;
            }
            catch (Exception)
            {
                // Non-fatal: without the kernel the FP16 conv stays on the FP32 path.
                _im2colFp16Available = false;
            }

            return _im2colFp16Available;
        }
    }

    /// <inheritdoc/>
    public void Im2colKNFp16(IGpuBuffer input, IGpuBuffer outputHalf,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (outputHalf is null) throw new ArgumentNullException(nameof(outputHalf));
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "Dimensions must be positive.");
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "Dimensions must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Dimensions must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Dimensions must be positive.");
        if (kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelH), "Kernel size must be positive.");
        if (kernelW <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW), "Kernel size must be positive.");
        if (strideH <= 0) throw new ArgumentOutOfRangeException(nameof(strideH), "Stride must be positive.");
        if (strideW <= 0) throw new ArgumentOutOfRangeException(nameof(strideW), "Stride must be positive.");
        if (padH < 0) throw new ArgumentOutOfRangeException(nameof(padH), "Padding must be non-negative.");
        if (padW < 0) throw new ArgumentOutOfRangeException(nameof(padW), "Padding must be non-negative.");
        if (dilationH <= 0) throw new ArgumentOutOfRangeException(nameof(dilationH), "Dilation must be positive.");
        if (dilationW <= 0) throw new ArgumentOutOfRangeException(nameof(dilationW), "Dilation must be positive.");

        // HOST computes the output spatial dims (same formula on every backend) and passes them in.
        int outH = (height + 2 * padH - ((kernelH - 1) * dilationH + 1)) / strideH + 1;
        int outW = (width + 2 * padW - ((kernelW - 1) * dilationW + 1)) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new ArgumentException(
                $"Computed output dims are non-positive (outH={outH}, outW={outW}); kernel/stride/pad/dilation " +
                "exceed the input size.");

        long n = (long)batch * outH * outW;      // N = batch*outH*outW
        long k = (long)channels * kernelH * kernelW; // K = channels*kernelH*kernelW
        long total = n * k;                       // one invocation per col element
        if (total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch),
                $"im2col element count N*K = {total} exceeds Int32.MaxValue.");
        if (total <= 0) return;

        // Im2colParams uniform: 14 u32 fields packed as a flat float[] (same convention as ConvParams /
        // MakeConvUniforms — all-scalar u32 members are tightly 4-byte packed in a WGSL uniform struct).
        // Padded to 16 floats (64 bytes) so the allocated uniform buffer is at least the 16-byte-rounded
        // struct size (mirrors MakeDeformConvUniforms padding).
        var uniforms = new float[16];
        uniforms[0] = BitConverter.Int32BitsToSingle(batch);
        uniforms[1] = BitConverter.Int32BitsToSingle(channels);
        uniforms[2] = BitConverter.Int32BitsToSingle(height);
        uniforms[3] = BitConverter.Int32BitsToSingle(width);
        uniforms[4] = BitConverter.Int32BitsToSingle(kernelH);
        uniforms[5] = BitConverter.Int32BitsToSingle(kernelW);
        uniforms[6] = BitConverter.Int32BitsToSingle(strideH);
        uniforms[7] = BitConverter.Int32BitsToSingle(strideW);
        uniforms[8] = BitConverter.Int32BitsToSingle(padH);
        uniforms[9] = BitConverter.Int32BitsToSingle(padW);
        uniforms[10] = BitConverter.Int32BitsToSingle(dilationH);
        uniforms[11] = BitConverter.Int32BitsToSingle(dilationW);
        uniforms[12] = BitConverter.Int32BitsToSingle(outH);
        uniforms[13] = BitConverter.Int32BitsToSingle(outW);
        // uniforms[14], [15] = 0 (padding to 16 floats / 64 bytes)

        // One thread per col element (t in [0, N*K)); workgroup_size(256), dispatch = ceil(N*K/256).
        // outputHalf is the truncated-f32 half buffer GemmFp16In32fOut consumes.
        Dispatch2BufferAsync("Im2colKnFp16", WebGpuKernels.Im2colKnFp16Source, "im2col_kn_fp16hw",
            input, outputHalf, uniforms, (int)total).GetAwaiter().GetResult();
    }

}

#endif
