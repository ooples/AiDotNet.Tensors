// Copyright (c) CrowdTrainer LLC. All rights reserved.
// FP16-NATIVE elementwise / activation ops for the WebGPU backend (Tensors #558): GELU / ReLU /
// residual add. The halves are packed binary16 (WebGpuBackend.PackedHalf): the element-wise ops read and write them
// in one pass, computing in f32; softmax and layernorm widen their operands, run the f32 row kernels and narrow back.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    /// <summary>
    /// True whenever the backend is initialized: the real WGSL ops that back the FP16-native path are
    /// always present (WebGPU has no CPU-fallback compute), so the only gate is device availability.
    /// </summary>
    public bool SupportsFp16NativeOps => IsAvailable;

    /// <summary>GELU over an FP16 buffer (truncated-f32): out = round16(gelu(in)).</summary>
    public void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int n)
    {
        ValidateUnary(input, output, n);
        HalfElementwise(HalfElementwiseOp.Gelu, input, input, output, n);
    }

    /// <summary>ReLU over an FP16 buffer (truncated-f32): out = round16(max(in, 0)).</summary>
    public void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int n)
    {
        ValidateUnary(input, output, n);
        HalfElementwise(HalfElementwiseOp.Relu, input, input, output, n);
    }

    /// <summary>Residual add over FP16 buffers (truncated-f32): out = round16(a + b).</summary>
    public void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
        HalfElementwise(HalfElementwiseOp.Add, a, b, output, n);
    }

    /// <summary>Row softmax over the last axis (truncated-f32 FP16 model): runs the real f32 WGSL softmax
    /// on the FP16-precision input and re-truncates the result. WebGPU counterpart of the CUDA
    /// fp16_softmax_native.</summary>
    public void Fp16Softmax(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
        int count = checked(rows * cols);
        using var wide = AllocateBuffer(count);
        using var scratch = AllocateBuffer(count);
        ConvertToFp32(input, wide, count);
        SoftmaxAsync(wide, scratch, rows, cols).GetAwaiter().GetResult();
        ConvertToFp16(scratch, output, count);
    }

    /// <summary>Row layernorm over the last axis with FP16 gamma/beta (truncated-f32 model): real f32 WGSL
    /// kernel that also emits per-row FP32 mean/variance, output re-truncated to FP16. WebGPU counterpart of
    /// the CUDA fp16_layernorm_native.</summary>
    public void Fp16LayerNorm(IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
        IGpuBuffer meanFp32, IGpuBuffer varFp32, int rows, int cols, float eps)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
        if (eps <= 0f || float.IsNaN(eps) || float.IsInfinity(eps))
            throw new ArgumentOutOfRangeException(nameof(eps), eps, "eps must be finite and positive.");

        int count = checked(rows * cols);
        using var wideInput = AllocateBuffer(count);
        using var wideGamma = AllocateBuffer(cols);
        using var wideBeta = AllocateBuffer(cols);
        ConvertToFp32(input, wideInput, count);
        ConvertToFp32(gamma, wideGamma, cols);
        ConvertToFp32(beta, wideBeta, cols);
        using var scratch = AllocateBuffer(count);
        IGpuBuffer? tmpMean = null, tmpVar = null;
        var meanBuf = meanFp32 ?? (tmpMean = AllocateBuffer(rows));
        var varBuf = varFp32 ?? (tmpVar = AllocateBuffer(rows));
        try
        {
            var prm = new float[] { BitConverter.Int32BitsToSingle(rows), BitConverter.Int32BitsToSingle(cols), eps, 0 };
            Dispatch6BufferAsync("Fp16LayerNorm", WebGpuKernels.Fp16LayerNormSource, "fp16_layernorm",
                wideInput, wideGamma, wideBeta, scratch, meanBuf, varBuf, prm, rows * 256).GetAwaiter().GetResult();
            ConvertToFp16(scratch, output, count);
        }
        finally
        {
            tmpMean?.Dispose();
            tmpVar?.Dispose();
        }
    }

    private static void ValidateUnary(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
    }
}

#endif
