// Copyright (c) AiDotNet. All rights reserved.
// FP16-NATIVE elementwise / activation ops for the WebGPU backend (Tensors #558): GELU / ReLU /
// residual add. WebGPU has no native 16-bit storage by default (the optional shader-f16 feature is not
// used), so this backend models FP16 as f32 with a truncated 10-bit mantissa (see ConvertToFp16). Under
// that model an FP16-native op is exactly the existing real WGSL f32 op run on the FP16-precision inputs,
// with the result re-truncated to FP16 precision — same contract the FP16 GEMM (Hgemm) uses. No separate
// FP16 kernel is needed because the data is already f32.

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
        using var scratch = AllocateBuffer(n);
        GeLUAsync(input, scratch, n).GetAwaiter().GetResult();
        ConvertToFp16(scratch, output, n); // re-truncate the result to FP16 precision
    }

    /// <summary>ReLU over an FP16 buffer (truncated-f32): out = round16(max(in, 0)).</summary>
    public void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int n)
    {
        ValidateUnary(input, output, n);
        using var scratch = AllocateBuffer(n);
        ReLUAsync(input, scratch, n).GetAwaiter().GetResult();
        ConvertToFp16(scratch, output, n);
    }

    /// <summary>Residual add over FP16 buffers (truncated-f32): out = round16(a + b).</summary>
    public void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
        using var scratch = AllocateBuffer(n);
        AddAsync(a, b, scratch, n).GetAwaiter().GetResult();
        ConvertToFp16(scratch, output, n);
    }

    private static void ValidateUnary(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
    }
}

#endif
