using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The global-norm clip as one work-group reduction per tensor (no device address tables in core Vulkan compute), the
// Vulkan port of CudaBackend's IMultiTensorKernels. The sum of squares is the finite guard's (VulkanFiniteGuardKernels):
// a float accumulation and non-finite mark finished as the double the other backends write, so the clip, the guarded
// optimizer updates and ReadSumOfSquares share one layout. (The clip used to read its own raw float sum, which became
// infinite past the float range -- an L2 norm above ~1.8e19 -- and left those gradients unclipped and reported as
// non-finite.)
public sealed partial class VulkanBackend : IMultiTensorKernels
{
    // min(1, maxNorm / (norm + 1e-6)) from the finished double, the norm taken from its exponent and mantissa
    // (sum = m * 2^k, norm = sqrt(m) * 2^(k/2)) so a sum past the float range (DBL_MAX: the float accumulator overflowed)
    // still clips, to ~0 as PyTorch's float32 norm (inf) does. A non-finite norm leaves the gradients unscaled.
    private const string ClipScaleFromSumSquaresGlsl = @"#version 450
layout(local_size_x = 1) in;
layout(set=0,binding=0) readonly buffer Accb { uint acc[]; };
layout(set=0,binding=1) buffer Scaleb { float scale[]; };
layout(push_constant) uniform PC { float maxNorm; };
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    uint lo = acc[0], hi = acc[1];
    uint ef = (hi >> 20) & 0x7ffu;
    float c = 1.0;
    if (ef == 0u) c = maxNorm / 1e-6;
    else if (ef != 0x7ffu) {
        int k = int(ef) - 1023;
        float m = uintBitsToFloat(0x3f800000u | ((hi & 0xfffffu) << 3) | (lo >> 29));
        if ((k & 1) != 0) { m *= 2.0; k -= 1; }
        int h = k / 2;
        float root = sqrt(m);
        c = h < 64 ? maxNorm / (ldexp(root, h) + 1e-6) : ldexp(maxNorm / root, -h);
    }
    scale[0] = c < 1.0 ? c : 1.0;
}";

    private const string ScaleByDeviceScalarGlsl = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Scaleb { float scale[]; };
layout(set=0,binding=1) buffer Xb { float x[]; };
layout(push_constant) uniform PC { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < n) x[i] *= scale[0];
}";

    public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        EnsureInitialized();
        MultiTensorSumOfSquaresOnDevice(tensors, sizes, sumOfSquares);
    }

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
    {
        EnsureInitialized();
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        GlslUnaryOp(ClipScaleFromSumSquaresGlsl, sumOfSquares, scale, 1,
            new[] { unchecked((uint)SingleToInt32BitsCompat(maxNorm)) }, sizeof(uint));
    }

    public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
    {
        EnsureInitialized();
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        for (int t = 0; t < tensors.Count; t++)
            GlslUnaryOp(ScaleByDeviceScalarGlsl, scale, tensors[t], sizes[t], new[] { (uint)sizes[t] }, sizeof(uint));
    }

    /// <inheritdoc />
    double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
    }
}