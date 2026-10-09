using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The global-norm clip as one work-group reduction per tensor (no device address tables in core Vulkan compute), the
// Vulkan port of CudaBackend's IMultiTensorKernels. acc[0] holds the float sum of squares.
public sealed partial class VulkanBackend : IMultiTensorKernels
{
    // Only work-group 0 runs; thread 0 adds the partials in order, and dispatches complete in order, so no atomics.
    private const string TensorSumSquaresAccumulateGlsl = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Xb { float x[]; };
layout(set=0,binding=1) buffer Accb { float acc[]; };
layout(push_constant) uniform PC { uint n; };
shared float partial[256];
void main() {
    if (gl_WorkGroupID.x != 0u) return;
    uint lid = gl_LocalInvocationID.x;
    float v = 0.0;
    for (uint i = lid; i < n; i += 256u) { float e = x[i]; v += e * e; }
    partial[lid] = v;
    barrier();
    if (lid == 0u) {
        float s = 0.0;
        for (uint j = 0u; j < 256u; ++j) s += partial[j];
        acc[0] += s;
    }
}";

    // min(1, maxNorm / (norm + 1e-6)); a non-finite norm (exponent bits all set) leaves the gradients unscaled.
    private const string ClipScaleFromSumSquaresGlsl = @"#version 450
layout(local_size_x = 1) in;
layout(set=0,binding=0) readonly buffer Accb { float acc[]; };
layout(set=0,binding=1) buffer Scaleb { float scale[]; };
layout(push_constant) uniform PC { float maxNorm; };
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    float norm = sqrt(acc[0]);
    float c = (floatBitsToUint(norm) & 0x7f800000u) != 0x7f800000u ? maxNorm / (norm + 1e-6) : 1.0;
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
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        for (int t = 0; t < tensors.Count; t++)
            GlslUnaryOp(TensorSumSquaresAccumulateGlsl, tensors[t], sumOfSquares, MultiTensorArgs.ReductionGroupSize,
                new[] { (uint)sizes[t] }, sizeof(uint));
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
}