// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// 16-bit float compute through the typed interfaces, for Float16 and BFloat16 alike: the packed kernels
// (GpuPackedKernelSource), two elements to a word, their arithmetic done on the bits so every device rounds as the
// host does.
public sealed partial class VulkanBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply, IGpuNarrowFloatReductions,
    IGpuNarrowFloatRounding, IGemmAutocastBackend
{
    private readonly GemmOperandAutocast _gemmAutocast = new();

    /// <inheritdoc/>
    IDisposable IGemmAutocastBackend.EnterGemmAutocast(Gpu.PrecisionMode precision, Gpu.MatMulAutocastMode mode)
        => _gemmAutocast.Enter(precision, mode);

    /// <inheritdoc/>
    bool IGpuNarrowFloatRounding.SupportsRoundToNarrowFloat(GpuElementType type) => HasNarrowFloatKernels(type);

    /// <inheritdoc/>
    void IGpuNarrowFloatRounding.RoundToNarrowFloat(GpuElementType type, IGpuBuffer source, IGpuBuffer destination, int count)
        => GpuPackedCompute.Round(LaunchPacked, type, source, destination, count);

    private bool HasNarrowFloatKernels(GpuElementType type) => GpuNarrowFloatLayout.IsNarrowFloat(type) && IsGlslCompilerAvailable;

    /// <inheritdoc/>
    public bool SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation operation, GpuElementType type) => HasNarrowFloatKernels(type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation operation, GpuElementType type) => HasNarrowFloatKernels(type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatMatrixMultiply(GpuElementType type) => HasNarrowFloatKernels(type);

    /// <inheritdoc/>
    public void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
        => GpuPackedCompute.Binary(LaunchPacked, operation, type, a, b, output, size);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatSum(GpuElementType type) => HasNarrowFloatKernels(type);

    /// <inheritdoc/>
    public float SumNarrowFloat(GpuElementType type, IGpuBuffer a, int size)
    {
        // The result is read back between passes, which resets the command buffer an open batch is recording into.
        ThrowIfHostSpanOpInsideOpenBatch();
        return GpuPackedCompute.Sum(this, LaunchPacked, type, a, size);
    }

    /// <inheritdoc/>
    public void ScalarNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, float scalar,
        IGpuBuffer output, int size)
        => GpuPackedCompute.Scalar(LaunchPacked, operation, type, a, scalar, output, size);

    /// <inheritdoc/>
    public void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size)
        => GpuPackedCompute.Unary(LaunchPacked, operation, type, a, output, size);

    /// <inheritdoc/>
    public void MatrixMultiplyNarrowFloat(GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        => GpuPackedCompute.MatrixMultiply(LaunchPacked, type, a, b, c, m, n, k);
}
