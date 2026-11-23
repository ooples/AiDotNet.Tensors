// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// 16-bit float compute through the typed interfaces. This backend's buffers are 4-byte words, so it has the half kernels
// it always had (add, relu, gelu, matrix multiply, over packed halves) and no BFloat16 ones.
public sealed partial class MetalBackend : IGpuNarrowFloatElementwise, IGpuNarrowFloatMatrixMultiply
{
    private bool HalfKernel(GpuNarrowFloatBinaryOperation operation, GpuElementType type)
        => type == GpuElementType.Float16 && operation == GpuNarrowFloatBinaryOperation.Add && SupportsFp16NativeOps;

    private bool HalfKernel(GpuNarrowFloatUnaryOperation operation, GpuElementType type)
        => type == GpuElementType.Float16 && SupportsFp16NativeOps
           && operation is GpuNarrowFloatUnaryOperation.Relu or GpuNarrowFloatUnaryOperation.Gelu;

    /// <inheritdoc/>
    public bool SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation operation, GpuElementType type) => HalfKernel(operation, type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatUnary(GpuNarrowFloatUnaryOperation operation, GpuElementType type) => HalfKernel(operation, type);

    /// <inheritdoc/>
    public bool SupportsNarrowFloatMatrixMultiply(GpuElementType type) => type == GpuElementType.Float16 && SupportsHgemm;

    /// <inheritdoc/>
    public void BinaryNarrowFloat(GpuNarrowFloatBinaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(BinaryNarrowFloat), type, size, (a, size), (b, size), (output, size));
        if (!HalfKernel(operation, type))
            throw new NotSupportedException($"{BackendName} has no {operation} kernel for {type} buffers.");
        if (size > 0) Fp16Add(a, b, output, size);
    }

    /// <inheritdoc/>
    public void UnaryNarrowFloat(GpuNarrowFloatUnaryOperation operation, GpuElementType type, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(UnaryNarrowFloat), type, size, (a, size), (output, size));
        if (!HalfKernel(operation, type))
            throw new NotSupportedException($"{BackendName} has no {operation} kernel for {type} buffers.");
        if (size == 0) return;
        if (operation == GpuNarrowFloatUnaryOperation.Relu) Fp16Relu(a, output, size);
        else Fp16Gelu(a, output, size);
    }

    /// <inheritdoc/>
    public void MatrixMultiplyNarrowFloat(GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
    {
        var (aCount, bCount, cCount) = GpuNarrowFloatLayout.MatrixCounts(m, n, k);
        GpuNarrowFloatLayout.Require(nameof(MatrixMultiplyNarrowFloat), type, cCount, (a, aCount), (b, bCount), (c, cCount));
        if (ReferenceEquals(c, a) || ReferenceEquals(c, b))
            throw new ArgumentException("The output of a matrix multiply must not be one of its operands.", nameof(c));
        if (!SupportsNarrowFloatMatrixMultiply(type))
            throw new NotSupportedException($"{BackendName} has no matrix multiply for {type} buffers.");
        Hgemm(a, b, c, m, n, k);
    }
}
