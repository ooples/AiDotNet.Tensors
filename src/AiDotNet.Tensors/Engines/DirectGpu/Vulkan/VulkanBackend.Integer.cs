// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Integer compute on Int32 buffers (this backend's buffers are 4-byte words; the narrower and wider integer types have
// no storage here). GLSL integer kernels, compiled at run time like the other parity kernels.
public sealed unsafe partial class VulkanBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type) => type == GpuElementType.Int32 && IsGlslCompilerAvailable;

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type) => SupportsIntegerElementwise(type);

    /// <inheritdoc/>
    public void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        GpuInt32ShaderSource.Require(nameof(BinaryInteger), size, output, a, b);
        string glsl = VulkanInt32Kernels.Binary(operation);
        if (size == 0) return;
        ParityDispatch(GpuInt32ShaderSource.Name(operation) + "_i32", glsl, size, new[] { (uint)size }, a, b, output);
    }

    /// <inheritdoc/>
    public void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuInt32ShaderSource.Require(nameof(UnaryInteger), size, output, a);
        string glsl = VulkanInt32Kernels.Unary(operation);
        if (size == 0) return;
        ParityDispatch(GpuInt32ShaderSource.Name(operation) + "_i32", glsl, size, new[] { (uint)size }, a, output);
    }

    /// <inheritdoc/>
    public void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        GpuInt32ShaderSource.Require(nameof(ScalarInteger), size, output, a);
        string glsl = VulkanInt32Kernels.Scalar(operation);
        if (size == 0) return;
        ParityDispatch(GpuInt32ShaderSource.Name(operation) + "_scalar_i32", glsl, size,
            new[] { (uint)size, unchecked((uint)(int)scalar) }, a, output);
    }

    /// <inheritdoc/>
    public long SumInteger(IGpuBuffer a, int size)
    {
        GpuInt32ShaderSource.RequireSum(a, size);
        string glsl = VulkanInt32Kernels.SumBlocks;
        if (size == 0) return 0;
        int block = GpuInt32ShaderSource.SumBlock;
        int partials = (size + block - 1) / block;
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = AllocateBuffer(Math.Max(1, (partials + block - 1) / block), GpuElementType.Int32);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        while (true)
        {
            int outputs = (count + block - 1) / block;
            ParityDispatch("sum_i32", glsl, outputs, new[] { (uint)outputs, (uint)count }, current, target);
            if (outputs == 1)
                return BitConverter.ToInt32(DownloadByteBuffer(target, sizeof(int)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = outputs;
        }
    }
}

/// <summary>GLSL sources of the Int32 kernels.</summary>
internal static class VulkanInt32Kernels
{
    private static readonly ConcurrentDictionary<string, string> Sources = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);

    private static readonly string Head = "#version 450\nlayout(local_size_x = " + VulkanTorchParityElementwiseKernels.LocalSize + ") in;\n";

    private const string TwoIn = @"
layout(set = 0, binding = 0) readonly buffer A { int a[]; };
layout(set = 0, binding = 1) readonly buffer B { int b[]; };
layout(set = 0, binding = 2) writeonly buffer O { int o[]; };
";

    private const string OneIn = @"
layout(set = 0, binding = 0) readonly buffer A { int a[]; };
layout(set = 0, binding = 1) writeonly buffer O { int o[]; };
";

    private const string Guard = "    uint i = gl_GlobalInvocationID.x;\n    if (i >= n) return;\n";

    /// <summary><c>o[i] = a[i] op b[i]</c>.</summary>
    public static string Binary(GpuIntegerBinaryOperation operation)
    {
        string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Glsl);
        return Sources.GetOrAdd("binary:" + operation, _ => Head + TwoIn + @"
layout(push_constant) uniform P { uint n; };
void main()
{
" + Guard + @"    int x = a[i];
    int y = b[i];
    o[i] = " + expression + @";
}
");
    }

    /// <summary><c>o[i] = a[i] op y</c>, the scalar a push constant.</summary>
    public static string Scalar(GpuIntegerBinaryOperation operation)
    {
        string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Glsl);
        return Sources.GetOrAdd("scalar:" + operation, _ => Head + OneIn + @"
layout(push_constant) uniform P { uint n; int y; };
void main()
{
" + Guard + @"    int x = a[i];
    o[i] = " + expression + @";
}
");
    }

    /// <summary><c>o[i] = op a[i]</c>.</summary>
    public static string Unary(GpuIntegerUnaryOperation operation)
    {
        string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Glsl);
        return Sources.GetOrAdd("unary:" + operation, _ => Head + OneIn + @"
layout(push_constant) uniform P { uint n; };
void main()
{
" + Guard + @"    int x = a[i];
    o[i] = " + expression + @";
}
");
    }

    /// <summary>Pass of a sum: <c>o[i]</c> is the wrapping sum of block <c>i</c> of the <c>count</c> input elements.</summary>
    public static string SumBlocks => Sources.GetOrAdd("sum", _ => Head + OneIn + @"
layout(push_constant) uniform P { uint n; uint count; };
void main()
{
" + Guard + @"    uint first = i * " + GpuInt32ShaderSource.SumBlock + @"u;
    uint last = min(first + " + GpuInt32ShaderSource.SumBlock + @"u, count);
    uint s = 0u;
    for (uint j = first; j < last; j++) { s += uint(a[j]); }
    o[i] = int(s);
}
");
}
