// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Integer compute. This backend's buffers are 4-byte words: Int32 is computed by the GLSL integer kernels below, and
// the other integer types and Bool by the packed kernels (GpuPackedKernelSource), which read and write whole words.
// All are compiled at run time like the other parity kernels.
public sealed unsafe partial class VulkanBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    private static bool PackedInteger(GpuElementType type) => Array.IndexOf(GpuPackedKernelSource.IntegerTypes, type) >= 0;

    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type)
        => (type == GpuElementType.Int32 || PackedInteger(type)) && IsGlslCompilerAvailable;

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type) => type != GpuElementType.Bool && SupportsIntegerElementwise(type);

    private void LaunchPacked(GpuPackedKernel kernel, int work, uint count, uint pa, uint pb, uint pc, IGpuBuffer[] buffers)
        => ParityDispatch(kernel.Name, GpuPackedKernelSource.Glsl(kernel, VulkanTorchParityElementwiseKernels.LocalSize), work,
            new[] { (uint)work, count, pa, pb, pc }, buffers);

    // Int32 has its own kernels; every other element type goes to the packed ones, which also refuse a non-integer.
    private static bool IsInt32(IGpuBuffer? buffer) => buffer is not null && buffer.ElementType == GpuElementType.Int32;

    /// <inheritdoc/>
    public void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (!IsInt32(output))
        {
            GpuPackedCompute.Binary(LaunchPacked, operation, a, b, output, size);
            return;
        }

        GpuInt32ShaderSource.Require(nameof(BinaryInteger), size, output, a, b);
        string glsl = VulkanInt32Kernels.Binary(operation);
        if (size == 0) return;
        ParityDispatch(GpuInt32ShaderSource.Name(operation) + "_i32", glsl, size, new[] { (uint)size }, a, b, output);
    }

    /// <inheritdoc/>
    public void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        if (!IsInt32(output))
        {
            GpuPackedCompute.Unary(LaunchPacked, operation, a, output, size);
            return;
        }

        GpuInt32ShaderSource.Require(nameof(UnaryInteger), size, output, a);
        string glsl = VulkanInt32Kernels.Unary(operation);
        if (size == 0) return;
        ParityDispatch(GpuInt32ShaderSource.Name(operation) + "_i32", glsl, size, new[] { (uint)size }, a, output);
    }

    /// <inheritdoc/>
    public void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        if (!IsInt32(output))
        {
            GpuPackedCompute.Scalar(LaunchPacked, operation, a, scalar, output, size);
            return;
        }

        GpuInt32ShaderSource.Require(nameof(ScalarInteger), size, output, a);
        string glsl = VulkanInt32Kernels.Scalar(operation);
        if (size == 0) return;
        ParityDispatch(GpuInt32ShaderSource.Name(operation) + "_scalar_i32", glsl, size,
            new[] { (uint)size, unchecked((uint)(int)scalar) }, a, output);
    }

    /// <inheritdoc/>
    public long SumInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Sum, a, size);

    /// <inheritdoc/>
    public long MinInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Min, a, size);

    /// <inheritdoc/>
    public long MaxInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Max, a, size);

    private long ReduceInteger(GpuIntegerReduction reduction, IGpuBuffer a, int size)
    {
        if (!IsInt32(a))
        {
            // The result is read back between passes, which resets the command buffer an open batch is recording into.
            ThrowIfHostSpanOpInsideOpenBatch();
            return GpuPackedCompute.Reduce(reduction, this, LaunchPacked, a, size, (partials, count) => ReduceInteger(reduction, partials, count));
        }

        GpuInt32ShaderSource.RequireSum(a, size);
        string glsl = VulkanInt32Kernels.ReduceBlocks(reduction);
        GpuIntegerLayout.RequireElements(reduction, size);
        if (size == 0) return 0;
        // The result is read back between passes, which resets the command buffer an open batch is recording into.
        ThrowIfHostSpanOpInsideOpenBatch();
        int block = GpuInt32ShaderSource.SumBlock;
        int partials = GpuIntegerLayout.Blocks(size, block);
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, block)), GpuElementType.Int32);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        while (true)
        {
            int outputs = GpuIntegerLayout.Blocks(count, block);
            ParityDispatch(GpuInt32ShaderSource.Name(reduction) + "_i32", glsl, outputs, new[] { (uint)outputs, (uint)count }, current, target);
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
    public static string ReduceBlocks(GpuIntegerReduction reduction) => Sources.GetOrAdd(GpuInt32ShaderSource.Name(reduction), _ => Head + OneIn + @"
layout(push_constant) uniform P { uint n; uint count; };
void main()
{
" + Guard + @"    uint first = i * " + GpuInt32ShaderSource.SumBlock + @"u;
    uint last = min(first + " + GpuInt32ShaderSource.SumBlock + @"u, count);
" + (reduction == GpuIntegerReduction.Sum
    ? @"    uint s = 0u;
    for (uint j = first; j < last; j++) { s += uint(a[j]); }
    o[i] = int(s);
"
    : @"    int s = a[first];
    for (uint j = first + 1u; j < last; j++) { s = " + (reduction == GpuIntegerReduction.Min ? "min" : "max") + @"(s, a[j]); }
    o[i] = s;
") + @"}
");
}
