// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Integer compute. This backend's buffers are 4-byte words: Int32 is computed by the kernels below, and the other
// integer types and Bool by the packed kernels (GpuPackedKernelSource), which read and write whole words.
public sealed partial class MetalBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    private const string Int32LibraryName = "Int32";
    private IntPtr _int32Library;
    private const string PackedLibraryName = "Packed";
    private IntPtr _packedLibrary;
    // Set when the compiler rejected the packed library: kept, so no later call compiles it again, and every
    // capability the library backs then reports false (the engine computes those types on the CPU).
    private Exception? _packedLibraryFailure;

    private bool PackedKernelsAvailable => IsAvailable && System.Threading.Volatile.Read(ref _packedLibraryFailure) is null;

    private static bool PackedInteger(GpuElementType type) => Array.IndexOf(GpuPackedKernelSource.IntegerTypes, type) >= 0;

    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type)
        => type == GpuElementType.Int32 ? IsAvailable : PackedInteger(type) && PackedKernelsAvailable;

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type) => type != GpuElementType.Bool && SupportsIntegerElementwise(type);

    private MetalPipelineState PackedPipeline(string kernel)
    {
        ThrowIfDisposed();
        // A rejected library is a capability gap (GpuCapabilityUnavailableException), recorded once.
        IntPtr library = Float64Library(ref _packedLibrary, ref _packedLibraryFailure, PackedLibraryName,
            () => _shaderLibrary.CompileLibrary(PackedLibraryName, GpuPackedKernelSource.MetalLibrary));
        return GetPipeline(PackedLibraryName, library, kernel);
    }

    // Runs a packed kernel and waits for it. An output that is also an operand is separated by the launch (Metal
    // buffer arguments cannot alias), which copies it back afterwards.
    private void LaunchPacked(GpuPackedKernel kernel, int work, uint count, uint pa, uint pb, uint pc, IGpuBuffer[] buffers)
    {
        var pipeline = PackedPipeline(kernel.Name);
        if (work <= 0) return;
        var bound = new MetalGpuBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) bound[i] = Int32Buffer(buffers[i]);
        LaunchFloat64(pipeline, work, bound, (uint)work, count, pa, pb, pc);
    }

    // Int32 has its own kernels; every other element type goes to the packed ones, which also refuse a non-integer.
    private static bool IsInt32(IGpuBuffer? buffer) => buffer is not null && buffer.ElementType == GpuElementType.Int32;

    private MetalPipelineState Int32Pipeline(string kernel)
    {
        ThrowIfDisposed();
        if (_int32Library == IntPtr.Zero)
        {
            lock (_float64LibraryLock)
            {
                if (_int32Library == IntPtr.Zero)
                    _int32Library = _shaderLibrary.CompileLibrary(Int32LibraryName, MetalInt32Kernels.Source);
            }
        }

        return GetPipeline(Int32LibraryName, _int32Library, kernel);
    }

    private static MetalGpuBuffer Int32Buffer(IGpuBuffer buffer)
        => buffer as MetalGpuBuffer ?? throw new ArgumentException("Buffer must be a MetalGpuBuffer", nameof(buffer));

    /// <inheritdoc/>
    public void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (!IsInt32(output))
        {
            GpuPackedCompute.Binary(LaunchPacked, operation, a, b, output, size);
            return;
        }

        GpuInt32ShaderSource.Require(nameof(BinaryInteger), size, output, a, b);
        string kernel = "i32k_" + GpuInt32ShaderSource.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Int32Pipeline(kernel), size, new[] { Int32Buffer(a), Int32Buffer(b), Int32Buffer(output) }, (uint)size);
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
        string kernel = "i32k_" + GpuInt32ShaderSource.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Int32Pipeline(kernel), size, new[] { Int32Buffer(a), Int32Buffer(output) }, (uint)size);
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
        string kernel = "i32k_scalar_" + GpuInt32ShaderSource.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Int32Pipeline(kernel), size, new[] { Int32Buffer(a), Int32Buffer(output) }, (uint)size, unchecked((uint)(int)scalar));
    }

    /// <inheritdoc/>
    public long SumInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Sum, a, size);

    /// <inheritdoc/>
    public long MinInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Min, a, size);

    /// <inheritdoc/>
    public long MaxInteger(IGpuBuffer a, int size) => ReduceInteger(GpuIntegerReduction.Max, a, size);

    private long ReduceInteger(GpuIntegerReduction reduction, IGpuBuffer a, int size)
    {
        if (!IsInt32(a)) return GpuPackedCompute.Reduce(reduction, this, LaunchPacked, a, size, (partials, count) => ReduceInteger(reduction, partials, count));
        GpuInt32ShaderSource.RequireSum(a, size);
        var pipeline = Int32Pipeline("i32k_" + GpuInt32ShaderSource.Name(reduction));
        GpuIntegerLayout.RequireElements(reduction, size);
        if (size == 0) return 0;
        int block = GpuInt32ShaderSource.SumBlock;
        int partials = GpuIntegerLayout.Blocks(size, block);
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, block)), GpuElementType.Int32);
        MetalGpuBuffer current = Int32Buffer(a);
        MetalGpuBuffer target = Int32Buffer(even);
        MetalGpuBuffer other = Int32Buffer(odd);
        int count = size;
        while (true)
        {
            int outputs = GpuIntegerLayout.Blocks(count, block);
            LaunchFloat64(pipeline, outputs, new[] { current, target }, (uint)outputs, (uint)count);
            if (outputs == 1)
                return BitConverter.ToInt32(DownloadByteBuffer(target, sizeof(int)), 0);
            current = target;
            (target, other) = (other, target);
            count = outputs;
        }
    }
}

/// <summary>Metal Shading Language source of the Int32 kernels.</summary>
internal static class MetalInt32Kernels
{
    private static readonly Lazy<string> Built = new Lazy<string>(Build);

    public static string Source => Built.Value;

    private static string Build()
    {
        var source = new StringBuilder(@"
#include <metal_stdlib>
using namespace metal;
");
        foreach (var operation in GpuInt32ShaderSource.BinaryOperations)
        {
            string name = GpuInt32ShaderSource.Name(operation);
            string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Metal);
            source.Append(@"
kernel void i32k_" + name + @"(
    device const int* a [[buffer(0)]],
    device const int* b [[buffer(1)]],
    device int* o [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    int x = a[gid];
    int y = b[gid];
    o[gid] = " + expression + @";
}

kernel void i32k_scalar_" + name + @"(
    device const int* a [[buffer(0)]],
    device int* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& scalarBits [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    int x = a[gid];
    int y = int(scalarBits);
    o[gid] = " + expression + @";
}
");
        }

        foreach (var operation in GpuInt32ShaderSource.UnaryOperations)
        {
            source.Append(@"
kernel void i32k_" + GpuInt32ShaderSource.Name(operation) + @"(
    device const int* a [[buffer(0)]],
    device int* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    int x = a[gid];
    o[gid] = " + GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Metal) + @";
}
");
        }

        source.Append(@"
// Pass of a sum: o[gid] is the wrapping sum of block gid of the `count` input elements.
kernel void i32k_sum(
    device const int* a [[buffer(0)]],
    device int* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    uint first = gid * " + GpuInt32ShaderSource.SumBlock + @"u;
    uint last = min(first + " + GpuInt32ShaderSource.SumBlock + @"u, count);
    uint s = 0u;
    for (uint j = first; j < last; j++) { s += uint(a[j]); }
    o[gid] = int(s);
}
");
        foreach (var reduction in new[] { GpuIntegerReduction.Min, GpuIntegerReduction.Max })
        {
            source.Append(@"
// Pass of a minimum or maximum: o[gid] is that of block gid of the `count` input elements (a block is never empty).
kernel void i32k_" + GpuInt32ShaderSource.Name(reduction) + @"(
    device const int* a [[buffer(0)]],
    device int* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    uint first = gid * " + GpuInt32ShaderSource.SumBlock + @"u;
    uint last = min(first + " + GpuInt32ShaderSource.SumBlock + @"u, count);
    int s = a[first];
    for (uint j = first + 1u; j < last; j++) { s = " + (reduction == GpuIntegerReduction.Min ? "min" : "max") + @"(s, a[j]); }
    o[gid] = s;
}
");
        }

        return source.ToString();
    }
}
