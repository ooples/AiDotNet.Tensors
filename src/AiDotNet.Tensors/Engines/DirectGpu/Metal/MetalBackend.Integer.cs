// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Integer compute on Int32 buffers (this backend's buffers are 4-byte words; the narrower and wider integer types have
// no storage here).
public sealed partial class MetalBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    private const string Int32LibraryName = "Int32";
    private IntPtr _int32Library;

    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type) => type == GpuElementType.Int32 && IsAvailable;

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type) => SupportsIntegerElementwise(type);

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
        GpuInt32ShaderSource.Require(nameof(BinaryInteger), size, output, a, b);
        string kernel = "i32k_" + GpuInt32ShaderSource.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Int32Pipeline(kernel), size, new[] { Int32Buffer(a), Int32Buffer(b), Int32Buffer(output) }, (uint)size);
    }

    /// <inheritdoc/>
    public void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuInt32ShaderSource.Require(nameof(UnaryInteger), size, output, a);
        string kernel = "i32k_" + GpuInt32ShaderSource.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Int32Pipeline(kernel), size, new[] { Int32Buffer(a), Int32Buffer(output) }, (uint)size);
    }

    /// <inheritdoc/>
    public void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        GpuInt32ShaderSource.Require(nameof(ScalarInteger), size, output, a);
        string kernel = "i32k_scalar_" + GpuInt32ShaderSource.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Int32Pipeline(kernel), size, new[] { Int32Buffer(a), Int32Buffer(output) }, (uint)size, unchecked((uint)(int)scalar));
    }

    /// <inheritdoc/>
    public long SumInteger(IGpuBuffer a, int size)
    {
        GpuInt32ShaderSource.RequireSum(a, size);
        var pipeline = Int32Pipeline("i32k_sum");
        if (size == 0) return 0;
        int block = GpuInt32ShaderSource.SumBlock;
        int partials = (size + block - 1) / block;
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = AllocateBuffer(Math.Max(1, (partials + block - 1) / block), GpuElementType.Int32);
        MetalGpuBuffer current = Int32Buffer(a);
        MetalGpuBuffer target = Int32Buffer(even);
        MetalGpuBuffer other = Int32Buffer(odd);
        int count = size;
        while (true)
        {
            int outputs = (count + block - 1) / block;
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
        return source.ToString();
    }
}
