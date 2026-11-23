#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Integer compute on Int32 buffers (this backend's buffers are 4-byte words; the narrower and wider integer types have
// no storage here). WGSL i32 arithmetic wraps by definition, as unchecked host arithmetic does.
public sealed partial class WebGpuBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type) => type == GpuElementType.Int32 && SynchronousRouteAvailable;

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type) => SupportsIntegerElementwise(type);

    /// <inheritdoc/>
    public void BinaryInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        GpuInt32ShaderSource.Require(nameof(BinaryInteger), size, output, a, b);
        string source = WebGpuInt32Kernels.Binary(operation);
        if (size == 0) return;
        Dispatch3BufferAsync("I32Binary" + operation, source, "main", a, b, output, Uniforms(size), size).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        GpuInt32ShaderSource.Require(nameof(UnaryInteger), size, output, a);
        string source = WebGpuInt32Kernels.Unary(operation);
        if (size == 0) return;
        Dispatch2BufferAsync("I32Unary" + operation, source, "main", a, output, Uniforms(size), size).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        GpuInt32ShaderSource.Require(nameof(ScalarInteger), size, output, a);
        string source = WebGpuInt32Kernels.Scalar(operation);
        if (size == 0) return;
        Dispatch2BufferAsync("I32Scalar" + operation, source, "main", a, output,
            UniformBits(size, 0, 0, unchecked((uint)(int)scalar)), size).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public long SumInteger(IGpuBuffer a, int size)
    {
        EnsureInitialized();
        GpuInt32ShaderSource.RequireSum(a, size);
        string source = WebGpuInt32Kernels.SumBlocks;
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
            Dispatch2BufferAsync("I32Sum", source, "main", current, target, Uniforms(outputs, count), outputs).GetAwaiter().GetResult();
            if (outputs == 1)
                return BitConverter.ToInt32(DownloadByteBuffer(target, sizeof(int)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = outputs;
        }
    }
}

/// <summary>WGSL sources of the Int32 kernels.</summary>
internal static class WebGpuInt32Kernels
{
    private static readonly ConcurrentDictionary<string, string> Sources = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);

    private const string Parameters = @"
struct Params { n: u32, p0: u32, p1: u32, pad0: u32, s0: u32, s1: u32, s2: u32, s3: u32 }
";

    private const string TwoIn = @"
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<i32>;
" + Parameters + @"@group(0) @binding(3) var<uniform> params: Params;
";

    private const string OneIn = @"
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
" + Parameters + @"@group(0) @binding(2) var<uniform> params: Params;
";

    private const string Entry = @"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
";

    /// <summary><c>o[i] = a[i] op b[i]</c>.</summary>
    public static string Binary(GpuIntegerBinaryOperation operation)
    {
        string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Wgsl);
        return Sources.GetOrAdd("binary:" + operation, _ => TwoIn + Entry + @"    let x = a[i];
    let y = b[i];
    o[i] = " + expression + @";
}
");
    }

    /// <summary><c>o[i] = a[i] op y</c>, the scalar in two 16-bit pieces of the uniform.</summary>
    public static string Scalar(GpuIntegerBinaryOperation operation)
    {
        string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Wgsl);
        return Sources.GetOrAdd("scalar:" + operation, _ => OneIn + Entry + @"    let x = a[i];
    let y = bitcast<i32>(params.s0 | (params.s1 << 16u));
    o[i] = " + expression + @";
}
");
    }

    /// <summary><c>o[i] = op a[i]</c>.</summary>
    public static string Unary(GpuIntegerUnaryOperation operation)
    {
        string expression = GpuInt32ShaderSource.Expression(operation, GpuInt32ShaderDialect.Wgsl);
        return Sources.GetOrAdd("unary:" + operation, _ => OneIn + Entry + @"    let x = a[i];
    o[i] = " + expression + @";
}
");
    }

    /// <summary>Pass of a sum: <c>o[i]</c> is the wrapping sum of block <c>i</c> of the <c>p0</c> input elements.</summary>
    public static string SumBlocks => Sources.GetOrAdd("sum", _ => OneIn + Entry + @"    let first = i * " + GpuInt32ShaderSource.SumBlock + @"u;
    let last = min(first + " + GpuInt32ShaderSource.SumBlock + @"u, params.p0);
    var s = 0u;
    for (var j = first; j < last; j++) { s += bitcast<u32>(a[j]); }
    o[i] = bitcast<i32>(s);
}
");
}
#endif
