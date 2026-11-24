#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Integer compute. This backend's buffers are 4-byte words: Int32 is computed by the WGSL i32 kernels below (WGSL i32
// arithmetic wraps by definition, as unchecked host arithmetic does), and the other integer types and Bool by the
// packed kernels (GpuPackedKernelSource), which read and write whole words.
public sealed partial class WebGpuBackend : IGpuIntegerElementwise, IGpuIntegerReductions
{
    private static bool PackedInteger(GpuElementType type) => Array.IndexOf(GpuPackedKernelSource.IntegerTypes, type) >= 0;

    /// <inheritdoc/>
    public bool SupportsIntegerElementwise(GpuElementType type)
        => (type == GpuElementType.Int32 || PackedInteger(type)) && SynchronousRouteAvailable;

    /// <inheritdoc/>
    public bool SupportsIntegerReductions(GpuElementType type) => type != GpuElementType.Bool && SupportsIntegerElementwise(type);

    // Runs a packed kernel and waits for it. WebGPU rejects a bind group that binds one buffer both read-only and
    // writable, so an output that is also an operand is computed in a copy of itself (the kernels keep the bits of
    // the output they do not compute) and copied back.
    private void LaunchPacked(GpuPackedKernel kernel, int work, uint count, uint pa, uint pb, uint pc, IGpuBuffer[] buffers)
    {
        EnsureInitialized();
        RequireSynchronousRoute();
        if (work <= 0) return;
        var output = AsWebGpu(buffers[buffers.Length - 1]);
        bool aliased = false;
        for (int i = 0; i < buffers.Length - 1; i++)
            aliased |= ReferenceEquals(buffers[i], output) || AsWebGpu(buffers[i]).BufferId == output.BufferId;
        if (aliased)
        {
            int words = checked((int)(output.SizeInBytes / sizeof(uint)));
            using var scratch = AllocateBufferCore(words, zeroInitialize: false);
            CopyWordRange(output, 0, scratch, 0, words);
            var separate = (IGpuBuffer[])buffers.Clone();
            separate[separate.Length - 1] = scratch;
            LaunchPacked(kernel, work, count, pa, pb, pc, separate);
            CopyWordRange(scratch, 0, output, 0, words);
            return;
        }

        string source = GpuPackedKernelSource.Wgsl(kernel);
        float[] uniforms = UniformBits(work, unchecked((int)count), unchecked((int)pa), (long)(((ulong)pc << 32) | pb));
        var launch = buffers.Length == 3
            ? Dispatch3BufferAsync(kernel.Name, source, "main", buffers[0], buffers[1], buffers[2], uniforms, work)
            : Dispatch2BufferAsync(kernel.Name, source, "main", buffers[0], buffers[1], uniforms, work);
        launch.GetAwaiter().GetResult();
    }

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

        EnsureInitialized();
        GpuInt32ShaderSource.Require(nameof(BinaryInteger), size, output, a, b);
        string source = WebGpuInt32Kernels.Binary(operation);
        if (size == 0) return;
        Dispatch3BufferAsync("I32Binary" + operation, source, "main", a, b, output, Uniforms(size), size).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public void UnaryInteger(GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        if (!IsInt32(output))
        {
            GpuPackedCompute.Unary(LaunchPacked, operation, a, output, size);
            return;
        }

        EnsureInitialized();
        GpuInt32ShaderSource.Require(nameof(UnaryInteger), size, output, a);
        string source = WebGpuInt32Kernels.Unary(operation);
        if (size == 0) return;
        Dispatch2BufferAsync("I32Unary" + operation, source, "main", a, output, Uniforms(size), size).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public void ScalarInteger(GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        if (!IsInt32(output))
        {
            GpuPackedCompute.Scalar(LaunchPacked, operation, a, scalar, output, size);
            return;
        }

        EnsureInitialized();
        GpuInt32ShaderSource.Require(nameof(ScalarInteger), size, output, a);
        string source = WebGpuInt32Kernels.Scalar(operation);
        if (size == 0) return;
        Dispatch2BufferAsync("I32Scalar" + operation, source, "main", a, output,
            UniformBits(size, 0, 0, unchecked((uint)(int)scalar)), size).GetAwaiter().GetResult();
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
        EnsureInitialized();
        GpuInt32ShaderSource.RequireSum(a, size);
        string source = WebGpuInt32Kernels.ReduceBlocks(reduction);
        GpuIntegerLayout.RequireElements(reduction, size);
        if (size == 0) return 0;
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
            Dispatch2BufferAsync("I32" + GpuInt32ShaderSource.Name(reduction), source, "main", current, target, Uniforms(outputs, count), outputs).GetAwaiter().GetResult();
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
    public static string ReduceBlocks(GpuIntegerReduction reduction) => Sources.GetOrAdd(GpuInt32ShaderSource.Name(reduction), _ => OneIn + Entry + @"    let first = i * " + GpuInt32ShaderSource.SumBlock + @"u;
    let last = min(first + " + GpuInt32ShaderSource.SumBlock + @"u, params.p0);
" + (reduction == GpuIntegerReduction.Sum
    ? @"    var s = 0u;
    for (var j = first; j < last; j++) { s += bitcast<u32>(a[j]); }
    o[i] = bitcast<i32>(s);
"
    : @"    var s = a[first];
    for (var j = first + 1u; j < last; j++) { s = " + (reduction == GpuIntegerReduction.Min ? "min" : "max") + @"(s, a[j]); }
    o[i] = s;
") + @"}
");
}
#endif
