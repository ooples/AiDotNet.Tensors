#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Double-precision compute over Float64 buffers.
//
// WGSL has no 64-bit float, so these kernels compute IEEE binary64 in 32-bit integer arithmetic (GpuSoftFloat64): the
// four arithmetic operations and the square root equal the host bit for bit (a NaN result is always the one quiet NaN
// 0xFFF8000000000000, whatever the sign and payload of the host's NaN), and exp, log, tanh and the logistic function are within a few ulp. A double is two 4-byte words of the buffer, low word first.
//
// Emulation costs hundreds of integer operations an element. The backend reports the capability as Emulated, and the
// engine sends doubles here only when a policy asks for device residency; by default they stay on the CPU.
public sealed partial class WebGpuBackend : IGpuFloat64Reductions, IGpuFloat64Elementwise, IGpuFloat64MatrixMultiply,
    IGpuFloat64UnaryAvailability
{
    // Every unary kernel is built with the rest of the double kernels here: none can be missing alone.
    bool IGpuFloat64UnaryAvailability.SupportsFloat64Unary(GpuFloat64UnaryOperation operation) => SupportsFloat64Elementwise;

    /// <inheritdoc/>
    public bool SupportsFloat64Reductions => IsAvailable;

    /// <inheritdoc/>
    public bool SupportsFloat64Elementwise => IsAvailable;

    /// <inheritdoc/>
    public bool SupportsFloat64MatrixMultiply => IsAvailable;

    private WebGpuBuffer Float64Operand(IGpuBuffer buffer, int size, string name)
    {
        EnsureInitialized();
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.ElementType != GpuElementType.Float64 || buffer is not WebGpuBuffer typed)
            throw new ArgumentException($"Expected a {GpuElementType.Float64} buffer of this backend, got {buffer.ElementType}.", name);
        if (size < 0 || size > buffer.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {buffer.Size} elements.");
        return typed;
    }

    private static float Word(int value) => BitConverter.Int32BitsToSingle(value);

    // A uniform reaches the device as floats, and a float does not carry every bit pattern through every host (a NaN
    // pattern has no JSON form). A double therefore travels as four 16-bit pieces, each a small integer.
    private static float[] Uniforms(int n, int p0 = 0, int p1 = 0, double scalar = 0.0)
    {
        long bits = BitConverter.DoubleToInt64Bits(scalar);
        return new[]
        {
            Word(n), Word(p0), Word(p1), 0f,
            Word((int)(bits & 0xFFFF)), Word((int)((bits >> 16) & 0xFFFF)), Word((int)((bits >> 32) & 0xFFFF)), Word((int)((bits >> 48) & 0xFFFF)),
        };
    }

    // WebGPU rejects a bind group that binds one buffer both read-only and writable. An output that is also an input
    // is computed into a scratch buffer and copied back.
    private void IntoOutput(WebGpuBuffer output, int size, Func<WebGpuBuffer, System.Threading.Tasks.Task> compute, params WebGpuBuffer[] inputs)
    {
        bool aliased = false;
        foreach (var input in inputs) aliased |= ReferenceEquals(input, output) || input.BufferId == output.BufferId;
        if (!aliased)
        {
            compute(output).GetAwaiter().GetResult();
            return;
        }

        using var scratch = AllocateBuffer(size, GpuElementType.Float64);
        compute((WebGpuBuffer)scratch).GetAwaiter().GetResult();
        Copy(scratch, output, size);
    }

    /// <inheritdoc/>
    public void BinaryFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        var left = Float64Operand(a, size, nameof(a));
        var right = Float64Operand(b, size, nameof(b));
        var destination = Float64Operand(output, size, nameof(output));
        string source = WebGpuFloat64Kernels.Binary(operation);
        if (size == 0) return;
        IntoOutput(destination, size, target => Dispatch3BufferAsync("F64Binary" + operation, source, "main", left, right, target,
            Uniforms(size), size), left, right);
    }

    /// <inheritdoc/>
    public void UnaryFloat64(GpuFloat64UnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        var destination = Float64Operand(output, size, nameof(output));
        string source = WebGpuFloat64Kernels.Unary(operation);
        if (size == 0) return;
        IntoOutput(destination, size, target => Dispatch2BufferAsync("F64Unary" + operation, source, "main", input, target,
            Uniforms(size), size), input);
    }

    /// <inheritdoc/>
    public void ScalarFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, double scalar, IGpuBuffer output, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        var destination = Float64Operand(output, size, nameof(output));
        string source = WebGpuFloat64Kernels.Scalar(operation);
        if (size == 0) return;
        IntoOutput(destination, size, target => Dispatch2BufferAsync("F64Scalar" + operation, source, "main", input, target,
            Uniforms(size, scalar: scalar), size), input);
    }

    /// <inheritdoc/>
    public double SumFloat64(IGpuBuffer a, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        return size == 0 ? 0.0 : ReduceFloat64(input, null, size);
    }

    /// <inheritdoc/>
    public double DotFloat64(IGpuBuffer a, IGpuBuffer b, int size)
    {
        var left = Float64Operand(a, size, nameof(a));
        var right = Float64Operand(b, size, nameof(b));
        return size == 0 ? 0.0 : ReduceFloat64(left, right, size);
    }

    // Each pass folds blocks of WebGpuFloat64Kernels.ReduceBlock elements into one partial each, until one value is
    // left. The first pass multiplies by `second` when it is given (the dot product); every later pass sums partials.
    private double ReduceFloat64(WebGpuBuffer first, WebGpuBuffer? second, int size)
    {
        int block = WebGpuFloat64Kernels.ReduceBlock;
        int partials = (size + block - 1) / block;
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Float64);
        using var odd = AllocateBuffer(Math.Max(1, (partials + block - 1) / block), GpuElementType.Float64);
        IGpuBuffer current = first;
        IGpuBuffer target = even;
        int count = size;
        bool firstPass = true;
        while (true)
        {
            int outputs = (count + block - 1) / block;
            if (firstPass && second is not null)
                Dispatch3BufferAsync("F64Dot", WebGpuFloat64Kernels.DotBlocks, "main", current, second, target,
                    Uniforms(outputs, count), outputs).GetAwaiter().GetResult();
            else
                Dispatch2BufferAsync("F64Sum", WebGpuFloat64Kernels.SumBlocks, "main", current, target,
                    Uniforms(outputs, count), outputs).GetAwaiter().GetResult();
            firstPass = false;
            if (outputs == 1)
                return BitConverter.ToDouble(DownloadByteBuffer(target, sizeof(double)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = outputs;
        }
    }

    /// <inheritdoc/>
    public void MatrixMultiplyFloat64(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), m, "Matrix dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), n, "Matrix dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), k, "Matrix dimensions must be positive.");
        var left = Float64Operand(a, checked(m * k), nameof(a));
        var right = Float64Operand(b, checked(k * n), nameof(b));
        var product = Float64Operand(c, checked(m * n), nameof(c));
        if (ReferenceEquals(product, left) || ReferenceEquals(product, right)
            || product.BufferId == left.BufferId || product.BufferId == right.BufferId)
            throw new ArgumentException("The product buffer must not be one of the operands.", nameof(c));
        int elements = checked(m * n);
        Dispatch3BufferAsync("F64MatMul", WebGpuFloat64Kernels.MatrixMultiply, "main", left, right, product,
            Uniforms(elements, n, k), elements).GetAwaiter().GetResult();
    }

    // A device copy of doubles moves their words as integers: a float load and store need not keep every bit pattern.
    // A copy within one buffer would bind it read-only and writable at once, which WebGPU rejects even for disjoint
    // ranges: it goes through a scratch buffer.
    private void CopyFloat64(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        if (ReferenceEquals(source, destination)
            || (source is WebGpuBuffer from && destination is WebGpuBuffer to && from.BufferId == to.BufferId))
        {
            using var scratch = AllocateBuffer(length, GpuElementType.Float64);
            CopyFloat64Words(source, sourceOffset, scratch, 0, length);
            CopyFloat64Words(scratch, 0, destination, destinationOffset, length);
            return;
        }

        CopyFloat64Words(source, sourceOffset, destination, destinationOffset, length);
    }

    private void CopyFloat64Words(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        Dispatch2BufferAsync("F64Copy", WebGpuFloat64Kernels.CopyWords, "main", source, destination,
            Uniforms(2 * length, 2 * sourceOffset, 2 * destinationOffset), 2 * length).GetAwaiter().GetResult();
    }
}

/// <summary>WGSL sources of the double-precision kernels (two words per double, low word first).</summary>
internal static class WebGpuFloat64Kernels
{
    /// <summary>Elements one invocation of a reduction pass folds.</summary>
    public const int ReduceBlock = 64;

    private static readonly ConcurrentDictionary<string, string> Sources = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);

    private const string Parameters = @"
struct Params { n: u32, p0: u32, p1: u32, pad0: u32, s0: u32, s1: u32, s2: u32, s3: u32 }
";

    private static string TwoIn => @"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
" + Parameters + @"@group(0) @binding(3) var<uniform> params: Params;
fn ld_a(i: u32) -> vec2<u32> { return vec2<u32>(a[2u * i], a[2u * i + 1u]); }
fn ld_b(i: u32) -> vec2<u32> { return vec2<u32>(b[2u * i], b[2u * i + 1u]); }
";

    private static string OneIn => @"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
" + Parameters + @"@group(0) @binding(2) var<uniform> params: Params;
fn ld_a(i: u32) -> vec2<u32> { return vec2<u32>(a[2u * i], a[2u * i + 1u]); }
";

    private const string Entry = @"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
";

    private const string Store = @"
    o[2u * i] = y.x;
    o[2u * i + 1u] = y.y;
}
";

    private static string Function(GpuFloat64BinaryOperation operation) => operation switch
    {
        GpuFloat64BinaryOperation.Add => "f64_add",
        GpuFloat64BinaryOperation.Subtract => "f64_sub",
        GpuFloat64BinaryOperation.Multiply => "f64_mul",
        GpuFloat64BinaryOperation.Divide => "f64_div",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
    };

    /// <summary><c>o[i] = a[i] op b[i]</c>.</summary>
    public static string Binary(GpuFloat64BinaryOperation operation)
    {
        string function = Function(operation);
        return Sources.GetOrAdd("binary:" + operation, _ => TwoIn + GpuSoftFloat64.WgslArithmetic + Entry + @"
    let y = " + function + @"(ld_a(i), ld_b(i));" + Store);
    }

    /// <summary><c>o[i] = a[i] op s</c>, the scalar in four 16-bit pieces of the uniform.</summary>
    public static string Scalar(GpuFloat64BinaryOperation operation)
    {
        string function = Function(operation);
        return Sources.GetOrAdd("scalar:" + operation, _ => OneIn + GpuSoftFloat64.WgslArithmetic + Entry + @"
    let s = vec2<u32>(params.s0 | (params.s1 << 16u), params.s2 | (params.s3 << 16u));
    let y = " + function + @"(ld_a(i), s);" + Store);
    }

    /// <summary><c>o[i] = f(a[i])</c>.</summary>
    public static string Unary(GpuFloat64UnaryOperation operation)
    {
        (string expression, bool elementary) = operation switch
        {
            GpuFloat64UnaryOperation.Negate => ("f64_neg(x)", false),
            GpuFloat64UnaryOperation.Abs => ("f64_abs(x)", false),
            GpuFloat64UnaryOperation.Sqrt => ("f64_sqrt(x)", false),
            // Math.Max(x, 0): NaN stays NaN, and anything not above zero (-0 included) becomes +0.
            GpuFloat64UnaryOperation.Relu => ("select(select(vec2<u32>(0u, 0u), x, f64_lt(vec2<u32>(0u, 0u), x)), x, f64_is_nan(x))", false),
            GpuFloat64UnaryOperation.Exp => ("f64_exp_fn(x)", true),
            GpuFloat64UnaryOperation.Log => ("f64_log_fn(x)", true),
            GpuFloat64UnaryOperation.Tanh => ("f64_tanh_fn(x)", true),
            GpuFloat64UnaryOperation.Sigmoid => ("f64_sigmoid_fn(x)", true),
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
        };
        return Sources.GetOrAdd("unary:" + operation, _ => OneIn + GpuSoftFloat64.WgslArithmetic
            + (elementary ? GpuSoftFloat64.WgslElementary : string.Empty) + Entry + @"
    let x = ld_a(i);
    let y = " + expression + @";" + Store);
    }

    /// <summary>Pass of a sum: <c>o[i]</c> is the sum of block <c>i</c> of the <c>p0</c> input elements.</summary>
    public static string SumBlocks => Sources.GetOrAdd("sum", _ => OneIn + GpuSoftFloat64.WgslArithmetic + Entry + @"
    let first = i * " + ReduceBlock + @"u;
    let last = min(first + " + ReduceBlock + @"u, params.p0);
    var y = vec2<u32>(0u, 0u);
    for (var j = first; j < last; j++) { y = f64_add(y, ld_a(j)); }" + Store);

    /// <summary>First pass of a dot product: <c>o[i]</c> is the sum of <c>a[j] * b[j]</c> over block <c>i</c>.</summary>
    public static string DotBlocks => Sources.GetOrAdd("dot", _ => TwoIn + GpuSoftFloat64.WgslArithmetic + Entry + @"
    let first = i * " + ReduceBlock + @"u;
    let last = min(first + " + ReduceBlock + @"u, params.p0);
    var y = vec2<u32>(0u, 0u);
    for (var j = first; j < last; j++) { y = f64_add(y, f64_mul(ld_a(j), ld_b(j))); }" + Store);

    /// <summary>Row-major <c>C[m x n] = A[m x k] * B[k x n]</c>, one invocation per element of C; <c>p0</c> is n and
    /// <c>p1</c> is k.</summary>
    public static string MatrixMultiply => Sources.GetOrAdd("matmul", _ => TwoIn + GpuSoftFloat64.WgslArithmetic + Entry + @"
    let row = i / params.p0;
    let col = i - row * params.p0;
    var y = vec2<u32>(0u, 0u);
    for (var j = 0u; j < params.p1; j++) { y = f64_add(y, f64_mul(ld_a(row * params.p1 + j), ld_b(j * params.p0 + col))); }" + Store);

    /// <summary>Copies <c>n</c> words from offset <c>p0</c> of the source to offset <c>p1</c> of the destination.</summary>
    public static string CopyWords => Sources.GetOrAdd("copy", _ => @"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
" + Parameters + @"@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    o[params.p1 + i] = a[params.p0 + i];
}
");
}
#endif
