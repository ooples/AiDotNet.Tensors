// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Double-precision compute over Float64 buffers.
//
// Metal Shading Language has no 64-bit float, so these kernels compute IEEE binary64 in 32-bit integer arithmetic
// (GpuSoftFloat64, the library the WebGPU backend runs): the four arithmetic operations and the square root equal the
// host bit for bit (a NaN result is always the one quiet NaN, not an operand's payload), and exp, log, tanh and the
// logistic function are within a few ulp. A double is two 4-byte words of
// the buffer, low word first.
//
// Emulation costs hundreds of integer operations an element. The backend reports the capability as Emulated, and the
// engine sends doubles here only when a policy asks for device residency; by default they stay on the CPU.
public sealed partial class MetalBackend : IGpuFloat64Reductions, IGpuFloat64Elementwise, IGpuFloat64MatrixMultiply
{
    private const string Float64LibraryName = "Float64";
    private const string Float64ElementaryLibraryName = "Float64Elementary";
    private readonly object _float64LibraryLock = new object();
    private IntPtr _float64Library;
    private IntPtr _float64ElementaryLibrary;
    // A library the Metal compiler rejected: kept so later calls report the gap without compiling it again.
    private Exception? _float64LibraryFailure;
    private Exception? _float64ElementaryLibraryFailure;

    /// <inheritdoc/>
    public bool SupportsFloat64Reductions => IsAvailable && Volatile.Read(ref _float64LibraryFailure) is null;

    /// <inheritdoc/>
    public bool SupportsFloat64Elementwise => IsAvailable && Volatile.Read(ref _float64LibraryFailure) is null;

    /// <inheritdoc/>
    public bool SupportsFloat64MatrixMultiply => IsAvailable && Volatile.Read(ref _float64LibraryFailure) is null;

    private MetalGpuBuffer Float64Operand(IGpuBuffer buffer, int size, string name)
    {
        ThrowIfDisposed();
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.ElementType != GpuElementType.Float64 || buffer is not MetalGpuBuffer typed)
            throw new ArgumentException($"Expected a {GpuElementType.Float64} buffer of this backend, got {buffer.ElementType}.", name);
        if (size < 0 || size > buffer.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {buffer.Size} elements.");
        return typed;
    }

    // The arithmetic library compiles in well under a second; the elementary functions are a larger program, compiled
    // only when one of them is first used. A library the compiler rejects is a capability gap, not a fault: the call
    // throws GpuCapabilityUnavailableException, which the engine answers on the CPU, and the rejection is kept, so a
    // later call neither compiles it again nor reports the capability. The arithmetic library backs every double
    // kernel, so its rejection also turns the Supports* properties off.
    private MetalPipelineState Float64Pipeline(string kernel, bool elementary = false)
    {
        IntPtr library = elementary
            ? Float64Library(ref _float64ElementaryLibrary, ref _float64ElementaryLibraryFailure, Float64ElementaryLibraryName,
                () => _shaderLibrary.CompileLibrary(Float64ElementaryLibraryName, MetalFloat64Kernels.ElementarySource))
            : Float64Library(ref _float64Library, ref _float64LibraryFailure, Float64LibraryName,
                () => _shaderLibrary.CompileLibrary(Float64LibraryName, MetalFloat64Kernels.ArithmeticSource));
        return GetPipeline(elementary ? Float64ElementaryLibraryName : Float64LibraryName, library, kernel);
    }

    private IntPtr Float64Library(ref IntPtr library, ref Exception? failure, string name, Func<IntPtr> compile)
    {
        if (library == IntPtr.Zero)
        {
            lock (_float64LibraryLock)
            {
                if (library == IntPtr.Zero && failure is null)
                {
                    try
                    {
                        library = compile();
                    }
                    catch (InvalidOperationException rejected)
                    {
                        Volatile.Write(ref failure, rejected);
                    }
                }

                if (failure is { } rejection)
                    throw new GpuCapabilityUnavailableException(
                        $"The Metal compiler rejected the double-precision library '{name}': {rejection.Message}", rejection);
            }
        }

        return library;
    }

    // Buffers bind at 0..n-1 and the 32-bit parameters after them; `work` invocations are launched. The last buffer is
    // the one the kernel writes.
    //
    // Buffers passed to separate arguments of one kernel cannot alias in Metal Shading Language, read-only ones
    // included (Metal Shading Language Specification, "Buffers ... cannot alias"), yet the IGpuFloat64* contracts let a
    // caller pass one buffer twice: Dot(a, a), Unary(op, a, a), a copy within one buffer. An argument whose storage an
    // earlier argument already binds therefore binds a copy of it; when that argument is the output, the copy is
    // copied back after the kernel (it starts as the output's contents, so elements the kernel leaves alone keep
    // theirs). Each launch completes before it returns, so the copies are released right after.
    private void LaunchFloat64(MetalPipelineState pipeline, int work, MetalGpuBuffer[] buffers, params uint[] parameters)
    {
        int last = buffers.Length - 1;
        MetalGpuBuffer[]? bound = null;
        List<IGpuBuffer>? copies = null;
        try
        {
            for (int i = 1; i < buffers.Length; i++)
            {
                if (!SharesStorageWithAnEarlierArgument(buffers, i)) continue;
                var copy = AllocateBuffer(buffers[i].Size, GpuElementType.Float64);
                (copies ??= new List<IGpuBuffer>()).Add(copy);
                var separate = (MetalGpuBuffer)copy;
                CopyFloat64(buffers[i], 0, separate, 0, buffers[i].Size);
                (bound ??= (MetalGpuBuffer[])buffers.Clone())[i] = separate;
            }

            DispatchFloat64(pipeline, work, bound ?? buffers, parameters);
            if (bound is not null && !ReferenceEquals(bound[last], buffers[last]))
                CopyFloat64(bound[last], 0, buffers[last], 0, buffers[last].Size);
        }
        finally
        {
            if (copies is not null)
                foreach (var copy in copies) copy.Dispose();
        }
    }

    // Views of one allocation share its handle, so equal handles are treated as overlapping storage.
    private static bool SharesStorageWithAnEarlierArgument(MetalGpuBuffer[] buffers, int index)
    {
        for (int j = 0; j < index; j++)
            if (ReferenceEquals(buffers[j], buffers[index]) || buffers[j].Handle == buffers[index].Handle) return true;
        return false;
    }

    private void DispatchFloat64(MetalPipelineState pipeline, int work, MetalGpuBuffer[] buffers, uint[] parameters)
    {
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(work);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        int index = 0;
        foreach (var buffer in buffers) encoder.SetBuffer(buffer, index++);
        foreach (uint parameter in parameters) encoder.SetBytes(parameter, (ulong)index++);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <inheritdoc/>
    public void BinaryFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        var left = Float64Operand(a, size, nameof(a));
        var right = Float64Operand(b, size, nameof(b));
        var destination = Float64Operand(output, size, nameof(output));
        string kernel = "f64k_" + MetalFloat64Kernels.Name(operation);
        if (size == 0) return;
        LaunchFloat64(Float64Pipeline(kernel), size, new[] { left, right, destination }, (uint)size);
    }

    /// <inheritdoc/>
    public void UnaryFloat64(GpuFloat64UnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        var destination = Float64Operand(output, size, nameof(output));
        string kernel = "f64k_" + MetalFloat64Kernels.Name(operation, out bool elementary);
        if (size == 0) return;
        LaunchFloat64(Float64Pipeline(kernel, elementary), size, new[] { input, destination }, (uint)size);
    }

    /// <inheritdoc/>
    public void ScalarFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, double scalar, IGpuBuffer output, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        var destination = Float64Operand(output, size, nameof(output));
        string kernel = "f64k_scalar_" + MetalFloat64Kernels.Name(operation);
        if (size == 0) return;
        ulong bits = unchecked((ulong)BitConverter.DoubleToInt64Bits(scalar));
        LaunchFloat64(Float64Pipeline(kernel), size, new[] { input, destination }, (uint)size, (uint)bits, (uint)(bits >> 32));
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

    // Each pass folds blocks of MetalFloat64Kernels.ReduceBlock elements into one partial each, until one value is left.
    // The first pass multiplies by `second` when it is given (the dot product); every later pass sums partials.
    private double ReduceFloat64(MetalGpuBuffer first, MetalGpuBuffer? second, int size)
    {
        int block = MetalFloat64Kernels.ReduceBlock;
        int partials = (size + block - 1) / block;
        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = AllocateBuffer(partials, GpuElementType.Float64);
        using var odd = AllocateBuffer(Math.Max(1, (partials + block - 1) / block), GpuElementType.Float64);
        MetalGpuBuffer current = first;
        MetalGpuBuffer target = Float64Operand(even, partials, nameof(even));
        MetalGpuBuffer other = Float64Operand(odd, 0, nameof(odd));
        int count = size;
        bool firstPass = true;
        while (true)
        {
            int outputs = (count + block - 1) / block;
            if (firstPass && second is not null)
                LaunchFloat64(Float64Pipeline("f64k_dot"), outputs, new[] { current, second, target }, (uint)outputs, (uint)count);
            else
                LaunchFloat64(Float64Pipeline("f64k_sum"), outputs, new[] { current, target }, (uint)outputs, (uint)count);
            firstPass = false;
            if (outputs == 1)
                return BitConverter.ToDouble(DownloadByteBuffer(target, sizeof(double)), 0);
            current = target;
            (target, other) = (other, target);
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
            || product.Handle == left.Handle || product.Handle == right.Handle)
            throw new ArgumentException("The product buffer must not be one of the operands.", nameof(c));
        int elements = checked(m * n);
        LaunchFloat64(Float64Pipeline("f64k_matmul"), elements, new[] { left, right, product }, (uint)elements, (uint)n, (uint)k);
    }

    // A device copy of doubles moves their words as integers: a float load and store need not keep every bit pattern.
    private void CopyFloat64(MetalGpuBuffer source, int sourceOffset, MetalGpuBuffer destination, int destinationOffset, int length)
    {
        LaunchFloat64(Float64Pipeline("f64k_copy"), 2 * length, new[] { source, destination },
            (uint)(2 * length), (uint)(2 * sourceOffset), (uint)(2 * destinationOffset));
    }
}

/// <summary>Metal Shading Language sources of the double-precision kernels (two words per double, low word first).</summary>
internal static class MetalFloat64Kernels
{
    /// <summary>Elements one invocation of a reduction pass folds.</summary>
    public const int ReduceBlock = 64;

    private static readonly Lazy<string> Arithmetic = new Lazy<string>(BuildArithmetic);
    private static readonly Lazy<string> Elementary = new Lazy<string>(BuildElementary);

    /// <summary>Arithmetic, scalar, sign, square root, relu, reduction, matrix multiply and copy kernels.</summary>
    public static string ArithmeticSource => Arithmetic.Value;

    /// <summary>The exp, log, tanh and logistic kernels.</summary>
    public static string ElementarySource => Elementary.Value;

    /// <summary>The kernel-name suffix of a binary operation, which is also its library function's.</summary>
    public static string Name(GpuFloat64BinaryOperation operation) => operation switch
    {
        GpuFloat64BinaryOperation.Add => "add",
        GpuFloat64BinaryOperation.Subtract => "sub",
        GpuFloat64BinaryOperation.Multiply => "mul",
        GpuFloat64BinaryOperation.Divide => "div",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
    };

    /// <summary>The kernel-name suffix of a unary operation, and which library holds it.</summary>
    public static string Name(GpuFloat64UnaryOperation operation, out bool elementary)
    {
        elementary = operation is GpuFloat64UnaryOperation.Exp or GpuFloat64UnaryOperation.Log
            or GpuFloat64UnaryOperation.Tanh or GpuFloat64UnaryOperation.Sigmoid;
        return operation switch
        {
            GpuFloat64UnaryOperation.Negate => "neg",
            GpuFloat64UnaryOperation.Abs => "abs",
            GpuFloat64UnaryOperation.Sqrt => "sqrt",
            GpuFloat64UnaryOperation.Relu => "relu",
            GpuFloat64UnaryOperation.Exp => "exp",
            GpuFloat64UnaryOperation.Log => "log",
            GpuFloat64UnaryOperation.Tanh => "tanh",
            GpuFloat64UnaryOperation.Sigmoid => "sigmoid",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
        };
    }

    private const string Header = @"
#include <metal_stdlib>
using namespace metal;
";

    private const string Load = @"
static inline uint2 f64k_load(device const uint* p, uint i) { return uint2(p[2u * i], p[2u * i + 1u]); }
static inline void f64k_store(device uint* p, uint i, uint2 v) { p[2u * i] = v.x; p[2u * i + 1u] = v.y; }
";

    private static string Unary(string name, string expression) => @"
kernel void f64k_" + name + @"(
    device const uint* a [[buffer(0)]],
    device uint* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    uint2 x = f64k_load(a, gid);
    f64k_store(o, gid, " + expression + @");
}
";

    private static string BuildArithmetic()
    {
        var source = new StringBuilder(Header).Append(GpuSoftFloat64.MetalArithmetic).Append(Load);
        foreach (string op in new[] { "add", "sub", "mul", "div" })
        {
            source.Append(@"
kernel void f64k_" + op + @"(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* o [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    f64k_store(o, gid, f64_" + op + @"(f64k_load(a, gid), f64k_load(b, gid)));
}

kernel void f64k_scalar_" + op + @"(
    device const uint* a [[buffer(0)]],
    device uint* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& scalarLo [[buffer(3)]],
    constant uint& scalarHi [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    f64k_store(o, gid, f64_" + op + @"(f64k_load(a, gid), uint2(scalarLo, scalarHi)));
}
");
        }

        source.Append(Unary("neg", "f64_neg(x)"));
        source.Append(Unary("abs", "f64_abs(x)"));
        source.Append(Unary("sqrt", "f64_sqrt(x)"));
        // Math.Max(x, 0): NaN stays NaN, and anything not above zero (-0 included) becomes +0.
        source.Append(Unary("relu", "(f64_is_nan(x) || f64_lt(uint2(0u, 0u), x)) ? x : uint2(0u, 0u)"));
        source.Append(@"
// Pass of a sum: o[gid] is the sum of block gid of the `count` input elements.
kernel void f64k_sum(
    device const uint* a [[buffer(0)]],
    device uint* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    uint first = gid * " + ReduceBlock + @"u;
    uint last = min(first + " + ReduceBlock + @"u, count);
    uint2 y = uint2(0u, 0u);
    for (uint j = first; j < last; j++) { y = f64_add(y, f64k_load(a, j)); }
    f64k_store(o, gid, y);
}

// First pass of a dot product: o[gid] is the sum of a[j] * b[j] over block gid.
kernel void f64k_dot(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* o [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    uint first = gid * " + ReduceBlock + @"u;
    uint last = min(first + " + ReduceBlock + @"u, count);
    uint2 y = uint2(0u, 0u);
    for (uint j = first; j < last; j++) { y = f64_add(y, f64_mul(f64k_load(a, j), f64k_load(b, j))); }
    f64k_store(o, gid, y);
}

// Row-major C[m x cols] = A[m x inner] * B[inner x cols], one invocation per element of C.
kernel void f64k_matmul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* o [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    uint row = gid / cols;
    uint col = gid - row * cols;
    uint2 y = uint2(0u, 0u);
    for (uint j = 0u; j < inner; j++) { y = f64_add(y, f64_mul(f64k_load(a, row * inner + j), f64k_load(b, j * cols + col))); }
    f64k_store(o, gid, y);
}

// Copies n words from word offset srcOffset of the source to word offset dstOffset of the destination.
kernel void f64k_copy(
    device const uint* a [[buffer(0)]],
    device uint* o [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& srcOffset [[buffer(3)]],
    constant uint& dstOffset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    o[dstOffset + gid] = a[srcOffset + gid];
}
");
        return source.ToString();
    }

    private static string BuildElementary()
    {
        var source = new StringBuilder(Header).Append(GpuSoftFloat64.MetalArithmetic).Append(GpuSoftFloat64.MetalElementary).Append(Load);
        source.Append(Unary("exp", "f64_exp_fn(x)"));
        source.Append(Unary("log", "f64_log_fn(x)"));
        source.Append(Unary("tanh", "f64_tanh_fn(x)"));
        source.Append(Unary("sigmoid", "f64_sigmoid_fn(x)"));
        return source.ToString();
    }
}
