// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Double-precision compute over Float64 buffers.
//
// A Vulkan storage buffer here is an array of 4-byte words, so a double is two of them, low word first (the layout the
// parity kernels' double probe already reads). The kernels compute in GLSL double behind shaderFloat64 and are compiled
// at run time, so the whole capability needs that device feature and libshaderc; a device without either reports the
// interfaces unsupported and the engine keeps double ops on the CPU.
//
// GLSL has no double exp / log / hyperbolic functions: those come from the parity kernels' double library
// (VulkanTorchParityMath). The four arithmetic operations and the square root are single IEEE operations and are
// declared `precise`, so the compiler may not contract a multiply into a following add or rewrite a division.
public sealed unsafe partial class VulkanBackend : IGpuFloat64Reductions, IGpuFloat64Elementwise, IGpuFloat64MatrixMultiply,
    IGpuFloat64UnaryAvailability
{
    // Every unary kernel is built with the rest of the double kernels here: none can be missing alone.
    bool IGpuFloat64UnaryAvailability.SupportsFloat64Unary(GpuFloat64UnaryOperation operation) => SupportsFloat64Elementwise;

    /// <inheritdoc/>
    public bool SupportsFloat64Reductions => SupportsFloat64Compute;

    /// <inheritdoc/>
    public bool SupportsFloat64Elementwise => SupportsFloat64Compute;

    /// <inheritdoc/>
    public bool SupportsFloat64MatrixMultiply => SupportsFloat64Compute;

    // The kernels must give the host's bits: subnormal results kept (2^-1022 * 0.5 is 2^-1023, not 0) and every result
    // rounded to nearest, ties to even. No float-controls bit promises all of that: DenormPreserve does not cover
    // division, and nothing a device reports fixes the rounding mode of a shader that does not request one. So the
    // compiled kernels are measured once on every device, whatever it reports: a device whose kernels flush or round
    // another way computes doubles on the CPU and keeps double storage.
    private bool SupportsFloat64Compute => SupportsFloat64Storage && KeepsSubnormals();

    private readonly object _subnormalProbeLock = new object();
    private bool? _keepsSubnormals;

    private bool KeepsSubnormals()
    {
        if (_keepsSubnormals is bool known) return known;
        lock (_subnormalProbeLock)
        {
            if (_keepsSubnormals is bool settled) return settled;
            bool kept;
            try
            {
                kept = MeasureSubnormals();
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is NotSupportedException)
            {
                // A kernel that cannot be built or run here cannot compute doubles either.
                kept = false;
            }
            _keepsSubnormals = kept;
            return kept;
        }
    }

    // Runs each arithmetic kernel on operands whose exact results are subnormal or come from subnormals, and on ones
    // whose results fall between two doubles (where round-to-nearest-even differs from every other mode: 1 + 3 * 2^-53
    // is a tie that goes up to 1 + 2^-51, 1 + 2^-53 a tie that stays at 1), and compares the bits with the host's.
    // Every double kernel here is built from these operations and the same header.
    internal bool MeasureSubnormals()
    {
        const double smallestNormal = 2.2250738585072014E-308, smallestSubnormal = double.Epsilon;
        const double halfUlp = 1.1102230246251565E-16;   // 2^-53
        var left = new[]
        {
            smallestNormal, smallestSubnormal, 3 * smallestSubnormal, smallestNormal, 1.0E-300,
            1.0, 1.0, 1.0, 0.1, 1.0, 0.7, -1.0, 1.0 + 4 * halfUlp, 2.0,
        };
        var right = new[]
        {
            0.5, smallestSubnormal, smallestSubnormal, 4.0, 1.0E-10,
            3 * halfUlp, halfUlp, 1.5 * halfUlp, 0.2, 3.0, 0.1, 3 * halfUlp, 1.0 + 2 * halfUlp, 3.0,
        };
        int n = left.Length;
        using var a = AllocateBuffer(n, GpuElementType.Float64);
        using var b = AllocateBuffer(n, GpuElementType.Float64);
        using var output = AllocateBuffer(n, GpuElementType.Float64);
        this.UploadBuffer<double>(a, left);
        this.UploadBuffer<double>(b, right);
        var actual = new double[n];
        foreach (GpuFloat64BinaryOperation operation in new[]
        {
            GpuFloat64BinaryOperation.Multiply, GpuFloat64BinaryOperation.Add,
            GpuFloat64BinaryOperation.Subtract, GpuFloat64BinaryOperation.Divide,
        })
        {
            ParityDispatch(operation + "_f64", VulkanFloat64Kernels.Binary(operation), n, new[] { (uint)n }, a, b, output);
            this.DownloadBuffer<double>(output, actual);
            for (int i = 0; i < n; i++)
            {
                double expected = operation switch
                {
                    GpuFloat64BinaryOperation.Multiply => left[i] * right[i],
                    GpuFloat64BinaryOperation.Add => left[i] + right[i],
                    GpuFloat64BinaryOperation.Subtract => left[i] - right[i],
                    _ => left[i] / right[i],
                };
                if (BitConverter.DoubleToInt64Bits(expected) != BitConverter.DoubleToInt64Bits(actual[i])) return false;
            }
        }
        return true;
    }

    private IGpuBuffer Float64Operand(IGpuBuffer buffer, int size, string name)
    {
        EnsureInitialized();
        if (!SupportsFloat64Compute)
            throw new GpuCapabilityUnavailableException("Double-precision kernels are not available on this Vulkan device.");
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.ElementType != GpuElementType.Float64 || buffer is not VulkanGpuBuffer)
            throw new ArgumentException($"Expected a {GpuElementType.Float64} buffer of this backend, got {buffer.ElementType}.", name);
        if (size < 0 || size > buffer.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {buffer.Size} elements.");
        return buffer;
    }

    /// <inheritdoc/>
    public void BinaryFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        var left = Float64Operand(a, size, nameof(a));
        var right = Float64Operand(b, size, nameof(b));
        var destination = Float64Operand(output, size, nameof(output));
        string glsl = VulkanFloat64Kernels.Binary(operation);
        if (size == 0) return;
        ParityDispatch(operation + "_f64", glsl, size, new[] { (uint)size }, left, right, destination);
    }

    /// <inheritdoc/>
    public void UnaryFloat64(GpuFloat64UnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        var destination = Float64Operand(output, size, nameof(output));
        string glsl = VulkanFloat64Kernels.Unary(operation);
        if (size == 0) return;
        ParityDispatch(operation + "_f64", glsl, size, new[] { (uint)size }, input, destination);
    }

    /// <inheritdoc/>
    public void ScalarFloat64(GpuFloat64BinaryOperation operation, IGpuBuffer a, double scalar, IGpuBuffer output, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        var destination = Float64Operand(output, size, nameof(output));
        string glsl = VulkanFloat64Kernels.Scalar(operation);
        if (size == 0) return;
        ParityDispatch(operation + "_scalar_f64", glsl, size, new[] { (uint)size, 0u, Low(scalar), High(scalar) }, input, destination);
    }

    /// <inheritdoc/>
    public double SumFloat64(IGpuBuffer a, int size)
    {
        var input = Float64Operand(a, size, nameof(a));
        return size == 0 ? 0.0 : ReduceFloat64(input, null, size);
    }

    /// <inheritdoc/>
    public double MaxFloat64(IGpuBuffer a, int size) => ExtremeFloat64(a, size, GpuFloat64Reduction.Max);

    /// <inheritdoc/>
    public double MinFloat64(IGpuBuffer a, int size) => ExtremeFloat64(a, size, GpuFloat64Reduction.Min);

    private double ExtremeFloat64(IGpuBuffer a, int size, GpuFloat64Reduction reduction)
    {
        var input = Float64Operand(a, size, nameof(a));
        if (size == 0) throw new ArgumentException("An empty range has no minimum or maximum.", nameof(size));
        return ReduceFloat64(input, null, size, reduction);
    }

    /// <inheritdoc/>
    public double DotFloat64(IGpuBuffer a, IGpuBuffer b, int size)
    {
        var left = Float64Operand(a, size, nameof(a));
        var right = Float64Operand(b, size, nameof(b));
        return size == 0 ? 0.0 : ReduceFloat64(left, right, size);
    }

    // Each pass folds blocks of VulkanFloat64Kernels.ReduceBlock elements into one partial each, until one value is
    // left. The first pass multiplies by `second` when it is given (the dot product); every later pass sums partials.
    private double ReduceFloat64(IGpuBuffer first, IGpuBuffer? second, int size, GpuFloat64Reduction reduction = GpuFloat64Reduction.Sum)
    {
        int block = VulkanFloat64Kernels.ReduceBlock;
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
                ParityDispatch("dot_f64", VulkanFloat64Kernels.DotBlocks, outputs, new[] { (uint)outputs, (uint)count }, current, second, target);
            else
                ParityDispatch(reduction == GpuFloat64Reduction.Sum ? "sum_f64" : reduction == GpuFloat64Reduction.Max ? "max_f64" : "min_f64",
                    reduction == GpuFloat64Reduction.Sum ? VulkanFloat64Kernels.SumBlocks : VulkanFloat64Kernels.ExtremeBlocks(reduction == GpuFloat64Reduction.Max),
                    outputs, new[] { (uint)outputs, (uint)count }, current, target);
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
        if (ReferenceEquals(product, left) || ReferenceEquals(product, right))
            throw new ArgumentException("The product buffer must not be one of the operands.", nameof(c));
        int elements = checked(m * n);
        ParityDispatch("matmul_f64", VulkanFloat64Kernels.MatrixMultiply, elements,
            new[] { (uint)elements, (uint)n, (uint)k }, left, right, product);
    }
}

/// <summary>GLSL sources of the double-precision kernels (two words per double, low word first).</summary>
internal static class VulkanFloat64Kernels
{
    /// <summary>Elements one invocation of a reduction pass folds.</summary>
    public const int ReduceBlock = 64;

    private const string TwoIn = @"
layout(set = 0, binding = 0) readonly buffer A { uint a[]; };
layout(set = 0, binding = 1) readonly buffer B { uint b[]; };
layout(set = 0, binding = 2) writeonly buffer O { uint o[]; };
";

    private const string OneIn = @"
layout(set = 0, binding = 0) readonly buffer A { uint a[]; };
layout(set = 0, binding = 1) writeonly buffer O { uint o[]; };
";

    private const string Store = @"
    uvec2 bits = unpackDouble2x32(y);
    o[2u * i] = bits.x;
    o[2u * i + 1u] = bits.y;
";

    private static string Symbol(GpuFloat64BinaryOperation operation) => operation switch
    {
        GpuFloat64BinaryOperation.Add => "+",
        GpuFloat64BinaryOperation.Subtract => "-",
        GpuFloat64BinaryOperation.Multiply => "*",
        GpuFloat64BinaryOperation.Divide => "/",
        _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
    };

    /// <summary><c>o[i] = a[i] op b[i]</c>.</summary>
    public static string Binary(GpuFloat64BinaryOperation operation)
    {
        string symbol = Symbol(operation);
        return VulkanTorchParityElementwiseKernels.DoubleKernel("binary:" + operation, TwoIn + @"
layout(push_constant) uniform P { uint n; };
", @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    double x = packDouble2x32(uvec2(a[2u * i], a[2u * i + 1u]));
    double z = packDouble2x32(uvec2(b[2u * i], b[2u * i + 1u]));
    precise double y = x " + symbol + @" z;
" + Store + @"
}
");
    }

    /// <summary><c>o[i] = a[i] op s</c>, the scalar passed as two push-constant words.</summary>
    // The two words sit on an 8-byte boundary, as every double push constant of the parity kernels does: packed straight
    // after `n` (offsets 4 and 8) the local AMD driver handed them to the shader swapped.
    public static string Scalar(GpuFloat64BinaryOperation operation)
    {
        string symbol = Symbol(operation);
        return VulkanTorchParityElementwiseKernels.DoubleKernel("scalar:" + operation, OneIn + @"
layout(push_constant) uniform P { uint n; uint pad0; uint scalarLo; uint scalarHi; };
", @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    double x = packDouble2x32(uvec2(a[2u * i], a[2u * i + 1u]));
    double z = packDouble2x32(uvec2(scalarLo, scalarHi));
    precise double y = x " + symbol + @" z;
" + Store + @"
}
");
    }

    /// <summary><c>o[i] = f(a[i])</c>.</summary>
    public static string Unary(GpuFloat64UnaryOperation operation)
    {
        string expression = operation switch
        {
            GpuFloat64UnaryOperation.Negate => "-x",
            GpuFloat64UnaryOperation.Abs => "abs(x)",
            GpuFloat64UnaryOperation.Sqrt => "f64_sqrt(x)",
            GpuFloat64UnaryOperation.Exp => "d_exp(x)",
            GpuFloat64UnaryOperation.Log => "d_log(x)",
            // sinh / cosh overflow to infinity long after tanh has reached 1 in double (|x| > 19.1).
            GpuFloat64UnaryOperation.Tanh => "(abs(x) > 20.0 ? (x < 0.0 ? -1.0 : 1.0) : d_sinh(x) / d_cosh(x))",
            GpuFloat64UnaryOperation.Sigmoid => "x < 0.0 ? d_exp(x) / (1.0 + d_exp(x)) : 1.0 / (1.0 + d_exp(-x))",
            // Math.Max(x, 0): NaN stays NaN, and -0 becomes +0.
            GpuFloat64UnaryOperation.Relu => "(x != x ? x : (x > 0.0 ? x : 0.0))",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown double-precision operation."),
        };
        return VulkanTorchParityElementwiseKernels.DoubleKernel("unary:" + operation, OneIn + @"
layout(push_constant) uniform P { uint n; };
", SqrtSource + @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    double x = packDouble2x32(uvec2(a[2u * i], a[2u * i + 1u]));
    precise double y = " + expression + @";
" + Store + @"
}
");
    }

    // The driver's double sqrt is not required to be correctly rounded. One step from its estimate, with the residual
    // x - r*r formed exactly (Dekker product, `precise`), brings an estimate that is an ulp or two off to within one ulp.
    // That step alone still rounds wrong where the root lies just below a power of two (sqrt(16777215.999999998) gave
    // 4096, not 4095.9999999999995), so Tuckerman's test then decides the last ulp: y is the rounded root exactly when
    // y * prev(y) < x <= y * next(y), each side's sign taken from an exact residual. The products must neither overflow
    // nor underflow, so an argument above 2^1000 is taken as 4 * (x / 4) and one below 2^-900 (a subnormal included)
    // as 2^-1000 * (x * 2^1000); every scaling is exact. Zero, infinity, NaN and negative arguments keep the driver's
    // answer.
    private const string SqrtSource = @"
double f64_sqrt_corrected(double x)
{
    double r = sqrt(x);
    if (!(x > 0.0) || r != r || (r - r) != 0.0 || r == 0.0) return r;
    precise double t = 134217729.0 * r;
    precise double rh = t - (t - r);
    precise double rl = r - rh;
    precise double p = r * r;
    precise double e = ((rh * rh - p) + 2.0 * rh * rl) + rl * rl;
    precise double residual = (x - p) - e;
    precise double corrected = r + residual / (2.0 * r);
    return corrected;
}

// The neighbours of a positive finite double, by its bit pattern.
double f64_next_up(double y)
{
    uvec2 b = unpackDouble2x32(y);
    b.x += 1u;
    if (b.x == 0u) b.y += 1u;
    return packDouble2x32(b);
}

double f64_next_down(double y)
{
    uvec2 b = unpackDouble2x32(y);
    if (b.x == 0u) b.y -= 1u;
    b.x -= 1u;
    return packDouble2x32(b);
}

// x - a * b with its exact sign, for a product within a factor of two of x (so x - p is exact) that neither
// overflows nor underflows (so the Dekker error term is exact).
double f64_residual(double x, double a, double b)
{
    precise double p = a * b;
    precise double ta = 134217729.0 * a;
    precise double ah = ta - (ta - a);
    precise double al = a - ah;
    precise double tb = 134217729.0 * b;
    precise double bh = tb - (tb - b);
    precise double bl = b - bh;
    precise double e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    precise double r = (x - p) - e;
    return r;
}

// The correctly rounded root of x in [2^-900, 2^1022]. A root is never a rounding midpoint, and x minus either
// product is a whole multiple of the squared ulp, so the two comparisons decide the rounding with no tie.
double f64_sqrt_rounded(double x)
{
    double y = f64_sqrt_corrected(x);
    double down = f64_next_down(y);
    if (f64_residual(x, y, down) <= 0.0) return down;
    double up = f64_next_up(y);
    if (f64_residual(x, y, up) > 0.0) return up;
    return y;
}

double f64_sqrt(double x)
{
    if (!(x > 0.0) || x != x || (x - x) != 0.0) return sqrt(x);
    if (x > ldexp(1.0, 1000)) return 2.0 * f64_sqrt_rounded(0.25 * x);
    if (x < ldexp(1.0, -900)) return ldexp(f64_sqrt_rounded(x * ldexp(1.0, 1000)), -500);
    return f64_sqrt_rounded(x);
}
";

    /// <summary>Pass of a sum: <c>o[i]</c> is the sum of block <c>i</c> of the <c>count</c> input elements.</summary>
    public static string SumBlocks => VulkanTorchParityElementwiseKernels.DoubleKernel("sum_blocks", OneIn + @"
layout(push_constant) uniform P { uint n; uint count; };
", @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    uint from = i * " + ReduceBlock + @"u;
    uint to = min(from + " + ReduceBlock + @"u, count);
    double y = 0.0;
    for (uint j = from; j < to; j++) y += packDouble2x32(uvec2(a[2u * j], a[2u * j + 1u]));
" + Store + @"
}
");

    /// <summary>
    /// Pass of a maximum or minimum: <c>o[i]</c> is that of block <c>i</c> of the <c>count</c> input elements, a NaN
    /// when the block holds one. A block is never empty.
    /// </summary>
    public static string ExtremeBlocks(bool largest) => VulkanTorchParityElementwiseKernels.DoubleKernel(largest ? "max_blocks" : "min_blocks", OneIn + @"
layout(push_constant) uniform P { uint n; uint count; };
", @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    uint from = i * " + ReduceBlock + @"u;
    uint to = min(from + " + ReduceBlock + @"u, count);
    double y = packDouble2x32(uvec2(a[2u * from], a[2u * from + 1u]));
    for (uint j = from + 1u; j < to; j++)
    {
        double v = packDouble2x32(uvec2(a[2u * j], a[2u * j + 1u]));
        if (y == y && (v != v || v " + (largest ? ">" : "<") + @" y)) y = v;
    }
" + Store + @"
}
");

    /// <summary>First pass of a dot product: <c>o[i]</c> is the sum of <c>a[j] * b[j]</c> over block <c>i</c>.</summary>
    public static string DotBlocks => VulkanTorchParityElementwiseKernels.DoubleKernel("dot_blocks", TwoIn + @"
layout(push_constant) uniform P { uint n; uint count; };
", @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    uint from = i * " + ReduceBlock + @"u;
    uint to = min(from + " + ReduceBlock + @"u, count);
    double y = 0.0;
    for (uint j = from; j < to; j++)
    {
        precise double product = packDouble2x32(uvec2(a[2u * j], a[2u * j + 1u])) * packDouble2x32(uvec2(b[2u * j], b[2u * j + 1u]));
        y += product;
    }
" + Store + @"
}
");

    /// <summary>Row-major <c>C[m x n] = A[m x k] * B[k x n]</c>, one invocation per element of C.</summary>
    public static string MatrixMultiply => VulkanTorchParityElementwiseKernels.DoubleKernel("matmul", TwoIn + @"
layout(push_constant) uniform P { uint n; uint cols; uint inner; };
", @"
void main()
{
" + VulkanTorchParityElementwiseKernels.IndexGuard + @"
    uint row = i / cols;
    uint col = i - row * cols;
    double y = 0.0;
    for (uint j = 0u; j < inner; j++)
    {
        uint left = row * inner + j;
        uint right = j * cols + col;
        precise double product = packDouble2x32(uvec2(a[2u * left], a[2u * left + 1u])) * packDouble2x32(uvec2(b[2u * right], b[2u * right + 1u]));
        y += product;
    }
" + Store + @"
}
");
}
