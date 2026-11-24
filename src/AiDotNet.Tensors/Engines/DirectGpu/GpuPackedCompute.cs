// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// How a backend runs a packed kernel of <see cref="GpuPackedKernelSource"/>: <paramref name="work"/> invocations over
/// <paramref name="buffers"/> (bound in order; the last is the one written), returning when it has completed. The
/// kernel's parameter words are <c>pn</c> = <paramref name="work"/>, <c>pcount</c>, <c>pa</c>, <c>pb</c> and <c>pc</c>.
/// </summary>
internal delegate void GpuPackedLaunch(GpuPackedKernel kernel, int work, uint count, uint pa, uint pb, uint pc, IGpuBuffer[] buffers);

/// <summary>
/// The typed-interface operations of the element types the word-addressed backends (Vulkan, Metal, WebGPU) compute
/// with packed kernels: argument checks, the launch geometry and the passes of a sum, written once for the three.
/// </summary>
internal static class GpuPackedCompute
{
    // Invocations of an element-wise kernel: one per output word, or per element for the two-word Int64.
    private static int Work(GpuElementType type, int size)
        => GpuPackedKernelSource.Lanes(type) == 0 ? size : GpuPackedKernelSource.Words(type, size);

    private static void RequireDefined(GpuIntegerBinaryOperation operation, GpuElementType type)
    {
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined on {type} buffers.", nameof(operation));
    }

    private static void RequireDefined(GpuIntegerUnaryOperation operation, GpuElementType type)
    {
        if (!GpuIntegerLayout.Defined(operation, type))
            throw new ArgumentException($"{operation} is not defined on {type} buffers.", nameof(operation));
    }

    /// <summary><c>output[i] = a[i] op b[i]</c> for the first <paramref name="size"/> elements.</summary>
    public static void Binary(GpuPackedLaunch launch, GpuIntegerBinaryOperation operation, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int size)
    {
        var type = GpuIntegerLayout.Require(nameof(IGpuIntegerElementwise.BinaryInteger), size, output, a, b);
        RequireDefined(operation, type);
        if (size == 0) return;
        launch(GpuPackedKernelSource.Binary(operation, type), Work(type, size), (uint)size, 0u, 0u, 0u, new[] { a, b, output });
    }

    /// <summary><c>output[i] = op a[i]</c> for the first <paramref name="size"/> elements.</summary>
    public static void Unary(GpuPackedLaunch launch, GpuIntegerUnaryOperation operation, IGpuBuffer a, IGpuBuffer output, int size)
    {
        var type = GpuIntegerLayout.Require(nameof(IGpuIntegerElementwise.UnaryInteger), size, output, a);
        RequireDefined(operation, type);
        if (size == 0) return;
        launch(GpuPackedKernelSource.Unary(operation, type), Work(type, size), (uint)size, 0u, 0u, 0u, new[] { a, output });
    }

    /// <summary><c>output[i] = a[i] op scalar</c>, the scalar truncated to the element type by the kernel.</summary>
    public static void Scalar(GpuPackedLaunch launch, GpuIntegerBinaryOperation operation, IGpuBuffer a, long scalar,
        IGpuBuffer output, int size)
    {
        var type = GpuIntegerLayout.Require(nameof(IGpuIntegerElementwise.ScalarInteger), size, output, a);
        RequireDefined(operation, type);
        if (size == 0) return;
        launch(GpuPackedKernelSource.Scalar(operation, type), Work(type, size), (uint)size, 0u,
            unchecked((uint)scalar), unchecked((uint)(scalar >> 32)), new[] { a, output });
    }

    /// <summary>
    /// The sum of the first <paramref name="size"/> elements, wrapped in the element type.
    /// <paramref name="sumInt32"/> is the backend's Int32 sum: the narrow types fold into Int32 partials it finishes.
    /// </summary>
    public static long Sum(IDirectGpuBackend backend, GpuPackedLaunch launch, IGpuBuffer a, int size, Func<IGpuBuffer, int, long> sumInt32)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        var type = a.ElementType;
        if (type == GpuElementType.Bool || Array.IndexOf(GpuPackedKernelSource.IntegerTypes, type) < 0)
            throw new ArgumentException($"A sum reduces integer buffers, got {type}.", nameof(a));
        if (size < 0 || size > a.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) is outside the buffer's {a.Size} elements.");
        if (size == 0) return 0;
        const int Block = GpuPackedKernelSource.SumBlock;
        var kernel = GpuPackedKernelSource.Sum(type);
        int partials = GpuIntegerLayout.Blocks(size, Block);
        if (type != GpuElementType.Int64)
        {
            // Each partial is a block's sum modulo 2^32; so is their Int32 sum, and its low bits are the sum in the type.
            using var words = backend.AllocateBuffer(partials, GpuElementType.Int32);
            launch(kernel, partials, (uint)size, 0u, 0u, 0u, new[] { a, words });
            return GpuIntegerLayout.Wrap(sumInt32(words, partials), type);
        }

        // Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = backend.AllocateBuffer(partials, GpuElementType.Int64);
        using var odd = backend.AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, Block)), GpuElementType.Int64);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        while (true)
        {
            int outputs = GpuIntegerLayout.Blocks(count, Block);
            launch(kernel, outputs, (uint)count, 0u, 0u, 0u, new[] { current, target });
            if (outputs == 1)
                return BitConverter.ToInt64(backend.DownloadByteBuffer(target, sizeof(long)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = outputs;
        }
    }

    /// <summary><c>output[i] = a[i] op b[i]</c> on 16-bit floats; the output may be an operand.</summary>
    public static void Binary(GpuPackedLaunch launch, GpuNarrowFloatBinaryOperation operation, GpuElementType type,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(IGpuNarrowFloatElementwise.BinaryNarrowFloat), type, size, (a, size), (b, size), (output, size));
        if (size == 0) return;
        launch(GpuPackedKernelSource.Binary(operation, type), Work(type, size), (uint)size, 0u, 0u, 0u, new[] { a, b, output });
    }

    /// <summary>The blocked binary32 sum of the first <paramref name="size"/> 16-bit float elements.</summary>
    public static float Sum(IDirectGpuBackend backend, GpuPackedLaunch launch, GpuElementType type, IGpuBuffer a, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(IGpuNarrowFloatReductions.SumNarrowFloat), type, size, (a, size));
        if (size == 0) return 0f;
        const int Block = GpuNarrowFloatLayout.SumBlock;
        int partials = GpuIntegerLayout.Blocks(size, Block);
        // Partials are binary32 bits, a word each. Two scratch buffers, alternated: a pass never reads the buffer it writes.
        using var even = backend.AllocateBuffer(partials, GpuElementType.Int32);
        using var odd = backend.AllocateBuffer(Math.Max(1, GpuIntegerLayout.Blocks(partials, Block)), GpuElementType.Int32);
        var kernel = GpuPackedKernelSource.FloatSum(type);
        IGpuBuffer current = a;
        IGpuBuffer target = even;
        int count = size;
        while (true)
        {
            int outputs = GpuIntegerLayout.Blocks(count, Block);
            launch(kernel, outputs, (uint)count, 0u, 0u, 0u, new[] { current, target });
            if (outputs == 1)
                return BitConverter.ToSingle(backend.DownloadByteBuffer(target, sizeof(float)), 0);
            current = target;
            target = ReferenceEquals(target, even) ? odd : even;
            count = outputs;
            kernel = GpuPackedKernelSource.FloatSumPartials();
        }
    }

    /// <summary><c>output[i] = a[i] op scalar</c> on 16-bit floats, the scalar a binary32 value; the output may be the operand.</summary>
    public static void Scalar(GpuPackedLaunch launch, GpuNarrowFloatBinaryOperation operation, GpuElementType type,
        IGpuBuffer a, float scalar, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(IGpuNarrowFloatElementwise.ScalarNarrowFloat), type, size, (a, size), (output, size));
        if (size == 0) return;
        launch(GpuPackedKernelSource.Scalar(operation, type), Work(type, size), (uint)size, 0u,
            GpuNarrowFloatLayout.Bits(scalar), 0u, new[] { a, output });
    }

    /// <summary><c>output[i] = op a[i]</c> on 16-bit floats; the output may be the operand.</summary>
    public static void Unary(GpuPackedLaunch launch, GpuNarrowFloatUnaryOperation operation, GpuElementType type,
        IGpuBuffer a, IGpuBuffer output, int size)
    {
        GpuNarrowFloatLayout.Require(nameof(IGpuNarrowFloatElementwise.UnaryNarrowFloat), type, size, (a, size), (output, size));
        if (size == 0) return;
        launch(GpuPackedKernelSource.Unary(operation, type), Work(type, size), (uint)size, 0u, 0u, 0u, new[] { a, output });
    }

    /// <summary><c>c = a b</c> on 16-bit floats (row-major, m x k times k x n).</summary>
    public static void MatrixMultiply(GpuPackedLaunch launch, GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k)
    {
        var (aCount, bCount, cCount) = GpuNarrowFloatLayout.MatrixCounts(m, n, k);
        GpuNarrowFloatLayout.Require(nameof(IGpuNarrowFloatMatrixMultiply.MatrixMultiplyNarrowFloat), type, cCount,
            (a, aCount), (b, bCount), (c, cCount));
        GpuNarrowFloatLayout.RequireDistinctOutput(c, a, b);
        launch(GpuPackedKernelSource.MatrixMultiply(type), Work(type, cCount), (uint)cCount, (uint)n, (uint)k, 0u, new[] { a, b, c });
    }

    /// <summary>
    /// Copies <paramref name="count"/> elements of <paramref name="elementBytes"/> (1 or 2) bytes between two distinct
    /// buffers, at any element offsets.
    /// </summary>
    public static void Copy(GpuPackedLaunch launch, IGpuBuffer source, int sourceOffset, IGpuBuffer destination,
        int destinationOffset, int count, int elementBytes)
    {
        if (count <= 0) return;
        int shift = elementBytes == 1 ? 2 : 1;
        int firstWord = destinationOffset >> shift;
        int lastWord = (int)(((long)destinationOffset + count - 1) >> shift);
        launch(GpuPackedKernelSource.Copy(elementBytes), lastWord - firstWord + 1, (uint)count, (uint)sourceOffset,
            (uint)destinationOffset, 0u, new[] { source, destination });
    }
}
