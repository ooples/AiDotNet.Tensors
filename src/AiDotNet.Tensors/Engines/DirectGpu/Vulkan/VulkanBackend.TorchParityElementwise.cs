// Copyright (c) AiDotNet. All rights reserved.
// Launches for the Vulkan PyTorch-parity elementwise kernels (VulkanTorchParityElementwiseKernels), #1127.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <remarks>
/// Most of these kernels compute in double, as the CPU reference does, and a few in int64: those need the device's
/// shaderFloat64 / shaderInt64 features (and the runtime GLSL compiler). On a device without them the call throws
/// <see cref="NotSupportedException"/>, which the engine records as that op's host fallback.
/// </remarks>
public sealed unsafe partial class VulkanBackend : ITorchParityElementwiseBackend
{
    // Serializes a validating launch with the merge of its flag slots (see VulkanTorchParityElementwiseKernels).
    private readonly object _parityFlagLock = new object();
    private IGpuBuffer? _parityFlagSlots;

    /// <summary>True when the device can run the parity kernels that compute in double.</summary>
    internal bool SupportsTorchParityDoubles => IsGlslCompilerAvailable && _device.SupportsShaderFloat64;

    /// <summary>True when the device can run the parity kernels that compute in int64.</summary>
    internal bool SupportsTorchParityInt64 => IsGlslCompilerAvailable && _device.SupportsShaderInt64;

    private void RequireParityDoubles(string kernel)
    {
        if (!_device.SupportsShaderFloat64)
            throw new NotSupportedException($"Vulkan parity kernel '{kernel}' computes in double; device '{DeviceName}' lacks shaderFloat64.");
    }

    private void RequireParityInt64(string kernel)
    {
        if (!_device.SupportsShaderInt64)
            throw new NotSupportedException($"Vulkan parity kernel '{kernel}' computes in int64; device '{DeviceName}' lacks shaderInt64.");
    }

    private static void RequireParityCapacity(IGpuBuffer buffer, long requiredElements, string parameter, string formula)
    {
        if (buffer is null) throw new ArgumentNullException(parameter);
        if (buffer.Size < requiredElements)
            throw new ArgumentException(
                $"{parameter} buffer too small: capacity={buffer.Size} elements, required={requiredElements} ({formula}).", parameter);
    }

    // One dispatch of `threads` invocations over the given buffers (bound in order), with integer push constants.
    private void ParityDispatch(string kernel, string glsl, int threads, uint[] push, int localSize, params IGpuBuffer[] buffers)
    {
        EnsureInitialized();
        if (threads <= 0) return;
        uint pushSize = (uint)(push.Length * sizeof(uint));
        var pipeline = GetOrCreateGlslPipeline(glsl, buffers.Length, pushSize);
        if (pipeline is null)
            throw new InvalidOperationException(
                $"Vulkan parity kernel '{kernel}' could not be built: {_glslCompiler?.LastError ?? "the runtime GLSL compiler (libshaderc) is unavailable"}");
        var storages = new VulkanBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) storages[i] = AsVulkan(buffers[i]).Storage;
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(storages);
            RecordAndExecuteWithPushData(pipeline, threads, push, pushSize, threadRes, localSize);
        }
    }

    private void ParityDispatch(string kernel, string glsl, int threads, uint[] push, params IGpuBuffer[] buffers)
        => ParityDispatch(kernel, glsl, threads, push, VulkanTorchParityElementwiseKernels.LocalSize, buffers);

    private static uint Low(double value) => unchecked((uint)BitConverter.DoubleToInt64Bits(value));

    private static uint High(double value) => unchecked((uint)(BitConverter.DoubleToInt64Bits(value) >> 32));

    private static uint Low(long value) => unchecked((uint)value);

    private static uint High(long value) => unchecked((uint)(value >> 32));

    private static uint Word(int value) => unchecked((uint)value);

    // The scratch slots a validating kernel lowers (all ones = no error); created on first use.
    private IGpuBuffer ParityFlagSlots()
    {
        if (_parityFlagSlots is null)
        {
            var ones = new float[8];
            fixed (float* p = ones)
            {
                for (int i = 0; i < ones.Length; i++) ((uint*)p)[i] = 0xFFFFFFFFu;
            }
            _parityFlagSlots = AllocateBuffer(ones);
        }
        return _parityFlagSlots;
    }

    // A validating launch: the kernel lowers the flag slots, then one merge folds them into the caller's 64-bit flag.
    private void ParityDispatchValidating(string kernel, string glsl, int threads, uint[] push, IGpuBuffer flag, params IGpuBuffer[] buffers)
    {
        RequireParityCapacity(flag, 2, nameof(flag), "2 (64-bit flag)");
        lock (_parityFlagLock)
        {
            var slots = ParityFlagSlots();
            var bound = new IGpuBuffer[buffers.Length + 1];
            Array.Copy(buffers, bound, buffers.Length);
            bound[buffers.Length] = slots;
            ParityDispatch(kernel, glsl, threads, push, bound);
            ParityDispatch("flag_merge", VulkanTorchParityElementwiseKernels.FlagMerge, 1, new uint[] { 0u }, slots, flag);
        }
    }

    /// <summary>
    /// Evaluates a GLSL double expression of <c>x</c> for <paramref name="n"/> doubles (two words each, low word
    /// first) with the parity kernels' double library in scope. Test hook: it is how the library's accuracy is
    /// measured against .NET on the device.
    /// </summary>
    internal void EvaluateTorchParityDoubleExpression(string expression, IGpuBuffer input, IGpuBuffer output, int n, bool special)
    {
        if (n <= 0) return;
        RequireParityCapacity(input, 2L * n, nameof(input), "2*n");
        RequireParityCapacity(output, 2L * n, nameof(output), "2*n");
        RequireParityDoubles("double_probe");
        ParityDispatch("double_probe", VulkanTorchParityElementwiseKernels.DoubleProbe(expression, special), n, new[] { (uint)n }, input, output);
    }

    void ITorchParityElementwiseBackend.TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1)
    {
        if (n <= 0) return;
        RequireParityCapacity(input, n, nameof(input), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        if (VulkanTorchParityElementwiseKernels.NeedsDoubles(op)) RequireParityDoubles(op.ToString());
        ParityDispatch(op.ToString(), VulkanTorchParityElementwiseKernels.Unary(op), n,
            new[] { (uint)n, FloatBits(p0), FloatBits(p1) }, input, output);
    }

    void ITorchParityElementwiseBackend.TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireParityCapacity(a, n, nameof(a), "n");
        RequireParityCapacity(b, n, nameof(b), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        if (VulkanTorchParityElementwiseKernels.NeedsDoubles(op)) RequireParityDoubles(op.ToString());
        ParityDispatch(op.ToString(), VulkanTorchParityElementwiseKernels.Binary(op), n, new[] { (uint)n }, a, b, output);
    }

    void ITorchParityElementwiseBackend.TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireParityCapacity(x, n, nameof(x), "n");
        RequireParityCapacity(degree, n, nameof(degree), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("polynomial");
        ParityDispatch("polynomial", VulkanTorchParityElementwiseKernels.Polynomial, n,
            new[] { (uint)n, Word(kind), shifted ? 1u : 0u }, x, degree, output);
    }

    void ITorchParityElementwiseBackend.TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator)
    {
        if (n <= 0) return;
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("range");
        ParityDispatch("range", VulkanTorchParityElementwiseKernels.Range, n,
            new[] { (uint)n, Low(start), High(start), Low(span), High(span), Low(denominator), High(denominator) }, output);
    }

    void ITorchParityElementwiseBackend.TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator,
        double logBase, bool single)
    {
        if (n <= 0) return;
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("logspace");
        ParityDispatch("logspace", VulkanTorchParityElementwiseKernels.Logspace, n,
            new[]
            {
                (uint)n, Low(start), High(start), Low(span), High(span), Low(denominator), High(denominator), Low(logBase), High(logBase),
                single ? 1u : 0u,
            }, output);
    }

    void ITorchParityElementwiseBackend.TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
    {
        if (n <= 0) return;
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("window");
        ParityDispatch("window", VulkanTorchParityElementwiseKernels.Window, n,
            new[] { (uint)n, Word(kind), Word(m), Low(alpha), High(alpha), Low(beta), High(beta) }, output);
    }

    void ITorchParityElementwiseBackend.TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total)
    {
        if (row <= 0 || total <= 0) return;
        RequireParityCapacity(output, 2L * total, nameof(output), "2*total");
        // Every (row, column) pair has column - row in [-(row - 1), col - 1], so an offset beyond that range selects
        // the same elements as the range's end: clamping keeps the kernel's row + offset inside int.
        int clamped = Math.Max(-row, Math.Min(col, offset));
        ParityDispatch("tri_indices", VulkanTorchParityElementwiseKernels.TriIndices, row,
            new[] { Word(row), Word(col), Word(clamped), lower ? 1u : 0u, Word(total) }, output);
    }

    void ITorchParityElementwiseBackend.TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
    {
        if (size <= 0) return;
        int total = checked(size * size);
        RequireParityCapacity(output, total, nameof(output), "size*size");
        RequireParityCapacity(input, n, nameof(input), "n");
        ParityDispatch("diagflat", VulkanTorchParityElementwiseKernels.Diagflat, total,
            new[] { Word(total), Word(size), Word(offset), Word(n) }, input, output);
    }

    void ITorchParityElementwiseBackend.TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count, int[] otherExtents,
        int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
    {
        if (otherExtents is null) throw new ArgumentNullException(nameof(otherExtents));
        if (otherStrides is null) throw new ArgumentNullException(nameof(otherStrides));
        if (otherStrides.Length != otherExtents.Length)
            throw new ArgumentException("otherStrides must hold one stride per other axis.", nameof(otherStrides));
        if (count <= 0) return;
        if (otherExtents.Length > 8) throw new ArgumentOutOfRangeException(nameof(otherExtents), "at most 8 other axes.");
        if (diagonal <= 0) throw new ArgumentOutOfRangeException(nameof(diagonal));
        RequireParityCapacity(src, count, nameof(src), "count");
        // The furthest element the scatter writes: every other axis and the diagonal at their last index.
        long furthest = (long)(diagonal - 1 + start1) * stride1 + (long)(diagonal - 1 + start2) * stride2;
        for (int k = 0; k < otherExtents.Length; k++) furthest += (long)(otherExtents[k] - 1) * otherStrides[k];
        RequireParityCapacity(output, furthest + 1, nameof(output), "furthest diagonal offset + 1");
        var push = new uint[23];
        push[0] = Word(count);
        push[1] = Word(otherExtents.Length);
        for (int k = 0; k < 8; k++)
        {
            // Unused axes get extent 1 so the kernel never divides by zero.
            push[2 + k] = k < otherExtents.Length ? Word(otherExtents[k]) : 1u;
            push[10 + k] = k < otherStrides.Length ? Word(otherStrides[k]) : 0u;
        }
        push[18] = Word(stride1); push[19] = Word(stride2); push[20] = Word(start1); push[21] = Word(start2); push[22] = Word(diagonal);
        ParityDispatch("diagonal_scatter", VulkanTorchParityElementwiseKernels.DiagonalScatter, count, push, src, output);
    }

    void ITorchParityElementwiseBackend.TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length, double q,
        int mode, bool ignoreNan)
    {
        if (rows <= 0) return;
        RequireParityCapacity(sorted, (long)rows * length, nameof(sorted), "rows*length");
        RequireParityCapacity(output, rows, nameof(output), "rows");
        RequireParityDoubles("quantile_rows");
        ParityDispatch("quantile_rows", VulkanTorchParityElementwiseKernels.QuantileRows, rows,
            new[] { Word(rows), Word(length), Low(q), High(q), Word(mode), ignoreNan ? 1u : 0u }, sorted, output);
    }

    void ITorchParityElementwiseBackend.TorchParityFlagReset(IGpuBuffer flag)
    {
        RequireParityCapacity(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("flag_reset", VulkanTorchParityElementwiseKernels.FlagReset, 1, new uint[] { 0u }, flag);
    }

    void ITorchParityElementwiseBackend.TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
    {
        if (n <= 0) return;
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("sample");
        ParityDispatch("sample", VulkanTorchParityElementwiseKernels.Sample, n,
            new[] { (uint)n, Word(kind), Low(a), High(a), Low(b), High(b), (uint)seed, (uint)(seed >> 32) }, output);
    }

    void ITorchParityElementwiseBackend.TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireParityCapacity(probabilities, n, nameof(probabilities), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("bernoulli");
        ParityDispatchValidating("bernoulli", VulkanTorchParityElementwiseKernels.Bernoulli, n,
            new[] { (uint)n, (uint)seed, (uint)(seed >> 32) }, flag, probabilities, output);
    }

    void ITorchParityElementwiseBackend.TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireParityCapacity(rates, n, nameof(rates), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("poisson");
        ParityDispatchValidating("poisson", VulkanTorchParityElementwiseKernels.Poisson, n,
            new[] { (uint)n, (uint)seed, (uint)(seed >> 32) }, flag, rates, output);
    }

    void ITorchParityElementwiseBackend.TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n,
        ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireParityCapacity(counts, n, nameof(counts), "n");
        RequireParityCapacity(probabilities, n, nameof(probabilities), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("binomial");
        RequireParityInt64("binomial");
        ParityDispatchValidating("binomial", VulkanTorchParityElementwiseKernels.Binomial, n,
            new[] { (uint)n, (uint)seed, (uint)(seed >> 32) }, flag, counts, probabilities, output);
    }

    void ITorchParityElementwiseBackend.TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows,
        int categories, int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
    {
        if (rows <= 0) return;
        RequireParityCapacity(weights, (long)rows * categories, nameof(weights), "rows*categories");
        RequireParityCapacity(scratch, 2L * rows * categories, nameof(scratch), "2*rows*categories (double prefix sums)");
        RequireParityCapacity(output, (long)rows * numSamples, nameof(output), "rows*numSamples");
        RequireParityDoubles("multinomial");
        ParityDispatchValidating("multinomial", VulkanTorchParityElementwiseKernels.Multinomial(replacement), rows,
            new[] { (uint)rows, Word(categories), Word(numSamples), (uint)seed, (uint)(seed >> 32) }, flag, weights, scratch, output);
    }

    void ITorchParityElementwiseBackend.TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p, double a,
        double sat, double b, bool alpha, ulong seed)
    {
        if (n <= 0) return;
        RequireParityCapacity(scale, n, nameof(scale), "n");
        if (alpha) RequireParityCapacity(offset, n, nameof(offset), "n");
        RequireParityDoubles("dropout_affine");
        var push = new[]
        {
            (uint)n, Word(block), Low(p), High(p), Low(a), High(a), Low(sat), High(sat), Low(b), High(b), (uint)seed, (uint)(seed >> 32),
        };
        // Channel dropout writes no offset (the argument may alias scale), so its kernel binds the scale only.
        if (alpha) ParityDispatch("dropout_affine", VulkanTorchParityElementwiseKernels.DropoutAffine(true), n, push, scale, offset);
        else ParityDispatch("dropout_affine", VulkanTorchParityElementwiseKernels.DropoutAffine(false), n, push, scale);
    }

    void ITorchParityElementwiseBackend.TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper,
        bool training, ulong seed)
    {
        if (n <= 0) return;
        RequireParityCapacity(input, n, nameof(input), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("rrelu");
        ParityDispatch("rrelu", VulkanTorchParityElementwiseKernels.Rrelu, n,
            new[] { (uint)n, Low(lower), High(lower), Low(upper), High(upper), training ? 1u : 0u, (uint)seed, (uint)(seed >> 32) },
            input, output);
    }

    void ITorchParityElementwiseBackend.TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed)
    {
        if (n <= 0) return;
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("random_keys");
        ParityDispatch("random_keys", VulkanTorchParityElementwiseKernels.RandomKeys, n,
            new[] { (uint)n, (uint)seed, (uint)(seed >> 32) }, output);
    }

    void ITorchParityElementwiseBackend.TorchParityIntBinary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireParityCapacity(a, n, nameof(a), "n");
        RequireParityCapacity(b, n, nameof(b), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityInt64("int_binary");
        ParityDispatchValidating("int_binary", VulkanTorchParityElementwiseKernels.IntBinary, n, new[] { (uint)n, Word(op) }, flag, a, b, output);
    }

    void ITorchParityElementwiseBackend.TorchParityInt64Binary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n,
        int typeCode, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireParityCapacity(a, 2 * n, nameof(a), "2n (int64 words)");
        RequireParityCapacity(b, 2 * n, nameof(b), "2n (int64 words)");
        RequireParityCapacity(output, 2 * n, nameof(output), "2n (int64 words)");
        RequireParityInt64("int64_binary");
        ParityDispatchValidating("int64_binary", VulkanTorchParityElementwiseKernels.Int64Binary, n,
            new[] { (uint)n, Word(op), Word(typeCode) }, flag, a, b, output);
    }

    void ITorchParityElementwiseBackend.TorchParityInt64Not(IGpuBuffer input, IGpuBuffer output, int n, int typeCode)
    {
        if (n <= 0) return;
        RequireParityCapacity(input, 2 * n, nameof(input), "2n (int64 words)");
        RequireParityCapacity(output, 2 * n, nameof(output), "2n (int64 words)");
        RequireParityInt64("int64_not");
        ParityDispatch("int64_not", VulkanTorchParityElementwiseKernels.Int64Not, n, new[] { (uint)n, Word(typeCode) }, input, output);
    }

    void ITorchParityElementwiseBackend.TorchParityIntNot(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireParityCapacity(input, n, nameof(input), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        ParityDispatch("int_not", VulkanTorchParityElementwiseKernels.IntNot, n, new[] { (uint)n }, input, output);
    }

    void ITorchParityElementwiseBackend.TorchParityInclusiveScan(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireParityCapacity(input, n, nameof(input), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        const int block = VulkanTorchParityElementwiseKernels.ScanBlock;
        int blocks = (int)(((long)n + block - 1) / block);
        // Each block scans in shared memory and reports its total; the totals are scanned the same way (recursively)
        // and added back as each later block's carry. Exact in float for counts below 2^24.
        using var sums = AllocateBuffer(blocks);
        ParityDispatch("scan_block", VulkanTorchParityElementwiseKernels.ScanBlockSource, n, new[] { (uint)n }, block, input, output, sums);
        if (blocks == 1) return;
        using var carries = AllocateBuffer(blocks);
        ((ITorchParityElementwiseBackend)this).TorchParityInclusiveScan(sums, carries, blocks);
        ParityDispatch("scan_add", VulkanTorchParityElementwiseKernels.ScanAdd, n, new[] { (uint)n }, output, carries);
    }

    void ITorchParityElementwiseBackend.TorchParityNonzeroStaticScatter(IGpuBuffer input, IGpuBuffer positions, IGpuBuffer output, int n,
        int size, int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (n <= 0) return;
        RequireParityCapacity(input, n, nameof(input), "n");
        RequireParityCapacity(positions, n, nameof(positions), "n");
        RequireParityCapacity(output, (long)size * shape.Length, nameof(output), "size*rank");
        if (shape.Length > 8) throw new ArgumentOutOfRangeException(nameof(shape), "at most 8 axes.");
        var push = new uint[11];
        push[0] = (uint)n;
        push[1] = Word(size);
        push[2] = Word(shape.Length);
        for (int k = 0; k < 8; k++) push[3 + k] = k < shape.Length ? Word(shape[k]) : 1u;
        ParityDispatch("nonzero_static_scatter", VulkanTorchParityElementwiseKernels.NonzeroStaticScatter, n, push, input, positions, output);
    }

    void ITorchParityElementwiseBackend.TorchParityUnravelAxis(IGpuBuffer indices, IGpuBuffer output, int n, long total, long stride,
        long extent, bool validate, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireParityCapacity(indices, n, nameof(indices), "n");
        RequireParityCapacity(output, n, nameof(output), "n");
        RequireParityDoubles("unravel_axis");
        RequireParityInt64("unravel_axis");
        ParityDispatchValidating("unravel_axis", VulkanTorchParityElementwiseKernels.UnravelAxis, n,
            new[] { (uint)n, Low(total), High(total), Low(stride), High(stride), Low(extent), High(extent), validate ? 1u : 0u },
            flag, indices, output);
    }

    void ITorchParityElementwiseBackend.TorchParitySliceBitonicStep(IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j,
        int length, int sliceCount)
    {
        if (rowLen <= 0) return;
        if (sliceCount > rowLen) throw new ArgumentOutOfRangeException(nameof(sliceCount), "more slices than order slots.");
        RequireParityCapacity(order, rowLen, nameof(order), "rowLen");
        RequireParityCapacity(slices, (long)sliceCount * length, nameof(slices), "sliceCount*length");
        ParityDispatch("slice_bitonic_step", VulkanTorchParityElementwiseKernels.SliceBitonicStep, rowLen,
            new[] { Word(rowLen), Word(k), Word(j), Word(length) }, slices, order);
    }

    void ITorchParityElementwiseBackend.TorchParitySliceFirstFlags(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length)
    {
        if (n <= 0) return;
        RequireParityCapacity(order, n, nameof(order), "n");
        RequireParityCapacity(flags, n, nameof(flags), "n");
        RequireParityCapacity(slices, (long)n * length, nameof(slices), "n*length");
        ParityDispatch("slice_first_flags", VulkanTorchParityElementwiseKernels.SliceFirstFlags, n,
            new[] { (uint)n, Word(length) }, slices, order, flags);
    }

    void ITorchParityElementwiseBackend.TorchParitySliceGatherUnique(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags,
        IGpuBuffer positions, IGpuBuffer output, int n, int length, int outputRows)
    {
        int total = checked(n * length);
        RequireParityCapacity(output, (long)outputRows * length, nameof(output), "outputRows*length");
        if (total <= 0) return;
        RequireParityCapacity(slices, (long)n * length, nameof(slices), "n*length");
        RequireParityCapacity(order, n, nameof(order), "n");
        RequireParityCapacity(flags, n, nameof(flags), "n");
        RequireParityCapacity(positions, n, nameof(positions), "n");
        ParityDispatch("slice_gather_unique", VulkanTorchParityElementwiseKernels.SliceGatherUnique, total,
            new[] { Word(total), Word(length) }, slices, order, flags, positions, output);
    }

    private static uint[] GridPush(int batches, int channels, int depth, int height, int width, int points, int mode, int padding,
        bool alignCorners)
        => new[]
        {
            Word(batches), Word(channels), Word(depth), Word(height), Word(width), Word(points), Word(mode), Word(padding),
            alignCorners ? 1u : 0u,
        };

    void ITorchParityElementwiseBackend.TorchParityGridSample3D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches,
        int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
    {
        int threads = checked(batches * points);
        if (threads <= 0) return;
        RequireParityCapacity(input, (long)batches * channels * depth * height * width, nameof(input), "N*C*D*H*W");
        RequireParityCapacity(grid, 3L * batches * points, nameof(grid), "N*points*3");
        RequireParityCapacity(output, (long)batches * channels * points, nameof(output), "N*C*points");
        RequireParityDoubles("grid_sample_3d");
        ParityDispatch("grid_sample_3d", VulkanTorchParityElementwiseKernels.GridSample3D, threads,
            GridPush(batches, channels, depth, height, width, points, mode, padding, alignCorners), input, grid, output);
    }

    void ITorchParityElementwiseBackend.TorchParityGridSample3DBackward(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer outputGrad,
        IGpuBuffer inputGrad, IGpuBuffer gridGrad, int batches, int channels, int depth, int height, int width, int points, int mode,
        int padding, bool alignCorners)
    {
        int threads = checked(batches * points);
        if (threads <= 0) return;
        RequireParityCapacity(input, (long)batches * channels * depth * height * width, nameof(input), "N*C*D*H*W");
        RequireParityCapacity(inputGrad, (long)batches * channels * depth * height * width, nameof(inputGrad), "N*C*D*H*W");
        RequireParityCapacity(grid, 3L * batches * points, nameof(grid), "N*points*3");
        RequireParityCapacity(gridGrad, 3L * batches * points, nameof(gridGrad), "N*points*3");
        RequireParityCapacity(outputGrad, (long)batches * channels * points, nameof(outputGrad), "N*C*points");
        RequireParityDoubles("grid_sample_3d_backward");
        ParityDispatch("grid_sample_3d_backward", VulkanTorchParityElementwiseKernels.GridSample3DBackward, threads,
            GridPush(batches, channels, depth, height, width, points, mode, padding, alignCorners),
            input, grid, outputGrad, inputGrad, gridGrad);
    }
}
