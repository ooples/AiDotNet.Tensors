using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The sweep data ops on the device through the backend's IDataTransformBackend (DataTransformKernelTemplate; WGSL on
    // WebGPU) and IDataDecodeBackend's encode kernel. The caller's System.Random is one sequential stream, so its draws
    // stay on the host; the per-element transform of those draws (Box-Muller, the exact threshold) runs on the device
    // and the float32 result stays device-resident. A draw whose count depends on its values (gamma) or an arbitrary
    // sampler delegate runs wholly on the host and uploads its result once. Float only, as every device kernel here;
    // other element types, a graph trace, or a size past the kernels' limits run the CPU op, and a float call the
    // backend refuses is recorded through GpuLaunchProbe.OnFallback. Once the stream has been drawn, a fallback reuses
    // those draws: it never draws twice.

    // The backend and its transform capability; false (recorded when a backend lacks the capability) for the host op.
    private bool TryGetTransformBackend(string op, [NotNullWhen(true)] out IDirectGpuBackend? backend,
        [NotNullWhen(true)] out IDataTransformBackend? transforms)
    {
        transforms = null;
        backend = null;
        if (!TryGetBackend(out var found)) return false;
        if (found is IDataTransformBackend t)
        {
            backend = found;
            transforms = t;
            return true;
        }
        GpuLaunchProbe.OnFallback(op, null);
        return false;
    }

    /// <summary>A float32 host result uploaded once and returned device-resident.</summary>
    private Tensor<T> UploadResult<T>(IDirectGpuBackend backend, float[] values, int[] shape)
        => DeferTensorResult<T>(backend, backend.AllocateBuffer(values), values.Length, (int[])shape.Clone());

    private static float[] ToSingles(double[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = (float)values[i];
        return result;
    }

    // Each drawn pair as (r, u2) with r = sqrt(-2 ln u1) taken in double: float32 cannot carry u1 itself, since every
    // u1 within 2^-25 of 1 rounds to 1f and its radius (up to 2.4e-4) would vanish; r and u2 as floats are within
    // float rounding of the host's, so the device's trig and affine steps agree with the host to float rounding.
    private static float[] BoxMullerRadii(double[] pairs)
    {
        var result = new float[pairs.Length];
        for (int i = 0; i < pairs.Length; i += 2)
        {
            result[i] = (float)Math.Sqrt(-2.0 * Math.Log(pairs[i]));
            result[i + 1] = (float)pairs[i + 1];
        }
        return result;
    }

    // Each double draw as (a, r): a = (float)u and r = (float)(u - a). u - a is exact in double, and r keeps its sign
    // (r is never subnormal for u in [2^-53, 1)), so the pair orders u against any float exactly.
    private static float[] SplitDraws(double[] draws)
    {
        var pairs = new float[2 * draws.Length];
        for (int i = 0; i < draws.Length; i++)
        {
            float a = (float)draws[i];
            pairs[2 * i] = a;
            pairs[2 * i + 1] = (float)(draws[i] - a);
        }
        return pairs;
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNormalFromRng<T>(Random generator, int[] shape, Func<Random, double> sampler)
    {
        using var __gpuOp = EnterGpuOp();
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (sampler is null) throw new ArgumentNullException(nameof(sampler));
        int count = ValidateShapeElements(shape);
        if (typeof(T) != typeof(float) || count == 0 || !TryGetBackend(out var backend))
            return base.TensorNormalFromRng<T>(generator, shape, sampler);
        // An arbitrary delegate over one sequential stream: drawn on the host, uploaded once.
        return UploadResult<T>(backend, ToSingles(DrawFromSampler(generator, sampler, count)), shape);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTruncatedNormalFromRng<T>(Random generator, int[] shape, BoxMullerVariant variant,
        double mean, double stdDev, double low, double high)
    {
        using var __gpuOp = EnterGpuOp();
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        int count = ValidateShapeElements(shape);
        ValidateVariant(variant);
        ValidateClampBounds(low, high);
        if (typeof(T) != typeof(float) || count == 0 || (long)count * 2 > int.MaxValue / 2
            || !AffineFitsInFloat(mean, stdDev)
            || !TryGetTransformBackend("TruncatedNormalFromRng", out var backend, out var transforms))
            return base.TensorTruncatedNormalFromRng<T>(generator, shape, variant, mean, stdDev, low, high);

        var pairs = DrawBoxMullerUniforms(generator, variant, count);
        var uv = new OwnedBuffer(backend.AllocateBuffer(BoxMullerRadii(pairs)), ownsBuffer: true);
        try
        {
            using var output = AllocateFullyWrittenOutputBuffer(backend, count);
            bool clamp = !(double.IsNegativeInfinity(low) && double.IsPositiveInfinity(high));
            if (!transforms.TryBoxMuller(uv.Buffer, output.Buffer, count, variant.Branch, (float)mean, (float)stdDev, clamp,
                    (float)low, (float)high))
            {
                GpuLaunchProbe.OnFallback("TruncatedNormalFromRng", null);
                return UploadResult<T>(backend, ToSingles(TransformBoxMullerPairs(pairs, variant.Branch, mean, stdDev, low, high)), shape);
            }
            var result = DeferTensorResult<T>(backend, output.Buffer, count, (int[])shape.Clone());
            output.RelinquishOwnership();
            return result;
        }
        finally
        {
            uv.DisposeAfterQueuedUse(backend);
        }
    }

    // The largest Box-Muller radius any positive double u1 can give: sqrt(-2 ln 2^-1074) = 38.6...
    private const double MaxBoxMullerRadius = 38.7;

    // Whether the kernel's float mean + stdDev * z stays finite for every finite z the host could produce. A mean or
    // stdDev past float range (1e40 casts to +inf, and +inf + -inf is NaN) would leave the double host result behind,
    // so those draw on the host, before any uniform is consumed.
    private static bool AffineFitsInFloat(double mean, double stdDev) =>
        Math.Abs(mean) + Math.Abs(stdDev) * MaxBoxMullerRadius <= float.MaxValue;

    /// <inheritdoc/>
    public override Tensor<T> TensorGammaFromRng<T>(Random generator, double shape, Tensor<T> scale, BoxMullerVariant normal)
    {
        using var __gpuOp = EnterGpuOp();
        var drawn = base.TensorGammaFromRng(generator, shape, scale, normal);
        return UploadIfFloat(drawn);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGammaFromRng<T>(Random generator, Tensor<T> shape, Tensor<T> scale, BoxMullerVariant normal)
    {
        using var __gpuOp = EnterGpuOp();
        var drawn = base.TensorGammaFromRng(generator, shape, scale, normal);
        return UploadIfFloat(drawn);
    }

    // A rejection sampler's draw count depends on its draws, so it ran on the host: the float result uploads once.
    private Tensor<T> UploadIfFloat<T>(Tensor<T> drawn)
    {
        if (drawn is not Tensor<float> floats || drawn.Length == 0 || !TryGetBackend(out var backend)) return drawn;
        return UploadResult<T>(backend, floats.GetDataArray(), drawn._shape);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBernoulliFromRng<T>(Random generator, Tensor<T> probabilities, UniformDrawComparison successWhen)
    {
        using var __gpuOp = EnterGpuOp();
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (probabilities is null) throw new ArgumentNullException(nameof(probabilities));
        ValidateComparison(successWhen);
        int count = probabilities.Length;
        if (typeof(T) != typeof(float) || count == 0 || (long)count * 2 > int.MaxValue / 2
            || !TryGetTransformBackend("BernoulliFromRng", out var backend, out var transforms))
            return base.TensorBernoulliFromRng(generator, probabilities, successWhen);

        var draws = DrawUniforms(generator, count);
        using var thresholds = GetOrAllocateBuffer(backend, probabilities);
        var result = ThresholdOnDevice<T>(backend, transforms, draws, thresholds.Buffer, 0f,
            successWhen == UniformDrawComparison.DrawAboveThreshold, 1f, 0f, probabilities._shape);
        if (result is not null) return result;
        GpuLaunchProbe.OnFallback("BernoulliFromRng", null);
        return UploadResult<T>(backend,
            ToSingles(CompareDraws(draws, RoIAlignPlan.ToDoubles(probabilities), 0.0, successWhen, 1.0, 0.0)), probabilities._shape);
    }

    /// <inheritdoc/>
    private protected override Tensor<T> ThresholdDraws<T>(Random generator, int[] shape, double threshold,
        UniformDrawComparison comparison, double onSuccess, double otherwise)
    {
        using var __gpuOp = EnterGpuOp();
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        int count = ValidateShapeElements(shape);
        ValidateComparison(comparison);
        // The kernel compares exactly against a float threshold; another double threshold is compared on the host.
        bool floatExact = double.IsNaN(threshold) || (double)(float)threshold == threshold;
        if (typeof(T) != typeof(float) || count == 0 || (long)count * 2 > int.MaxValue / 2 || !floatExact
            || !TryGetTransformBackend("ThresholdFromRng", out var backend, out var transforms))
            return base.ThresholdDraws<T>(generator, shape, threshold, comparison, onSuccess, otherwise);

        var draws = DrawUniforms(generator, count);
        var result = ThresholdOnDevice<T>(backend, transforms, draws, null, (float)threshold,
            comparison == UniformDrawComparison.DrawAboveThreshold, (float)onSuccess, (float)otherwise, shape);
        if (result is not null) return result;
        GpuLaunchProbe.OnFallback("ThresholdFromRng", null);
        return UploadResult<T>(backend, ToSingles(CompareDraws(draws, null, threshold, comparison, onSuccess, otherwise)), shape);
    }

    // The drawn uniforms compared on the device; null when the backend refused (nothing launched).
    private Tensor<T>? ThresholdOnDevice<T>(IDirectGpuBackend backend, IDataTransformBackend transforms, double[] draws,
        IGpuBuffer? thresholds, float threshold, bool above, float onSuccess, float otherwise, int[] shape)
    {
        int count = draws.Length;
        var uv = new OwnedBuffer(backend.AllocateBuffer(SplitDraws(draws)), ownsBuffer: true);
        try
        {
            using var output = AllocateFullyWrittenOutputBuffer(backend, count);
            if (!transforms.TryUniformThreshold(uv.Buffer, thresholds, output.Buffer, count, above, threshold, onSuccess, otherwise))
                return null;
            var result = DeferTensorResult<T>(backend, output.Buffer, count, (int[])shape.Clone());
            output.RelinquishOwnership();
            return result;
        }
        finally
        {
            uv.DisposeAfterQueuedUse(backend);
        }
    }

    // Whether the tensor's values are already on the device: a host tensor is encoded / cast on the host, where an
    // upload would only be read back.
    private static bool IsOnDevice<T>(Tensor<T> tensor) => tensor.IsGpuResident || tensor.HasPendingGpuData;

    /// <inheritdoc/>
    /// <remarks>A device-resident float32 tensor is encoded on the device and only the packed bytes are read back; a
    /// host tensor is encoded on the host.</remarks>
    public override byte[] TensorToBytes<T>(Tensor<T> input, NumericStorageType targetType, bool littleEndian = true)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        int width = targetType.GetByteWidth();
        int count = input.Length;
        if (typeof(T) != typeof(float) || count == 0 || !BitConverter.IsLittleEndian || !IsOnDevice(input)
            || (long)count * width > int.MaxValue - 3 || !TryGetBackend(out var backend))
            return base.TensorToBytes(input, targetType, littleEndian);
        if (backend is not IDataDecodeBackend decode)
        {
            GpuLaunchProbe.OnFallback("TensorToBytes", null);
            return base.TensorToBytes(input, targetType, littleEndian);
        }

        int byteCount = count * width;
        int nWords = (byteCount + 3) / 4;
        using var source = GetOrAllocateBuffer(backend, input);
        var words = new OwnedBuffer(backend.AllocateIntBuffer(nWords), ownsBuffer: true);
        try
        {
            if (!decode.TryEncodeBytes(source.Buffer, words.Buffer, count, targetType, littleEndian))
            {
                GpuLaunchProbe.OnFallback("TensorToBytes", null);
                return base.TensorToBytes(input, targetType, littleEndian);
            }
            var packed = DownloadInts(backend, words.Buffer, nWords);
            var bytes = new byte[byteCount];
            Buffer.BlockCopy(packed, 0, bytes, 0, byteCount);
            return bytes;
        }
        finally
        {
            words.DisposeAfterQueuedUse(backend);
        }
    }

    /// <inheritdoc/>
    /// <remarks>A device-resident float32 source into an integral type is rounded and range-limited on the device; the
    /// narrowing of the (already integral, in-range) values into the result array is the host's.</remarks>
    public override Tensor<TOut> TensorCast<TIn, TOut>(Tensor<TIn> input, RoundingMode mode = RoundingMode.TowardZero,
        CastOverflowBehavior overflow = CastOverflowBehavior.Saturate)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        ValidateRounding(mode, overflow);
        int count = input.Length;
        if (typeof(TIn) != typeof(float) || !IsIntegral(typeof(TOut)) || overflow != CastOverflowBehavior.Saturate
            || count == 0 || !IsOnDevice(input) || !TryGetTransformBackend("TensorCast", out var backend, out var transforms))
            return base.TensorCast<TIn, TOut>(input, mode, overflow);

        var (min, maxExclusive) = IntegralRange(typeof(TOut));
        using var source = GetOrAllocateBuffer(backend, input);
        using var rounded = AllocateTransientScratch(backend, count, fullyWritten: true);
        if (!transforms.TryRoundIntegral(source.Buffer, rounded.Buffer, count, mode, (float)min, (float)maxExclusive))
        {
            GpuLaunchProbe.OnFallback("TensorCast", null);
            return base.TensorCast<TIn, TOut>(input, mode, overflow);
        }
        var values = DownloadFloats(backend, rounded.Buffer, count);
        var result = new Tensor<TOut>((int[])input._shape.Clone());
        var destination = result.GetDataArray();
        for (int i = 0; i < count; i++) StoreRounded(destination, i, values[i], overflow);
        return result;
    }

    /// <summary>
    /// The ids as exactly-valued floats for the gather / scatter kernels: an id outside [0, vocab) becomes -1, and for a
    /// scatter (<paramref name="lastWins"/>) so does every repeat of an id but its last. Null when the vocabulary is past
    /// the floats' exact integers.
    /// </summary>
    private static float[]? DeviceRowIds(long[] ids, int vocab, bool lastWins)
    {
        if (vocab > DataTransformKernelTemplate.MaxExactId) return null;
        var result = new float[ids.Length];
        HashSet<long>? seen = lastWins ? new HashSet<long>() : null;
        for (int k = ids.Length - 1; k >= 0; k--)
        {
            long row = ids[k];
            bool keep = row >= 0 && row < vocab && (seen is null || seen.Add(row));
            result[k] = keep ? row : -1f;
        }
        return result;
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGatherRows<T>(Tensor<T> table, Tensor<long> rows)
    {
        using var __gpuOp = EnterGpuOp();
        var (vocab, rowSize, outShape) = GatherRowsShape(table, rows);
        // A host table is gathered on the host: uploading it would move all vocab * rowSize values to read k rows, the
        // O(vocab * rowSize) cost the row gather exists to avoid (the scatter takes the same rule).
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || rows.Length == 0 || rowSize == 0
            || !IsOnDevice(table))
            return base.TensorGatherRows(table, rows);
        var ids = ReadAll(rows);
        var deviceIds = DeviceRowIds(ids, vocab, lastWins: false);
        if (deviceIds is null || !TryGetTransformBackend("GatherRows", out var backend, out var transforms))
        {
            if (deviceIds is null) GpuLaunchProbe.OnFallback("GatherRows", null);
            return base.TensorGatherRows(table, rows);
        }

        int count = ids.Length * rowSize;
        using var tableBuffer = GetOrAllocateBuffer(backend, table);
        var idBuffer = new OwnedBuffer(backend.AllocateBuffer(deviceIds), ownsBuffer: true);
        try
        {
            using var output = AllocateFullyWrittenOutputBuffer(backend, count);
            if (!transforms.TryGatherRows(tableBuffer.Buffer, idBuffer.Buffer, output.Buffer, ids.Length, rowSize))
            {
                GpuLaunchProbe.OnFallback("GatherRows", null);
                return base.TensorGatherRows(table, rows);
            }
            var result = DeferTensorResult<T>(backend, output.Buffer, count, outShape);
            output.RelinquishOwnership();
            RecordGatherRows(result, table, ids);
            return result;
        }
        finally
        {
            idBuffer.DisposeAfterQueuedUse(backend);
        }
    }

    /// <inheritdoc/>
    /// <remarks>A device-resident float32 table is written in its own device buffer (the kernel copies each kept row;
    /// which repeat of an id is kept is decided on the host); a host table is written on the host.</remarks>
    public override void TensorScatterRowsInPlace<T>(Tensor<T> table, Tensor<long> rows, Tensor<T> source)
    {
        using var __gpuOp = EnterGpuOp();
        var (vocab, rowSize, expected) = GatherRowsShape(table, rows);
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (!SameShapeAs(source._shape, expected) || typeof(T) != typeof(float) || rows.Length == 0 || rowSize == 0
            || table.IsSparse || !table.IsContiguous || !IsOnDevice(table))
        {
            base.TensorScatterRowsInPlace(table, rows, source);
            return;
        }
        var deviceIds = DeviceRowIds(ReadAll(rows), vocab, lastWins: true);
        if (deviceIds is null || !TryGetTransformBackend("ScatterRows", out var backend, out var transforms))
        {
            if (deviceIds is null) GpuLaunchProbe.OnFallback("ScatterRows", null);
            base.TensorScatterRowsInPlace(table, rows, source);
            return;
        }

        int k = deviceIds.Length;
        // A source sharing the table's storage would hand the kernel the buffer it writes: parallel row copies race.
        source = SnapshotIfAliased(table, source);
        using var sourceBuffer = GetOrAllocateBuffer(backend, source);
        var idBuffer = new OwnedBuffer(backend.AllocateBuffer(deviceIds), ownsBuffer: true);
        try
        {
            var src = sourceBuffer.Buffer;
            var ids = idBuffer.Buffer;
            bool done = TryRunUnaryInPlaceResident(table, (b, tableBuffer, n) =>
            {
                if (!transforms.TryScatterRows(src, ids, tableBuffer, k, rowSize))
                    throw new NotSupportedException("The backend refused the row scatter.");
            });
            if (!done) base.TensorScatterRowsInPlace(table, rows, source);
        }
        finally
        {
            idBuffer.DisposeAfterQueuedUse(backend);
        }
    }

    private static bool SameShapeAs(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLinearFilter<T>(Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi = null)
    {
        using var __gpuOp = EnterGpuOp();
        var result = LinearFilterOnDevice(b, a, x, zi, guard: false, 0.0);
        if (result is null) return base.TensorLinearFilter(b, a, x, zi);
        RecordLinearFilter(result, b, a, x, zi);
        return result;
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLinearFilterGuarded<T>(Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi, double maxAbs)
    {
        using var __gpuOp = EnterGpuOp();
        ValidateGuard(maxAbs);
        return LinearFilterOnDevice(b, a, x, zi, guard: true, maxAbs) ?? base.TensorLinearFilterGuarded(b, a, x, zi, maxAbs);
    }

    // The filter on the device, one thread per signal; null for the host op (non-float, a graph trace, an empty signal,
    // a zero a[0] the host op reports, or a refused launch, which is recorded).
    private Tensor<T>? LinearFilterOnDevice<T>(Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi, bool guard, double maxAbs)
    {
        var (signals, length, order) = LinearFilterShape(b, a, x, zi);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || signals == 0 || length == 0) return null;
        double a0 = RoIAlignPlan.ToDoubles(a)[0];
        if (a0 == 0.0 || double.IsNaN(a0)) return null;
        if (!TryGetTransformBackend("LinearFilter", out var backend, out var transforms)) return null;

        using var xB = GetOrAllocateBuffer(backend, x);
        using var bB = GetOrAllocateBuffer(backend, b);
        using var aB = GetOrAllocateBuffer(backend, a);
        using var ziB = zi is not null && order > 1 ? GetOrAllocateBuffer(backend, zi) : (OwnedBuffer?)null;
        using var work = AllocateTransientScratch(backend, Math.Max(1, signals * (order - 1)), fullyWritten: true);
        using var yB = AllocateFullyWrittenOutputBuffer(backend, signals * length);
        float bound = double.IsPositiveInfinity(maxAbs) ? float.PositiveInfinity : (float)maxAbs;
        if (!transforms.TryLinearFilter(xB.Buffer, bB.Buffer, aB.Buffer, ziB?.Buffer, yB.Buffer, work.Buffer, signals, length,
                b.Length, a.Length, guard, bound))
        {
            GpuLaunchProbe.OnFallback("LinearFilter", null);
            return null;
        }
        var result = DeferTensorResult<T>(backend, yB.Buffer, signals * length, (int[])x._shape.Clone());
        yB.RelinquishOwnership();
        // The queued filter still reads the scratch and the uploaded inputs: release them in queue order (each
        // `using` above then finds nothing left to own). A refused launch queued nothing, so it disposes at once.
        work.DisposeAfterQueuedUse(backend);
        xB.DisposeAfterQueuedUse(backend);
        bB.DisposeAfterQueuedUse(backend);
        aB.DisposeAfterQueuedUse(backend);
        ziB?.DisposeAfterQueuedUse(backend);
        return result;
    }
}
