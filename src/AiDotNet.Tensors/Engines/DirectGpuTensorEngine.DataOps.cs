using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The data-loading sources on the device through the backend's IDataDecodeBackend, which every backend implements
    // (DataDecodeKernelTemplate; WGSL on WebGPU). The raw bytes or text upload once, packed four bytes per word, and the
    // float32 result stays device-resident. Other element types, and anything the backend refuses, run the host op.

    /// <inheritdoc/>
    public override Tensor<T> TensorFromBytes<T>(ReadOnlySpan<byte> data, NumericStorageType sourceType, int[] shape,
        bool littleEndian = true)
    {
        using var __gpuOp = EnterGpuOp();
        int count = ValidateFromBytes(data.Length, sourceType, shape);
        if (typeof(T) != typeof(float) || count == 0 || !BitConverter.IsLittleEndian || !TryGetBackend(out var backend)
            || backend is not IDataDecodeBackend decode)
            return base.TensorFromBytes<T>(data, sourceType, shape, littleEndian);

        var words = new OwnedBuffer(backend.AllocateIntBuffer(PackWords(data)), ownsBuffer: true);
        try
        {
            var output = AllocateFullyWrittenOutputBuffer(backend, count);
            bool launched;
            try
            {
                launched = decode.TryDecodeBytes(words.Buffer, output.Buffer, count, sourceType, littleEndian);
            }
            catch
            {
                output.Dispose();
                throw;
            }
            if (!launched)
            {
                output.Dispose();
                GpuLaunchProbe.OnFallback("TensorFromBytes", null);
                return base.TensorFromBytes<T>(data, sourceType, shape, littleEndian);
            }
            var result = DeferTensorResult<T>(backend, output.Buffer, count, (int[])shape.Clone());
            output.RelinquishOwnership();
            return result;
        }
        finally
        {
            words.DisposeAfterQueuedUse(backend);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorParse<T>(ReadOnlySpan<byte> utf8Text, byte delimiter, int[] shape, NumberStyles styles,
        IFormatProvider provider, T missingValue)
    {
        using var __gpuOp = EnterGpuOp();
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        int styleMask = NumericTextFastPath.StyleMask(styles, provider);
        int length = utf8Text.Length;
        bool trailingNewline = length > 0 && utf8Text[length - 1] == (byte)'\n';
        if (trailingNewline) length--;
        if (typeof(T) != typeof(float) || styleMask < 0 || length == 0 || !BitConverter.IsLittleEndian
            || !TryGetBackend(out var backend) || backend is not IDataDecodeBackend decode)
            return base.TensorParse(utf8Text, delimiter, shape, styles, provider, missingValue);

        int expected = ValidateShapeElements(shape);
        float missing = missingValue is float missingFloat ? missingFloat : float.NaN;
        var words = new OwnedBuffer(backend.AllocateIntBuffer(PackWords(utf8Text)), ownsBuffer: true);
        var offsets = default(OwnedBuffer);
        var total = default(OwnedBuffer);
        var bounds = default(OwnedBuffer);
        try
        {
            int segments = (length + DataDecodeKernelTemplate.SegmentBytes - 1) / DataDecodeKernelTemplate.SegmentBytes;
            offsets = new OwnedBuffer(backend.AllocateIntBuffer(segments), ownsBuffer: true);
            total = new OwnedBuffer(backend.AllocateIntBuffer(1), ownsBuffer: true);
            if (!decode.TryCountSeparators(words.Buffer, offsets.Buffer, length, delimiter)
                || !decode.TryExclusiveScan(offsets.Buffer, total.Buffer, segments))
            {
                GpuLaunchProbe.OnFallback("TensorParse", null);
                return base.TensorParse(utf8Text, delimiter, shape, styles, provider, missingValue);
            }
            int fields = DownloadInts(backend, total.Buffer, 1)[0] + 1;
            ValidateParseShape(fields, shape);
            if (expected == 0 || fields > int.MaxValue / 2)
                return base.TensorParse(utf8Text, delimiter, shape, styles, provider, missingValue);
            bounds = new OwnedBuffer(backend.AllocateIntBuffer(2 * fields), ownsBuffer: true);
            if (!decode.TryScatterFieldBounds(words.Buffer, offsets.Buffer, bounds.Buffer, length, delimiter, trailingNewline))
            {
                GpuLaunchProbe.OnFallback("TensorParse", null);
                return base.TensorParse(utf8Text, delimiter, shape, styles, provider, missingValue);
            }
            var parsed = ParseFieldsOnDevice<T>(backend, decode, words.Buffer, bounds.Buffer, total.Buffer, fields, shape,
                styleMask, missing, out var undecided);
            if (parsed is not null) return parsed;
            if (undecided is null)
            {
                GpuLaunchProbe.OnFallback("TensorParse", null);
                return base.TensorParse(utf8Text, delimiter, shape, styles, provider, missingValue);
            }
            // Some fields are outside the exact fast path: the host parses just those.
            var fieldBounds = DownloadInts(backend, bounds.Buffer, 2 * fields);
            foreach (int f in undecided.Indices)
            {
                int start = fieldBounds[2 * f];
                var token = Encoding.UTF8.GetString(utf8Text.Slice(start, fieldBounds[2 * f + 1] - start).ToArray());
                undecided.Values[f] = ParseToken(token, styles, provider, missing);
            }
            return new Tensor<T>((T[])(object)undecided.Values, (int[])shape.Clone());
        }
        finally
        {
            bounds.DisposeAfterQueuedUse(backend);
            total.DisposeAfterQueuedUse(backend);
            offsets.DisposeAfterQueuedUse(backend);
            words.DisposeAfterQueuedUse(backend);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorParse<T>(IReadOnlyList<string?> tokens, int[] shape, NumberStyles styles,
        IFormatProvider provider, T missingValue)
    {
        using var __gpuOp = EnterGpuOp();
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        int styleMask = NumericTextFastPath.StyleMask(styles, provider);
        int count = ValidateParseShape(tokens.Count, shape);
        if (typeof(T) != typeof(float) || styleMask < 0 || count == 0 || count > int.MaxValue / 2
            || !BitConverter.IsLittleEndian || !TryGetBackend(out var backend) || backend is not IDataDecodeBackend decode)
            return base.TensorParse(tokens, shape, styles, provider, missingValue);

        // The tokens are already split: their UTF-8 bytes back to back, and each one's range from the host.
        var hostBounds = new int[2 * count];
        var utf8 = new List<byte>();
        for (int i = 0; i < count; i++)
        {
            hostBounds[2 * i] = utf8.Count;
            var token = tokens[i];
            if (token is not null) utf8.AddRange(Encoding.UTF8.GetBytes(token));
            hostBounds[2 * i + 1] = utf8.Count;
        }
        float missing = missingValue is float missingFloat ? missingFloat : float.NaN;
        var words = new OwnedBuffer(backend.AllocateIntBuffer(PackWords(utf8.ToArray())), ownsBuffer: true);
        var bounds = default(OwnedBuffer);
        var total = default(OwnedBuffer);
        try
        {
            bounds = new OwnedBuffer(backend.AllocateIntBuffer(hostBounds), ownsBuffer: true);
            total = new OwnedBuffer(backend.AllocateIntBuffer(1), ownsBuffer: true);
            var parsed = ParseFieldsOnDevice<T>(backend, decode, words.Buffer, bounds.Buffer, total.Buffer, count, shape,
                styleMask, missing, out var undecided);
            if (parsed is not null) return parsed;
            if (undecided is null)
            {
                GpuLaunchProbe.OnFallback("TensorParse", null);
                return base.TensorParse(tokens, shape, styles, provider, missingValue);
            }
            foreach (int f in undecided.Indices)
                undecided.Values[f] = ParseToken(tokens[f], styles, provider, missing);
            return new Tensor<T>((T[])(object)undecided.Values, (int[])shape.Clone());
        }
        finally
        {
            total.DisposeAfterQueuedUse(backend);
            bounds.DisposeAfterQueuedUse(backend);
            words.DisposeAfterQueuedUse(backend);
        }
    }

    /// <summary>The device values and the fields the parse kernel left to the host.</summary>
    private sealed class UndecidedFields
    {
        internal UndecidedFields(float[] values, List<int> indices)
        {
            Values = values;
            Indices = indices;
        }

        internal float[] Values { get; }
        internal List<int> Indices { get; }
    }

    /// <summary>
    /// Parses every field on the device. Returns the device-resident result when the kernel decided every field; null
    /// with <paramref name="undecided"/> set (the downloaded values and the fields to parse on the host) when it did not;
    /// null with undecided null when the backend refused.
    /// </summary>
    private Tensor<T>? ParseFieldsOnDevice<T>(IDirectGpuBackend backend, IDataDecodeBackend decode, IGpuBuffer words,
        IGpuBuffer bounds, IGpuBuffer total, int fields, int[] shape, int styleMask, float missing,
        out UndecidedFields? undecided)
    {
        undecided = null;
        var output = AllocateFullyWrittenOutputBuffer(backend, fields);
        var status = default(OwnedBuffer);
        try
        {
            status = new OwnedBuffer(backend.AllocateIntBuffer(fields), ownsBuffer: true);
            if (!decode.TryParseFields(words, bounds, output.Buffer, status.Buffer, fields, styleMask, missing)
                || !decode.TryExclusiveScan(status.Buffer, total, fields))
                return null;
            int left = DownloadInts(backend, total, 1)[0];
            if (left == 0)
            {
                var result = DeferTensorResult<T>(backend, output.Buffer, fields, (int[])shape.Clone());
                output.RelinquishOwnership();
                return result;
            }
            var values = DownloadFloats(backend, output.Buffer, fields);
            var prefix = DownloadInts(backend, status.Buffer, fields);
            var indices = new List<int>(left);
            for (int f = 0; f < fields; f++)
            {
                int next = f + 1 < fields ? prefix[f + 1] : left;
                if (next != prefix[f]) indices.Add(f);
            }
            undecided = new UndecidedFields(values, indices);
            return null;
        }
        finally
        {
            status.DisposeAfterQueuedUse(backend);
            output.DisposeAfterQueuedUse(backend);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandomUniformFromGenerator<T>(Random generator, int[] shape, double min, double max)
    {
        using var __gpuOp = EnterGpuOp();
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        int count = ValidateShapeElements(shape);
        if (typeof(T) != typeof(float) || count == 0 || !TryGetBackend(out var backend))
            return base.TensorRandomUniformFromGenerator<T>(generator, shape, min, max);
        // The draw is one sequential stream on the host; the result then uploads once and stays on the device.
        var values = new float[count];
        FillUniformFromGenerator(generator, values.AsSpan(), min, max);
        var buffer = backend.AllocateBuffer(values);
        return DeferTensorResult<T>(backend, buffer, count, (int[])shape.Clone());
    }

    private static int ValidateShapeElements(int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        return (int)ElementCountOf(shape);
    }

    /// <summary>The bytes packed four per int, little-endian, at least one word (a backend may refuse an empty buffer).</summary>
    private static int[] PackWords(ReadOnlySpan<byte> bytes)
    {
        var words = new int[Math.Max(1, (bytes.Length + 3) / 4)];
        bytes.CopyTo(MemoryMarshal.AsBytes(words.AsSpan()));
        return words;
    }

    private static int[] DownloadInts(IDirectGpuBackend backend, IGpuBuffer buffer, int count)
    {
        var raw = DownloadFloats(backend, buffer, count);
        var ints = new int[count];
        Buffer.BlockCopy(raw, 0, ints, 0, count * sizeof(int));
        return ints;
    }

    private static float[] DownloadFloats(IDirectGpuBackend backend, IGpuBuffer buffer, int count)
    {
        backend.Synchronize();
        var raw = backend.DownloadBuffer(buffer);
        if (raw.Length < count)
            throw new InvalidOperationException($"Downloaded {raw.Length} values from a {count}-element device buffer.");
        return raw;
    }
}
