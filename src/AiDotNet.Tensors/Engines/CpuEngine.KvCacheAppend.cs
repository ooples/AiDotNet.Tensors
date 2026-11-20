using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// A fixed-capacity decode cache whose write position lives in a tensor, so a compiled decode plan appends on the device
// with no host round trip (the static-cache pattern: one persistent buffer, written at a device-side position).
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> KvCacheAppend<T>(Tensor<T> cache, Tensor<T> position, Tensor<T> value)
    {
        var (batch, capacity, width) = ValidateKvCacheAppend(cache, position, value);
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var (cc, cp, cv) = (cache, position, value);
            return scope.RecordVariadicWritingInPlace(LazyNodeType.Custom, "KvCacheAppend", new[] { cache, position, value },
                new[] { capacity }, new[] { cache, position },
                (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, eng.KvCacheAppend(cc, cp, cv), output));
        }
        return KvCacheAppendHost(cache, position, value, batch, capacity, width);
    }

    /// <summary>The host implementation of <see cref="KvCacheAppend{T}"/>: row copies and one position read.</summary>
    private protected Tensor<T> KvCacheAppendHost<T>(Tensor<T> cache, Tensor<T> position, Tensor<T> value, int batch,
        int capacity, int width)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double raw = ops.ToDouble(position.AsSpanUnmarked()[0]);
        int p = (int)raw;
        if (p != raw || p < 0) throw new InvalidOperationException($"The cache position must be a non-negative integer; it is {raw}.");
        if (p >= capacity) throw new InvalidOperationException($"The cache is full: {capacity} positions.");
        if (!value.IsContiguous) value = value.Contiguous();
        using (new KeepAliveScope(cache))
        using (new KeepAliveScope(value))
        {
            var src = value.AsSpanUnmarked();
            var dst = cache.AsWritableSpanUnmarked();
            for (int b = 0; b < batch; b++)
                src.Slice(b * width, width).CopyTo(dst.Slice((b * capacity + p) * width, width));
        }
        cache.IncrementVersion();
        InvalidatePersistentTensor(cache);
        position.AsWritableSpanUnmarked()[0] = ops.FromDouble(p + 1);
        position.IncrementVersion();
        InvalidatePersistentTensor(position);
        var mask = new Tensor<T>(new[] { capacity });
        var m = mask.AsWritableSpanUnmarked();
        m.Slice(0, p + 1).Fill(ops.Zero);
        m.Slice(p + 1).Fill(ops.FromDouble(double.NegativeInfinity));
        return mask;
    }

    private protected static (int Batch, int Capacity, int Width) ValidateKvCacheAppend<T>(Tensor<T> cache, Tensor<T> position,
        Tensor<T> value)
    {
        if (cache is null) throw new ArgumentNullException(nameof(cache));
        if (position is null) throw new ArgumentNullException(nameof(position));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (cache.Rank != 3) throw new ArgumentException("cache must be [batch, capacity, width].", nameof(cache));
        if (!cache.IsContiguous) throw new ArgumentException("cache is written in place and must be contiguous.", nameof(cache));
        if (position.Length != 1) throw new ArgumentException("position must hold one value.", nameof(position));
        if (!position.IsContiguous) throw new ArgumentException("position is written in place and must be contiguous.", nameof(position));
        int batch = cache.Shape[0], capacity = cache.Shape[1], width = cache.Shape[2];
        if (capacity <= 0 || width <= 0 || batch <= 0) throw new ArgumentException("cache dimensions must be positive.", nameof(cache));
        if (value.Length != batch * width) throw new ArgumentException($"value must hold batch x width = {batch * width} values.", nameof(value));
        return (batch, capacity, width);
    }
}
