// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// An <see cref="IAsyncEngine"/> over an <see cref="IAwaitableGpuBackend"/>: each operation uploads its operands,
/// awaits the kernel and the read-back, and never blocks the calling thread.
/// </summary>
/// <remarks>
/// <para>
/// Float tensors of matching shape (and rank-2 matrix products) run on the device. Everything else gives the same
/// answer through <see cref="CpuEngine"/>: other element types, operands that broadcast, a backend that reports a
/// capability gap, and any call made while a <see cref="Autodiff.GradientTape{T}"/> or a graph trace is active on
/// the calling thread. The tape is thread-local, and an awaited continuation may resume on another thread, so a
/// recorded operation must complete on the thread that started it.
/// </para>
/// </remarks>
public sealed class AwaitableGpuTensorEngine : IAsyncEngine, IDisposable
{
    private readonly IAwaitableGpuBackend _backend;
    private readonly bool _ownsBackend;
    private readonly CpuEngine _cpu = new();
    private int _disposed;

    /// <summary>Wraps <paramref name="backend"/>, which must already be initialized.</summary>
    /// <param name="backend">The device.</param>
    /// <param name="ownsBackend">Whether disposing this engine disposes the backend.</param>
    public AwaitableGpuTensorEngine(IAwaitableGpuBackend backend, bool ownsBackend = false)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (!backend.IsAvailable)
            throw new ArgumentException("The backend is not initialized or not available.", nameof(backend));
        _ownsBackend = ownsBackend;
    }

    /// <summary>The device this engine runs on.</summary>
    public IAwaitableGpuBackend Backend => _backend;

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorAddAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
        => Binary(a, b, cancellationToken, (be, x, y, o, n) => be.AddAsync(x, y, o, n),
            _cpu.TensorAddAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorSubtractAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
        => Binary(a, b, cancellationToken, (be, x, y, o, n) => be.SubtractAsync(x, y, o, n),
            _cpu.TensorSubtractAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorMultiplyAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
        => Binary(a, b, cancellationToken, (be, x, y, o, n) => be.MultiplyAsync(x, y, o, n),
            _cpu.TensorMultiplyAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorDivideAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
        => Binary(a, b, cancellationToken, (be, x, y, o, n) => be.DivideAsync(x, y, o, n),
            _cpu.TensorDivideAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorExpAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.ExpAsync(x, o, n), _cpu.TensorExpAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorLogAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.LogAsync(x, o, n), _cpu.TensorLogAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorSqrtAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.SqrtAsync(x, o, n), _cpu.TensorSqrtAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorAbsAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.AbsAsync(x, o, n), _cpu.TensorAbsAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorNegateAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.NegateAsync(x, o, n), _cpu.TensorNegateAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> ReLUAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.ReluAsync(x, o, n), _cpu.ReLUAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> SigmoidAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.SigmoidAsync(x, o, n), _cpu.SigmoidAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TanhAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
        => Unary(tensor, cancellationToken, (be, x, o, n) => be.TanhAsync(x, o, n), _cpu.TanhAsync);

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorMatMulAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        var left = a.Shape.ToArray();
        var right = b.Shape.ToArray();
        if (typeof(T) != typeof(float) || RecordsOnThisThread<T>() || left.Length != 2 || right.Length != 2
            || left[1] != right[0] || left[0] == 0 || left[1] == 0 || right[1] == 0
            // A product whose element count does not fit an int has no device buffer (m * n would wrap to a small
            // one, undersized for the kernel): the CPU engine answers for it, as for every other shape it owns.
            || (long)left[0] * right[1] > int.MaxValue)
            return _cpu.TensorMatMulAsync(a, b, cancellationToken);
        return MatMulOnDevice(a, b, left[0], right[1], left[1], cancellationToken);
    }

    private async ValueTask<Tensor<T>> MatMulOnDevice<T>(Tensor<T> a, Tensor<T> b, int m, int n, int k, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        float[]? product;
        try
        {
            using var left = _backend.AllocateBuffer(Floats(a));
            using var right = _backend.AllocateBuffer(Floats(b));
            using var output = _backend.AllocateBuffer(m * n);
            await _backend.GemmAsync(left, right, output, m, n, k).ConfigureAwait(false);
            product = await _backend.DownloadBufferAsync(output).ConfigureAwait(false);
        }
        catch (NotSupportedException)
        {
            // A capability gap (GpuCapabilityUnavailableException included): the CPU computes the same product.
            product = null;
        }
        catch (GpuBufferTooLargeException) when (BufferCapFallsBackToCpu)
        {
            product = null;
        }
        if (product is null) return _cpu.TensorMatMul(a, b);
        return FromFloats<T>(product, m * n, new[] { m, n });
    }

    private delegate Task BinaryKernel(IAwaitableGpuBackend backend, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);

    private delegate Task UnaryKernel(IAwaitableGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size);

    private ValueTask<Tensor<T>> Binary<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken, BinaryKernel kernel,
        Func<Tensor<T>, Tensor<T>, CancellationToken, ValueTask<Tensor<T>>> onCpu)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        var shape = a.Shape.ToArray();
        if (typeof(T) != typeof(float) || RecordsOnThisThread<T>() || !SameShape(shape, b.Shape.ToArray()) || a.Length == 0)
            return onCpu(a, b, cancellationToken);
        return BinaryOnDevice(a, b, shape, kernel, onCpu, cancellationToken);
    }

    private async ValueTask<Tensor<T>> BinaryOnDevice<T>(Tensor<T> a, Tensor<T> b, int[] shape, BinaryKernel kernel,
        Func<Tensor<T>, Tensor<T>, CancellationToken, ValueTask<Tensor<T>>> onCpu, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        int size = a.Length;
        float[]? result;
        try
        {
            using var left = _backend.AllocateBuffer(Floats(a));
            using var right = _backend.AllocateBuffer(Floats(b));
            using var output = _backend.AllocateBuffer(size);
            await kernel(_backend, left, right, output, size).ConfigureAwait(false);
            result = await _backend.DownloadBufferAsync(output).ConfigureAwait(false);
        }
        catch (NotSupportedException)
        {
            result = null;
        }
        catch (GpuBufferTooLargeException) when (BufferCapFallsBackToCpu)
        {
            result = null;
        }
        if (result is null) return await onCpu(a, b, cancellationToken).ConfigureAwait(false);
        return FromFloats<T>(result, size, shape);
    }

    private ValueTask<Tensor<T>> Unary<T>(Tensor<T> tensor, CancellationToken cancellationToken, UnaryKernel kernel,
        Func<Tensor<T>, CancellationToken, ValueTask<Tensor<T>>> onCpu)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(T) != typeof(float) || RecordsOnThisThread<T>() || tensor.Length == 0)
            return onCpu(tensor, cancellationToken);
        return UnaryOnDevice(tensor, kernel, onCpu, cancellationToken);
    }

    private async ValueTask<Tensor<T>> UnaryOnDevice<T>(Tensor<T> tensor, UnaryKernel kernel,
        Func<Tensor<T>, CancellationToken, ValueTask<Tensor<T>>> onCpu, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        int size = tensor.Length;
        var shape = tensor.Shape.ToArray();
        float[]? result;
        try
        {
            using var input = _backend.AllocateBuffer(Floats(tensor));
            using var output = _backend.AllocateBuffer(size);
            await kernel(_backend, input, output, size).ConfigureAwait(false);
            result = await _backend.DownloadBufferAsync(output).ConfigureAwait(false);
        }
        catch (NotSupportedException)
        {
            result = null;
        }
        catch (GpuBufferTooLargeException) when (BufferCapFallsBackToCpu)
        {
            result = null;
        }
        if (result is null) return await onCpu(tensor, cancellationToken).ConfigureAwait(false);
        return FromFloats<T>(result, size, shape);
    }

    // A buffer past the device's per-allocation cap follows the process's chunking policy, as on the synchronous
    // engine: the caller sees the refusal under NeverChunk_FailFast, and the CPU answers otherwise (this engine has no
    // chunked path, so every other policy ends at the CPU).
    private static bool BufferCapFallsBackToCpu
        => GpuFallbackOptionsHolder.Current.EffectiveChunkingPolicy != GpuChunkingPolicy.NeverChunk_FailFast;

    // Recording state is thread-local; see the remarks. Anomaly mode is too, and takes the CPU as the synchronous engine
    // does: a device may leave a float division by zero indeterminate, and the detector exists to report the CPU's Inf.
    private static bool RecordsOnThisThread<T>()
        => Autodiff.GradientTape<T>.Current is not null || Compilation.GraphMode.Current is not null
           || Autodiff.AnomalyModeScope.IsActive;

    private static bool SameShape(int[] left, int[] right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
            if (left[i] != right[i]) return false;
        return true;
    }

    // The engine computes float tensors only; its entry points refuse any other element type before reaching here.
    private static float[] Floats<T>(Tensor<T> tensor)
        => tensor.ToArray() is float[] floats
            ? floats
            : throw new NotSupportedException($"The awaitable GPU engine computes float tensors, not {typeof(T).Name}.");

    // A pooled buffer can be longer than the tensor: keep the first size elements.
    private static Tensor<T> FromFloats<T>(float[] data, int size, int[] shape)
    {
        if (data.Length != size)
        {
            var exact = new float[size];
            Array.Copy(data, exact, size);
            data = exact;
        }
        return new Tensor<float>(data, shape) is Tensor<T> typed
            ? typed
            : throw new NotSupportedException($"The awaitable GPU engine computes float tensors, not {typeof(T).Name}.");
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(AwaitableGpuTensorEngine));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_ownsBackend) _backend.Dispose();
    }
}
