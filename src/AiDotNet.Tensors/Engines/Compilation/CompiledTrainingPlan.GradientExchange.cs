using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// The gradients of one training step, handed to a data-parallel reduction between the backward and the optimizer
/// update (PyTorch DDP's all-reduce point). <see cref="Read"/> gathers every gradient into one host vector in parameter
/// order, wherever the step left it (the device buffer the backward wrote, or the host array); <see cref="Write"/> puts
/// the reduced values back in the same places, so the clip and the fused update that follow read them.
/// </summary>
internal sealed class GradientExchange<T>
{
    private readonly Tensor<T>[] _parameters;
    private readonly Tensor<T>[] _gradients;
    private readonly Engines.DirectGpu.IGpuBuffer?[] _deviceGradients;
    private readonly Engines.DirectGpu.IDirectGpuBackend? _backend;

    private readonly Tensor<T> _loss;
    private readonly Engines.DirectGpu.IGpuBuffer? _deviceLoss;

    internal GradientExchange(Tensor<T>[] parameters, Tensor<T>[] gradients,
        Engines.DirectGpu.IGpuBuffer?[] deviceGradients, Engines.DirectGpu.IDirectGpuBackend? backend,
        Tensor<T> loss, Engines.DirectGpu.IGpuBuffer? deviceLoss)
    {
        _loss = loss;
        _deviceLoss = deviceLoss;
        _parameters = parameters;
        _gradients = gradients;
        _deviceGradients = deviceGradients;
        _backend = backend;
        long total = 0;
        foreach (var g in gradients) total += g.Length;
        TotalLength = checked((int)total);
    }

    /// <summary>The exchanged parameters, in plan order (only those with a gradient this step).</summary>
    public IReadOnlyList<Tensor<T>> Parameters => _parameters;

    /// <summary>The number of gradient elements over every exchanged parameter.</summary>
    public int TotalLength { get; }

    /// <summary>How many gradients the step left on the device (the rest are host arrays).</summary>
    public int DeviceGradientCount
    {
        get { int n = 0; foreach (var b in _deviceGradients) if (b is not null) n++; return n; }
    }

    /// <summary>This rank's loss for the step (the plan's loss output, read where the step left it).</summary>
    public T ReadLoss()
    {
        if (_deviceLoss is { } buffer)
        {
            var host = new float[buffer.Size];
            _backend!.DownloadBuffer(buffer, host);
            return (T)(object)host[0];
        }
        using var lease = _loss.LeaseArray();
        return lease.Array[0];
    }

    /// <summary>Copies every gradient into <paramref name="destination"/> (length <see cref="TotalLength"/>).</summary>
    public void Read(T[] destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (destination.Length < TotalLength) throw new ArgumentException("The destination is shorter than the gradients.", nameof(destination));
        int offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is { } buffer)
            {
                // Synchronizes the compute stream first, so the backward that wrote the buffer has finished.
                var host = new float[buffer.Size];
                _backend!.DownloadBuffer(buffer, host);
                Array.Copy(host, 0, (float[])(object)destination, offset, length);   // device gradients are float
            }
            else
            {
                using var lease = _gradients[i].LeaseArray();
                ((ReadOnlySpan<T>)lease.Array.AsSpan(0, length)).CopyTo(destination.AsSpan(offset, length));
            }
            offset += length;
        }
    }

    /// <summary>Writes the reduced gradients back where <see cref="Read"/> found them.</summary>
    public void Write(T[] source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (source.Length < TotalLength) throw new ArgumentException("The source is shorter than the gradients.", nameof(source));
        int offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is { } buffer)
            {
                var host = new float[length];
                Array.Copy((float[])(object)source, offset, host, 0, length);
                _backend!.UploadBufferInPlace(host, buffer);   // synchronous: done before the update launches
            }
            else
            {
                using var lease = _gradients[i].LeaseWritable();
                source.AsSpan(offset, length).CopyTo(lease.Span.Slice(0, length));
            }
            offset += length;
        }
    }
}

internal sealed partial class CompiledTrainingPlan<T>
{
    private Action<GradientExchange<T>>? _gradientExchange;

    // With an exchange the captured step is two graphs: the backward (this plan's _stepGraphExec) and the update (clip
    // and the captured fused optimizer), with the exchange between their launches.
    private IntPtr _updateGraphExec;

    /// <summary>Steps whose gradients went through the exchange.</summary>
    internal long GradientExchangeCount { get; private set; }

    /// <summary>Launches of the captured update graph (the update ran as a graph, after the exchange).</summary>
    internal long UpdateGraphReplayCount { get; private set; }

    /// <summary>Whether the step currently runs as a backward graph and an update graph.</summary>
    internal bool HasSplitStepGraph => _stepGraphExec != IntPtr.Zero && _updateGraphExec != IntPtr.Zero;

    /// <summary>
    /// Runs <paramref name="exchange"/> on every step's gradients after the backward and before the regularization,
    /// clip and optimizer update -- on the eager, resident and captured paths alike. On the captured path the step is
    /// captured as two graphs, the backward and the update, and the exchange runs between their launches, so the
    /// fused optimizer still runs as a graph on the reduced gradients. Null removes it. Drops any captured graph.
    /// </summary>
    internal void SetGradientExchange(Action<GradientExchange<T>>? exchange)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledTrainingPlan<T>));
        lock (_stepSync)
        {
            if (_stepGraphExec != IntPtr.Zero) InvalidateCapturedStepGraph();
            _gradientExchange = exchange;
            // The weight-buffer alternation would need a second pair of graphs; one buffer set keeps the split simple.
            if (exchange is not null) _pingPongRejected = true;
        }
    }

    // deviceAuthoritative: the step just ran as a graph, which writes the gradients' device buffers without the
    // host version bookkeeping an eager op keeps, so a bound device buffer is the current gradient.
    private void ExchangeGradients(bool deviceAuthoritative)
    {
        if (_gradientExchange is not { } exchange) return;
        var engine = _engine.As<Engines.DirectGpuTensorEngine>();
        var backend = engine?.GetBackend();
        var parameters = new List<Tensor<T>>(_parameters.Length);
        var gradients = new List<Tensor<T>>(_parameters.Length);
        var device = new List<Engines.DirectGpu.IGpuBuffer?>(_parameters.Length);
        for (int p = 0; p < _parameters.Length; p++)
        {
            var grad = _gradients[p];
            if (grad is null) continue;
            parameters.Add(_parameters[p]);
            gradients.Add(grad);
            // The same rule the L2 term and the optimizer use for where the gradient is current.
            device.Add(typeof(T) == typeof(float) && backend is not null
                && grad.TryGetGpuBuffer() is { } buffer
                && (deviceAuthoritative || grad._gpuBufferVersion == grad.GpuCacheVersion) ? buffer : null);
        }
        var loss = _lossOutput;
        var deviceLoss = typeof(T) == typeof(float) && backend is not null && loss.TryGetGpuBuffer() is { } lossBuffer
            && (deviceAuthoritative || loss._gpuBufferVersion == loss.GpuCacheVersion) ? lossBuffer : null;
        exchange(new GradientExchange<T>(parameters.ToArray(), gradients.ToArray(), device.ToArray(), backend, loss, deviceLoss));
        GradientExchangeCount++;
    }

    private void DestroyUpdateGraph(Engines.DirectGpuTensorEngine? gte)
    {
        if (_updateGraphExec != IntPtr.Zero && gte is not null) gte.DestroyGpuGraph(_updateGraphExec);
        _updateGraphExec = IntPtr.Zero;
    }
}
