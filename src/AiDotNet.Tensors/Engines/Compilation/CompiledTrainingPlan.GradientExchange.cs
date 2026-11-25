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

    // A device gradient or loss is only ever recorded with the backend that holds it.
    private Engines.DirectGpu.IDirectGpuBackend Backend
        => _backend ?? throw new InvalidOperationException("A gradient on the device was exchanged without its backend.");

    private readonly Tensor<T> _loss;
    private readonly Engines.DirectGpu.IGpuBuffer? _deviceLoss;

    // Host staging for the device gradients, owned by the plan and kept across its steps: [0] what a download fills
    // (the whole device buffer), [1] what an upload sends (exactly the gradient), each indexed like the gradients.
    // A step allocated both for every gradient before, twice the model's gradients in garbage per step.
    private readonly float[]?[][] _staging;

    // Page-locked staging for the whole step (every gradient in order, then the loss), owned by the plan and kept
    // across its steps; null where the backend has none. See TryReadStaged.
    private readonly Engines.DirectGpu.IGradientHostStaging? _pinned;
    private bool _readStaged;

    internal GradientExchange(Tensor<T>[] parameters, Tensor<T>[] gradients,
        Engines.DirectGpu.IGpuBuffer?[] deviceGradients, Engines.DirectGpu.IDirectGpuBackend? backend,
        Tensor<T> loss, Engines.DirectGpu.IGpuBuffer? deviceLoss, float[]?[][] staging,
        Engines.DirectGpu.IGradientHostStaging? pinned = null)
    {
        _pinned = pinned;
        _loss = loss;
        _deviceLoss = deviceLoss;
        _staging = staging;
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

    private float[] Staging(int kind, int index, int size)
    {
        var held = _staging[kind];
        if (held[index] is { } array && array.Length == size) return array;
        return held[index] = new float[size];
    }

    /// <summary>This rank's loss for the step (the plan's loss output, read where the step left it).</summary>
    public T ReadLoss()
    {
        if (_deviceLoss is { } buffer)
        {
            var host = new float[buffer.Size];
            Backend.DownloadBuffer(buffer, host);
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
                var host = Staging(0, i, buffer.Size);
                Backend.DownloadBuffer(buffer, host);
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

    /// <summary>
    /// Reads every gradient, in plan order, and the loss into the plan's page-locked staging: one copy per device
    /// gradient queued on the compute stream after the backward, then a single host wait. Returns false where there
    /// is no page-locked staging (a backend without it, or a non-float plan); <see cref="Read"/> and
    /// <see cref="Write"/> are the path then.
    /// </summary>
    /// <param name="gradients">The staged gradients (length <see cref="TotalLength"/>), valid until
    /// <see cref="WriteStaged"/>; the caller may overwrite them in place with the reduced gradients.</param>
    /// <param name="loss">This rank's loss for the step.</param>
    public bool TryReadStaged(out Span<float> gradients, out float loss)
    {
        gradients = default;
        loss = 0f;
        if (_pinned is null || typeof(T) != typeof(float)) return false;
        var host = _pinned.Host;
        if (host.Length < TotalLength + 1)
            throw new InvalidOperationException("The page-locked gradient staging is smaller than the step's gradients.");
        int offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is { } buffer) _pinned.EnqueueDownload(buffer, offset, length);
            offset += length;
        }
        if (_deviceLoss is { } lossBuffer) _pinned.EnqueueDownload(lossBuffer, TotalLength, 1);
        _pinned.WaitForTransfers();
        offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is null)
            {
                using var lease = _gradients[i].LeaseArray();
                ((float[])(object)lease.Array).AsSpan(0, length).CopyTo(host.Slice(offset, length));
            }
            offset += length;
        }
        if (_deviceLoss is null)
        {
            using var lease = _loss.LeaseArray();
            loss = ((float[])(object)lease.Array)[0];
        }
        else
        {
            loss = host[TotalLength];
        }
        gradients = host.Slice(0, TotalLength);
        _readStaged = true;
        return true;
    }

    /// <summary>
    /// Writes the staged gradients (as the caller left them after <see cref="TryReadStaged"/>) back where they were
    /// found. Device gradients are copied on the compute stream, ordered before the update that reads them, without
    /// a host wait.
    /// </summary>
    public void WriteStaged()
    {
        if (!_readStaged || _pinned is null)
            throw new InvalidOperationException("WriteStaged needs a TryReadStaged of the same step first.");
        var host = _pinned.Host;
        int offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is { } buffer)
            {
                _pinned.EnqueueUpload(buffer, offset, length);
            }
            else
            {
                using var lease = _gradients[i].LeaseWritable();
                var destination = lease.Span;
                var source = host.Slice(offset, length);
                for (int k = 0; k < length; k++)
                    destination[k] = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref source[k]);
            }
            offset += length;
        }
        _readStaged = false;
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
                // The upload sends the whole array, so it is exactly the gradient (a pooled device buffer can be larger).
                var host = buffer.Size == length ? Staging(0, i, length) : Staging(1, i, length);
                Array.Copy((float[])(object)source, offset, host, 0, length);
                Backend.UploadBufferInPlace(host, buffer);   // synchronous: done before the update launches
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

    // The exchange's host staging (see GradientExchange): one slot per exchanged gradient, kept across steps.
    private readonly float[]?[][] _exchangeStaging = { Array.Empty<float[]?>(), Array.Empty<float[]?>() };

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
        for (int kind = 0; kind < _exchangeStaging.Length; kind++)
            if (_exchangeStaging[kind].Length != gradients.Count) _exchangeStaging[kind] = new float[]?[gradients.Count];
        exchange(new GradientExchange<T>(parameters.ToArray(), gradients.ToArray(), device.ToArray(), backend, loss, deviceLoss,
            _exchangeStaging, PinnedExchangeStaging(backend, gradients)));
        GradientExchangeCount++;
    }

    // The exchange's page-locked staging (every gradient, then the loss), sized to the step and kept across steps.
    private Engines.DirectGpu.IGradientHostStaging? _exchangePinned;
    private int _exchangePinnedFloats;

    private Engines.DirectGpu.IGradientHostStaging? PinnedExchangeStaging(
        Engines.DirectGpu.IDirectGpuBackend? backend, List<Tensor<T>> gradients)
    {
        if (typeof(T) != typeof(float) || backend is not Engines.DirectGpu.IGradientHostStagingSource source) return null;
        long total = 1;
        foreach (var g in gradients) total += g.Length;
        if (total > int.MaxValue / sizeof(float)) return null;
        if (_exchangePinned is not null && _exchangePinnedFloats == total) return _exchangePinned;
        ReleasePinnedExchangeStaging();
        _exchangePinned = source.TryCreateGradientHostStaging((int)total);
        _exchangePinnedFloats = _exchangePinned is null ? 0 : (int)total;
        return _exchangePinned;
    }

    private void ReleasePinnedExchangeStaging()
    {
        _exchangePinned?.Dispose();
        _exchangePinned = null;
        _exchangePinnedFloats = 0;
    }

    private void DestroyUpdateGraph(Engines.DirectGpuTensorEngine? gte)
    {
        if (_updateGraphExec != IntPtr.Zero && gte is not null) gte.DestroyGpuGraph(_updateGraphExec);
        _updateGraphExec = IntPtr.Zero;
    }
}
