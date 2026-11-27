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

    // One contiguous device buffer of every gradient in plan order (TotalLength floats), owned by the plan; null when
    // the step does not coalesce. The device gradients are gathered into it on the compute stream (device-to-device),
    // so a host transfer is one large copy instead of one per gradient (many of them a few KB, which run at a fraction
    // of the link's bandwidth), and scattered back from it the same way, ordered before the update.
    private readonly Engines.DirectGpu.IGpuBuffer? _flat;

    internal GradientExchange(Tensor<T>[] parameters, Tensor<T>[] gradients,
        Engines.DirectGpu.IGpuBuffer?[] deviceGradients, Engines.DirectGpu.IDirectGpuBackend? backend,
        Tensor<T> loss, Engines.DirectGpu.IGpuBuffer? deviceLoss, float[]?[][] staging,
        Engines.DirectGpu.IGradientHostStaging? pinned = null, Engines.DirectGpu.IGpuBuffer? flat = null,
        GradientAccumulation<T>? accumulation = null, string? deferRefusal = null)
    {
        _accumulation = accumulation;
        _deferRefusal = accumulation is null ? deferRefusal ?? "the plan keeps no gradient accumulator" : deferRefusal;
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
        _flat = flat is not null && backend is not null && flat.Size >= TotalLength ? flat : null;
    }

    // The plan's accumulator across deferred steps (see DeferUpdate), and why this step cannot defer (null: it can).
    private readonly GradientAccumulation<T>? _accumulation;
    private readonly string? _deferRefusal;
    private bool _combined;
    // Whether reduced gradients were written back this step (Write, WriteStaged, UploadRangeFrom).
    private bool _written;

    /// <summary>Whether <see cref="DeferUpdate"/> ran on this step: the plan skips its update and keeps the gradients.</summary>
    internal bool UpdateDeferred { get; private set; }

    /// <summary>Whether <see cref="CombineAccumulated"/> ran on this step.</summary>
    internal bool CombinedAccumulation => _combined;

    /// <summary>The steps whose gradients the accumulator holds (deferred and not yet combined).</summary>
    public int AccumulatedSteps => _accumulation?.Steps ?? 0;

    /// <summary>Whether <see cref="DeferUpdate"/> or <see cref="CombineAccumulated"/> began writing the accumulator.</summary>
    internal bool AccumulatorTouched { get; private set; }

    /// <summary>
    /// Whether an accumulation write began and did not finish (it threw after its validation, and the callback may have
    /// caught that): the accumulator and this step's gradients are part-written, so the step cannot update or retry.
    /// </summary>
    internal bool AccumulationFaulted => AccumulatorTouched && !UpdateDeferred && !_combined;

    /// <summary>Why this step's update cannot be deferred, or null when it can.</summary>
    public string? DeferUnavailableReason => _deferRefusal;

    /// <summary>
    /// Gradient accumulation without a reduction (PyTorch DDP's <c>no_sync</c>): adds this step's gradients, multiplied
    /// by <paramref name="weight"/>, to the plan's accumulator, and the plan then skips this step's clip and optimizer
    /// update (its step count and learning-rate schedule do not advance; the loss is still published). The first
    /// deferred step sets the accumulator to <c>g * weight</c>, each later one adds <c>g * weight</c> to it; the multiply
    /// and the add are separate roundings (no fused multiply-add), so the sum is bitwise the eager
    /// <c>acc = g1 * w1; acc += gm * wm</c> computed with the same backend Scale and Add operations. The accumulator is
    /// per parameter, so a window may run its micro-batches through different plans that share it (see
    /// <see cref="CompiledTrainingPlan{T}.SetGradientExchange"/>). The step's own gradient buffers are left scaled. A
    /// later step ends the accumulation with <see cref="CombineAccumulated"/> before its exchange. An exchange callback
    /// that throws after this call drops the window (the accumulator already holds this step).
    /// </summary>
    /// <param name="weight">The step's weight (for a token-weighted mean, its valid-token count), rounded to the
    /// gradient type.</param>
    public void DeferUpdate(double weight)
    {
        if (_deferRefusal is { } why) throw new InvalidOperationException("This step's update cannot be deferred: " + why + ".");
        if (UpdateDeferred || _combined)
            throw new InvalidOperationException("This step already deferred its update or combined the accumulated gradients.");
        if (AccumulationFaulted)
            throw new InvalidOperationException(
                "An earlier accumulation write on this step failed part-way; the step cannot write the accumulator again.");
        if (double.IsNaN(weight) || double.IsInfinity(weight))
            throw new ArgumentOutOfRangeException(nameof(weight), "The weight must be finite.");
        var acc = _accumulation!;
        AccumulatorTouched = true;   // from here a throw leaves it part-written: the plan drops the window
        var slots = acc.Prepare(_parameters, _gradients, _deviceGradients, _backend);
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            var slot = slots[i];
            // Per parameter: a parameter the window has not held yet (its first micro-batch, or the first that reached
            // it) sets its slot; one it holds adds to it.
            bool first = !slot.Holds;
            slot.Holds = true;
            if (length == 0) continue;
            if (_deviceGradients[i] is { } buffer)
            {
                var held = slot.Device!;
                Backend.Scale(buffer, buffer, (float)weight, length);
                if (first) Backend.Copy(buffer, held, length);
                else Backend.Add(held, buffer, held, length);
            }
            else
            {
                var held = slot.Host!;
                using var lease = _gradients[i].LeaseWritable();
                var g = lease.Span.Slice(0, length);
                GradientAccumulation<T>.ScaleInPlace(g, weight);
                if (first) g.CopyTo(held);
                else GradientAccumulation<T>.AddInto(held, g, held);
            }
        }
        acc.Steps++;
        UpdateDeferred = true;
    }

    /// <summary>
    /// Ends a gradient accumulation: replaces this step's gradients with
    /// <c>acc * <paramref name="accumulatedScale"/> + g * <paramref name="weight"/></c> (the accumulator is scaled in
    /// place first, then this step's gradient, then the two are added: the eager <c>acc *= 1/N; acc += g * (n/N)</c>,
    /// each a separate rounding), and empties the accumulator. The exchange then reduces the combined gradients as it
    /// would any step's, and the plan's update runs on them.
    /// </summary>
    public void CombineAccumulated(double accumulatedScale, double weight)
    {
        if (_written)
            throw new InvalidOperationException(
                "The accumulated gradients must be combined before the exchange writes reduced gradients back: combined "
                + "after, the local accumulator would be added to the reduced gradient.");
        if (_accumulation is not { Steps: > 0 } acc)
            throw new InvalidOperationException("There are no accumulated gradients to combine.");
        if (UpdateDeferred || _combined)
            throw new InvalidOperationException("This step already deferred its update or combined the accumulated gradients.");
        if (AccumulationFaulted)
            throw new InvalidOperationException(
                "An earlier accumulation write on this step failed part-way; the step cannot write the accumulator again.");
        if (double.IsNaN(accumulatedScale) || double.IsInfinity(accumulatedScale))
            throw new ArgumentOutOfRangeException(nameof(accumulatedScale), "The scale must be finite.");
        if (double.IsNaN(weight) || double.IsInfinity(weight))
            throw new ArgumentOutOfRangeException(nameof(weight), "The weight must be finite.");
        AccumulatorTouched = true;   // from here a throw leaves it part-scaled: the plan drops the window
        acc.RequireHeldParametersIn(_parameters);
        var slots = acc.Prepare(_parameters, _gradients, _deviceGradients, _backend);
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (length == 0) continue;
            var slot = slots[i];
            if (_deviceGradients[i] is { } buffer)
            {
                Backend.Scale(buffer, buffer, (float)weight, length);
                if (!slot.Holds) continue;   // no deferred step reached it: acc is zero, so the sum is g * weight
                var held = slot.Device!;
                Backend.Scale(held, held, (float)accumulatedScale, length);
                Backend.Add(held, buffer, buffer, length);
            }
            else
            {
                using var lease = _gradients[i].LeaseWritable();
                var g = lease.Span.Slice(0, length);
                GradientAccumulation<T>.ScaleInPlace(g, weight);
                if (!slot.Holds) continue;
                var held = slot.Host!;
                GradientAccumulation<T>.ScaleInPlace(held, accumulatedScale);
                GradientAccumulation<T>.AddInto(held, g, g);
            }
        }
        acc.EndWindow();
        _combined = true;
    }

    /// <summary>Whether host transfers go through one contiguous device buffer (see the field).</summary>
    internal bool Coalesced => _flat is not null;

    // Device gradients -> the flat buffer, on the compute stream after the backward.
    private void GatherIntoFlat()
    {
        var flat = _flat!;
        int offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is { } buffer && length > 0) Backend.Copy(buffer, 0, flat, offset, length);
            offset += length;
        }
    }

    // The flat range [lo, hi) -> the device gradients it covers, on the compute stream (after the upload into it).
    private void ScatterFromFlat(int lo, int hi)
    {
        var flat = _flat!;
        int start = 0;
        for (int i = 0; i < _gradients.Length && start < hi; i++)
        {
            int length = _gradients[i].Length;
            int a = Math.Max(start, lo), b = Math.Min(start + length, hi);
            if (a < b && _deviceGradients[i] is { } buffer) Backend.Copy(flat, a, buffer, a - start, b - a);
            start += length;
        }
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
        if (_flat is not null)
        {
            GatherIntoFlat();
            if (TotalLength > 0) _pinned.EnqueueDownload(_flat, 0, TotalLength);
        }
        else
        {
            for (int i = 0; i < _gradients.Length; i++)
            {
                int length = _gradients[i].Length;
                if (_deviceGradients[i] is { } buffer) _pinned.EnqueueDownload(buffer, offset, length);
                offset += length;
            }
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
        _written = true;
        if (!_readStaged || _pinned is null)
            throw new InvalidOperationException("WriteStaged needs a TryReadStaged of the same step first.");
        var host = _pinned.Host;
        if (_flat is not null && TotalLength > 0)
        {
            _pinned.EnqueueUploadAt(_flat, 0, 0, TotalLength);
            ScatterFromFlat(0, TotalLength);
        }
        int offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is { } buffer)
            {
                if (_flat is null) _pinned.EnqueueUpload(buffer, offset, length);
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

    /// <summary>
    /// Page-locks caller-owned host memory (for example a shared-memory region every rank maps) as a staging area for
    /// <see cref="DownloadInto"/> and <see cref="UploadRangeFrom"/>, through the backend that holds the device
    /// gradients. Null, with the reason, where that is not possible (no device backend, a non-float plan, a backend
    /// without registration, or the driver refused). The caller disposes the result, which unregisters the memory,
    /// before the memory goes away.
    /// </summary>
    public Engines.DirectGpu.IGradientHostStaging? TryRegisterHostStaging(IntPtr host, int floats, out string? failure)
    {
        if (typeof(T) != typeof(float)) { failure = "the plan is not float"; return null; }
        if (_backend is not Engines.DirectGpu.IGradientHostStagingSource source)
        {
            failure = _backend is null ? "the step has no device backend" : "the backend has no page-locked staging";
            return null;
        }
        return source.TryRegisterGradientHostStaging(host, floats, out failure);
    }

    /// <summary>
    /// Copies every gradient, in plan order, into <paramref name="target"/> from float <paramref name="hostOffset"/>:
    /// one async copy per device gradient on the compute stream (after the backward), then a single wait; host
    /// gradients are copied directly.
    /// </summary>
    public void DownloadInto(Engines.DirectGpu.IGradientHostStaging target, int hostOffset)
    {
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (typeof(T) != typeof(float)) throw new NotSupportedException("Staged transfers are float only.");
        if (hostOffset < 0 || (long)hostOffset + TotalLength > target.Host.Length)
            throw new ArgumentOutOfRangeException(nameof(hostOffset), "The gradients do not fit the staging area.");
        int offset = 0;
        if (_flat is not null)
        {
            GatherIntoFlat();
            if (TotalLength > 0) target.EnqueueDownload(_flat, hostOffset, TotalLength);
        }
        else
        {
            for (int i = 0; i < _gradients.Length; i++)
            {
                int length = _gradients[i].Length;
                if (_deviceGradients[i] is { } buffer) target.EnqueueDownload(buffer, hostOffset + offset, length);
                offset += length;
            }
        }
        target.WaitForTransfers();
        var host = target.Host;
        offset = 0;
        for (int i = 0; i < _gradients.Length; i++)
        {
            int length = _gradients[i].Length;
            if (_deviceGradients[i] is null)
            {
                using var lease = _gradients[i].LeaseArray();
                ((float[])(object)lease.Array).AsSpan(0, length).CopyTo(host.Slice(hostOffset + offset, length));
            }
            offset += length;
        }
    }

    /// <summary>
    /// Writes the flat gradient range [<paramref name="flatOffset"/>, <paramref name="flatOffset"/> +
    /// <paramref name="count"/>) (plan order, as <see cref="DownloadInto"/> lays it out) back from
    /// <paramref name="source"/> at float <paramref name="hostOffset"/> + <paramref name="flatOffset"/>. Device parts
    /// are queued on the compute stream, ordered before the update; the host memory must not change until
    /// <see cref="Engines.DirectGpu.IGradientHostStaging.WaitForTransfers"/> returns.
    /// </summary>
    public void UploadRangeFrom(Engines.DirectGpu.IGradientHostStaging source, int hostOffset, int flatOffset, int count)
    {
        _written = true;
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (typeof(T) != typeof(float)) throw new NotSupportedException("Staged transfers are float only.");
        if (flatOffset < 0 || count < 0 || (long)flatOffset + count > TotalLength)
            throw new ArgumentOutOfRangeException(nameof(count), "The range is outside the gradients.");
        int end = flatOffset + count;
        if (_flat is not null && count > 0)
        {
            // One upload of the range into the flat buffer, then device-to-device into the gradients it covers.
            source.EnqueueUploadAt(_flat, flatOffset, hostOffset + flatOffset, count);
            ScatterFromFlat(flatOffset, end);
        }
        int start = 0;
        for (int i = 0; i < _gradients.Length && start < end; i++)
        {
            int length = _gradients[i].Length;
            int lo = Math.Max(start, flatOffset), hi = Math.Min(start + length, end);
            if (lo < hi)
            {
                if (_deviceGradients[i] is { } buffer)
                {
                    if (_flat is null) source.EnqueueUploadAt(buffer, lo - start, hostOffset + lo, hi - lo);
                }
                else
                {
                    using var lease = _gradients[i].LeaseWritable();
                    var destination = lease.Span;
                    var from = source.Host.Slice(hostOffset + lo, hi - lo);
                    for (int k = 0; k < from.Length; k++)
                        destination[lo - start + k] = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref from[k]);
                }
            }
            start += length;
        }
    }

    /// <summary>Writes the reduced gradients back where <see cref="Read"/> found them.</summary>
    public void Write(T[] source)
    {
        _written = true;
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

/// <summary>
/// A compiled plan's gradient accumulator across steps whose update was deferred (<see
/// cref="GradientExchange{T}.DeferUpdate"/>): one buffer per exchanged gradient, on the device where the gradient is a
/// device buffer and on the host otherwise, owned by the plan and kept across accumulation windows.
/// </summary>
internal sealed class GradientAccumulation<T> : IDisposable
{
    /// <summary>One parameter's accumulated gradient: on the device (float only) or on the host, where its gradient is.</summary>
    internal sealed class Slot
    {
        internal readonly int Length;
        internal Engines.DirectGpu.IGpuBuffer? Device;
        internal T[]? Host;
        // Whether the slot holds a deferred step's gradient in this window (false: its contents are stale, and the next
        // deferral sets it instead of adding to it, as PyTorch accumulates into a parameter whose grad was None).
        internal bool Holds;
        internal Slot(int length) { Length = length; }
    }

    // Keyed by the parameter, not by its position in a plan: plans compiled for different batch shapes order and
    // select their parameters on their own, and a window may run its micro-batches through more than one of them
    // (PyTorch's no_sync accumulates into param.grad whatever graph produced it).
    private readonly Dictionary<Tensor<T>, Slot> _slots =
        new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private Engines.DirectGpu.IDirectGpuBackend? _backend;

    /// <summary>The deferred steps the accumulator holds (0: none; the slots' contents are then stale).</summary>
    internal int Steps { get; set; }

    /// <summary>The parameters whose slots hold accumulated gradients in the current window, for tests.</summary>
    internal int HeldParameterCountForTests
    {
        get { int n = 0; foreach (var slot in _slots.Values) if (slot.Holds) n++; return n; }
    }

    /// <summary>
    /// This step's slots, one per gradient and in its order, each where the step's gradient is (a held slot whose
    /// gradient moved between the host and the device moves with it). A parameter whose gradient length changed while
    /// the window holds it, or a parameter listed twice, drops the window and throws.
    /// </summary>
    internal Slot[] Prepare(Tensor<T>[] parameters, Tensor<T>[] gradients, Engines.DirectGpu.IGpuBuffer?[] device,
        Engines.DirectGpu.IDirectGpuBackend? backend)
    {
        if (_backend is not null && !ReferenceEquals(backend, _backend))
        {
            // Another backend cannot read these buffers: the held values go to the host first.
            foreach (var held in _slots.Values) MoveToHost(held);
            _backend = null;
        }
        var result = new Slot[gradients.Length];
        var seen = new HashSet<Tensor<T>>(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
        for (int i = 0; i < gradients.Length; i++)
        {
            var parameter = parameters[i];
            int length = gradients[i].Length;
            if (!seen.Add(parameter))
            {
                Release();
                throw new InvalidOperationException(
                    "A parameter appears twice in one step's gradients; the accumulation window was dropped.");
            }
            if (_slots.TryGetValue(parameter, out var slot) && slot.Length != length)
            {
                if (slot.Holds)
                {
                    Release();
                    throw new InvalidOperationException(
                        "A parameter's gradient changed length during a gradient accumulation; the accumulated steps were dropped.");
                }
                slot.Device?.Dispose();
                _slots.Remove(parameter);
                slot = null;
            }
            if (slot is null) _slots[parameter] = slot = new Slot(length);
            result[i] = slot;
            if (length == 0) continue;
            if (typeof(T) == typeof(float) && device[i] is not null && backend is not null)
            {
                if (slot.Device is not null) continue;
                Engines.DirectGpu.IGpuBuffer buffer;
                // The accumulator's own memory for its life: never a loan of idle step memory.
                using (Engines.DirectGpu.CUDA.CudaBackend.NoIdleLoans())
                    buffer = backend.AllocateBuffer(length);
                if (slot.Holds && slot.Host is { } values)
                {
                    // The buffer is owned by nothing until Device[i] takes it: a failed upload must not leak it.
                    try
                    {
                        var upload = new float[length];
                        Array.Copy(values, upload, length);   // T is float on this path
                        backend.UploadBufferInPlace(upload, buffer);
                    }
                    catch
                    {
                        buffer.Dispose();
                        throw;
                    }
                }
                slot.Device = buffer;
                slot.Host = null;
                _backend = backend;
            }
            else
            {
                if (slot.Device is not null) MoveToHost(slot);
                slot.Host ??= new T[length];
            }
        }
        return result;
    }

    /// <summary>
    /// Throws (dropping the window) when a parameter accumulated gradients in this window but has no gradient in the
    /// step that ends it: that step's plan can neither exchange nor update it, and dropping its accumulated gradient
    /// would end the window for the other parameters only.
    /// </summary>
    internal void RequireHeldParametersIn(Tensor<T>[] parameters)
    {
        var present = new HashSet<Tensor<T>>(parameters, Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
        foreach (var pair in _slots)
        {
            if (!pair.Value.Holds || present.Contains(pair.Key)) continue;
            Release();
            throw new InvalidOperationException(
                "A parameter accumulated gradients in this window but has no gradient in the step that ends it, so the "
                + "update cannot include it; the accumulation window was dropped.");
        }
    }

    /// <summary>Ends the window: every slot's contents become stale (the buffers are kept for the next window).</summary>
    internal void EndWindow()
    {
        foreach (var slot in _slots.Values) slot.Holds = false;
        Steps = 0;
    }

    private void MoveToHost(Slot slot)
    {
        if (slot.Device is not { } buffer) return;
        if (typeof(T) != typeof(float))
            throw new InvalidOperationException("A device gradient accumulator exists only for a float plan.");
        var values = new T[slot.Length];
        if (slot.Holds)
        {
            var owner = _backend
                ?? throw new InvalidOperationException("A device gradient accumulator has no backend to download it from.");
            var host = new float[buffer.Size];
            owner.DownloadBuffer(buffer, host);
            Array.Copy(host, values, values.Length);   // T is float on this path
        }
        buffer.Dispose();
        slot.Device = null;
        slot.Host = values;
    }

    // values *= factor (rounded to T), element by element: one rounding per element.
    [Helpers.ScalarKernel("Element-wise scale of a host-resident accumulated gradient; float and double are specialized")]
    internal static void ScaleInPlace(Span<T> values, double factor)
    {
        if (typeof(T) == typeof(float))
        {
            float s = (float)factor;
            for (int k = 0; k < values.Length; k++)
            {
                ref float f = ref System.Runtime.CompilerServices.Unsafe.As<T, float>(ref values[k]);
                f *= s;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            for (int k = 0; k < values.Length; k++)
            {
                ref double d = ref System.Runtime.CompilerServices.Unsafe.As<T, double>(ref values[k]);
                d *= factor;
            }
        }
        else
        {
            var ops = Helpers.MathHelper.GetNumericOperations<T>();
            T s = ops.FromDouble(factor);
            for (int k = 0; k < values.Length; k++) values[k] = ops.Multiply(values[k], s);
        }
    }

    // destination = a + b, element by element (destination may be either operand).
    [Helpers.ScalarKernel("Element-wise add of host-resident accumulated gradients; float and double are specialized")]
    internal static void AddInto(Span<T> a, Span<T> b, Span<T> destination)
    {
        if (typeof(T) == typeof(float))
        {
            for (int k = 0; k < destination.Length; k++)
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref destination[k]) =
                    System.Runtime.CompilerServices.Unsafe.As<T, float>(ref a[k])
                    + System.Runtime.CompilerServices.Unsafe.As<T, float>(ref b[k]);
        }
        else if (typeof(T) == typeof(double))
        {
            for (int k = 0; k < destination.Length; k++)
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref destination[k]) =
                    System.Runtime.CompilerServices.Unsafe.As<T, double>(ref a[k])
                    + System.Runtime.CompilerServices.Unsafe.As<T, double>(ref b[k]);
        }
        else
        {
            var ops = Helpers.MathHelper.GetNumericOperations<T>();
            for (int k = 0; k < destination.Length; k++) destination[k] = ops.Add(a[k], b[k]);
        }
    }

    /// <summary>Drops the held steps and frees the buffers.</summary>
    internal void Release()
    {
        foreach (var slot in _slots.Values) slot.Device?.Dispose();
        _slots.Clear();
        _backend = null;
        Steps = 0;
    }

    public void Dispose() => Release();
}

internal sealed partial class CompiledTrainingPlan<T>
{
    private Action<GradientExchange<T>>? _gradientExchange;

    // The accumulator for deferred updates (gradient accumulation through the exchange): the one the caller handed to
    // SetGradientExchange (shared by every plan a window's micro-batches may run through), or else the plan's own,
    // made on the first exchange.
    private GradientAccumulation<T>? _gradientAccumulation;
    private bool _ownsGradientAccumulation;

    // Frees the accumulator only when it is the plan's own; a caller's belongs to the caller (and to its other plans).
    private void ReleaseOwnGradientAccumulation()
    {
        if (_ownsGradientAccumulation) _gradientAccumulation?.Release();
        _gradientAccumulation = null;
        _ownsGradientAccumulation = false;
    }

    // Whether the last exchange deferred its step's update (the plan then skipped the clip and the optimizer).
    private bool _lastStepUpdateDeferred;

    // Whether the last launched split replay ran the epilogue-only graph (no update, so no finiteness verdict).
    private bool _lastLaunchDeferred;

    // The replay index of the last captured launch that ran the update (whose slot holds the step's verdict); null
    // before any since the capture.
    private uint? _lastUpdateReplay;

    // The epilogue-only graph a split step launches instead of the update graph when the exchange deferred the update:
    // it publishes the loss and the progress word without counting an applied step.
    private IntPtr _epilogueGraphExec;

    /// <summary>Steps whose exchange deferred the update (their gradients were accumulated instead).</summary>
    internal long DeferredUpdateCount { get; private set; }

    /// <summary>The captured epilogue-only graph (zero when there is none), for tests.</summary>
    internal IntPtr EpilogueGraphHandleForTests => _epilogueGraphExec;

    /// <summary>The deferred steps whose gradients the plan's accumulator holds now.</summary>
    internal int AccumulatedGradientSteps => _gradientAccumulation?.Steps ?? 0;

    // Set while the exchange callback runs (see SetGradientExchange).
    private bool _exchangeCallbackRunning;

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
    /// <paramref name="accumulation"/> is where deferred steps accumulate (see <see cref="GradientExchange{T}.DeferUpdate"/>):
    /// one accumulator handed to every plan a training loop switches between (a plan per batch shape) carries a window
    /// across them, per parameter. The caller owns it (the plan never frees it); null gives the plan its own.
    /// </summary>
    internal void SetGradientExchange(Action<GradientExchange<T>>? exchange, GradientAccumulation<T>? accumulation = null)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledTrainingPlan<T>));
        lock (_stepSync)
        {
            // The running callback still holds its step's exchange, whose accumulator the replacement would clear and
            // the old callback could then refill for the new exchange to combine. Replace between steps.
            if (_exchangeCallbackRunning)
                throw new InvalidOperationException("The gradient exchange cannot be replaced from inside its own callback.");
            if (_stepGraphExec != IntPtr.Zero) InvalidateCapturedStepGraph();
            // The plan's own accumulated gradients belong to the exchange that deferred them; a caller's stays theirs.
            bool droppedWindow = _ownsGradientAccumulation && _gradientAccumulation is { Steps: > 0 };
            _gradientExchange = exchange;
            ReleaseOwnGradientAccumulation();
            if (exchange is not null && accumulation is not null) _gradientAccumulation = accumulation;
            _lastStepUpdateDeferred = false;
            // The weight-buffer alternation would need a second pair of graphs; one buffer set keeps the split simple.
            if (exchange is not null) _pingPongRejected = true;
            if (droppedWindow)
                throw new InvalidOperationException(
                    "The gradient exchange was replaced with accumulated steps pending in the plan's own accumulator; the "
                    + "accumulation window was dropped. Hand one accumulator to SetGradientExchange to carry a window "
                    + "across a change of exchange or plan.");
        }
    }

    // deviceAuthoritative: the step just ran as a graph, which writes the gradients' device buffers without the
    // host version bookkeeping an eager op keeps, so a bound device buffer is the current gradient.
    // Returns whether the exchange deferred the step's update (the caller then skips the clip and the optimizer).
    private bool ExchangeGradients(bool deviceAuthoritative)
    {
        _lastStepUpdateDeferred = false;
        if (_gradientExchange is not { } exchange) return false;
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
        var deviceArray = device.ToArray();
        if (_gradientAccumulation is null)
        {
            _gradientAccumulation = new GradientAccumulation<T>();
            _ownsGradientAccumulation = true;
        }
        string? deferRefusal = _lossScaling is not null
            ? "loss scaling is configured (a deferred step's overflow check is not supported)"
            : _preForwardParamTransform is not null
            // Schedule-Free writes its evaluation point y into the live parameters before the forward, and only the
            // update puts x back: a deferred step would leave y in the parameters until the next accepted update.
            ? "the optimizer evaluates the forward at transformed parameters (Schedule-Free), which only its update restores"
            : null;
        var step = new GradientExchange<T>(parameters.ToArray(), gradients.ToArray(), deviceArray, backend, loss, deviceLoss,
            _exchangeStaging, PinnedExchangeStaging(backend, gradients), FlatExchangeBuffer(backend, gradients, deviceArray),
            _gradientAccumulation, deferRefusal);
        try
        {
            _exchangeCallbackRunning = true;
            try { exchange(step); }
            finally { _exchangeCallbackRunning = false; }
        }
        catch when (step.AccumulatorTouched)
        {
            // The accumulator took (some of) this step's gradients, or was (partly) folded into them: a retried step
            // would count twice, and a combined window lives only in this step's gradients. The window is dropped.
            _gradientAccumulation.Release();
            throw;
        }
        if (step.AccumulationFaulted)
        {
            // An accumulation write threw part-way and the callback returned anyway: the gradients are part-scaled, so
            // the update must not run on them, and the part-written window is dropped.
            _gradientAccumulation.Release();
            throw new InvalidOperationException(
                "An accumulation write failed part-way inside the gradient exchange; the step was refused and the "
                + "accumulation window was dropped.");
        }
        if (!step.UpdateDeferred && !step.CombinedAccumulation && _gradientAccumulation.Steps > 0)
        {
            // An update would run with the window still open, and the next combine would mix gradients from before and
            // after it. The window is dropped and the step refused before its update.
            _gradientAccumulation.Release();
            throw new InvalidOperationException(
                "The exchange neither deferred this step's update nor combined the accumulated gradients, with "
                + "accumulated steps pending; the accumulation window was dropped.");
        }
        GradientExchangeCount++;
        if (step.UpdateDeferred)
        {
            DeferredUpdateCount++;
            _lastStepUpdateDeferred = true;
        }
        return step.UpdateDeferred;
    }

    /// <summary>
    /// Whether the exchange's host transfers go through one contiguous device buffer of the gradients (on by default:
    /// one large copy each way instead of one per gradient). Off keeps a copy per gradient and no extra device memory.
    /// </summary>
    internal bool CoalesceExchangeTransfers { get; set; } = true;

    // The flat device buffer (every gradient in plan order), sized to the step and kept across steps, for the backend
    // that made it.
    private Engines.DirectGpu.IGpuBuffer? _exchangeFlat;
    private Engines.DirectGpu.IDirectGpuBackend? _exchangeFlatBackend;

    private Engines.DirectGpu.IGpuBuffer? FlatExchangeBuffer(Engines.DirectGpu.IDirectGpuBackend? backend,
        List<Tensor<T>> gradients, Engines.DirectGpu.IGpuBuffer?[] device)
    {
        if (!CoalesceExchangeTransfers || typeof(T) != typeof(float) || backend is null) return null;
        long total = 0;
        int onDevice = 0;
        for (int i = 0; i < gradients.Count; i++)
        {
            total += gradients[i].Length;
            if (device[i] is not null) onDevice++;
        }
        if (onDevice < 2 || total <= 0 || total > int.MaxValue) return null;   // nothing to coalesce
        if (_exchangeFlat is { } held && held.Size >= total && ReferenceEquals(_exchangeFlatBackend, backend)) return held;
        ReleaseFlatExchangeBuffer();
        // The plan's own memory for its life: never a loan of idle step memory (a loan would be moved at the next step).
        using (Engines.DirectGpu.CUDA.CudaBackend.NoIdleLoans())
            _exchangeFlat = backend.AllocateBuffer((int)total);
        _exchangeFlatBackend = backend;
        return _exchangeFlat;
    }

    private void ReleaseFlatExchangeBuffer()
    {
        _exchangeFlat?.Dispose();
        _exchangeFlat = null;
        _exchangeFlatBackend = null;
    }

    // The exchange's page-locked staging (every gradient, then the loss), sized to the step and kept across steps.
    // Kept for the backend that made it: a different (or recreated) backend gets its own.
    private Engines.DirectGpu.IGradientHostStaging? _exchangePinned;
    private int _exchangePinnedFloats;
    private Engines.DirectGpu.IDirectGpuBackend? _exchangePinnedBackend;

    /// <summary>Why the exchange has no page-locked staging (null while it has one, or before the first exchange).</summary>
    internal string? PinnedExchangeStagingUnavailableReason { get; private set; }

    private Engines.DirectGpu.IGradientHostStaging? PinnedExchangeStaging(
        Engines.DirectGpu.IDirectGpuBackend? backend, List<Tensor<T>> gradients)
    {
        string? why = typeof(T) != typeof(float) ? "the plan is not float"
            : backend is null ? "the step has no device backend"
            : backend is not Engines.DirectGpu.IGradientHostStagingSource ? "the backend has no page-locked staging"
            : null;
        long total = 1;
        foreach (var g in gradients) total += g.Length;
        if (why is null && total > int.MaxValue) why = $"the gradients ({total} floats) exceed one staging area";
        if (why is not null) return NoPinnedExchangeStaging(why);
        if (_exchangePinned is not null && _exchangePinnedFloats == total && ReferenceEquals(_exchangePinnedBackend, backend))
            return _exchangePinned;
        ReleasePinnedExchangeStaging();
        try
        {
            _exchangePinned = ((Engines.DirectGpu.IGradientHostStagingSource)backend!).TryCreateGradientHostStaging((int)total);
        }
        catch (Exception ex) when (ex is InvalidOperationException or OutOfMemoryException or DllNotFoundException)
        {
            return NoPinnedExchangeStaging("the page-locked allocation failed: " + ex.Message);
        }
        if (_exchangePinned is null) return NoPinnedExchangeStaging("the backend could not create page-locked staging");
        _exchangePinnedFloats = (int)total;
        _exchangePinnedBackend = backend;
        PinnedExchangeStagingUnavailableReason = null;
        return _exchangePinned;
    }

    private Engines.DirectGpu.IGradientHostStaging? NoPinnedExchangeStaging(string why)
    {
        if (PinnedExchangeStagingUnavailableReason != why)
            System.Diagnostics.Trace.TraceWarning("Gradient exchange: no page-locked staging (" + why
                + "); the exchange copies each gradient on its own.");
        PinnedExchangeStagingUnavailableReason = why;
        return null;
    }

    private void ReleasePinnedExchangeStaging()
    {
        _exchangePinned?.Dispose();
        _exchangePinned = null;
        _exchangePinnedFloats = 0;
        _exchangePinnedBackend = null;
    }

    private void DestroyUpdateGraph(Engines.DirectGpuTensorEngine? gte)
    {
        var update = _updateGraphExec;
        var epilogue = _epilogueGraphExec;
        // Cleared first: a destroy that throws must not leave a handle to be destroyed again.
        _updateGraphExec = IntPtr.Zero;
        _epilogueGraphExec = IntPtr.Zero;
        try
        {
            if (update != IntPtr.Zero && gte is not null) gte.DestroyGpuGraph(update);
        }
        finally
        {
            if (epilogue != IntPtr.Zero && gte is not null) gte.DestroyGpuGraph(epilogue);
        }
    }
}
