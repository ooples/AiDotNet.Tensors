using System;
using System.Buffers;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Shared scaffolding for the concrete optimizers: param groups, default hyper-parameters,
/// per-parameter state slots, and state-dict serialisation. Subclasses only have to:
///   1) declare the hyper-parameter <see cref="Defaults"/>,
///   2) declare which state slots each parameter needs in <see cref="StateNames"/>,
///   3) implement <see cref="Step"/> by walking <see cref="ParamGroups"/>.
/// </summary>
public abstract class OptimizerBase : IOptimizer
{
    private readonly List<ParamGroup> _groups = new List<ParamGroup>();
    /// <summary>Per-parameter state buffers, keyed by (group index, param index) → state name → value.</summary>
    protected readonly Dictionary<(int g, int p), Dictionary<string, OptimizerStateValue>> _state =
        new Dictionary<(int, int), Dictionary<string, OptimizerStateValue>>();

    /// <summary>Access to the state map for sharded-optimizer wrappers that need to snapshot
    /// and restore non-local parameter state across <see cref="Step"/> calls.</summary>
    internal Dictionary<(int g, int p), Dictionary<string, OptimizerStateValue>> StateInternal => _state;

    /// <inheritdoc />
    public IReadOnlyList<ParamGroup> ParamGroups => _groups;

    /// <summary>Default hyper-parameters injected into every newly added group.</summary>
    protected abstract IReadOnlyDictionary<string, double> Defaults { get; }

    /// <summary>Names of the per-parameter state slots required by this optimizer.</summary>
    protected abstract IReadOnlyList<string> StateNames { get; }

    /// <summary>
    /// Names of state slots that hold a single scalar per parameter (e.g. <c>"step"</c>,
    /// <c>"eta"</c>, <c>"mu"</c>) rather than a length-N tensor. <see cref="GetOrCreateState"/>
    /// allocates these as zero-initialised <see cref="OptimizerStateValue"/> placeholders
    /// instead of full tensor buffers, eliminating wasted memory on every parameter.
    /// Default: only <c>"step"</c> is treated as a scalar.
    /// </summary>
    protected virtual IReadOnlyList<string> ScalarStateNames { get; } = new[] { "step" };

    /// <inheritdoc />
    public abstract void Step();

    // Elements per parallel chunk of an element-wise update. Fixed (not derived from the core count) and the update
    // is element-wise, so the result does not depend on the thread count.
    private protected const int ElementwiseChunk = 64 * 1024;

    // The gradients bound by Step(gradients), keyed by parameter tensor; null outside that call.
    private IReadOnlyDictionary<Tensor<float>, Tensor<float>>? _boundGradients;

    // The pooled buffer holding the current parameter's effective gradient (negated for maximize, plus coupled weight
    // decay) or a contiguous copy of a strided one. Returned when the next parameter needs one and when a step ends.
    private float[]? _gradientScratch;

    // Negated sparse values for a maximize group; grown, never shrunk.
    private float[] _sparseScratch = Array.Empty<float>();

    /// <summary>
    /// Which parameters a step updates, as (group index, parameter index) → true; null updates every parameter.
    /// Set by <see cref="ZeroShardedOptimizer"/> to restrict a step to its rank's shard.
    /// </summary>
    internal Func<int, int, bool>? StepFilter { get; set; }

    /// <summary>
    /// Combines a group-wide statistic across ranks in place (a sum). Set by <see cref="ZeroShardedOptimizer"/> so an
    /// optimizer whose update depends on statistics over every parameter of a group (<see cref="HasGroupStatistics"/>)
    /// sees the whole group while each rank steps only its shard.
    /// </summary>
    internal Action<double[]>? GroupStatisticsReducer { get; set; }

    /// <summary>True when an update reads statistics summed over all of a group's parameters (D-Adaptation, Prodigy).</summary>
    internal virtual bool HasGroupStatistics => false;

    /// <summary>Sums <paramref name="statistics"/> across ranks when the step is sharded; a no-op otherwise.</summary>
    private protected void ReduceGroupStatistics(double[] statistics) => GroupStatisticsReducer?.Invoke(statistics);

    /// <summary>
    /// One optimization step that reads each tensor parameter's gradient straight from <paramref name="gradients"/>
    /// (typically the dictionary <c>GradientTape.ComputeGradients</c> returns), with no copy into
    /// <see cref="ParamGroup.Gradients"/>. A tensor parameter missing from the dictionary is skipped, as PyTorch skips a
    /// parameter whose <c>.grad</c> is <c>None</c>. Parameters added as arrays keep reading their own gradient buffers.
    /// The gradients are never written.
    /// </summary>
    /// <param name="gradients">Gradient per parameter tensor; each must have its parameter's element count.</param>
    public void Step(IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients)
    {
        if (gradients == null) throw new ArgumentNullException(nameof(gradients));
        _boundGradients = gradients;
        try { Step(); }
        finally { _boundGradients = null; }
    }

    // Device parameters a host-computed step downloaded into their staging arrays; written back when the step ends.
    private readonly List<(Tensor<float> Tensor, float[] Staging)> _stagedDeviceParameters =
        new List<(Tensor<float> Tensor, float[] Staging)>();

    /// <summary>
    /// The host array the update of <c>group[gi].param[pi]</c> writes. For a GPU-resident parameter the optimizer
    /// had no device kernel for, that is a staging copy downloaded now and written back to the device when the step
    /// ends (recorded as a fallback: correct, but it crosses the device boundary).
    /// </summary>
    private protected float[] HostParameter(int gi, int pi)
    {
        var group = _groups[gi];
        var parameter = group.Parameters[pi];
        var tensor = group.ParameterTensor(pi);
        if (tensor is null || !group.IsDeviceParameter(pi)) return parameter;
        var current = Gpu.GpuOptimizer.TryDownload(tensor)
            ?? throw new InvalidOperationException("A GPU parameter has no device buffer to update.");
        Array.Copy(current, parameter, parameter.Length);
        _stagedDeviceParameters.Add((tensor, parameter));
        DirectGpu.GpuLaunchProbe.OnFallback($"{GetType().Name}-host-step-of-a-device-parameter", null);
        return parameter;
    }

    /// <summary>
    /// Runs this step's update of <c>group[gi].param[pi]</c> on the GPU when the parameter and its gradient are both
    /// there and the optimizer has a device kernel for its configuration; false sends it down the host path.
    /// </summary>
    private protected bool StepOnDevice(int gi, int pi)
    {
        var group = _groups[gi];
        var tensor = group.ParameterTensor(pi);
        if (tensor is null || !group.IsDeviceParameter(pi) || _boundGradients is null || HasSparseGradient(gi, pi))
            return false;
        if (!_boundGradients.TryGetValue(tensor, out var gradient) || !gradient.IsGpuResident) return false;
        if (gradient.Length != tensor.Length)
            throw new ArgumentException(
                $"A gradient has {gradient.Length} elements but its parameter has {tensor.Length}.", "gradients");
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine engine)) return false;
        // The kernels descend; ascent descends the negated gradient (a new device tensor, the caller's is untouched).
        if (group.GetOption("maximize", 0.0) != 0.0) gradient = engine.TensorNegate(gradient);
        return TryStepOnDevice(gi, pi, tensor, gradient);
    }

    /// <summary>
    /// The optimizer's device update of one GPU parameter with a GPU gradient (maximize already applied), using
    /// <see cref="DeviceState"/> for its state. False when it has no kernel for the group's configuration.
    /// </summary>
    private protected virtual bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
        => false;

    /// <summary>
    /// The state record of a device-stepped parameter: scalars on the host as usual, every buffer slot a GPU tensor
    /// (<see cref="OptimizerStateValue.DeviceTensor"/>), created zeroed or uploaded from a host value it already had.
    /// </summary>
    private protected Dictionary<string, OptimizerStateValue> DeviceState(int gi, int pi, int length)
    {
        var slot = GetOrCreateStateRecord(gi, pi, length, onDevice: true);
        foreach (var value in slot.Values)
        {
            if (value.DeviceTensor is not null) continue;
            if (value.Tensor is not null)
            {
                var device = Gpu.GpuOptimizer.CreateStateTensor(new[] { value.Tensor.Length });
                if (!Gpu.GpuOptimizer.TryUpload(device, value.Tensor))
                    throw new InvalidOperationException("Optimizer state could not be placed on the GPU.");
                value.DeviceTensor = device;
                value.Tensor = null;
            }
        }
        return slot;
    }

    /// <summary>A device state record's GPU buffer for <paramref name="name"/>.</summary>
    private protected static Tensor<float> DeviceSlot(Dictionary<string, OptimizerStateValue> slot, string name)
        => slot[name].DeviceTensor ?? throw new InvalidOperationException($"State '{name}' is not on the GPU.");

    // Moves every device-resident buffer of a state record back to the host (a host step is about to read it).
    private static void MoveStateToHost(Dictionary<string, OptimizerStateValue> slot)
    {
        foreach (var value in slot.Values)
        {
            if (value.DeviceTensor is null) continue;
            value.Tensor = Gpu.GpuOptimizer.TryDownload(value.DeviceTensor) ?? value.DeviceTensor.ToArray();
            value.DeviceTensor = null;
        }
    }

    /// <summary>Starts a step: re-reads tensor parameters' storage. Every <see cref="Step()"/> calls it first.</summary>
    private protected void BeginStep()
    {
        foreach (var group in _groups) group.RefreshTensorParameters();
    }

    /// <summary>
    /// Ends a step (from a <c>finally</c>): marks every updated tensor parameter modified, so data derived from it
    /// (packed weights, device copies) is refreshed, and drops this step's sparse gradients and scratch.
    /// </summary>
    private protected void EndStep()
    {
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var group = _groups[gi];
            for (int pi = 0; pi < group.Parameters.Count; pi++)
            {
                // A device-stepped parameter was marked current on the device by its kernel; marking it modified
                // here would make the stale host copy look newer.
                var tensor = group.ParameterTensor(pi);
                if (tensor is not null && !group.IsDeviceParameter(pi) && ShouldStep(gi, pi)) tensor.MarkModified();
            }
        }
        foreach (var (tensor, staging) in _stagedDeviceParameters)
            if (!Gpu.GpuOptimizer.TryUpload(tensor, staging))
                throw new InvalidOperationException("A GPU parameter's host-computed update could not be written back.");
        _stagedDeviceParameters.Clear();
        ReturnGradientScratch();
        ClearAutoClearSparseGrads();
    }

    /// <summary>
    /// Whether this step updates <c>group[gi].param[pi]</c>: false outside the <see cref="StepFilter"/> shard, and for
    /// a tensor parameter that <see cref="Step(IReadOnlyDictionary{Tensor{float}, Tensor{float}})"/> was given no
    /// gradient for.
    /// </summary>
    private protected bool ShouldStep(int gi, int pi)
    {
        if (StepFilter is not null && !StepFilter(gi, pi)) return false;
        var tensor = _groups[gi].ParameterTensor(pi);
        return tensor is null || _boundGradients is null || _boundGradients.ContainsKey(tensor);
    }

    /// <summary>A read-only gradient: <see cref="Length"/> elements of <see cref="Array"/> from <see cref="Offset"/>.</summary>
    private protected readonly struct GradientBuffer
    {
        public GradientBuffer(float[] array, int offset, int length)
        {
            Array = array;
            Offset = offset;
            Length = length;
        }

        public float[] Array { get; }

        public int Offset { get; }

        public int Length { get; }

        public ReadOnlySpan<float> Span => new ReadOnlySpan<float>(Array, Offset, Length);
    }

    /// <summary>
    /// The dense gradient the update of <c>group[gi].param[pi]</c> reads: the bound tensor's storage (or a contiguous
    /// copy of a strided one) or the group's gradient buffer — negated when the group maximizes, plus
    /// <paramref name="coupledWeightDecay"/>·parameter for L2-coupled weight decay (PyTorch's
    /// <c>grad = grad.add(param, alpha=weight_decay)</c>). Those two land in scratch: the caller's gradient is never
    /// written. Valid until the next call or the end of the step.
    /// </summary>
    private protected GradientBuffer DenseGradient(int gi, int pi, float[] parameter, float coupledWeightDecay = 0f)
    {
        var group = _groups[gi];
        int length = parameter.Length;
        float[] source;
        int offset = 0;
        var tensor = group.ParameterTensor(pi);
        if (tensor is not null && _boundGradients is not null)
        {
            if (!_boundGradients.TryGetValue(tensor, out var bound))
                throw new InvalidOperationException("No gradient was given for this parameter; ShouldStep skips it.");
            if (bound.Length != length)
                throw new ArgumentException(
                    $"A gradient has {bound.Length} elements but its parameter has {length}.", "gradients");
            var backing = bound.IsContiguous ? bound.GetCpuBackingForStridedRead(out offset) : null;
            if (backing is null)
            {
                source = RentGradientScratch(length);
                bound.CopyLogicalTo(new Span<float>(source, 0, length));
                offset = 0;
            }
            else
            {
                source = backing;
            }
        }
        else
        {
            source = group.Gradients[pi];
        }

        bool negate = group.GetOption("maximize", 0.0) != 0.0;
        if (!negate && coupledWeightDecay == 0f) return new GradientBuffer(source, offset, length);

        var effective = ReferenceEquals(source, _gradientScratch) ? source : RentGradientScratch(length);
        float sign = negate ? -1f : 1f;
        int sourceOffset = offset;
        ForEachChunk(length, (start, count) =>
            SignedAddScaled(source, sourceOffset + start, parameter, start, effective, start, count, sign, coupledWeightDecay));
        return new GradientBuffer(effective, 0, length);
    }

    // destination[i] = sign·gradient[i] + decay·parameter[i], rounded per operation as the scalar form is.
    private static void SignedAddScaled(float[] gradient, int gradientOffset, float[] parameter, int parameterOffset,
        float[] destination, int destinationOffset, int count, float sign, float decay)
    {
        int i = 0;
        int width = System.Numerics.Vector<float>.Count;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            var signVector = new System.Numerics.Vector<float>(sign);
            var decayVector = new System.Numerics.Vector<float>(decay);
            for (; i <= count - width; i += width)
            {
                var g = new System.Numerics.Vector<float>(gradient, gradientOffset + i) * signVector;
                var p = new System.Numerics.Vector<float>(parameter, parameterOffset + i) * decayVector;
                (g + p).CopyTo(destination, destinationOffset + i);
            }
        }
        for (; i < count; i++)
            destination[destinationOffset + i] = sign * gradient[gradientOffset + i] + decay * parameter[parameterOffset + i];
    }

    private float[] RentGradientScratch(int length)
    {
        ReturnGradientScratch();
        _gradientScratch = ArrayPool<float>.Shared.Rent(length);
        return _gradientScratch;
    }

    private void ReturnGradientScratch()
    {
        if (_gradientScratch is null) return;
        ArrayPool<float>.Shared.Return(_gradientScratch);
        _gradientScratch = null;
    }

    /// <summary>Runs <paramref name="body"/>(start, count) over fixed chunks of <paramref name="length"/> across the pool.</summary>
    private protected static void ForEachChunk(int length, Action<int, int> body)
    {
        int chunks = (length + ElementwiseChunk - 1) / ElementwiseChunk;
        if (chunks <= 1)
        {
            body(0, length);
            return;
        }
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)length * 4, c =>
        {
            int start = c * ElementwiseChunk;
            body(start, Math.Min(ElementwiseChunk, length - start));
        }, deterministicSafe: true);
    }

    /// <summary>Add a parameter group; <paramref name="overrides"/> override <see cref="Defaults"/>.</summary>
    public ParamGroup AddParamGroup(IDictionary<string, double>? overrides = null)
    {
        var g = new ParamGroup();
        foreach (var kv in Defaults) g.Options[kv.Key] = kv.Value;
        if (overrides != null)
            foreach (var kv in overrides) g.Options[kv.Key] = kv.Value;
        _groups.Add(g);
        return g;
    }

    /// <summary>Convenience: single-group setup that adds all params under default hyper-params.</summary>
    public ParamGroup AddParameters(IEnumerable<(float[] param, float[] grad)> pairs)
    {
        var group = AddParamGroup();
        foreach (var (p, g) in pairs) group.AddParameter(p, g);
        return group;
    }

    /// <summary>Get or lazily create the state record for <c>group[gi].param[pi]</c>.</summary>
    protected Dictionary<string, OptimizerStateValue> GetOrCreateState(int gi, int pi, int paramLen)
    {
        var slot = GetOrCreateStateRecord(gi, pi, paramLen);
        MoveStateToHost(slot);
        return slot;
    }

    // The state record without moving its buffers anywhere; new buffer slots are zeroed on the host.
    private Dictionary<string, OptimizerStateValue> GetOrCreateStateRecord(int gi, int pi, int paramLen, bool onDevice = false)
    {
        var key = (gi, pi);
        if (_state.TryGetValue(key, out var dict)) return dict;
        dict = new Dictionary<string, OptimizerStateValue>();
        var scalarSet = new HashSet<string>(ScalarStateNames, StringComparer.Ordinal);
        foreach (var name in StateNames)
        {
            if (name == "step")
                dict[name] = OptimizerStateValue.FromInt(0);
            else if (scalarSet.Contains(name))
                dict[name] = OptimizerStateValue.FromFloat(0f);
            else if (onDevice)
                dict[name] = new OptimizerStateValue { DeviceTensor = Gpu.GpuOptimizer.CreateStateTensor(new[] { paramLen }) };
            else
                dict[name] = OptimizerStateValue.FromTensor(new float[paramLen]);
        }
        _state[key] = dict;
        return dict;
    }

    /// <inheritdoc />
    public void ZeroGrad()
    {
        foreach (var g in _groups)
            for (int i = 0; i < g.Parameters.Count; i++)
            {
                var gradient = g.PeekGradient(i);
                if (gradient is not null) Array.Clear(gradient, 0, gradient.Length);
            }
    }

    // ------------------------------------------------------------------
    // Sparse-gradient plumbing (shared by every concrete optimizer).
    //
    // The autodiff side (BackwardFunctions / DifferentiableOps / SparseEmbeddingGradient)
    // records embedding-style gradients as (row-indices, row-values) instead of dense
    // [vocab, dim] tensors. Consumers (PyTorch-style training loops) bridge that sparse
    // representation onto the optimizer by calling SetSparseGradient(gi, pi, idx, vals)
    // before Step(). Each Step() then sees the sparse view via TryGetSparseGradient and
    // scatter-updates only the touched indices, skipping the full-parameter scan that
    // is wasteful when the dense gradient is mostly zero.
    //
    // This mirrors what SparseAdamOptimizer (the only sparse-aware optimizer prior to
    // PR #567) already did, but lifted into the base class so EVERY optimizer
    // automatically gains the same fast path with zero per-subclass boilerplate.
    // Optimizers whose update math doesn't decompose elementwise (Rprop sign tracking,
    // LBFGS / Shampoo matrix state, Asgd's averaged-weight buffer) opt out by simply
    // not calling TryGetSparseGradient.
    // ------------------------------------------------------------------

    private readonly Dictionary<(int gi, int pi), (int[] idx, float[] val, bool autoClear)> _sparseGrads =
        new Dictionary<(int gi, int pi), (int[], float[], bool)>();

    /// <summary>Publish a sparse gradient (row-indices + row-values) for the next <see cref="Step"/> call.
    /// Indices are flat positions into the parameter buffer (not row indices into a 2D view) so this works
    /// for any rank/layout; embedding callers feed in <c>row*embDim + col</c> flat indices.
    /// When <paramref name="autoClear"/> is true (the default) the entry is removed at the end of <see cref="Step"/>
    /// so each step re-publishes — matching the AccumulateGrad / SparseEmbeddingGradient lifecycle on the
    /// autodiff side. Pass false for static masks that persist across steps.</summary>
    public void SetSparseGradient(int paramGroupIndex, int paramIndex, int[] indices, float[] values, bool autoClear = true)
    {
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (indices.Length != values.Length)
            throw new ArgumentException("indices and values must be the same length.", nameof(values));
        _sparseGrads[(paramGroupIndex, paramIndex)] = (indices, values, autoClear);
    }

    /// <summary>Drop a sparse gradient previously published via <see cref="SetSparseGradient"/>.</summary>
    public void ClearSparseGradient(int paramGroupIndex, int paramIndex)
        => _sparseGrads.Remove((paramGroupIndex, paramIndex));

    /// <summary>Drop every sparse-gradient entry (both auto-clear and sticky).</summary>
    public void ClearAllSparseGradients() => _sparseGrads.Clear();

    /// <summary>Subclass probe: returns true and yields the published (indices, values) when
    /// a sparse override has been wired for this parameter. Subclasses that can decompose
    /// their update elementwise should consume the sparse view; others should fall through
    /// to their existing dense kernel.</summary>
    protected bool TryGetSparseGradient(int paramGroupIndex, int paramIndex, out int[] idx, out float[] val, out int nnz)
    {
        if (_sparseGrads.TryGetValue((paramGroupIndex, paramIndex), out var pair))
        {
            idx = pair.idx;
            val = pair.val;
            nnz = pair.idx.Length;
            if (_groups[paramGroupIndex].GetOption("maximize", 0.0) != 0.0)
            {
                // Ascent: hand the update negated values, leaving the published ones untouched.
                if (_sparseScratch.Length < nnz) _sparseScratch = new float[nnz];
                for (int k = 0; k < nnz; k++) _sparseScratch[k] = -pair.val[k];
                val = _sparseScratch;
            }
            return true;
        }
        idx = null!;
        val = null!;
        nnz = 0;
        return false;
    }

    /// <summary>Returns true iff a sparse gradient is wired for the given param. Cheap probe
    /// for subclasses that want to short-circuit before touching the dense buffer.</summary>
    protected bool HasSparseGradient(int paramGroupIndex, int paramIndex)
        => _sparseGrads.ContainsKey((paramGroupIndex, paramIndex));

    /// <summary>Materialize the published sparse gradient (if any) into the supplied dense
    /// buffer: zeros <paramref name="dense"/> first, then scatter-adds <c>(idx, val)</c> pairs.
    /// Used by optimizers whose update math does NOT decompose elementwise (LAMB / LARS /
    /// ASGD / Rprop — trust-ratio, averaged-weight, sign-tracking) so they can still consume
    /// a sparse-published gradient via the regular dense kernel. No-op if no sparse grad is
    /// published for this (gi, pi).</summary>
    protected void MaterializeSparseIntoDense(int paramGroupIndex, int paramIndex, float[] dense)
    {
        if (dense == null) throw new ArgumentNullException(nameof(dense));
        if (!_sparseGrads.TryGetValue((paramGroupIndex, paramIndex), out var pair)) return;
        Array.Clear(dense, 0, dense.Length);
        var idx = pair.idx;
        var val = pair.val;
        for (int k = 0; k < idx.Length; k++) dense[idx[k]] += val[k];
    }

    /// <summary>Remove all auto-clear sparse-grad entries. Subclasses MUST call this from
    /// the <c>finally</c> block of <see cref="Step"/> so each step starts with a clean slate.</summary>
    protected void ClearAutoClearSparseGrads()
    {
        if (_sparseGrads.Count == 0) return;
        List<(int, int)>? toRemove = null;
        foreach (var kv in _sparseGrads)
        {
            if (kv.Value.autoClear)
            {
                toRemove ??= new List<(int, int)>();
                toRemove.Add(kv.Key);
            }
        }
        if (toRemove != null)
            foreach (var key in toRemove) _sparseGrads.Remove(key);
    }

    /// <summary>If a group's <c>"maximize"</c> option is true, flip the sign of each gradient
    /// in-place so the downstream descent kernel performs an ascent step. Gradients are written
    /// back to their original sign at the end of <see cref="Step"/> via <see cref="UnflipMaximize"/>.</summary>
    /// <returns>True if any group had maximize active (caller must call <see cref="UnflipMaximize"/>).</returns>
    [Obsolete("Writes the caller's gradients. The built-in optimizers read DenseGradient, which negates into scratch.")]
    protected bool ApplyMaximize()
    {
        bool any = false;
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var g = _groups[gi];
            if (g.GetOption("maximize", 0.0) == 0.0) continue;
            any = true;
            for (int pi = 0; pi < g.Gradients.Count; pi++)
            {
                var grad = g.Gradients[pi];
                for (int i = 0; i < grad.Length; i++) grad[i] = -grad[i];
            }
        }
        return any;
    }

    /// <summary>Restore the original sign of gradients flipped by <see cref="ApplyMaximize"/>.</summary>
    [Obsolete("Pairs with ApplyMaximize; the built-in optimizers no longer flip gradients in place.")]
    protected void UnflipMaximize()
    {
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var g = _groups[gi];
            if (g.GetOption("maximize", 0.0) == 0.0) continue;
            for (int pi = 0; pi < g.Gradients.Count; pi++)
            {
                var grad = g.Gradients[pi];
                for (int i = 0; i < grad.Length; i++) grad[i] = -grad[i];
            }
        }
    }

    /// <summary>
    /// Hook for subclasses to publish optimizer-level per-group state into the saved dict
    /// (e.g. D-Adaptation's <c>CurrentD</c>, Prodigy's <c>DNumerator</c>). Default: empty.
    /// Mirror this on the load side via <see cref="SetGroupExtraState"/>.
    /// </summary>
    protected virtual Dictionary<string, OptimizerStateValue> GetGroupExtraState(int groupIndex) =>
        new Dictionary<string, OptimizerStateValue>();

    /// <summary>Restore the per-group state captured by <see cref="GetGroupExtraState"/>.</summary>
    protected virtual void SetGroupExtraState(int groupIndex, Dictionary<string, OptimizerStateValue> extraState) { }

    /// <inheritdoc />
    public OptimizerStateDict StateDict()
    {
        var sd = new OptimizerStateDict();
        int paramCounter = 0;
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var group = _groups[gi];
            var groupState = new OptimizerGroupState();
            foreach (var kv in group.Options) groupState.Options[kv.Key] = kv.Value;
            // Subclass-level state lives in ExtraState so save/load round-trips it.
            foreach (var kv in GetGroupExtraState(gi))
            {
                var v = kv.Value;
                groupState.ExtraState[kv.Key] = new OptimizerStateValue
                {
                    IntValue = v.IntValue,
                    FloatValue = v.FloatValue,
                    Tensor = v.Tensor == null ? null : (float[])v.Tensor.Clone(),
                };
            }
            for (int pi = 0; pi < group.Parameters.Count; pi++)
            {
                int id = paramCounter++;
                groupState.ParamIds.Add(id);
                if (_state.TryGetValue((gi, pi), out var slots))
                {
                    var copy = new Dictionary<string, OptimizerStateValue>();
                    foreach (var kv in slots)
                    {
                        var v = kv.Value;
                        copy[kv.Key] = new OptimizerStateValue
                        {
                            IntValue = v.IntValue,
                            FloatValue = v.FloatValue,
                            // A device-resident buffer is read back; the saved dict is always host data.
                            Tensor = v.DeviceTensor is not null
                                ? Gpu.GpuOptimizer.TryDownload(v.DeviceTensor) ?? v.DeviceTensor.ToArray()
                                : v.Tensor == null ? null : (float[])v.Tensor.Clone()
                        };
                    }
                    sd.State[id] = copy;
                }
            }
            sd.ParamGroups.Add(groupState);
        }
        return sd;
    }

    /// <inheritdoc />
    public void LoadStateDict(OptimizerStateDict state)
    {
        if (state == null) throw new ArgumentNullException(nameof(state));
        if (state.ParamGroups.Count != _groups.Count)
            throw new InvalidOperationException(
                $"state-dict has {state.ParamGroups.Count} groups but optimizer has {_groups.Count}.");

        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var group = _groups[gi];
            var gs = state.ParamGroups[gi];
            foreach (var kv in gs.Options) group.Options[kv.Key] = kv.Value;
            // Restore subclass-level per-group state before per-parameter slots so a
            // freshly-loaded optimizer's first Step() observes the saved values rather than
            // recomputing from defaults.
            if (gs.ExtraState.Count > 0)
            {
                var clone = new Dictionary<string, OptimizerStateValue>();
                foreach (var kv in gs.ExtraState)
                {
                    var v = kv.Value;
                    clone[kv.Key] = new OptimizerStateValue
                    {
                        IntValue = v.IntValue,
                        FloatValue = v.FloatValue,
                        Tensor = v.Tensor == null ? null : (float[])v.Tensor.Clone(),
                    };
                }
                SetGroupExtraState(gi, clone);
            }
            if (gs.ParamIds.Count != group.Parameters.Count)
                throw new InvalidOperationException(
                    $"group {gi} has {group.Parameters.Count} params; state-dict has {gs.ParamIds.Count}.");
            for (int pi = 0; pi < group.Parameters.Count; pi++)
            {
                // Use the serialized param id (from the state-dict's ParamIds list) rather than
                // a fresh counter. This makes load symmetric with save (both sides honor the
                // explicit id mapping) and works with non-contiguous IDs that arise from
                // sharded / partial state-dict loads.
                int id = gs.ParamIds[pi];
                if (!state.State.TryGetValue(id, out var slots)) continue;
                // Host-side: a device parameter's next device step uploads what is loaded here.
                var dst = GetOrCreateState(gi, pi, group.Parameters[pi].Length);
                foreach (var kv in slots)
                {
                    if (!dst.TryGetValue(kv.Key, out var existing))
                    {
                        dst[kv.Key] = new OptimizerStateValue
                        {
                            IntValue = kv.Value.IntValue,
                            FloatValue = kv.Value.FloatValue,
                            Tensor = kv.Value.Tensor == null ? null : (float[])kv.Value.Tensor.Clone()
                        };
                        continue;
                    }
                    existing.IntValue = kv.Value.IntValue;
                    existing.FloatValue = kv.Value.FloatValue;
                    if (kv.Value.Tensor != null && existing.Tensor != null)
                    {
                        if (kv.Value.Tensor.Length != existing.Tensor.Length)
                            throw new InvalidOperationException(
                                $"state '{kv.Key}' length {kv.Value.Tensor.Length} != {existing.Tensor.Length}.");
                        Array.Copy(kv.Value.Tensor, existing.Tensor, existing.Tensor.Length);
                    }
                    else if (kv.Value.Tensor != null)
                    {
                        existing.Tensor = (float[])kv.Value.Tensor.Clone();
                    }
                }
            }
        }
    }
}
