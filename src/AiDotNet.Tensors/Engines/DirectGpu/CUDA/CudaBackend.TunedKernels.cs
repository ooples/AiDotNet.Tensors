using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>Runs a candidate on arguments passed by reference.</summary>
internal delegate void TunedKernelInvoker<TArgs>(in TArgs args) where TArgs : struct;

/// <summary>A registry candidate backed by a delegate on a CUDA backend.</summary>
internal sealed class CudaTunedKernelCandidate<TArgs> : ITunedKernelCandidate<TArgs> where TArgs : struct
{
    private readonly Func<TunedShape, bool> _applicable;
    private readonly TunedKernelInvoker<TArgs> _execute;

    internal CudaTunedKernelCandidate(string id, TunedKernelOrigin origin, bool deterministic,
        Func<TunedShape, bool> applicable, TunedKernelInvoker<TArgs> execute)
    {
        Id = id;
        Origin = origin;
        IsDeterministic = deterministic;
        _applicable = applicable;
        _execute = execute;
    }

    public string Id { get; }
    public TunedKernelOrigin Origin { get; }
    public bool IsDeterministic { get; }
    public bool IsApplicable(in TunedShape shape) => _applicable(shape);
    public void Execute(in TArgs args) => _execute(args);
}

/// <summary>CUDA measurement services for one op family: host snapshots of outputs, device-event timing.</summary>
internal sealed class CudaTunedKernelHarness<TArgs> : ITunedKernelHarness<TArgs> where TArgs : struct
{
    private readonly CudaBackend _backend;
    private readonly Func<TArgs, (IGpuBuffer Buffer, int Count)[]> _outputs;
    private readonly Func<TArgs, bool> _aliased;

    internal CudaTunedKernelHarness(CudaBackend backend, double relativeTolerance,
        Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs, Func<TArgs, bool> aliased)
    {
        _backend = backend;
        RelativeTolerance = relativeTolerance;
        _outputs = outputs;
        _aliased = aliased;
    }

    public double RelativeTolerance { get; }

    public bool CanMeasure(in TArgs args) => _backend.CanMeasureTunedKernels && !_aliased(args);

    public void PoisonOutput(in TArgs args)
    {
        foreach (var o in _outputs(args)) _backend.Fill(o.Buffer, float.NaN, o.Count);
    }

    public float[] SnapshotOutput(in TArgs args)
    {
        var outs = _outputs(args);
        int total = 0;
        foreach (var o in outs) total += o.Count;
        var result = new float[total];
        int offset = 0;
        foreach (var o in outs)
        {
            float[] host = _backend.DownloadBuffer(o.Buffer);
            Array.Copy(host, 0, result, offset, o.Count);
            offset += o.Count;
        }
        return result;
    }

    public double MeasureMilliseconds(ITunedKernelCandidate<TArgs> candidate, in TArgs args, int repetitions)
    {
        TArgs local = args;
        return _backend.MeasureTunedKernelMilliseconds(() => candidate.Execute(local), repetitions);
    }
}

/// <summary>Row softmax arguments.</summary>
internal readonly struct CudaSoftmaxArgs
{
    internal CudaSoftmaxArgs(IGpuBuffer input, IGpuBuffer output, int rows, int n)
    { Input = input; Output = output; Rows = rows; N = n; }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Output { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>Row softmax backward arguments.</summary>
internal readonly struct CudaSoftmaxBackwardArgs
{
    internal CudaSoftmaxBackwardArgs(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
    { GradOutput = gradOutput; Output = output; GradInput = gradInput; Rows = rows; N = n; }
    internal IGpuBuffer GradOutput { get; }
    internal IGpuBuffer Output { get; }
    internal IGpuBuffer GradInput { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>LayerNorm forward arguments.</summary>
internal readonly struct CudaLayerNormArgs
{
    internal CudaLayerNormArgs(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int rows, int n, float epsilon)
    {
        Input = input; Output = output; Gamma = gamma; Beta = beta; SaveMean = saveMean; SaveInvVar = saveInvVar;
        Rows = rows; N = n; Epsilon = epsilon;
    }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Output { get; }
    internal IGpuBuffer Gamma { get; }
    internal IGpuBuffer Beta { get; }
    internal IGpuBuffer SaveMean { get; }
    internal IGpuBuffer SaveInvVar { get; }
    internal int Rows { get; }
    internal int N { get; }
    internal float Epsilon { get; }
}

/// <summary>Normalization backward arguments (LayerNorm input gradient and the affine column reductions).</summary>
internal readonly struct CudaNormBackwardArgs
{
    internal CudaNormBackwardArgs(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer? gamma,
        IGpuBuffer stat0, IGpuBuffer? stat1, IGpuBuffer out0, IGpuBuffer? out1, int rows, int n)
    {
        GradOutput = gradOutput; Input = input; Gamma = gamma; Stat0 = stat0; Stat1 = stat1; Out0 = out0; Out1 = out1;
        Rows = rows; N = n;
    }
    internal IGpuBuffer GradOutput { get; }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer? Gamma { get; }
    /// <summary>Mean (LayerNorm) or RMS (RMSNorm) per row.</summary>
    internal IGpuBuffer Stat0 { get; }
    /// <summary>Inverse standard deviation per row (LayerNorm only).</summary>
    internal IGpuBuffer? Stat1 { get; }
    /// <summary>Input gradient, or gradGamma for a column reduction.</summary>
    internal IGpuBuffer Out0 { get; }
    /// <summary>gradBeta for the LayerNorm column reduction.</summary>
    internal IGpuBuffer? Out1 { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>
/// The CUDA backend's tuned-kernel slots: every op family below dispatches through
/// <see cref="TunedKernelSlot{TArgs}"/>, with the established kernel as the reference and the generated variants of
/// <see cref="CudaTunedRowKernels"/> (and any externally evolved artifact registered later) as candidates.
/// </summary>
public sealed partial class CudaBackend
{
    private const double RowOpTolerance = 1e-4;
    private IntPtr _tuneStartEvent, _tuneEndEvent;
    private readonly object _tunedSlotsLock = new();
    private readonly object _tuneEventLock = new();
    private string? _tunedDeviceKey;
    private TunedKernelSlot<CudaSoftmaxArgs>? _softmaxSlot;
    private TunedKernelSlot<CudaSoftmaxBackwardArgs>? _softmaxBackwardSlot;
    private TunedKernelSlot<CudaLayerNormArgs>? _layerNormSlot;
    private TunedKernelSlot<CudaNormBackwardArgs>? _layerNormBackwardSlot;
    private TunedKernelSlot<CudaNormBackwardArgs>? _layerNormGradParamsSlot;
    private TunedKernelSlot<CudaNormBackwardArgs>? _rmsNormGradGammaSlot;

    /// <summary>Device key used by the tuned-kernel registry: <c>cuda:sm{major}{minor}:{device name}</c>.</summary>
    internal string TunedDeviceKey =>
        _tunedDeviceKey ??= $"cuda:sm{_ccMajor}{_ccMinor}:{DeviceName}";

    internal bool CanMeasureTunedKernels => IsAvailable && !IsStreamCapturing();

    /// <summary>Compiles the generated row/column variants. Optional: a failure leaves only the references.</summary>
    private void CompileTunedRowKernels(int device)
    {
        try
        {
            CompileKernelModule(device, CudaTunedRowKernels.GetSource(), "tuned_row_kernels",
                CudaTunedRowKernels.GetKernelNames());
        }
        catch (OutOfMemoryException)
        {
            throw;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning($"CUDA tuned row kernels unavailable: {ex.Message}");
        }
    }

    /// <summary>Device time of <paramref name="repetitions"/> back-to-back runs, on the backend's stream.</summary>
    internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
    {
        using var _ = PushContext();
        // Its own lock: a gate runs under its slot's lock, and slot creation (which may register external
        // artifacts into other slots) runs under _tunedSlotsLock, so sharing that lock here could deadlock.
        lock (_tuneEventLock)
        {
            if (_tuneStartEvent == IntPtr.Zero)
            {
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventCreate(out _tuneStartEvent, 0), "cuEventCreate(tune)");
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventCreate(out _tuneEndEvent, 0), "cuEventCreate(tune)");
            }
        }
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(_tuneStartEvent, _stream), "cuEventRecord(tune)");
        for (int i = 0; i < repetitions; i++) run();
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(_tuneEndEvent, _stream), "cuEventRecord(tune)");
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventSynchronize(_tuneEndEvent), "cuEventSynchronize(tune)");
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuEventElapsedTime(out float ms, _tuneStartEvent, _tuneEndEvent), "cuEventElapsedTime(tune)");
        return ms;
    }

    private void DisposeTunedKernelResources()
    {
        DisposeExternalKernelModules();
        if (_tuneStartEvent != IntPtr.Zero) { try { CudaNativeBindings.cuEventDestroy(_tuneStartEvent); } catch { } _tuneStartEvent = IntPtr.Zero; }
        if (_tuneEndEvent != IntPtr.Zero) { try { CudaNativeBindings.cuEventDestroy(_tuneEndEvent); } catch { } _tuneEndEvent = IntPtr.Zero; }
    }

    private bool HasTunedKernel(string name) => _kernelCache.ContainsKey(name);

    // Every slot getter returns through here so the environment's external artifacts join the candidate pools
    // the first time any slot exists (re-entrant: registering an artifact re-reads the already-assigned slot).
    private TSlot Created<TSlot>(TSlot slot)
    {
        EnsureExternalKernelArtifactsLoaded();
        return slot;
    }

    private static uint RowGrid(int rows, int lanes) => (uint)(((long)rows * lanes + 255) / 256);

    // ---------------------------------------------------------------- softmax

    internal TunedKernelSlot<CudaSoftmaxArgs> SoftmaxSlot
    {
        get
        {
            if (_softmaxSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_softmaxSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<CudaSoftmaxArgs>>
                    {
                        new CudaTunedKernelCandidate<CudaSoftmaxArgs>("nvrtc.softmax.block256", TunedKernelOrigin.Builtin,
                            true, _ => true, (in CudaSoftmaxArgs a) => LaunchSoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                    };
                    foreach (int lanes in CudaTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = CudaTunedRowKernels.SoftmaxName(l);
                        candidates.Add(new CudaTunedKernelCandidate<CudaSoftmaxArgs>("generated.softmax.lanes" + l,
                            TunedKernelOrigin.Generated, true,
                            shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in CudaSoftmaxArgs a) => LaunchTunedRow(name, l, a.Input, a.Output, a.Rows, a.N)));
                    }
                    _softmaxSlot = new TunedKernelSlot<CudaSoftmaxArgs>(TunedKernelOp.Softmax, TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaSoftmaxArgs>(this, RowOpTolerance,
                            a => new[] { (a.Output, a.Rows * a.N) },
                            a => a.Input.Handle == a.Output.Handle),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_softmaxSlot);
            }
        }
    }

    private unsafe void LaunchTunedRow(string kernelName, int lanes, IGpuBuffer input, IGpuBuffer output, int rows, int n)
    {
        IntPtr kernel = _kernelCache[kernelName];
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &rows; args[3] = &n;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    // ---------------------------------------------------------------- softmax backward

    internal TunedKernelSlot<CudaSoftmaxBackwardArgs> SoftmaxBackwardSlot
    {
        get
        {
            if (_softmaxBackwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_softmaxBackwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<CudaSoftmaxBackwardArgs>>
                    {
                        new CudaTunedKernelCandidate<CudaSoftmaxBackwardArgs>("nvrtc.softmax_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in CudaSoftmaxBackwardArgs a) => LaunchSoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                    };
                    foreach (int lanes in CudaTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = CudaTunedRowKernels.SoftmaxBackwardName(l);
                        candidates.Add(new CudaTunedKernelCandidate<CudaSoftmaxBackwardArgs>(
                            "generated.softmax_backward.lanes" + l, TunedKernelOrigin.Generated, true,
                            shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in CudaSoftmaxBackwardArgs a) => LaunchTunedSoftmaxBackward(name, l, a)));
                    }
                    _softmaxBackwardSlot = new TunedKernelSlot<CudaSoftmaxBackwardArgs>(TunedKernelOp.SoftmaxBackward,
                        TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaSoftmaxBackwardArgs>(this, RowOpTolerance,
                            a => new[] { (a.GradInput, a.Rows * a.N) },
                            a => a.GradInput.Handle == a.GradOutput.Handle || a.GradInput.Handle == a.Output.Handle),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_softmaxBackwardSlot);
            }
        }
    }

    private unsafe void LaunchTunedSoftmaxBackward(string kernelName, int lanes, in CudaSoftmaxBackwardArgs a)
    {
        IntPtr kernel = _kernelCache[kernelName];
        using var _ = PushContext();
        IntPtr g = a.GradOutput.Handle, o = a.Output.Handle, gi = a.GradInput.Handle;
        int rows = a.Rows, n = a.N;
        void** args = stackalloc void*[5];
        args[0] = &g; args[1] = &o; args[2] = &gi; args[3] = &rows; args[4] = &n;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    // ---------------------------------------------------------------- layer norm forward

    internal TunedKernelSlot<CudaLayerNormArgs> LayerNormSlot
    {
        get
        {
            if (_layerNormSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_layerNormSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<CudaLayerNormArgs>>
                    {
                        new CudaTunedKernelCandidate<CudaLayerNormArgs>("nvrtc.layernorm_forward.block256",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in CudaLayerNormArgs a) => LaunchLayerNormReference(a)),
                    };
                    foreach (int lanes in CudaTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = CudaTunedRowKernels.LayerNormName(l);
                        candidates.Add(new CudaTunedKernelCandidate<CudaLayerNormArgs>(
                            "generated.layernorm_forward.lanes" + l, TunedKernelOrigin.Generated, true,
                            shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in CudaLayerNormArgs a) => LaunchTunedLayerNorm(name, l, a)));
                    }
                    candidates.Add(new CudaTunedKernelCandidate<CudaLayerNormArgs>(
                        "directptx.layernorm_forward.d64", TunedKernelOrigin.Generated, true, IsDirectPtxD64Shape,
                        (in CudaLayerNormArgs a) => RequireDirectPtx(TryDirectPtxRowNormalizationCandidate(
                            DirectPtxRowNormalizationOperation.LayerNormForward, a.Rows, a.Epsilon,
                            a.Input, a.Gamma, a.Beta, a.Output, a.SaveMean, a.SaveInvVar))));
                    _layerNormSlot = new TunedKernelSlot<CudaLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaLayerNormArgs>(this, RowOpTolerance,
                            a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                            a => a.Output.Handle == a.Input.Handle),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_layerNormSlot);
            }
        }
    }

    private unsafe void LaunchLayerNormReference(in CudaLayerNormArgs a)
    {
        if (!_kernelCache.TryGetValue("layernorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_forward");
        using var _ = PushContext();
        IntPtr inputPtr = a.Input.Handle, outputPtr = a.Output.Handle, gammaPtr = a.Gamma.Handle, betaPtr = a.Beta.Handle;
        IntPtr saveMeanPtr = a.SaveMean.Handle, saveInvVarPtr = a.SaveInvVar.Handle;
        int batchSize = a.Rows, normalizedSize = a.N;
        float epsilon = a.Epsilon;
        void** args = stackalloc void*[9];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &gammaPtr; args[3] = &betaPtr;
        args[4] = &saveMeanPtr; args[5] = &saveInvVarPtr; args[6] = &batchSize; args[7] = &normalizedSize;
        args[8] = &epsilon;
        // 1 block per batch element, 1 shared array
        LaunchKernelWithSharedMem(kernel, (uint)batchSize, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    private unsafe void LaunchTunedLayerNorm(string kernelName, int lanes, in CudaLayerNormArgs a)
    {
        IntPtr kernel = _kernelCache[kernelName];
        using var _ = PushContext();
        IntPtr inputPtr = a.Input.Handle, outputPtr = a.Output.Handle, gammaPtr = a.Gamma.Handle, betaPtr = a.Beta.Handle;
        IntPtr saveMeanPtr = a.SaveMean.Handle, saveInvVarPtr = a.SaveInvVar.Handle;
        int rows = a.Rows, n = a.N;
        float epsilon = a.Epsilon;
        void** args = stackalloc void*[9];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &gammaPtr; args[3] = &betaPtr;
        args[4] = &saveMeanPtr; args[5] = &saveInvVarPtr; args[6] = &rows; args[7] = &n; args[8] = &epsilon;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    // ---------------------------------------------------------------- layer norm backward (input gradient)

    internal TunedKernelSlot<CudaNormBackwardArgs> LayerNormBackwardSlot
    {
        get
        {
            if (_layerNormBackwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_layerNormBackwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<CudaNormBackwardArgs>>
                    {
                        new CudaTunedKernelCandidate<CudaNormBackwardArgs>("nvrtc.layernorm_backward.block256",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in CudaNormBackwardArgs a) => LaunchLayerNormBackwardReference(a)),
                    };
                    foreach (int lanes in CudaTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = CudaTunedRowKernels.LayerNormBackwardName(l);
                        candidates.Add(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(
                            "generated.layernorm_backward.lanes" + l, TunedKernelOrigin.Generated, true,
                            shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in CudaNormBackwardArgs a) => LaunchTunedLayerNormBackward(name, l, a)));
                    }
                    candidates.Add(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(
                        "directptx.layernorm_backward.d64", TunedKernelOrigin.Generated, true, IsDirectPtxD64Shape,
                        (in CudaNormBackwardArgs a) => RequireDirectPtx(TryDirectPtxRowNormalizationCandidate(
                            DirectPtxRowNormalizationOperation.LayerNormBackwardInput, a.Rows, 0f,
                            a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0))));
                    _layerNormBackwardSlot = new TunedKernelSlot<CudaNormBackwardArgs>(TunedKernelOp.LayerNormBackward,
                        TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaNormBackwardArgs>(this, RowOpTolerance,
                            a => new[] { (a.Out0, a.Rows * a.N) },
                            a => a.Out0.Handle == a.GradOutput.Handle || a.Out0.Handle == a.Input.Handle),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_layerNormBackwardSlot);
            }
        }
    }

    private unsafe void LaunchLayerNormBackwardReference(in CudaNormBackwardArgs a)
    {
        if (!_kernelCache.TryGetValue("layernorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_backward");
        using var _ = PushContext();
        IntPtr gradOutputPtr = a.GradOutput.Handle, inputPtr = a.Input.Handle;
        IntPtr gammaPtr = Required(a.Gamma).Handle, saveMeanPtr = a.Stat0.Handle, saveInvVarPtr = Required(a.Stat1).Handle;
        IntPtr gradInputPtr = a.Out0.Handle;
        // layernorm_backward declares gradGamma/gradBeta parameters but never writes them; any valid pointer works.
        IntPtr unusedGamma = gradInputPtr, unusedBeta = gradInputPtr;
        int batchSize = a.Rows, normalizedSize = a.N;
        float epsilon = 0f;
        void** args = stackalloc void*[11];
        args[0] = &gradOutputPtr; args[1] = &inputPtr; args[2] = &gammaPtr; args[3] = &saveMeanPtr;
        args[4] = &saveInvVarPtr; args[5] = &gradInputPtr; args[6] = &unusedGamma; args[7] = &unusedBeta;
        args[8] = &batchSize; args[9] = &normalizedSize; args[10] = &epsilon;
        // 2 shared arrays for sumDy and sumDyXmu
        LaunchKernelWithSharedMem(kernel, (uint)batchSize, DefaultBlockSize, (uint)(2 * DefaultBlockSize * sizeof(float)), args);
    }

    private unsafe void LaunchTunedLayerNormBackward(string kernelName, int lanes, in CudaNormBackwardArgs a)
    {
        IntPtr kernel = _kernelCache[kernelName];
        using var _ = PushContext();
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, gamma = Required(a.Gamma).Handle;
        IntPtr mean = a.Stat0.Handle, invVar = Required(a.Stat1).Handle, gi = a.Out0.Handle;
        int rows = a.Rows, n = a.N;
        void** args = stackalloc void*[8];
        args[0] = &g; args[1] = &x; args[2] = &gamma; args[3] = &mean; args[4] = &invVar; args[5] = &gi;
        args[6] = &rows; args[7] = &n;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    // ---------------------------------------------------------------- column reductions (affine gradients)

    internal TunedKernelSlot<CudaNormBackwardArgs> LayerNormGradParametersSlot
    {
        get
        {
            if (_layerNormGradParamsSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _layerNormGradParamsSlot ??= CreateColumnGradSlot(TunedKernelOp.LayerNormGradParameters,
                    "nvrtc.layernorm_grad_params.thread_per_column", "layernorm_grad_params", rms: false);
                return Created(_layerNormGradParamsSlot);
            }
        }
    }

    internal TunedKernelSlot<CudaNormBackwardArgs> RmsNormGradGammaSlot
    {
        get
        {
            if (_rmsNormGradGammaSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                    "nvrtc.rmsnorm_grad_gamma.thread_per_column", "rmsnorm_grad_gamma", rms: true);
                return Created(_rmsNormGradGammaSlot);
            }
        }
    }

    private TunedKernelSlot<CudaNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId,
        string referenceKernel, bool rms)
    {
        var candidates = new List<ITunedKernelCandidate<CudaNormBackwardArgs>>
        {
            new CudaTunedKernelCandidate<CudaNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true, _ => true,
                (in CudaNormBackwardArgs a) => LaunchColumnGradReference(referenceKernel, rms, a)),
        };
        foreach (int rowLanes in CudaTunedRowKernels.ColumnRowLanes)
        {
            int r = rowLanes;
            string name = rms ? CudaTunedRowKernels.RmsNormGradGammaName(r) : CudaTunedRowKernels.LayerNormGradParamsName(r);
            candidates.Add(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(
                (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                TunedKernelOrigin.Generated, true, _ => HasTunedKernel(name),
                (in CudaNormBackwardArgs a) => LaunchTunedColumnGrad(name, r, rms, a)));
        }
        candidates.Add(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(
            rms ? "directptx.rmsnorm_grad_gamma.d64" : "directptx.layernorm_grad_params.d64",
            TunedKernelOrigin.Generated, true, IsDirectPtxD64Shape,
            rms
                ? new TunedKernelInvoker<CudaNormBackwardArgs>((in CudaNormBackwardArgs a) => RequireDirectPtx(TryDirectPtxRowNormalizationCandidate(
                    DirectPtxRowNormalizationOperation.RmsNormGradGamma, a.Rows, 0f,
                    a.GradOutput, a.Input, a.Stat0, a.Out0)))
                : new TunedKernelInvoker<CudaNormBackwardArgs>((in CudaNormBackwardArgs a) => RequireDirectPtx(TryDirectPtxRowNormalizationCandidate(
                    DirectPtxRowNormalizationOperation.LayerNormGradParameters, a.Rows, 0f,
                    a.GradOutput, a.Input, a.Stat0, Required(a.Stat1), a.Out0, Required(a.Out1))))));
        return new TunedKernelSlot<CudaNormBackwardArgs>(op, TunedDeviceKey,
            new CudaTunedKernelHarness<CudaNormBackwardArgs>(this, RowOpTolerance,
                a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (Required(a.Out1), a.N) },
                a => a.Out0.Handle == a.GradOutput.Handle || a.Out0.Handle == a.Input.Handle ||
                     (a.Out1 is { } o1 && (o1.Handle == a.GradOutput.Handle || o1.Handle == a.Input.Handle))),
            () => GpuDeterminism.IsActive, candidates.ToArray());
    }

    private unsafe void LaunchColumnGradReference(string kernelName, bool rms, in CudaNormBackwardArgs a)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: " + kernelName);
        using var _ = PushContext();
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, s0 = a.Stat0.Handle, o0 = a.Out0.Handle;
        int rows = a.Rows, n = a.N;
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        if (rms)
        {
            void** args = stackalloc void*[6];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &o0; args[4] = &rows; args[5] = &n;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
        else
        {
            IntPtr s1 = Required(a.Stat1).Handle, o1 = Required(a.Out1).Handle;
            void** args = stackalloc void*[8];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &s1; args[4] = &o0; args[5] = &o1;
            args[6] = &rows; args[7] = &n;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
    }

    private unsafe void LaunchTunedColumnGrad(string kernelName, int rowLanes, bool rms, in CudaNormBackwardArgs a)
    {
        IntPtr kernel = _kernelCache[kernelName];
        using var _ = PushContext();
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, s0 = a.Stat0.Handle, o0 = a.Out0.Handle;
        int rows = a.Rows, n = a.N;
        uint grid = (uint)((n + 31) / 32);
        if (rms)
        {
            void** args = stackalloc void*[6];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &o0; args[4] = &rows; args[5] = &n;
            lock (GpuDispatchLock) LaunchKernel3D(kernel, grid, 1, 1, 32, (uint)rowLanes, 1, args);
        }
        else
        {
            IntPtr s1 = Required(a.Stat1).Handle, o1 = Required(a.Out1).Handle;
            void** args = stackalloc void*[8];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &s1; args[4] = &o0; args[5] = &o1;
            args[6] = &rows; args[7] = &n;
            lock (GpuDispatchLock) LaunchKernel3D(kernel, grid, 1, 1, 32, (uint)rowLanes, 1, args);
        }
    }

    // The Direct-PTX row-normalization kernels exist for d = 64 and a fixed set of row counts. As registry
    // candidates they are admitted on any architecture: the gate's correctness and timing replace the
    // validated-architecture list and the environment flag.
    private static bool IsDirectPtxD64Shape(TunedShape shape) =>
        shape.Count == 2 && shape[1] == PtxRowNormalizationD64Kernel.Dimension &&
        PtxRowNormalizationD64Kernel.IsSupportedRows(shape[0]);

    // A Direct-PTX entry point reports failure by returning false without writing; a registry candidate must
    // fail loudly instead, so the gate rejects it rather than timing a no-op.
    private void RequireDirectPtx(bool launched)
    {
        if (!launched)
            throw new InvalidOperationException("Direct-PTX candidate did not launch: " + DirectPtxLastError);
    }

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    private static IGpuBuffer Required(IGpuBuffer? buffer) =>
        buffer ?? throw new InvalidOperationException("A required tuned-kernel buffer argument is missing.");
}
