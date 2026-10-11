using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Runs a tuned-kernel candidate on arguments passed by reference.</summary>
internal delegate void TunedCandidateInvoker<TArgs>(in TArgs args) where TArgs : struct;

/// <summary>A registry candidate backed by delegates, for the backends without CUDA's own candidate type.</summary>
internal sealed class DelegateTunedKernelCandidate<TArgs> : ITunedKernelCandidate<TArgs> where TArgs : struct
{
    private readonly Func<TunedShape, bool> _applicable;
    private readonly TunedCandidateInvoker<TArgs> _execute;

    internal DelegateTunedKernelCandidate(string id, TunedKernelOrigin origin, bool deterministic,
        Func<TunedShape, bool> applicable, TunedCandidateInvoker<TArgs> execute)
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

/// <summary>
/// Measurement services for one op family on any <see cref="IDirectGpuBackend"/>: output poisoning and host snapshots
/// through the backend's Fill / Download, and timing through a backend-supplied measure function.
/// </summary>
internal sealed class BackendTunedKernelHarness<TArgs> : ITunedKernelHarness<TArgs> where TArgs : struct
{
    private readonly IDirectGpuBackend _backend;
    private readonly Func<bool> _canMeasure;
    private readonly Func<Action, int, double> _measure;
    private readonly Func<TArgs, (IGpuBuffer Buffer, int Count)[]> _outputs;
    private readonly Func<TArgs, bool> _aliased;

    internal BackendTunedKernelHarness(IDirectGpuBackend backend, double relativeTolerance, Func<bool> canMeasure,
        Func<Action, int, double> measure, Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs, Func<TArgs, bool> aliased)
    {
        _backend = backend;
        RelativeTolerance = relativeTolerance;
        _canMeasure = canMeasure;
        _measure = measure;
        _outputs = outputs;
        _aliased = aliased;
    }

    public double RelativeTolerance { get; }

    public bool CanMeasure(in TArgs args) => _canMeasure() && !_aliased(args);

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
        return _measure(() => candidate.Execute(local), repetitions);
    }
}

/// <summary>Row softmax arguments (any backend).</summary>
internal readonly struct TunedSoftmaxArgs
{
    internal TunedSoftmaxArgs(IGpuBuffer input, IGpuBuffer output, int rows, int n)
    { Input = input; Output = output; Rows = rows; N = n; }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Output { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>Row softmax backward arguments (any backend).</summary>
internal readonly struct TunedSoftmaxBackwardArgs
{
    internal TunedSoftmaxBackwardArgs(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
    { GradOutput = gradOutput; Output = output; GradInput = gradInput; Rows = rows; N = n; }
    internal IGpuBuffer GradOutput { get; }
    internal IGpuBuffer Output { get; }
    internal IGpuBuffer GradInput { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>LayerNorm forward arguments (any backend).</summary>
internal readonly struct TunedLayerNormArgs
{
    internal TunedLayerNormArgs(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
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

/// <summary>
/// Normalization backward arguments (any backend): the LayerNorm input gradient (Gamma, Stat0 = mean, Stat1 = inverse
/// standard deviation, Out0 = gradInput) or a column reduction (Out0 = gradGamma, Out1 = gradBeta; RMSNorm: Stat0 = RMS).
/// </summary>
internal readonly struct TunedNormBackwardArgs
{
    internal TunedNormBackwardArgs(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer? gamma,
        IGpuBuffer stat0, IGpuBuffer? stat1, IGpuBuffer out0, IGpuBuffer? out1, int rows, int n)
    {
        GradOutput = gradOutput; Input = input; Gamma = gamma; Stat0 = stat0; Stat1 = stat1; Out0 = out0; Out1 = out1;
        Rows = rows; N = n;
    }
    internal IGpuBuffer GradOutput { get; }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer? Gamma { get; }
    internal IGpuBuffer Stat0 { get; }
    internal IGpuBuffer? Stat1 { get; }
    internal IGpuBuffer Out0 { get; }
    internal IGpuBuffer? Out1 { get; }
    internal int Rows { get; }
    internal int N { get; }
}