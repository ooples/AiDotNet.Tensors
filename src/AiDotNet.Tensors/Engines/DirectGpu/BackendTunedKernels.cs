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
        using var _ = GpuLaunchProbe.TuningScope();
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
        using var _ = GpuLaunchProbe.TuningScope();
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
        IGpuBuffer stat0, IGpuBuffer? stat1, IGpuBuffer out0, IGpuBuffer? out1, int rows, int n, float epsilon = 0f)
    {
        GradOutput = gradOutput; Input = input; Gamma = gamma; Stat0 = stat0; Stat1 = stat1; Out0 = out0; Out1 = out1;
        Rows = rows; N = n; Epsilon = epsilon;
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

    /// <summary>
    /// The forward pass's epsilon, for a reference kernel that recomputes the row statistics instead of reading the saved
    /// ones (WebGPU's). Zero where no reference needs it.
    /// </summary>
    internal float Epsilon { get; }
}
/// <summary>
/// Conv2D backward arguments. Weight gradient: First = input, Second = gradOutput, Output = gradKernel. Input gradient:
/// First = gradOutput, Second = kernel, Output = gradInput. Sizes as in <see cref="TunedConv2DArgs"/>. Shared by every
/// backend's backward slots.
/// </summary>
internal readonly struct TunedConv2DBackwardArgs
{
    internal TunedConv2DBackwardArgs(IGpuBuffer first, IGpuBuffer second, IGpuBuffer output, bool weightGradient,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        First = first; Second = second; Output = output; WeightGradient = weightGradient;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW;
        PadH = padH; PadW = padW; DilationH = dilationH; DilationW = dilationW;
    }

    internal IGpuBuffer First { get; }
    internal IGpuBuffer Second { get; }
    internal IGpuBuffer Output { get; }
    internal bool WeightGradient { get; }
    internal int Batch { get; }
    internal int InChannels { get; }
    internal int InHeight { get; }
    internal int InWidth { get; }
    internal int OutChannels { get; }
    internal int OutHeight { get; }
    internal int OutWidth { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int StrideH { get; }
    internal int StrideW { get; }
    internal int PadH { get; }
    internal int PadW { get; }
    internal int DilationH { get; }
    internal int DilationW { get; }

    internal int OutputCount => WeightGradient
        ? OutChannels * InChannels * KernelH * KernelW
        : Batch * InChannels * InHeight * InWidth;

    /// <summary>Shape class: every geometric parameter, laid out as <see cref="TunedConv2DArgs.Shape"/>.</summary>
    internal unsafe TunedShape Shape()
    {
        int* d = stackalloc int[15];
        d[0] = Batch; d[1] = InChannels; d[2] = InHeight; d[3] = InWidth; d[4] = OutChannels;
        d[5] = KernelH; d[6] = KernelW; d[7] = StrideH; d[8] = StrideW; d[9] = PadH; d[10] = PadW;
        d[11] = DilationH; d[12] = DilationW; d[13] = OutHeight; d[14] = OutWidth;
        return TunedShape.Create(TunedKernelDType.Float32, new ReadOnlySpan<int>(d, 15));
    }
}

/// <summary>
/// A convolution's epilogue (<see cref="TunedConv2DArgs.Bias"/> and <see cref="TunedConv2DArgs.Activation"/>) on every
/// backend: the flags an output kernel takes (bit 0 adds the per-channel bias, bit 1 applies ReLU, the activations the
/// kernels fuse), and the bias add and activation as separate passes for a route whose kernels do not take them.
/// </summary>
internal static class ConvEpilogue
{
    /// <summary>Whether the output kernels can apply <paramref name="activation"/> (no activation, or ReLU).</summary>
    internal static bool Fuses(FusedActivationType activation)
        => activation == FusedActivationType.None || activation == FusedActivationType.ReLU;

    internal static int Flags(in TunedConv2DArgs a)
        => (a.Bias is null ? 0 : 1) | (a.Activation == FusedActivationType.ReLU ? 2 : 0);

    /// <summary>The bias add and the ReLU after a convolution that wrote <see cref="TunedConv2DArgs.Output"/> without them.</summary>
    internal static void RunPasses(IDirectGpuBackend backend, in TunedConv2DArgs a)
    {
        if (a.Bias is { } bias) backend.Conv2DBiasAdd(a.Output, bias, a.Batch, a.OutChannels, a.OutHeight * a.OutWidth);
        if (a.Activation == FusedActivationType.ReLU) backend.Relu(a.Output, a.Output, a.OutputCount);
    }
}

/// <summary>2-D convolution arguments (any backend): NCHW input, [Cout, Cin, kH, kW] filter.</summary>
internal readonly struct TunedConv2DArgs
{
    internal TunedConv2DArgs(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        IGpuBuffer? bias = null, FusedActivationType activation = FusedActivationType.None)
    {
        Input = input; Kernel = kernel; Output = output; Bias = bias; Activation = activation;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW;
        PadH = padH; PadW = padW; DilationH = dilationH; DilationW = dilationW;
    }

    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Kernel { get; }
    internal IGpuBuffer Output { get; }

    /// <summary>A per-output-channel bias added to the convolution's result, or null.</summary>
    internal IGpuBuffer? Bias { get; }

    /// <summary>The activation applied after the bias. Every route produces the same result: one that cannot apply them
    /// in its own output kernel runs the bias add and the activation after it.</summary>
    internal FusedActivationType Activation { get; }

    internal int Batch { get; }
    internal int InChannels { get; }
    internal int InHeight { get; }
    internal int InWidth { get; }
    internal int OutChannels { get; }
    internal int OutHeight { get; }
    internal int OutWidth { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int StrideH { get; }
    internal int StrideW { get; }
    internal int PadH { get; }
    internal int PadW { get; }
    internal int DilationH { get; }
    internal int DilationW { get; }

    internal int OutputCount => Batch * OutChannels * OutHeight * OutWidth;

    /// <summary>Shape class: every geometric parameter, in CudaConv2DArgs' order (so profiles share keys).</summary>
    internal TunedShape Shape()
    {
        var d = new[]
        {
            Batch, InChannels, InHeight, InWidth, OutChannels, KernelH, KernelW, StrideH, StrideW, PadH, PadW,
            DilationH, DilationW, OutHeight, OutWidth,
        };
        return TunedShape.Create(TunedKernelDType.Float32, d);
    }

    /// <summary>The im2col scratch (batch * Cin*kH*kW * outH*outW floats) stays under 256 MiB.</summary>
    internal static bool GemmScratchFits(TunedShape s)
    {
        long floats = (long)s[0] * s[1] * s[5] * s[6] * s[13] * s[14];
        return floats > 0 && floats <= 64L * 1024 * 1024;
    }
}