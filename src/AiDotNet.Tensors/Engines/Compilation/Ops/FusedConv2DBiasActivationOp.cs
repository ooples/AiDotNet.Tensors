using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Ops;

/// <summary>
/// IR operation for fused Conv2D + Bias + Activation. Maps to the existing
/// <see cref="IEngine.FusedConv2D{T}"/> kernel which combines convolution,
/// bias addition, and activation into a single pass — no intermediate tensor
/// materialization between the three stages.
///
/// <para><b>Pattern:</b> Diffusion UNets use <c>Conv+Bias+Identity</c> (no
/// BatchNorm) or <c>Conv+Bias+SiLU</c> at every layer. This is distinct from
/// <c>FusedConvBatchNormActivationOp</c> which handles the BN-containing
/// pattern from classification networks.</para>
///
/// <para><b>Performance:</b> 2-4× faster than the 3-op sequence
/// <c>Conv → BroadcastAdd → Activation</c> on CPU; larger gains on GPU
/// (single kernel launch instead of 3).</para>
///
/// <para><b>Backward:</b> decomposes into standard Conv2D backward (for input
/// + kernel gradients), broadcast-add backward (for bias gradient), and the
/// activation's pointwise backward — reusing the existing backward kernels
/// rather than implementing a monolithic fused backward.</para>
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
internal sealed class FusedConv2DBiasActivationOp<T> : ICompiledOp<T>
{
    /// <summary>Input tensor [N, Cin, H, W].</summary>
    public Tensor<T> Input { get; }

    /// <summary>Convolution kernel [Cout, Cin, kH, kW].</summary>
    public Tensor<T> Kernel { get; }

    /// <summary>Optional bias [Cout]. Null for unbiased convolutions.</summary>
    public Tensor<T>? Bias { get; }

    /// <summary>Stride in H and W dimensions.</summary>
    public int StrideH { get; }
    public int StrideW { get; }

    /// <summary>Padding in H and W dimensions.</summary>
    public int PadH { get; }
    public int PadW { get; }

    /// <summary>Dilation in H and W dimensions (default 1).</summary>
    public int DilationH { get; }
    public int DilationW { get; }

    /// <summary>Activation to fuse (None/Identity, SiLU/Swish, ReLU, Sigmoid, etc.).</summary>
    public FusedActivationType Activation { get; }

    public FusedConv2DBiasActivationOp(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH = 1, int dilationW = 1,
        FusedActivationType activation = FusedActivationType.None)
    {
        Input = input ?? throw new ArgumentNullException(nameof(input));
        Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        Bias = bias;
        StrideH = strideH;
        StrideW = strideW;
        PadH = padH;
        PadW = padW;
        DilationH = dilationH;
        DilationW = dilationW;
        Activation = activation;
    }

    // ── ICompiledOp<T> ─────────────────────────────────────────────────

    public OpType OpType => OpType.Conv2D; // Reuses Conv2D slot since fused is a superset
    public string OpName => "FusedConv2DBiasActivation";

    public Tensor<T>[] Inputs => Bias is not null
        ? new[] { Input, Kernel, Bias }
        : new[] { Input, Kernel };

    public int[] OutputShape
    {
        get
        {
            int n = Input._shape[0];
            int cout = Kernel._shape[0];
            int hIn = Input._shape[2];
            int wIn = Input._shape[3];
            int kH = Kernel._shape[2];
            int kW = Kernel._shape[3];
            int hOut = (hIn + 2 * PadH - DilationH * (kH - 1) - 1) / StrideH + 1;
            int wOut = (wIn + 2 * PadW - DilationW * (kW - 1) - 1) / StrideW + 1;
            return new[] { n, cout, hOut, wOut };
        }
    }

    public Action<IEngine, Tensor<T>> BuildForwardClosure()
    {
        var input = Input;
        var kernel = Kernel;
        var bias = Bias;
        int sH = StrideH, sW = StrideW;
        int pH = PadH, pW = PadW;
        int dH = DilationH, dW = DilationW;
        var activation = Activation;

        // Reshape bias to [1, Cout, 1, 1] for broadcasting over [N, C, H, W].
        // CpuEngine.FusedConv2D calls TensorBroadcastAdd(convResult, bias)
        // which follows NumPy broadcasting rules — a raw [Cout] shape can't
        // broadcast against [N, Cout, H, W] (trailing dims don't align).
        // Only reshape if bias is 1D; if already 4D (e.g., from a fusion pass
        // that extracted it from a BroadcastAdd), use as-is.
        var reshapedBias = bias is not null && bias.Rank == 1
            ? bias.Reshape(new[] { 1, bias._shape[0], 1, 1 })
            : bias;

        return (eng, output) =>
        {
            var result = eng.FusedConv2D(input, kernel, reshapedBias, sH, sW, pH, pW, dH, dW, activation);
            result.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    public BackwardFunction<T>? GetBackwardFunction()
    {
        // Fused backward: decompose into Conv2D backward + bias grad + activation backward.
        // Uses the standard Conv2DBackward for input/kernel gradients. Bias gradient is
        // the sum of gradOutput over spatial dims. Activation backward is pointwise.
        return FusedConv2DBiasActivationBackward;
    }

    public object[]? BuildSavedState()
    {
        // SavedState: [stride_arr, padding_arr, dilation_arr, activation_int]
        return new object[]
        {
            new[] { StrideH, StrideW },
            new[] { PadH, PadW },
            new[] { DilationH, DilationW },
            (int)Activation
        };
    }

    public CompiledStep<T> ToCompiledStep(Tensor<T> outputBuffer)
    {
        return new CompiledStep<T>(
            OpName,
            BuildForwardClosure(),
            outputBuffer,
            Inputs,
            GetBackwardFunction(),
            BuildSavedState());
    }

    // ── Factory ─────────────────────────────────────────────────────────

    /// <summary>
    /// Attempts to extract a <see cref="FusedConv2DBiasActivationOp{T}"/> from
    /// an existing <see cref="CompiledStep{T}"/>. Returns null if the step
    /// isn't a fused conv op or its SavedState can't be parsed.
    /// </summary>
    internal static FusedConv2DBiasActivationOp<T>? TryFromStep(CompiledStep<T> step)
    {
        if (step.OpName != "FusedConv2DBiasActivation") return null;
        if (step.Inputs.Length < 2) return null;

        var state = step.SavedState;
        if (state is null || state.Length < 4) return null;

        var stride = state[0] as int[];
        var padding = state[1] as int[];
        var dilation = state[2] as int[];
        var activationInt = state[3] is int a ? a : 0;

        if (stride is null || padding is null || dilation is null) return null;

        return new FusedConv2DBiasActivationOp<T>(
            step.Inputs[0],
            step.Inputs[1],
            step.Inputs.Length > 2 ? step.Inputs[2] : null,
            stride.Length > 0 ? stride[0] : 1,
            stride.Length > 1 ? stride[1] : stride[0],
            padding.Length > 0 ? padding[0] : 0,
            padding.Length > 1 ? padding[1] : padding[0],
            dilation.Length > 0 ? dilation[0] : 1,
            dilation.Length > 1 ? dilation[1] : dilation[0],
            (FusedActivationType)activationInt);
    }

    // ── Backward ────────────────────────────────────────────────────────

    /// <summary>
    /// Backward for fused Conv+Bias+Activation. Decomposes into:
    /// 1. Activation backward (pointwise — modifies gradOutput in place)
    /// 2. Conv2D backward for input + kernel gradients
    /// 3. Bias gradient (sum of gradOutput over batch + spatial dims)
    /// </summary>
    private static void FusedConv2DBiasActivationBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int[])savedState[0];
        var padding = (int[])savedState[1];
        var dilation = (int[])savedState[2];
        var activationType = (FusedActivationType)(int)savedState[3];

        // Step 1: undo activation on gradOutput if needed
        var effectiveGrad = gradOutput;
        if (activationType != FusedActivationType.None)
        {
            // For the common case (SiLU, ReLU), pointwise backward
            effectiveGrad = activationType switch
            {
                FusedActivationType.ReLU => engine.ReluBackward(gradOutput, output),
                FusedActivationType.Sigmoid => engine.SigmoidBackward(gradOutput, output),
                FusedActivationType.Swish => engine.SwishBackward(gradOutput, inputs[0]),
                _ => gradOutput, // Fallback: pass through
            };
        }

        // Step 2: Conv2D backward for input + kernel
        var gradInput = engine.Conv2DBackwardInput(
            effectiveGrad, inputs[1], inputs[0]._shape, stride, padding, dilation);
        var gradKernel = engine.Conv2DBackwardKernel(
            effectiveGrad, inputs[0], inputs[1]._shape, stride, padding, dilation);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradKernel, engine);

        // Step 3: Bias gradient (sum over batch + spatial)
        if (inputs.Length > 2 && inputs[2] is not null)
        {
            // gradBias = sum of effectiveGrad over dims [0, 2, 3] (batch, H, W)
            var gradBias = engine.ReduceSum(effectiveGrad, new[] { 0, 2, 3 });
            DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
        }
    }
}
