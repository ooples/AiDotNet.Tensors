using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxPointwiseCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxPointwiseCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxPointwiseCoverageStatus Status);

/// <summary>
/// Executable issue-#839 inventory. It assigns every CUDA pointwise,
/// activation, derivative, gated activation, scalar/broadcast, and common
/// fused-epilogue family to a direct-PTX blueprint lane.
/// </summary>
internal static class DirectPtxPointwiseCoverageManifest
{
    private const string Vector = "canonical contiguous vector or flattened row-major tensor";
    private const string SplitRow = "canonical contiguous [outer,2D] split as [value|gate]";

    internal static IReadOnlyList<DirectPtxPointwiseCoverageCell> All { get; } =
    [
        Planned("CudaBackend.Add", "NVRTC add", "a+b", Vector, "FP32", "binary-arithmetic-vector-families"),
        Planned("CudaBackend.Subtract", "NVRTC subtract", "a-b", Vector, "FP32", "binary-arithmetic-vector-families"),
        Planned("CudaBackend.Multiply", "NVRTC multiply", "a*b", Vector, "FP32", "binary-arithmetic-vector-families"),
        Planned("CudaBackend.Divide", "NVRTC divide", "a/b", Vector, "FP32", "binary-arithmetic-vector-families"),
        Planned("CudaBackend.AddScalar", "NVRTC add_scalar", "x+scalar", Vector, "FP32", "scalar-arithmetic-baked-value-families"),
        Planned("CudaBackend.SubScalar", "NVRTC sub_scalar", "x-scalar", Vector, "FP32", "scalar-arithmetic-baked-value-families"),
        Planned("CudaBackend.DivScalar", "NVRTC div_scalar", "x/scalar", Vector, "FP32", "scalar-arithmetic-baked-value-families"),
        Planned("CudaBackend.PowScalar", "NVRTC pow_scalar", "pow(x,scalar)", Vector, "FP32", "scalar-special-math-families"),
        Planned("CudaBackend.ScalarMinusTensor", "NVRTC scalar_minus_tensor", "scalar-x", Vector, "FP32", "scalar-arithmetic-baked-value-families"),
        Planned("CudaBackend.BroadcastAddLast", "NVRTC broadcast_add_last", "last-axis broadcast add", "canonical contiguous [outer,inner] plus vector[inner]", "FP32", "broadcast-last-arithmetic-families"),
        Planned("CudaBackend.BroadcastSubLast", "NVRTC broadcast_sub_last", "last-axis broadcast subtract", "canonical contiguous [outer,inner] plus vector[inner]", "FP32", "broadcast-last-arithmetic-families"),
        Planned("CudaBackend.BroadcastMulLast", "NVRTC broadcast_mul_last", "last-axis broadcast multiply", "canonical contiguous [outer,inner] plus vector[inner]", "FP32", "broadcast-last-arithmetic-families"),
        Planned("CudaBackend.BroadcastDivLast", "NVRTC broadcast_div_last", "last-axis broadcast divide", "canonical contiguous [outer,inner] plus vector[inner]", "FP32", "broadcast-last-arithmetic-families"),
        Planned("CudaBackend.BroadcastAddFirst", "NVRTC broadcast_add_first", "first-axis broadcast add", "canonical contiguous [outer,inner] plus vector[outer]", "FP32", "broadcast-first-arithmetic-families"),
        Planned("CudaBackend.BroadcastMulFirst", "NVRTC broadcast_mul_first", "first-axis broadcast multiply", "canonical contiguous [outer,inner] plus vector[outer]", "FP32", "broadcast-first-arithmetic-families"),
        Planned("CudaBackend.EqualsKernel", "NVRTC equals_kernel", "a==b mask", Vector, "FP32 mask", "comparison-mask-vector-families"),
        Planned("CudaBackend.NotEqualsKernel", "NVRTC not_equals_kernel", "a!=b mask", Vector, "FP32 mask", "comparison-mask-vector-families"),
        Planned("CudaBackend.ClipKernel", "NVRTC clip_kernel", "clamp(x,min,max)", Vector, "FP32", "clamp-baked-bound-families"),
        Planned("CudaBackend.FracKernel", "NVRTC frac_kernel", "fractional part", Vector, "FP32", "special-math-vector-families"),
        Planned("CudaBackend.RsqrtKernel", "NVRTC rsqrt_kernel", "reciprocal square root", Vector, "FP32", "special-math-accuracy-mode-families"),
        Planned("CudaBackend.SinCosKernel", "NVRTC sincos_kernel", "paired sine and cosine", Vector, "FP32", "sincos-dual-output-families"),
        Planned("CudaBackend.Relu", "NVRTC relu", "max(x,0)", Vector, "FP32", "activation-forward-vector-families"),
        Planned("CudaBackend.Sigmoid", "NVRTC sigmoid", "1/(1+exp(-x))", Vector, "FP32", "activation-forward-accuracy-families"),
        Planned("CudaBackend.Tanh", "NVRTC tanh", "tanh(x)", Vector, "FP32", "activation-forward-accuracy-families"),
        Planned("CudaBackend.Gelu", "NVRTC gelu", "tanh-approximate GELU", Vector, "FP32", "activation-forward-accuracy-families"),
        Planned("CudaBackend.Swish", "NVRTC swish", "x*sigmoid(x)", Vector, "FP32", "activation-forward-accuracy-families"),
        Planned("CudaBackend.Silu", "NVRTC silu", "x*sigmoid(x)", Vector, "FP32", "activation-forward-accuracy-families"),
        Planned("CudaBackend.Mish", "NVRTC mish", "x*tanh(softplus(x))", Vector, "FP32", "activation-forward-accuracy-families"),
        Planned("CudaBackend.Softplus", "NVRTC softplus", "log(1+exp(x))", Vector, "FP32", "activation-forward-threshold-families"),
        Planned("CudaBackend.LeakyRelu", "NVRTC leaky_relu", "x>=0?x:alpha*x", Vector, "FP32", "activation-forward-baked-alpha-families"),
        Planned("CudaBackend.Elu", "NVRTC elu", "ELU with alpha", Vector, "FP32", "activation-forward-baked-alpha-families"),
        Planned("CudaBackend.Selu", "NVRTC selu", "scaled ELU", Vector, "FP32", "activation-forward-baked-parameter-families"),
        Planned("CudaBackend.Hardswish", "NVRTC hardswish", "hard-Swish", Vector, "FP32", "hard-activation-forward-families"),
        Planned("CudaBackend.Hardsigmoid", "NVRTC hardsigmoid", "hard-sigmoid", Vector, "FP32", "hard-activation-forward-families"),
        Planned("CudaBackend.Hardtanh", "NVRTC hardtanh", "clamped tanh surrogate", Vector, "FP32", "hard-activation-forward-families"),
        Planned("CudaBackend.Relu6", "NVRTC relu6", "clamp(x,0,6)", Vector, "FP32", "hard-activation-forward-families"),
        Planned("CudaBackend.PRelu", "NVRTC prelu", "channel/scalar parameterized ReLU", Vector, "FP32", "prelu-layout-and-alpha-families"),
        Planned("CudaBackend.RRelu", "NVRTC rrelu", "randomized slope using supplied noise", Vector, "FP32", "rrelu-forward-noise-layout-families"),
        Planned("CudaBackend.Threshold", "NVRTC threshold", "x>threshold?x:value", Vector, "FP32", "threshold-baked-parameter-families"),
        Planned("CudaBackend.ReluBackward", "NVRTC relu_backward", "ReLU input gradient", Vector, "FP32", "activation-backward-vector-families"),
        Planned("CudaBackend.SigmoidBackward", "NVRTC sigmoid_backward", "sigmoid output gradient", Vector, "FP32", "activation-backward-vector-families"),
        Planned("CudaBackend.TanhBackward", "NVRTC tanh_backward", "tanh output gradient", Vector, "FP32", "activation-backward-vector-families"),
        Planned("CudaBackend.GeluBackward", "NVRTC gelu_backward", "tanh-GELU input gradient", Vector, "FP32", "activation-backward-accuracy-families"),
        Planned("CudaBackend.SwishBackward", "NVRTC swish_backward", "Swish input gradient", Vector, "FP32", "activation-backward-accuracy-families"),
        Planned("CudaBackend.SiluBackward", "NVRTC silu_backward", "SiLU input gradient", Vector, "FP32", "activation-backward-accuracy-families"),
        Planned("CudaBackend.MishBackward", "NVRTC mish_backward", "Mish input gradient", Vector, "FP32", "activation-backward-accuracy-families"),
        Planned("CudaBackend.SoftplusBackward", "NVRTC softplus_backward", "Softplus input gradient", Vector, "FP32", "activation-backward-threshold-families"),
        Planned("CudaBackend.HardswishBackward", "NVRTC hardswish_backward", "hard-Swish input gradient", Vector, "FP32", "hard-activation-backward-families"),
        Planned("CudaBackend.SeluBackward", "NVRTC selu_backward", "SELU input gradient", Vector, "FP32", "activation-backward-baked-parameter-families"),
        Planned("CudaBackend.HardsigmoidBackward", "NVRTC hardsigmoid_backward", "hard-sigmoid input gradient", Vector, "FP32", "hard-activation-backward-families"),
        Planned("CudaBackend.HardtanhBackward", "NVRTC hardtanh_backward", "hard-tanh input gradient", Vector, "FP32", "hard-activation-backward-families"),
        Planned("CudaBackend.Relu6Backward", "NVRTC relu6_backward", "ReLU6 input gradient", Vector, "FP32", "hard-activation-backward-families"),
        Planned("CudaBackend.PReluBackwardInput", "NVRTC prelu_backward_input", "PReLU input gradient", Vector, "FP32", "prelu-backward-input-families"),
        Planned("CudaBackend.PReluBackwardAlpha", "NVRTC prelu_backward_alpha", "PReLU alpha reduction", Vector, "FP32", "prelu-backward-alpha-reduction-families"),
        Planned("CudaBackend.RReluBackward", "NVRTC rrelu_backward", "RReLU input gradient using saved noise", Vector, "FP32", "rrelu-backward-noise-families"),
        Planned("CudaBackend.ThresholdBackward", "NVRTC threshold_backward", "threshold input gradient", Vector, "FP32", "threshold-backward-families"),
        Planned("CudaBackend.ReciprocalBackward", "NVRTC reciprocal_backward", "-grad/x^2", Vector, "FP32", "reciprocal-backward-families"),
        Planned("CudaBackend.GluForward", "NVRTC glu_forward", "value*sigmoid(gate)", SplitRow, "FP32", "glu-forward-split-row-families"),
        Experimental("CudaBackend.GeGluForward", "direct PTX or NVRTC geglu_forward", "value*gelu_tanh(gate)", SplitRow, "FP32", "v1 Ampere split-row FP32x4 candidates"),
        Planned("CudaBackend.ReGluForward", "NVRTC reglu_forward", "value*relu(gate)", SplitRow, "FP32", "reglu-forward-split-row-families"),
        Experimental("CudaBackend.SwiGluForward", "direct PTX or NVRTC swiglu_forward", "value*silu(gate)", SplitRow, "FP32", "v1 Ampere split-row FP32x4 performance-gated candidates"),
        Planned("CudaBackend.GluBackward", "NVRTC glu_backward", "GLU gradients for both halves", SplitRow, "FP32", "glu-backward-split-row-families"),
        Experimental("CudaBackend.GeGluBackward", "direct PTX or NVRTC geglu_backward", "GeGLU gradients for both halves", SplitRow, "FP32", "v1 Ampere split-row FP32x4 derivative candidate"),
        Planned("CudaBackend.ReGluBackward", "NVRTC reglu_backward", "ReGLU gradients for both halves", SplitRow, "FP32", "reglu-backward-split-row-families"),
        Planned("CudaBackend.SwiGluBackward", "NVRTC swiglu_backward", "SwiGLU gradients for both halves", SplitRow, "FP32", "swiglu-backward-split-row-families"),
        Planned("CudaBackend.AddRelu", "NVRTC add then in-place ReLU", "relu(a+b)", Vector, "FP32", "fused-add-activation-vocabulary"),
        Planned("CudaBackend.AddSigmoid", "NVRTC add then in-place sigmoid", "sigmoid(a+b)", Vector, "FP32", "fused-add-activation-vocabulary"),
        Planned("CudaBackend.AddGelu", "NVRTC add then in-place GELU", "gelu(a+b)", Vector, "FP32", "fused-add-activation-vocabulary"),
        Planned("CudaBackend.BiasAddRelu", "NVRTC bias add plus ReLU composition", "relu(matrix+bias)", "canonical contiguous [rows,cols] plus vector[cols]", "FP32", "fused-bias-activation-vocabulary"),
        Planned("CudaBackend.BiasAddGelu", "NVRTC bias add plus GELU composition", "gelu(matrix+bias)", "canonical contiguous [rows,cols] plus vector[cols]", "FP32", "fused-bias-activation-vocabulary"),
        Planned("CudaBackend.BiasAddSilu", "NVRTC bias add plus SiLU composition", "silu(matrix+bias)", "canonical contiguous [rows,cols] plus vector[cols]", "FP32", "fused-bias-activation-vocabulary"),
        Existing("CudaFusedKernels.pointwise chains", "resident NVRTC fused elementwise catalog", "selected add/bias/activation chains", Vector, "FP32", "bounded-direct-ptx-expression-vocabulary"),
        Existing("CudaFp16Kernels activations", "resident NVRTC FP16 activation catalog", "native FP16 I/O activation forward/backward", Vector, "FP16", "fp16-pointwise-and-activation-families"),
        Existing("DirectGpuTensorEngine activation routes", "resident backend route or CPU fallback", "public tensor activation semantics", Vector, "generic public; GPU FP32/FP16", "public-activation-routing-families")
    ];

    internal static DirectPtxPointwiseCoverageCell Get(string api) =>
        All.FirstOrDefault(cell => string.Equals(cell.Api, api, StringComparison.Ordinal)) ??
        throw new KeyNotFoundException(
            $"No #839 pointwise coverage cell is assigned to '{api}'.");

    private static DirectPtxPointwiseCoverageCell Planned(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxPointwiseCoverageStatus.PlannedDirectPtx);

    private static DirectPtxPointwiseCoverageCell Existing(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxPointwiseCoverageStatus.ExistingBackend);

    private static DirectPtxPointwiseCoverageCell Experimental(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx);
}
