using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxOptimizerCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxOptimizerCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxOptimizerCoverageStatus Status);

/// <summary>
/// Executable issue-#848 inventory. Every CUDA optimizer step, gradient-scaling,
/// clipping, and sparse-update boundary has one explicit direct-PTX lane. Shape
/// and hyperparameter families remain closed sets: hyperparameters that vary at
/// runtime are baked into module identity, and a planned row must be split into
/// exact physical ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxOptimizerCoverageManifest
{
    private const string Vector = "canonical contiguous vector";

    internal static IReadOnlyList<DirectPtxOptimizerCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.SgdMomentumUpdate", "NVRTC sgd_momentum_update", "g'=grad+wd*p; v=mom*v+g'; p-=lr*v", Vector, "FP32", "v1 Ampere linear-vec4 exact-size cells (baked lr/mom/wd, no materialized temp)"),
        Planned("CudaBackend.SgdUpdate", "NVRTC sgd_update", "p -= lr*(grad + wd*p)", Vector, "FP32", "linear-vec4-baked-hyperparam-families"),
        Planned("CudaBackend.NesterovMomentumUpdate", "NVRTC nesterov_momentum_update", "look-ahead momentum step", Vector, "FP32", "linear-vec4-nesterov-families"),
        Planned("CudaBackend.AdamUpdate", "NVRTC adam_update", "bias-corrected first/second moment step", Vector, "FP32", "baked-beta-eps-step-corrected-lr-families"),
        Planned("CudaBackend.AdamWUpdate", "NVRTC adamw_update", "Adam with decoupled weight decay", Vector, "FP32", "baked-beta-eps-decoupled-wd-families"),
        Planned("CudaBackend.RmsPropUpdate", "NVRTC rmsprop_update", "root-mean-square propagation step", Vector, "FP32", "baked-alpha-eps-families"),
        Planned("CudaBackend.AdagradUpdate", "NVRTC adagrad_update", "accumulated-square adaptive step", Vector, "FP32", "linear-vec4-adagrad-families"),
        Planned("CudaBackend.AdadeltaUpdate", "NVRTC adadelta_update", "running-delta adaptive step", Vector, "FP32", "baked-rho-eps-families"),
        Planned("CudaBackend.LambUpdate", "NVRTC lamb_update", "layerwise adaptive trust-ratio step", Vector, "FP32", "per-tensor-trust-ratio-families"),
        Planned("CudaBackend.LionUpdate", "NVRTC lion_update", "sign-momentum step", Vector, "FP32", "baked-beta-lion-families"),
        Planned("CudaBackend.ClipGradNorm", "NVRTC clip_grad_norm", "scale gradients by global-norm ratio", Vector, "FP32", "two-stage-global-norm-clip-families"),
        Planned("CudaBackend.ClipGradValue", "NVRTC clip_grad_value", "clamp gradients to a baked range", Vector, "FP32", "baked-clamp-range-families"),
        Planned("CudaBackend.ScaleGradients", "NVRTC scale_gradients", "multiply gradients by a baked scale", Vector, "FP32", "baked-loss-scale-families"),
        Planned("CudaBackend.UnscaleGradientsCheckFinite", "NVRTC unscale_and_check_finite", "AMP unscale with inf/nan detection", Vector, "FP32", "baked-inv-scale-finite-check-families"),
        Planned("CudaBackend.ZeroGradients", "NVRTC memset / zero", "reset gradients to zero", Vector, "FP32", "linear-vec4-zero-families"),
        Planned("CudaBackend.SparseSgdUpdate", "NVRTC sparse_sgd_update", "SGD over gathered parameter rows", "canonical contiguous param/index rows", "FP32; INT32 indices", "indexed-sparse-update-families"),
        Planned("CudaBackend.EmaUpdate", "NVRTC ema_update", "exponential moving average of parameters", Vector, "FP32", "baked-decay-ema-families"),
        Planned("DirectGpuEngine.SgdMomentumUpdate", "array upload + CudaBackend.SgdMomentumUpdate + download", "public array SGD-momentum step", Vector, "generic public; CUDA FP32", "public-array-optimizer-routing"),
        Planned("DirectGpuTensorEngine.TensorSgdStep", "resident CudaBackend.SgdMomentumUpdate or CPU fallback", "public tensor optimizer step", "logical strided tensors; canonical contiguous fast path", "generic public; CUDA FP32", "public-tensor-optimizer-routing"),
        Planned("DirectGpuTensorEngine.TensorAdamStep", "CudaBackend.AdamUpdate or CPU fallback", "public tensor Adam step", "logical strided tensors; canonical contiguous fast path", "generic public; CUDA FP32", "public-adam-routing"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxOptimizerCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxOptimizerCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxOptimizerCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX optimizer coverage cell for '{api}'.");

    private static DirectPtxOptimizerCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxOptimizerCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxOptimizerCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxOptimizerCoverageStatus.PlannedDirectPtx);
}
