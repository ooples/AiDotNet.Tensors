using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxPointwiseCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

/// <summary>
/// One auditable assignment in issue #839 (pointwise arithmetic, activation, GLU/gating and
/// backward fusions). The inventory is code rather than a markdown snapshot so tests reject
/// duplicate, missing, or accidentally unassigned pointwise entry points as the backend
/// surface evolves.
/// </summary>
internal sealed record DirectPtxPointwiseCoverageCell(
    string Api,
    string ExistingCudaImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxPointwiseCoverageStatus Status,
    string DirectPtxAssignment);

internal static class DirectPtxPointwiseCoverageManifest
{
    private const string Flat = "contiguous flat element buffer";
    private const string Planned =
        "shape/dtype-specific PTX pointwise kernel; part of the bounded fusion vocabulary";
    private const string ActFwd =
        "elementwise activation-forward PTX kernel (PtxActivationForwardKernel) selected by " +
        "the shared PtxActivationEmit; fails closed until GPU-validated and promoted";

    internal static IReadOnlyList<DirectPtxPointwiseCoverageCell> All { get; } =
    [
        // Forward activations covered by the parameterized PTX kernel.
        new("CudaBackend.Relu", "NVRTC relu", "max(x, 0)", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),
        new("CudaBackend.LeakyRelu", "NVRTC leaky-relu", "x > 0 ? x : 0.01x", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),
        new("CudaBackend.Sigmoid", "NVRTC sigmoid", "1/(1+exp(-x))", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),
        new("CudaBackend.Tanh", "NVRTC tanh", "tanh(x)", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),
        new("CudaBackend.Gelu", "NVRTC gelu", "gelu-tanh approximation", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),
        new("CudaBackend.Silu", "NVRTC silu", "x * sigmoid(x)", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),
        new("CudaBackend.Swish", "NVRTC swish", "x * sigmoid(x) (== silu)", Flat, "FP32", DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx, ActFwd),

        // Backward activations (Jacobian gating) — planned.
        new("CudaBackend.ReluBackward", "NVRTC relu backward", "dY * (x > 0)", Flat + ", plus upstream grad", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.LeakyReluBackward", "NVRTC leaky-relu backward", "dY * (x > 0 ? 1 : 0.01)", Flat + ", plus upstream grad", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.SigmoidBackward", "NVRTC sigmoid backward", "dY * s(1-s)", Flat + ", plus upstream grad", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.TanhBackward", "NVRTC tanh backward", "dY * (1 - t^2)", Flat + ", plus upstream grad", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.GeluBackward", "NVRTC gelu backward", "dY * gelu'(x)", Flat + ", plus upstream grad", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.SiluBackward", "NVRTC silu backward", "dY * (s + x s(1-s))", Flat + ", plus upstream grad", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),

        // Exp-based activations and their backward — planned.
        new("CudaBackend.Elu", "NVRTC elu", "x > 0 ? x : exp(x)-1", Flat, "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.Softplus", "NVRTC softplus", "log(1 + exp(x))", Flat, "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.Mish", "NVRTC mish", "x * tanh(softplus(x))", Flat, "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),

        // GLU family — planned.
        new("CudaBackend.GluForward", "NVRTC glu", "a * sigmoid(b) over a split axis", "paired halves", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.GeGluForward", "NVRTC geglu", "a * gelu(b)", "paired halves", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.SwiGluForward", "NVRTC swiglu", "a * silu(b)", "paired halves", "FP32", DirectPtxPointwiseCoverageStatus.PlannedDirectPtx, Planned)
    ];

    internal static DirectPtxPointwiseCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxPointwiseCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Pointwise API '{api}' is not assigned in the #839 coverage manifest.");
    }
}
