#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxRecurrentCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxRecurrentCoverageCell(
    string Api,
    string ExistingCudaImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxRecurrentCoverageStatus Status,
    string DirectPtxAssignment);

/// <summary>
/// Executable issue-#846 inventory. Public tensor/device operations and every
/// CUDA recurrent/scan backend entry point have an explicit PTX assignment.
/// </summary>
internal static class DirectPtxRecurrentCoverageManifest
{
    internal static IReadOnlyList<DirectPtxRecurrentCoverageCell> All { get; } =
    [
        Public("CpuEngine.LstmSequenceForward(inference)", "CudaBackend.LstmForwardSequence (NVRTC)",
            "LSTM full-sequence forward and optional final h/c", "input [B,S,I], state [B,H], gates [S,B,4H]", "generic public; CUDA FP32",
            "exact-shape projected-gate + recurrent update families; inference first"),
        Public("CpuEngine.LstmSequenceForward(training)", "CudaBackend.LstmForwardSequence + LstmBackwardSequence (NVRTC)",
            "LSTM tape-aware sequence forward/BPTT", "sequence/state/gate caches", "FP32 CUDA training",
            "forward/backward specializations with deterministic cache ABI"),
        Public("CpuEngine.GlaScanForward", "CudaBackend.GlaScanForward/GlaScanBackward (NVRTC)",
            "GLA recurrent matrix-state forward and BPTT", "[B,S,D] plus [B,S,H] gates", "generic public; CUDA FP32",
            "forward scan then recompute/backward families"),
        Public("CpuEngine.XLstmScanForward", "CudaBackend.XLstmScanForward (NVRTC)",
            "xLSTM matrix-memory scan forward", "[B,S,D] plus per-head gates", "generic public; CUDA FP32",
            "head-persistent forward then backward"),
        Public("CpuEngine.GatedDeltaNetScanForward", "CudaBackend.GatedDeltaNetScanForward (NVRTC)",
            "gated delta-rule state scan", "[B,S,D] plus alpha/beta", "generic public; CUDA FP32",
            "row-persistent exact-head families"),
        new("CpuEngine.RgLruScanForward", "CudaBackend.RgLruScanForward (NVRTC)",
            "RG-LRU channel recurrence forward and CPU-tape backward", "dense [B,S,D], decay [D]", "generic public; CUDA FP32",
            DirectPtxRecurrentCoverageStatus.ExperimentalDirectPtx,
            "v1 direct PTX forward [1,128,256] on exact SM86; backward and other shapes planned"),
        Public("CpuEngine.Rwkv4WkvForward", "CudaBackend.Rwkv4WkvForward (NVRTC)",
            "RWKV-4 numerically stable WKV scan", "[B,S,D], time vectors [D]", "generic public; CUDA FP32",
            "channel-persistent forward/backward families"),
        Public("CpuEngine.Rwkv7SequenceForward", "CudaBackend.Rwkv7Forward (NVRTC)",
            "RWKV-7 generalized delta-rule scan", "[B,S,D] with [B,H,Dh,Dh] state", "generic public; CUDA FP32",
            "head-persistent forward/backward families"),
        Public("CpuEngine.MambaSelectiveScanForward", "CudaBackend.MambaSelectiveScanForward (NVRTC)",
            "Mamba S6 selective scan forward/BPTT", "x/delta [B,S,D], A [D,N], B/C [B,S,N]", "generic public; CUDA FP32",
            "channel-persistent forward and chunked BPTT"),
        Public("CpuEngine.Mamba2SsdScanForward", "CudaBackend.Mamba2SsdScanForward (NVRTC)",
            "Mamba-2 SSD scan forward/BPTT", "x [B,S,D], head delta/A/D, token B/C", "generic public; CUDA FP32",
            "head/channel persistent forward and chunked BPTT"),

        Public("IDeviceRnn.ForwardRnn", "CuDnnRnn CPU fallback; cuDNN v8 bindings not wired", "plain RNN/LSTM/GRU forward", "[S,B,I] packed weight bundle", "generic", "direct PTX per cell/layout; cuDNN resident competitor"),
        Public("IDeviceRnn.ForwardLstm", "CuDnnRnn CPU fallback; cuDNN v8 bindings not wired", "LSTM convenience forward", "[S,B,I] packed weight bundle", "generic", "same LSTM forward families"),
        Public("IDeviceRnn.BackwardRnn", "CuDnnRnn CPU fallback; cuDNN v8 bindings not wired", "RNN/LSTM/GRU backward", "saved sequence plus packed weights", "generic", "cell-specific deterministic BPTT families"),

        Backend("IDirectGpuBackend.LstmForwardSequence", "lstm_forward_sequence (NVRTC)", "LSTM sequence forward", "[S,B,*] + allH/allC/gates", "FP32", "shape-specialized fused forward"),
        Backend("IDirectGpuBackend.LstmBackwardSequence", "lstm_backward_sequence (NVRTC)", "LSTM BPTT", "saved states/gates and gradient outputs", "FP32", "deterministic backward family"),
        Backend("IDirectGpuBackend.GruForwardSequence", "gru_forward_sequence (NVRTC)", "GRU sequence forward", "[S,B,*] + allH/gates", "FP32", "shape-specialized fused forward"),
        Backend("IDirectGpuBackend.GruBackwardSequence", "gru_backward_sequence (NVRTC cooperative)", "GRU BPTT", "saved states/gates and gradients", "FP32", "deterministic backward family"),
        Backend("IDirectGpuBackend.GruCellBackward", "gru_cell_backward_unified + gru_backward_prevh_unified (NVRTC)", "single-cell GRU backward", "[B,H] gates/state", "FP32", "single-launch fused cell backward"),
        Backend("IDirectGpuBackend.GlaScanForward", "gla_scan_forward (NVRTC)", "GLA forward scan", "dense token/head", "FP32", "persistent state-row forward"),
        Backend("IDirectGpuBackend.GlaScanBackward", "gla_scan_recompute + gla_scan_backward (NVRTC)", "GLA recompute BPTT", "trajectory workspace", "FP32", "chunked zero-avoidable-workspace backward"),
        Backend("IDirectGpuBackend.XLstmScanForward", "xlstm_scan_forward (NVRTC)", "xLSTM forward", "dense token/head", "FP32", "head-persistent forward"),
        Backend("IDirectGpuBackend.GatedDeltaNetScanForward", "gated_delta_scan_forward (NVRTC)", "delta-rule forward", "dense token/head", "FP32", "row-persistent forward"),
        new("IDirectGpuBackend.RgLruScanForward", "rglru_scan_forward (NVRTC)", "RG-LRU forward", "dense [B,S,D], decay [D]", "FP32",
            DirectPtxRecurrentCoverageStatus.ExperimentalDirectPtx,
            "v1 direct PTX exact [1,128,256]/SM86 with deterministic NVRTC fallback"),
        Backend("IDirectGpuBackend.Rwkv4WkvForward", "rwkv4_wkv_forward (NVRTC)", "RWKV-4 forward", "dense [B,S,D]", "FP32", "channel-persistent forward"),
        Backend("IDirectGpuBackend.Rwkv7Forward", "rwkv7_forward (NVRTC)", "RWKV-7 forward", "dense token/head state", "FP32", "head-persistent forward"),
        Backend("IDirectGpuBackend.MambaSelectiveScanForward", "mamba_selective_scan_forward (NVRTC)", "Mamba S6 forward", "dense token/channel/state", "FP32", "channel-persistent forward"),
        Backend("IDirectGpuBackend.Mamba2SsdScanForward", "mamba2_ssd_scan_forward (NVRTC)", "Mamba-2 SSD forward", "dense token/head/state", "FP32", "head/channel-persistent forward")
    ];

    internal static DirectPtxRecurrentCoverageCell Get(string api)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(api);
        foreach (DirectPtxRecurrentCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException(
            $"Recurrent API '{api}' is not assigned in the #846 coverage manifest.");
    }

    private static DirectPtxRecurrentCoverageCell Public(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes,
            DirectPtxRecurrentCoverageStatus.PlannedDirectPtx, assignment);

    private static DirectPtxRecurrentCoverageCell Backend(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes,
            DirectPtxRecurrentCoverageStatus.PlannedDirectPtx, assignment);
}
#endif
