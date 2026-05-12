// Copyright (c) AiDotNet. All rights reserved.
// PR #333 / issue #337 — mixed-precision routing policy for compute-bound
// GPU ops. Distinct from the TF32 path (CudaDispatchPolicy.AllowTF32) which
// stays on fp32 but rounds the multiply on the Tensor Cores; this enum
// controls whether to actively cast tensors down to fp16 / bf16 for the
// compute step while keeping fp32 master weights.

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Routing policy that hot-path GPU kernels consult to decide their compute
/// dtype. Set via <see cref="TensorCodecOptions.MixedPrecisionPolicy"/> for
/// thread-local effect, or as a default in <see cref="TensorCodecOptions.Default"/>.
/// </summary>
public enum MixedPrecisionPolicy
{
    /// <summary>
    /// No autocast. Kernels run in the model's declared dtype throughout.
    /// Default. Use this for numerical-reproducibility runs, baseline
    /// benchmarks, and platforms where the autocast paths aren't yet
    /// kernel-supported on the loaded backend.
    /// </summary>
    None = 0,

    /// <summary>
    /// Autocast compute-bound ops to fp16 (System.Half) while keeping fp32
    /// master weights, gradients, and reductions. Forward casts fp32 → fp16
    /// at op entry; backward emits gradients in fp32 from the autocast op's
    /// internal fp16 state. Requires loss scaling (typically ×2¹⁵..2¹⁷)
    /// to prevent small-magnitude gradients from underflowing fp16's narrow
    /// exponent range; the loss-scaling state lives on the optimizer.
    /// </summary>
    /// <remarks>
    /// Hardware availability: Volta+, Turing+, Ampere+ for fp16 Tensor Cores.
    /// All compute-capability 7.0+ NVIDIA GPUs.
    /// </remarks>
    AutoCastFP16 = 1,

    /// <summary>
    /// Autocast compute-bound ops to bfloat16 while keeping fp32 master weights.
    /// BF16 has the same 8-bit exponent as fp32 — no loss scaling required —
    /// but trades 7 mantissa bits (vs fp16's 10) for the dynamic range. Best
    /// choice for transformer training when the hardware supports it.
    /// </summary>
    /// <remarks>
    /// Hardware availability: Ampere (sm_80+) for full BF16 Tensor Core
    /// support, Hopper (sm_90+) for BF16 on every op. On Volta / Turing /
    /// pre-Ampere consumer cards (RTX 2080, RTX 3090 selected ops), BF16
    /// kernels may emulate via fp32 — check vendor docs.
    /// </remarks>
    AutoCastBF16 = 2,
}
