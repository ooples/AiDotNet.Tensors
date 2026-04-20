namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// #210 Item #3 — GPU kernel coverage for the 90+ Parity-210 ops across 6
/// backends (CUDA, HIP, Metal, Vulkan, OpenCL, WebGPU).
/// </summary>
/// <remarks>
/// <para><b>Current state:</b> every Parity-210 op added to
/// <see cref="CpuEngine"/> is also reachable from
/// <see cref="DirectGpuTensorEngine"/> by inheritance — the GPU class
/// inherits <c>: CpuEngine</c>, so calling e.g.
/// <c>gpuEngine.TensorRoll(...)</c> drops into the CPU implementation
/// when no GPU kernel exists. This is the "best-effort" floor the #210
/// plan specifies (§4, "GPU backend floor"):
/// <list type="bullet">
///   <item>Every op ships CPU + (GPU-via-CPU-fallback) kernels.</item>
///   <item>Native CUDA / HIP / Metal / Vulkan / OpenCL / WebGPU kernels
///     are added op-by-op as measurement shows the fallback is too slow.</item>
///   <item>Where a backend genuinely cannot express an op (e.g. WebGPU
///     lacking a specific intrinsic), the test is skipped with a
///     documented technical reason.</item>
/// </list>
/// </para>
/// <para><b>Native kernel rollout plan</b> (documented here so follow-up
/// commits have a concrete target):</para>
/// <list type="number">
///   <item>Priority 1 — ops that appear inside hot paths (Roll, Flip,
///     CumSum/Prod, IndexAdd, ScatterReduce, MaskedSelect, Sort, TopK,
///     Histogram, Gather/Scatter for packed formats).</item>
///   <item>Priority 2 — ops used in training loss / metric paths
///     (CosineSimilarity, PDist, CDist, Kthvalue, Median).</item>
///   <item>Priority 3 — auxiliary ops (Trace, DiagEmbed, Cross, Meshgrid,
///     CartesianProd, NanToNum).</item>
///   <item>Priority 4 — rare/scientific ops (Bessel I0/I1/I0e/I1e,
///     Erfinv, Polygamma, Lgamma, Digamma, Xlogy, Xlog1py) where the CPU
///     fallback is typically fast enough.</item>
/// </list>
/// <para><b>Per-backend directory layout</b> (established here; each
/// backend's Parity210/ subfolder holds kernels as they land):</para>
/// <list type="bullet">
///   <item><c>Engines/DirectGpu/CUDA/Parity210/</c> — .cu PTX blobs +
///     P/Invoke wrappers.</item>
///   <item><c>Engines/DirectGpu/HIP/Parity210/</c> — HIP kernels.</item>
///   <item><c>Engines/DirectGpu/Metal/Parity210/</c> — MSL kernels.</item>
///   <item><c>Engines/DirectGpu/Vulkan/Parity210/</c> — GLSL compute
///     shaders.</item>
///   <item><c>Engines/DirectGpu/OpenCL/Parity210/</c> — CL C kernels.</item>
///   <item><c>Engines/DirectGpu/WebGpu/Parity210/</c> — WGSL shaders.</item>
/// </list>
/// <para><b>Skip-with-reason convention</b>: when a backend genuinely
/// can't express an op, the GPU-parity test that compares it to the CPU
/// reference includes <c>[Fact(Skip="reason")]</c> with a sentence
/// explaining why (e.g. "WebGPU lacks native popcount" for bit-parity
/// ops on int1 masks).</para>
/// </remarks>
internal static class Parity210KernelRollout
{
    // This class intentionally has no members — it's a documentation
    // anchor linking source-code navigation to the #210 item #3 rollout
    // plan.
}
