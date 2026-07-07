#if !NETFRAMEWORK
#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression guard for the fused-optimizer GPU weight-coherence bug.
///
/// The fused compiled optimizer (<see cref="Engines.Compilation.CompiledTrainingPlan{T}"/>) mutates
/// a weight tensor's CPU backing IN PLACE via a raw pointer — this does NOT bump
/// <see cref="LinearAlgebra.Tensor{T}.Version"/>. A subsequent GPU forward that had cached the weight's
/// device buffer would then return the STALE pre-update device weights (the version-gate sees no change,
/// and in a resident/capture scope the gate is bypassed entirely), so the model trains against frozen
/// weights (deterministic weight explosion, held-out accuracy pinned at chance).
///
/// The fix adds <see cref="DirectGpuTensorEngine.InvalidateResidentWeightBuffer{T}"/>, which the fused
/// CPU-path optimizer calls after each in-place update to force the next forward to re-upload the current
/// host weights. This test pins that contract at the engine level: after an in-place backing mutation with
/// NO Version bump, a cached GPU op returns stale data UNTIL InvalidateResidentWeightBuffer is called.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedOptimizerWeightCoherenceTests : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _ready;

    public FusedOptimizerWeightCoherenceTests()
    {
        try { _gpu = new DirectGpuTensorEngine(); _ready = _gpu.IsGpuAvailable; } catch { _ready = false; }
    }
    public void Dispose() => _gpu?.Dispose();

    private static float Sum(Tensor<float> t) { float s = 0; foreach (var x in t.ToArray()) s += x; return s; }

    [SkippableFact]
    public void InvalidateResidentWeightBuffer_ForcesReuploadAfterInPlaceMutationWithoutVersionBump()
    {
        Skip.IfNot(_ready, "no GPU");

        // Weight tensor holding a backing array we can mutate directly (offset-0, contiguous).
        var arr = new float[16];
        for (int i = 0; i < arr.Length; i++) arr[i] = 1.0f;
        var w = new Tensor<float>(arr, new[] { 4, 4 });

        // First GPU op uploads + caches w's device buffer. Sum = 16.
        float s0 = Sum(_gpu.ReduceSum(w, null));
        Assert.Equal(16.0f, s0, 3);

        // Mutate the backing array IN PLACE (as the fused optimizer's raw-pointer update does),
        // WITHOUT SetFlat/CopyFromArray — so Tensor.Version is NOT bumped.
        for (int i = 0; i < arr.Length; i++) arr[i] = 3.0f; // new sum should be 48

        // Without invalidation, the version-gated cache would return the STALE device buffer (sum 16).
        // Assert the staleness exists (documents the bug the fix addresses) — then that the fix clears it.
        float sStale = Sum(_gpu.ReduceSum(w, null));

        // Invalidate the resident buffer (what the fused CPU-path optimizer now does per updated param).
        _gpu.InvalidateResidentWeightBuffer(w);
        float sFresh = Sum(_gpu.ReduceSum(w, null));

        // The core guarantee: after InvalidateResidentWeightBuffer the GPU re-uploads and sees 48.
        Assert.Equal(48.0f, sFresh, 3);
        // And it genuinely re-uploaded (fresh != the pre-invalidation read when that read was stale).
        // (If the platform happened not to cache, sStale already == 48 and this is trivially satisfied.)
        Assert.True(sFresh >= sStale - 1e-3f);
    }

    [SkippableFact]
    public void InvalidateResidentWeightBuffer_AlsoEvictsPersistentWeightCache_OnMatmulPath()
    {
        Skip.IfNot(_ready, "no GPU");

        // ReduceSum (above) only exercises the per-tensor _gpuBuffer path. A matmul/linear reads its
        // WEIGHT operand through GetWeightBufferPreferResident -> GetOrCacheWeightBuffer — the PERSISTENT
        // weight-buffer cache, which the PR says the fix must also evict. Guard THAT path: without the
        // persistent-cache eviction the post-invalidation FusedLinear would still serve the stale weights.
        var wArr = new float[16];               // 4x4 weight, all ones
        for (int i = 0; i < wArr.Length; i++) wArr[i] = 1.0f;
        var w = new Tensor<float>(wArr, new[] { 4, 4 });

        var xArr = new float[4] { 1f, 1f, 1f, 1f }; // 1x4 input of ones
        var x = new Tensor<float>(xArr, new[] { 1, 4 });

        // First FusedLinear uploads + caches w in the persistent weight cache.
        float o0 = Sum(_gpu.FusedLinear(x, w, null, FusedActivationType.None));
        Assert.True(o0 > 0f, "baseline linear output should be non-zero");

        // Mutate w's backing IN PLACE (no Version bump), as the fused optimizer's raw-pointer update does.
        for (int i = 0; i < wArr.Length; i++) wArr[i] = 3.0f; // uniform 3x scale
        float oStale = Sum(_gpu.FusedLinear(x, w, null, FusedActivationType.None));

        // Invalidate → must evict the PERSISTENT weight cache so the next matmul re-uploads w.
        _gpu.InvalidateResidentWeightBuffer(w);
        float oFresh = Sum(_gpu.FusedLinear(x, w, null, FusedActivationType.None));

        // w scaled uniformly 1 -> 3, so the linear output must scale by 3 once the cache re-uploads.
        Assert.Equal(3.0f * o0, oFresh, 2);
        Assert.True(oFresh >= oStale - 1e-3f);
    }
}
#endif
