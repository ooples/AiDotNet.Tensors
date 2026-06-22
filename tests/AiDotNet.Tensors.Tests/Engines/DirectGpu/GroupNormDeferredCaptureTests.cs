using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On-device (#642 P3) regression coverage for the GPU diffusion ResBlock through the deferred
/// execution graph and CUDA-graph capture. These run only on a real CUDA device (SkippableFact)
/// since CI has no GPU; they lock in three fixes that the mock-only graph tests could not catch:
/// <list type="number">
/// <item>The GroupNorm argument-order bug (numGroups/channels/spatial scrambled) — exercised at
/// channels != numGroups (64 channels, 32 groups), the case that overflowed the save buffers.</item>
/// <item>Deferred-graph correctness for a full ResBlock (GroupNorm-recording + GPU-resident in-place
/// activations) — deferred output must match eager exactly.</item>
/// <item>CUDA-graph capture+replay of the resident graph through the production
/// <see cref="GraphedInferenceStep"/> API replays the captured compute kernels correctly.</item>
/// </list>
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GroupNormDeferredCaptureTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _gpuAvailable;
    private const float Tolerance = 1e-3f;

    // Flip to true when the deferred-graph buffer-management bug is fixed. Root-caused on a
    // GTX 1660 Ti: in a DEEP deferred graph, a tensor produced early and held across several
    // intervening ops, then consumed by TensorConcatenate, reads STALE data — its pooled GPU
    // buffer is recycled for an intervening op before the concat consumer runs (the UNet
    // encoder→decoder skip-connection pattern; the cause of the #1650 end-to-end DDPM divergence).
    // Every individual op + short chain + an 8-deep chain WITHOUT a long-lived skip are correct;
    // only the long-lived-skip-into-concat case fails. NOTE: MemoryPlanningPass.BufferReuseMap is
    // dead code (computed, never consumed), so the bad reuse is in the recording/release-deferral
    // path, not that pass.
    private const bool DeferredLongLivedSkipFixed = true;
    private const int C = 64;       // channels != numGroups (the GroupNorm arg-order overflow case)
    private const int Groups = 32;
    private const int Sp = 16;

    public GroupNormDeferredCaptureTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuAvailable = _gpu.IsGpuAvailable;
        }
        catch (PlatformNotSupportedException) { _gpuAvailable = false; }
        catch (DllNotFoundException) { _gpuAvailable = false; }
    }

    public void Dispose() => (_gpu as IDisposable)?.Dispose();

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // SD-style ResBlock: GroupNorm -> Swish -> Conv -> GroupNorm -> Swish -> Conv -> residual add.
    private Tensor<float> ResBlock(Tensor<float> input, Tensor<float> gamma, Tensor<float> beta,
        Tensor<float> k1, Tensor<float> k2)
    {
        var gpu = _gpu!;
        var h = gpu.GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _);
        gpu.SwishInPlace(h);
        h = gpu.Conv2D(h, k1, 1, 1, 1);
        h = gpu.GroupNorm(h, Groups, gamma, beta, 1e-5, out _, out _);
        gpu.SwishInPlace(h);
        h = gpu.Conv2D(h, k2, 1, 1, 1);
        return gpu.TensorAdd(input, h);
    }

    // Pure-CPU unit test (no GPU) for the GpuBufferReleaseDeferral depth-counting fix (#642 review,
    // Critical). Runs in CI and locks in the nesting contract: an inner scope's EndAndRelease must
    // NOT flush the outer scope's still-pending releases (the use-after-free the depth counter fixes).
    [Fact]
    public void DeferralGate_NestedScopes_FlushOnlyAtOutermostEnd()
    {
        // (Serialized with the GPU tests via the collection, so no concurrent DeferredScope pollutes
        // the thread-local depth on this thread.)
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        var flushed = new List<string>();

        GpuBufferReleaseDeferral.Begin();                                   // outer
        Assert.True(GpuBufferReleaseDeferral.IsActive);
        Assert.True(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("outer")));

        GpuBufferReleaseDeferral.Begin();                                   // inner
        Assert.True(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("inner")));

        GpuBufferReleaseDeferral.EndAndRelease();                           // inner ends — must NOT flush
        Assert.True(GpuBufferReleaseDeferral.IsActive);
        Assert.Empty(flushed);

        GpuBufferReleaseDeferral.EndAndRelease();                           // outer ends — flush both
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        Assert.Equal(new[] { "outer", "inner" }, flushed);

        // Balanced + idempotent: an extra EndAndRelease is a no-op, and TryDefer returns false.
        GpuBufferReleaseDeferral.EndAndRelease();
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        Assert.False(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("after")));
        Assert.Equal(2, flushed.Count);
    }

    [SkippableFact]
    public void GroupNorm_ChannelsNeNumGroups_MatchesCpu()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var input = Rand(new[] { 1, C, Sp, Sp }, 1);
        var gamma = Rand(new[] { C }, 2);
        var beta = Rand(new[] { C }, 3);

        var gpuOut = gpu.GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _);
        var cpuOut = new CpuEngine().GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _);

        float maxDiff = 0;
        for (int i = 0; i < gpuOut.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(gpuOut[i] - cpuOut[i]));
        Assert.True(maxDiff < Tolerance, $"GPU GroupNorm (channels={C}, groups={Groups}) diverged from CPU: maxDiff={maxDiff}");
    }

    [SkippableFact]
    public void DeferredResBlock_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var input = Rand(new[] { 1, C, Sp, Sp }, 1);
        var gamma = Rand(new[] { C }, 2);
        var beta = Rand(new[] { C }, 3);
        var k1 = Rand(new[] { C, C, 3, 3 }, 4);
        var k2 = Rand(new[] { C, C, 3, 3 }, 5);

        var eager = ResBlock(input, gamma, beta, k1, k2);
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        Tensor<float> deferred;
        using (var scope = gpu.BeginDeferredScope())
        {
            Skip.If(scope is null, "Deferred execution unsupported.");
            deferred = ResBlock(input, gamma, beta, k1, k2);
            scope!.Execute();
        }

        // Shape is part of the contract — Math.Min would mask a wrong-shape regression.
        Assert.Equal(eagerCopy.Length, deferred.Length);
        float maxDiff = 0;
        for (int i = 0; i < eagerCopy.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - deferred[i]));
        Assert.True(maxDiff < Tolerance, $"Deferred ResBlock diverged from eager: maxDiff={maxDiff}");
    }

    // Pure-CPU async test (no GPU → runs in CI) for the AsyncLocal deferral gate (#642 review,
    // Major): Begin on this thread, hop threads via an await, then TryDefer/EndAndRelease on the
    // continuation thread must see the SAME flowed state. With the old [ThreadStatic] storage the
    // continuation thread saw an empty gate → the originating thread's gate leaked and its queued
    // releases never ran. (NOTE: full async GPU *execution* via DeferredScope.ExecuteAsync is a
    // separate, pre-existing limitation — GPU ops need the CUDA context current on the continuation
    // thread; the production deferred path uses synchronous Execute. This test isolates the gate.)
    [Fact]
    public async System.Threading.Tasks.Task DeferralGate_FlowsAcrossAwaitAndThreadHop()
    {
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        var flushed = new List<string>();
        int beginThread = System.Environment.CurrentManagedThreadId;

        GpuBufferReleaseDeferral.Begin();
        try
        {
            Assert.True(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("a")));

            // LongRunning + the default scheduler forces a DEDICATED new thread (deterministic hop),
            // and the ExecutionContext capture flows the AsyncLocal gate to it.
            await System.Threading.Tasks.Task.Factory.StartNew(() =>
            {
                Assert.NotEqual(beginThread, System.Environment.CurrentManagedThreadId);
                Assert.True(GpuBufferReleaseDeferral.IsActive);
                Assert.True(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("b")));
            },
            System.Threading.CancellationToken.None,
            System.Threading.Tasks.TaskCreationOptions.LongRunning,
            System.Threading.Tasks.TaskScheduler.Default).ConfigureAwait(false);
        }
        finally
        {
            GpuBufferReleaseDeferral.EndAndRelease();   // outermost end → flush both, regardless of thread
        }
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        Assert.Equal(new[] { "a", "b" }, flushed);
    }

    [SkippableFact]
    public void ChannelConcat_MatchesCpu_EagerAndDeferred()
    {
        // The UNet decoder skip-connection: TensorConcatenate(new[]{a,b}, axis:1) on NCHW.
        // Channel-axis concat was the #642 gap — the old GPU path only handled the last axis, so
        // this fell to CPU (breaking the device-resident chain). Validate GPU eager + deferred == CPU.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var a = Rand(new[] { 1, C, Sp, Sp }, 7);
        var b = Rand(new[] { 1, C / 2, Sp, Sp }, 8);

        var cpu = new CpuEngine().TensorConcatenate(new[] { a, b }, axis: 1);
        Assert.Equal(C + C / 2, cpu.Shape[1]);

        var eager = gpu.TensorConcatenate(new[] { a, b }, axis: 1);
        float eagerDiff = 0;
        for (int i = 0; i < cpu.Length; i++) eagerDiff = Math.Max(eagerDiff, Math.Abs(cpu[i] - eager[i]));
        Assert.True(eagerDiff < Tolerance, $"Eager GPU channel-concat diverged from CPU: maxDiff={eagerDiff}");

        Tensor<float> deferred;
        using (var scope = gpu.BeginDeferredScope())
        {
            Skip.If(scope is null, "Deferred execution unsupported.");
            deferred = gpu.TensorConcatenate(new[] { a, b }, axis: 1);
            scope!.Execute();
        }
        float defDiff = 0;
        for (int i = 0; i < cpu.Length; i++) defDiff = Math.Max(defDiff, Math.Abs(cpu[i] - deferred[i]));
        Assert.True(defDiff < Tolerance, $"Deferred GPU channel-concat diverged from CPU: maxDiff={defDiff}");
    }

    // Runs `op` eagerly, then inside a deferred scope; asserts the deferred result matches eager.
    // Proves the op's backend kernel is RECORDED (replays in order) rather than running eagerly
    // mid-record on not-yet-computed buffers (#642 structural-op coverage).
    private void AssertDeferredMatchesEager(string name, Func<Tensor<float>> op)
    {
        var eager = op();
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        Tensor<float> deferred;
        using (var scope = _gpu!.BeginDeferredScope())
        {
            Skip.If(scope is null, "Deferred execution unsupported.");
            deferred = op();
            scope!.Execute();
        }
        // Shape is part of the contract — Math.Min would mask a wrong-shape regression.
        Assert.Equal(eagerCopy.Length, deferred.Length);
        float maxDiff = 0;
        for (int i = 0; i < eagerCopy.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - deferred[i]));
        Assert.True(maxDiff < Tolerance, $"Deferred {name} diverged from eager: maxDiff={maxDiff}");
    }

    [SkippableFact]
    public void DeferredUpsample_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 11);
        AssertDeferredMatchesEager("Upsample", () => _gpu!.Upsample(input, 2, 2));
    }

    [SkippableFact]
    public void DeferredConvWithBias_MatchesEager()
    {
        // The conv+bias path previously downloaded the conv output, added bias in a CPU loop, and
        // re-uploaded — broken under a DeferredScope. Now uses the recorded Conv2DBiasAdd kernel.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 21);
        var kernel = Rand(new[] { C, C, 3, 3 }, 22);
        var bias = Rand(new[] { C }, 23);
        AssertDeferredMatchesEager("FusedConv2D+bias",
            () => _gpu!.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.None));
    }

    [SkippableFact]
    public void DeferredTensorTranspose_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var m = Rand(new[] { 64, 96 }, 24);   // 2-D → backend.Transpose
        AssertDeferredMatchesEager("TensorTranspose", () => _gpu!.TensorTranspose(m));
    }

    [SkippableFact]
    public void DeferredAdaptiveAvgPool2D_MatchesEager()
    {
        // 1x1 adaptive avg-pool → GlobalAvgPool2D (SE blocks / attention pooling).
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 25);
        AssertDeferredMatchesEager("AdaptiveAvgPool2D", () => _gpu!.AdaptiveAvgPool2D(input, 1, 1));
    }

    [SkippableFact]
    public void DeferredEmbedding_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var indices = new Tensor<int>(new[] { 4 });
        for (int i = 0; i < 4; i++) indices[i] = i % 3;
        var table = Rand(new[] { 3, 16 }, 26);
        AssertDeferredMatchesEager("Embedding", () => _gpu!.Embedding(indices, table));
    }

    [SkippableFact]
    public void DeferredMaxPool2D_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 12);
        AssertDeferredMatchesEager("MaxPool2D", () => _gpu!.MaxPool2D(input, 2, 2, 0));
    }

    [SkippableFact]
    public void DeferredAvgPool2D_MatchesEager()
    {
        // Regression for the CudaBackend.AvgPool2D launch bug (#642): the kernel takes 15 params
        // (… countIncludePad) but the launcher allocated 14 arg slots, so cuLaunchKernel read a
        // garbage 15th pointer → 0xC0000005 on small inputs (surfaced under deferred replay).
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 13);
        AssertDeferredMatchesEager("AvgPool2D", () => _gpu!.AvgPool2D(input, 2, 2, 0));
    }

    [SkippableFact]
    public void DeferredConvTranspose2D_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 14);
        var kernel = Rand(new[] { C, C, 2, 2 }, 15);   // [inC, outC, kH, kW]
        AssertDeferredMatchesEager("ConvTranspose2D",
            () => _gpu!.ConvTranspose2D(input, kernel, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 0, 0 }));
    }

    [SkippableFact]
    public void DeferredDeepResBlockChain_MatchesEager()
    {
        // The isolated op + single-ResBlock tests are SHORT graphs (1-6 ops); the real DDPM UNet
        // forward is a DEEP graph (~100+ ops). A deferred-graph buffer-reuse / aliasing bug can
        // manifest only at depth (a recycled scratch buffer overwritten by a later node before its
        // consumer runs). This chains 8 ResBlocks + a channel-concat skip + a merge conv (~55 ops)
        // to stress that path — the end-to-end DDPM deferred denoise diverged ~100% (#1650), so a
        // deep-graph regression is the leading remaining suspect once every isolated op passes.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        Skip.IfNot(DeferredLongLivedSkipFixed,
            "Known deferred-graph buffer bug: long-lived skip + concat in a deep graph reads stale data (#1650 root cause).");
        var input = Rand(new[] { 1, C, Sp, Sp }, 61);
        var gamma = Rand(new[] { C }, 62);
        var beta = Rand(new[] { C }, 63);
        var k1 = Rand(new[] { C, C, 3, 3 }, 64);
        var k2 = Rand(new[] { C, C, 3, 3 }, 65);
        var kMerge = Rand(new[] { C, 2 * C, 3, 3 }, 66);  // merges the concatenated skip back to C

        Func<Tensor<float>> deep = () =>
        {
            var gpu = _gpu!;
            var h = input;
            var skip = h;
            for (int i = 0; i < 8; i++) h = ResBlock(h, gamma, beta, k1, k2);
            var cat = gpu.TensorConcatenate(new[] { h, skip }, axis: 1);   // [1, 2C, Sp, Sp]
            var merged = gpu.Conv2D(cat, kMerge, 1, 1, 1);                 // [1, C, Sp, Sp]
            return ResBlock(merged, gamma, beta, k1, k2);
        };
        AssertDeferredMatchesEager("DeepResBlockChain(8x + concat-skip)", deep);
    }

    [SkippableFact]
    public void DeferredDeepResBlockChain_NoSkip_MatchesEager()
    {
        // Disambiguates the DeepResBlockChain failure: same 8-deep ResBlock chain but WITHOUT the
        // long-lived concat-skip. Each ResBlock's own residual is short-lived (6 ops). If THIS
        // passes while the concat-skip variant fails, the bug is specifically a long-lived tensor
        // (held across the deep span) whose buffer the deferred graph recycles before its consumer
        // runs — not depth/accumulation per se.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 71);
        var gamma = Rand(new[] { C }, 72);
        var beta = Rand(new[] { C }, 73);
        var k1 = Rand(new[] { C, C, 3, 3 }, 74);
        var k2 = Rand(new[] { C, C, 3, 3 }, 75);
        AssertDeferredMatchesEager("DeepResBlockChain(8x, no skip)", () =>
        {
            var h = input;
            for (int i = 0; i < 8; i++) h = ResBlock(h, gamma, beta, k1, k2);
            return h;
        });
    }

    [SkippableFact]
    public void DeferredLongLivedSkipConcat_MatchesEager()
    {
        // Minimal repro of the suspected root cause: a tensor produced early, held across several
        // intervening ops, then consumed by a concat. If the deferred graph recycles `skip`'s buffer
        // for one of the intervening ResBlocks' scratch, the concat reads stale data. Smaller/faster
        // than the 8-deep chain — a clean bug report repro if it fails.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        Skip.IfNot(DeferredLongLivedSkipFixed,
            "Known deferred-graph buffer bug: long-lived skip + concat reads stale data (#1650 root cause; minimal repro).");
        var input = Rand(new[] { 1, C, Sp, Sp }, 81);
        var gamma = Rand(new[] { C }, 82);
        var beta = Rand(new[] { C }, 83);
        var k1 = Rand(new[] { C, C, 3, 3 }, 84);
        var k2 = Rand(new[] { C, C, 3, 3 }, 85);
        var kMerge = Rand(new[] { C, 2 * C, 3, 3 }, 86);
        AssertDeferredMatchesEager("LongLivedSkipConcat(3x intervening)", () =>
        {
            var gpu = _gpu!;
            var skip = gpu.GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _); // produced early
            var h = skip;
            for (int i = 0; i < 3; i++) h = ResBlock(h, gamma, beta, k1, k2);          // intervening ops
            var cat = gpu.TensorConcatenate(new[] { h, skip }, axis: 1);               // consumes the old skip
            return gpu.Conv2D(cat, kMerge, 1, 1, 1);
        });
    }

    [SkippableFact]
    public void DeferredSoftmax_MatchesEager()
    {
        // The UNet self-attention softmax over scores [heads, S, S]. Not covered by the
        // structural/ResBlock tests above.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var scores = Rand(new[] { 4, 64, 64 }, 31);
        AssertDeferredMatchesEager("Softmax(-1)", () => _gpu!.Softmax(scores, -1));
    }

    [SkippableFact]
    public void DeferredSelfAttention_MatchesEager()
    {
        // The DDPM UNet self-attention at 16x16: scores = Q·Kᵀ, softmax(-1), ctx = scores·V.
        // This batched-matmul → softmax → batched-matmul chain is NOT exercised by the existing
        // deferred coverage (conv/groupnorm/pool/concat/upsample/transpose/embedding), and is the
        // prime suspect for the end-to-end deferred-graph denoise divergence observed for #1650.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        int heads = 4, s = 64, dh = 16;
        var q = Rand(new[] { heads, s, dh }, 41);
        var kT = Rand(new[] { heads, dh, s }, 42);
        var v = Rand(new[] { heads, s, dh }, 43);
        AssertDeferredMatchesEager("SelfAttention(QKᵀ·softmax·V)", () =>
        {
            var scores = _gpu!.BatchMatMul(q, kT);   // [heads, s, s]
            scores = _gpu!.Softmax(scores, -1);
            return _gpu!.BatchMatMul(scores, v);     // [heads, s, dh]
        });
    }

    [SkippableFact]
    public void DeferredBatchMatMulChain_MatchesEager()
    {
        // Isolates whether a *chained* BatchMatMul (output of one feeds the next) is deferred-correct
        // — separating a matmul-recording bug from a softmax-recording bug in the attention chain.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        int heads = 4, s = 64, dh = 16;
        var q = Rand(new[] { heads, s, dh }, 51);
        var kT = Rand(new[] { heads, dh, s }, 52);
        var v = Rand(new[] { heads, s, dh }, 53);
        AssertDeferredMatchesEager("BatchMatMulChain", () =>
        {
            var scores = _gpu!.BatchMatMul(q, kT);   // [heads, s, s]
            return _gpu!.BatchMatMul(scores, v);     // [heads, s, dh] (no softmax in between)
        });
    }

    [SkippableFact]
    public void CudaGraphCapture_ReplaysResBlockCorrectly()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var backend = gpu.GetBackend();
        Skip.If(backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Capture path is CUDA-only.");
        var cudaBackend = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)backend!;

        var input = Rand(new[] { 1, C, Sp, Sp }, 1);
        var gamma = Rand(new[] { C }, 2);
        var beta = Rand(new[] { C }, 3);
        var k1 = Rand(new[] { C, C, 3, 3 }, 4);
        var k2 = Rand(new[] { C, C, 3, 3 }, 5);

        var eager = ResBlock(input, gamma, beta, k1, k2);
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        // Record the resident graph (buffers allocated at record time → Execute is alloc-free).
        var scope = gpu.BeginDeferredScope() as DeferredScope;
        Skip.If(scope is null, "Deferred execution unsupported.");
        ExecutionGraph? graph = null;
        try
        {
            Tensor<float> result = ResBlock(input, gamma, beta, k1, k2);
            graph = scope!.Compile();   // owns the stream pool (GraphCompiler transferred ownership)

            // Warmup: full graph once (H2D populates resident buffers + JIT).
            foreach (var node in graph.TopologicalOrder) node.Execute(backend);
            backend.Synchronize();

            using var step = new GraphedInferenceStep(
                backend, cudaBackend.DefaultStream.Handle,
                () => graph.ExecuteComputeKernelsNoSync(backend),
                new GraphedInferenceStepOptions { ThrowOnUnsupported = false });
            step.Prepare();
            step.Capture();
            Skip.IfNot(step.HasGraph, "CUDA graph capture unsupported on this driver.");

            step.Replay();   // recomputes into the captured output buffer; `result` downloads it

            Assert.Equal(eagerCopy.Length, result.Length);
            float maxDiff = 0;
            for (int i = 0; i < eagerCopy.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - result[i]));
            Assert.True(maxDiff < Tolerance, $"CUDA-graph replay diverged from eager: maxDiff={maxDiff}");
        }
        finally
        {
            // Dispose the scope first (its Dispose may touch the compiled graph), then the graph —
            // which disposes the stream pool the GraphCompiler transferred to it (no leak).
            scope?.Dispose();
            graph?.Dispose();
        }
    }
}
