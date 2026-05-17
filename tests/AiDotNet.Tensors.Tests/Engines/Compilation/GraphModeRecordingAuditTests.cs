using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Audit-driven regression tests for AiDotNet#1328. The root-cause fix
/// for TensorEmbeddingLookup uncovered six other CpuEngine ops with the
/// same bug pattern: they recorded themselves to the eager
/// <see cref="Autodiff.GradientTape{T}"/> but never to
/// <see cref="GraphMode"/>'s lazy-graph tracer, so any
/// <see cref="CompiledTrainingPlan{T}"/> that included them produced an
/// empty / partial backward graph and silently dropped gradients on the
/// fused-compiled training path.
///
/// <para>Each test below builds a minimal forward that includes the
/// suspect op under <see cref="GraphMode.Enable"/> and asserts the
/// returned tensor has a non-null <see cref="Tensor{T}.LazySource"/> —
/// the minimum invariant required for the compiler to see and back-
/// propagate through the op. The corresponding eager-tape tests live
/// in <c>tests/.../Autodiff/</c> and are unaffected by this PR.</para>
///
/// <para>Audit scope (each op had no GraphMode.IsActive branch in
/// CpuEngine.cs pre-fix):</para>
/// <list type="number">
/// <item><b>TensorEmbeddingLookup</b> — covered separately in
///   <see cref="TensorEmbeddingLookupCompiledPlanTests"/>.</item>
/// <item><b>FusedConv2D</b></item>
/// <item><b>FusedConv3D</b></item>
/// <item><b>FusedConvTranspose2D</b></item>
/// <item><b>FusedBatchNorm</b></item>
/// <item><b>DeformableConv2D</b> (DCN v1, mask null)</item>
/// <item><b>GraphAttention</b></item>
/// <item><b>ReduceVariance</b></item>
/// </list>
/// </summary>
public class GraphModeRecordingAuditTests
{
    /// <summary>
    /// FusedConv2D = Conv2D + BroadcastAdd + Activation. Under GraphMode
    /// the function decomposes into the recorded sequence (each
    /// component op records itself), so the returned tensor MUST have a
    /// LazySource. Pre-fix the fused fast-path bypassed recording and
    /// the result had no LazySource.
    /// </summary>
    [Fact]
    public void FusedConv2D_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([1, 3, 4, 4]);
        var kernel = Tensor<float>.CreateRandom([2, 3, 3, 3]);
        var bias = Tensor<float>.CreateRandom([2]);

        using var scope = GraphMode.Enable();
        var output = engine.FusedConv2D<float>(
            input, kernel, bias,
            strideH: 1, strideW: 1,
            padH: 1, padW: 1,
            dilationH: 1, dilationW: 1,
            activation: FusedActivationType.ReLU);

        Assert.NotNull(output.LazySource);
    }

    /// <summary>
    /// FusedConvTranspose2D — same pattern as FusedConv2D but transposed.
    /// </summary>
    [Fact]
    public void FusedConvTranspose2D_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([1, 2, 4, 4]);
        var kernel = Tensor<float>.CreateRandom([2, 3, 3, 3]);
        var bias = Tensor<float>.CreateRandom([3]);

        using var scope = GraphMode.Enable();
        var output = engine.FusedConvTranspose2D<float>(
            input, kernel, bias,
            strideH: 1, strideW: 1,
            padH: 0, padW: 0,
            outputPadH: 0, outputPadW: 0,
            activation: FusedActivationType.None);

        Assert.NotNull(output.LazySource);
    }

    /// <summary>
    /// FusedBatchNorm decomposes into BatchNorm + ApplyActivationRecorded.
    /// </summary>
    [Fact]
    public void FusedBatchNorm_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        const int N = 2, C = 3, H = 4, W = 4;
        var input = Tensor<float>.CreateRandom([N, C, H, W]);
        var gamma = Tensor<float>.CreateRandom([C]);
        var beta = Tensor<float>.CreateRandom([C]);
        var runningMean = new Tensor<float>([C]);
        var runningVar = new Tensor<float>([C]);
        var rvSpan = runningVar.AsWritableSpan();
        for (int i = 0; i < rvSpan.Length; i++) rvSpan[i] = 1f;

        using var scope = GraphMode.Enable();
        var output = engine.FusedBatchNorm<float>(
            input, gamma, beta, runningMean, runningVar,
            epsilon: 1e-5, momentum: 0.1, training: true,
            activation: FusedActivationType.ReLU,
            out _, out _);

        Assert.NotNull(output.LazySource);
    }

    /// <summary>
    /// DeformableConv2D (DCN v1 — mask null).
    /// </summary>
    [Fact]
    public void DeformableConv2D_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        const int N = 1, Cin = 2, Cout = 2, H = 4, W = 4, K = 3;
        var input = Tensor<float>.CreateRandom([N, Cin, H, W]);
        var kernel = Tensor<float>.CreateRandom([Cout, Cin, K, K]);
        // Offset has shape [N, 2*K*K, Hout, Wout] for DCN. With stride=1, pad=1,
        // dilation=1 the output spatial dims equal input's.
        var offset = Tensor<float>.CreateRandom([N, 2 * K * K, H, W]);

        using var scope = GraphMode.Enable();
        var output = engine.DeformableConv2D<float>(
            input, kernel, offset, mask: null,
            new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

        Assert.NotNull(output.LazySource);
    }

    /// <summary>
    /// GraphAttention — 3 trainable inputs, complex saved state.
    /// </summary>
    [Fact]
    public void GraphAttention_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        const int batch = 1, nodes = 4, features = 3, edges = 4;
        var nodeFeatures = Tensor<float>.CreateRandom([batch, nodes, features]);
        var src = new Tensor<int>([edges]);
        var tgt = new Tensor<int>([edges]);
        src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3;
        tgt[0] = 1; tgt[1] = 2; tgt[2] = 3; tgt[3] = 0;
        var attnSrc = Tensor<float>.CreateRandom([features]);
        var attnTgt = Tensor<float>.CreateRandom([features]);

        using var scope = GraphMode.Enable();
        var output = engine.GraphAttention<float>(
            nodeFeatures, src, tgt, attnSrc, attnTgt,
            leakyReluAlpha: 0.2,
            out _);

        Assert.NotNull(output.LazySource);
    }

    /// <summary>
    /// ReduceVariance — single-input op with a separately-allocated
    /// `mean` tensor saved into savedState for backward.
    /// </summary>
    [Fact]
    public void ReduceVariance_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 4, 8]);

        using var scope = GraphMode.Enable();
        var output = engine.ReduceVariance<float>(input, new[] { 2 }, keepDims: false);

        Assert.NotNull(output.LazySource);
    }

    /// <summary>
    /// FlashAttention — Q/K/V multi-head, out-param softmaxStats saved
    /// into savedState for backward. Same pattern as LayerNorm
    /// (mean/variance recomputed on replay). Pre-fix the eager fast
    /// path produced a materialised tensor with no LazySource, and the
    /// downstream Permute/Reshape ops in the calling FlashAttentionLayer
    /// captured the FA output as a frozen compile-time constant —
    /// breaking gradient flow to the upstream Q/K/V projections in the
    /// fused compiled training path (HarmonicEngine PathB reproducer
    /// at dModel=128 / L=2 / ctx=64 / 10KB WikiText-2 byte-LM:
    /// top-1 0% / top-5 100% / ppl=V uniform + 3.76× slowdown vs MHA).
    /// </summary>
    [Fact]
    public void FlashAttention_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        const int batch = 1, heads = 2, seqQ = 4, headDim = 8;
        var query = Tensor<float>.CreateRandom([batch, heads, seqQ, headDim]);
        var key   = Tensor<float>.CreateRandom([batch, heads, seqQ, headDim]);
        var value = Tensor<float>.CreateRandom([batch, heads, seqQ, headDim]);

        using var scope = GraphMode.Enable();
        var output = engine.FlashAttention<float>(
            query, key, value,
            scale: null,
            isCausal: false,
            out _);

        Assert.NotNull(output.LazySource);
    }
}
