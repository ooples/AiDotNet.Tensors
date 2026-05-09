// Copyright (c) AiDotNet. All rights reserved.
// Issue #318 — diagnostic + integration tests for the Clone-after-
// train forward-output drift on multi-layer models.
//
// The issue's empirical signature:
//   * Two tensors with byte-identical LOGICAL contents
//   * One created via Engine.TensorSubtract (training UpdateParameters)
//   * The other created via SetFlat into a fresh allocation (Clone via
//     SetParameters)
//   * Subsequent Engine.TensorMatMul / FusedLinear / TensorTranspose
//     produces DIFFERENT output across the two paths
//
// PR #315 zeroed the padding region in TensorAllocator.Rent, but the
// drift persists at 3.5%. The user lists three plausible root causes;
// these tests narrow it down by direct byte-level comparison.

using System;
using System.Buffers;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue318CloneDriftDiagnosticTests
{
    private readonly ITestOutputHelper _output;
    public Issue318CloneDriftDiagnosticTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Builds two tensors with logically-identical contents through
    /// the two paths the issue describes — one via engine-op result
    /// (TensorSubtract), one via SetFlat into a fresh tensor — then
    /// diffs the underlying float[] byte-for-byte. The output of
    /// this test tells us exactly what's different post-#315: either
    /// padding bytes (#315 didn't fully cover) or alignment / stride
    /// metadata (#315 wasn't the right fix at all).
    /// </summary>
    [Fact]
    public void EngineOpWrittenTensor_VsSetFlatWrittenTensor_FullBackingArrayDiff()
    {
        // ArrayPool-sized AND non-power-of-two: 401,408 elements
        // → bucket pads to 524,288 (next power of 2), producing a
        // 122,880-element padding region. Same regime as the
        // consumer DBM's middle-layer weight (500×500 = 250K → 256K).
        const int Rows = 392;
        const int Cols = 1024;
        const int Total = Rows * Cols;  // 401,408 — non-power-of-two

        var engine = new CpuEngine();

        // ── Path A — engine-op result.
        // Build A as TensorSubtract(zero, negA). Same arithmetic as a
        // training step's UpdateParameters: w_new = w - α·grad.
        // Result lives in pool-allocated storage with whatever side-
        // effects the engine kernel produces.
        var rng = new Random(42);
        var values = new float[Total];
        for (int i = 0; i < Total; i++) values[i] = (float)((rng.NextDouble() - 0.5) * 0.1);

        var zero = new Tensor<float>(new[] { Rows, Cols });
        var neg = new Tensor<float>(new[] { Rows, Cols });
        var negSpan = neg.AsWritableSpan();
        for (int i = 0; i < Total; i++) negSpan[i] = -values[i];
        var pathA = engine.TensorSubtract(zero, neg);

        // ── Path B — SetFlat into fresh tensor.
        // Same arithmetic the deserialization path uses: SetParameters
        // calls Tensor.SetFlat(i, value) on a freshly-allocated tensor
        // for each parameter element. No engine ops.
        var pathB = new Tensor<float>(new[] { Rows, Cols });
        var bSpan = pathB.AsWritableSpan();
        for (int i = 0; i < Total; i++) bSpan[i] = values[i];

        // ── Diff.
        var aBacking = pathA.GetDataArray();
        var bBacking = pathB.GetDataArray();
        var aSpan = pathA.AsSpan();
        var bASpan = pathB.AsSpan();

        // Logical content first.
        int logicalDiffs = 0;
        float maxLogicalDiff = 0;
        for (int i = 0; i < Total; i++)
        {
            float d = MathF.Abs(aSpan[i] - bASpan[i]);
            if (d > 0) { logicalDiffs++; if (d > maxLogicalDiff) maxLogicalDiff = d; }
        }

        // Padding region (if any).
        int aPadStart = pathA.Length;
        int bPadStart = pathB.Length;
        int aPadLen = aBacking.Length - aPadStart;
        int bPadLen = bBacking.Length - bPadStart;

        int aNonzeroPaddingBytes = 0;
        for (int i = aPadStart; i < aBacking.Length; i++)
            if (aBacking[i] != 0f) aNonzeroPaddingBytes++;
        int bNonzeroPaddingBytes = 0;
        for (int i = bPadStart; i < bBacking.Length; i++)
            if (bBacking[i] != 0f) bNonzeroPaddingBytes++;

        _output.WriteLine($"Path A (engine-op result): backing.Length={aBacking.Length}, "
            + $"logical={pathA.Length}, padding={aPadLen}, non-zero-padding-bytes={aNonzeroPaddingBytes}");
        _output.WriteLine($"Path B (SetFlat):          backing.Length={bBacking.Length}, "
            + $"logical={pathB.Length}, padding={bPadLen}, non-zero-padding-bytes={bNonzeroPaddingBytes}");
        _output.WriteLine($"Logical diffs: {logicalDiffs} elements, max abs diff {maxLogicalDiff}");

        // Logical content is the user's stated property: byte-identical.
        Assert.True(logicalDiffs == 0,
            $"Logical content already diverges at construction time — that's a different bug. "
            + $"Path A and B should have byte-identical logical values; got {logicalDiffs} differing elements.");

        // The actual issue #318 hypothesis: post-#315 padding still
        // carries non-zero garbage on path A but is zero on path B
        // (or some other backing-array discrepancy). If this assert
        // fails, the diagnostic output above tells us where the
        // divergence is.
        Assert.True(aNonzeroPaddingBytes == 0 && bNonzeroPaddingBytes == 0,
            $"Padding region carries non-zero bytes after #315's zero-on-Rent fix. "
            + $"Path A non-zero padding bytes: {aNonzeroPaddingBytes} of {aPadLen}. "
            + $"Path B non-zero padding bytes: {bNonzeroPaddingBytes} of {bPadLen}. "
            + "Engine.TensorSubtract is writing past the logical extent (SIMD overhang) "
            + "and overwriting padding that #315 zeroed at allocation time.");
    }

    /// <summary>
    /// Issue #318 closing contract: <see cref="TensorBase{T}.Canonicalize"/>
    /// must produce backing arrays of EXACTLY <see cref="TensorBase{T}.Length"/>
    /// elements, regardless of how the source tensor was constructed.
    /// Two tensors with byte-equal logical content must have byte-equal
    /// backing arrays after both go through Canonicalize. This is the
    /// fix the issue explicitly requested.
    /// </summary>
    [Fact]
    public void Canonicalize_ProducesByteEqualBackingArrays_AcrossConstructionPaths()
    {
        // Pool-allocation regime to surface the asymmetry between
        // engine-op-produced (bucket-padded) and SetFlat-produced
        // (exact-length) tensors. 401,408 elements → ArrayPool
        // bucket pads to 524,288.
        const int Rows = 392, Cols = 1024;
        const int Total = Rows * Cols;
        var engine = new CpuEngine();

        var rng = new Random(42);
        var values = new float[Total];
        for (int i = 0; i < Total; i++) values[i] = (float)((rng.NextDouble() - 0.5) * 0.1);

        // Path A — engine-op result (lives in pool-allocated storage).
        var zero = new Tensor<float>(new[] { Rows, Cols });
        var neg = new Tensor<float>(new[] { Rows, Cols });
        var negSpan = neg.AsWritableSpan();
        for (int i = 0; i < Total; i++) negSpan[i] = -values[i];
        var pathA = engine.TensorSubtract(zero, neg);

        // Path B — fresh SetFlat-style allocation.
        var pathB = new Tensor<float>(new[] { Rows, Cols });
        var bSpan = pathB.AsWritableSpan();
        for (int i = 0; i < Total; i++) bSpan[i] = values[i];

        var aBacking = pathA.GetDataArray();
        var bBacking = pathB.GetDataArray();
        _output.WriteLine($"Pre-canonicalize: A backing={aBacking.Length}, B backing={bBacking.Length}");

        // Pre-Canonicalize the backing arrays may differ in length
        // (the issue's signature). Sanity check: at this scale Path A
        // typically has a longer backing than Path B.
        Assert.True(aBacking.Length >= Total, "Path A backing must hold at least logical extent.");
        Assert.True(bBacking.Length >= Total, "Path B backing must hold at least logical extent.");

        // Canonicalize both. After this, BOTH backing arrays must be
        // EXACTLY Total elements long with byte-identical contents.
        var canonA = (Tensor<float>)pathA.Canonicalize();
        var canonB = (Tensor<float>)pathB.Canonicalize();

        var caBacking = canonA.GetDataArray();
        var cbBacking = canonB.GetDataArray();
        _output.WriteLine($"Post-canonicalize: A backing={caBacking.Length}, B backing={cbBacking.Length}");

        Assert.True(caBacking.Length == Total,
            $"Canonicalize must produce backing array of EXACTLY logical Length ({Total}); "
            + $"got {caBacking.Length} on path A.");
        Assert.True(cbBacking.Length == Total,
            $"Canonicalize must produce backing array of EXACTLY logical Length ({Total}); "
            + $"got {cbBacking.Length} on path B.");

        // The full backing arrays must now be byte-equal.
        for (int i = 0; i < Total; i++)
        {
            Assert.True(caBacking[i] == cbBacking[i],
                $"Canonicalized backing arrays differ at idx {i}: A={caBacking[i]}, B={cbBacking[i]}.");
        }
    }

    /// <summary>
    /// End-to-end forward-chain test mirroring the consumer's DBM
    /// Clone scenario. Engine-op-produced weights and SetFlat-style
    /// fresh-allocated weights with byte-equal logical content.
    /// Asserts forward outputs are bit-identical (or within fp64
    /// rounding for the double case). Without this passing, #318
    /// isn't fixed regardless of what unit-level diagnostics show.
    /// </summary>
    [Theory]
    [InlineData(typeof(float))]
    [InlineData(typeof(double))]
    public void ThreeLayerForwardChain_TrainedWeights_VsClonedWeights_OutputsMatch(Type elementType)
    {
        if (elementType == typeof(float))
            RunFloatChain();
        else
            RunDoubleChain();
    }

    private void RunFloatChain()
    {
        const int Dim0 = 256, Dim1 = 1024, Dim2 = 256;
        var engine = new CpuEngine();
        var rng = new Random(7);

        var w0Trained = TrainedWeightF(engine, new[] { Dim0, Dim1 }, rng);
        var w1Trained = TrainedWeightF(engine, new[] { Dim1, Dim1 }, rng);
        var w2Trained = TrainedWeightF(engine, new[] { Dim1, Dim2 }, rng);
        var w0Cloned = SetFlatTensorF(w0Trained);
        var w1Cloned = SetFlatTensorF(w1Trained);
        var w2Cloned = SetFlatTensorF(w2Trained);
        AssertLogicalEqualF(w0Trained, w0Cloned, "w0");
        AssertLogicalEqualF(w1Trained, w1Cloned, "w1");
        AssertLogicalEqualF(w2Trained, w2Cloned, "w2");

        var input = new Tensor<float>(new[] { 1, Dim0 });
        var inSpan = input.AsWritableSpan();
        for (int i = 0; i < Dim0; i++) inSpan[i] = (float)((rng.NextDouble() - 0.5));

        var tOut = engine.Sigmoid(engine.TensorMatMul(
            engine.Sigmoid(engine.TensorMatMul(
                engine.Sigmoid(engine.TensorMatMul(input, w0Trained)),
                w1Trained)),
            w2Trained));
        var cOut = engine.Sigmoid(engine.TensorMatMul(
            engine.Sigmoid(engine.TensorMatMul(
                engine.Sigmoid(engine.TensorMatMul(input, w0Cloned)),
                w1Cloned)),
            w2Cloned));

        var tS = tOut.AsSpan();
        var cS = cOut.AsSpan();
        float maxDiff = 0; int worstIdx = -1;
        for (int i = 0; i < tS.Length; i++)
        {
            float d = MathF.Abs(tS[i] - cS[i]);
            if (d > maxDiff) { maxDiff = d; worstIdx = i; }
        }
        _output.WriteLine($"fp32 3-layer forward diff: max={maxDiff:E4} at idx {worstIdx}");
        Assert.True(maxDiff < 1e-5f,
            $"#318 fp32 — 3-layer forward output diverges. Max abs diff {maxDiff:E4} (tolerance 1e-5). "
            + "This is the consumer-side Clone-after-train signature.");
    }

    private void RunDoubleChain()
    {
        const int Dim0 = 256, Dim1 = 1024, Dim2 = 256;
        var engine = new CpuEngine();
        var rng = new Random(7);

        var w0Trained = TrainedWeightD(engine, new[] { Dim0, Dim1 }, rng);
        var w1Trained = TrainedWeightD(engine, new[] { Dim1, Dim1 }, rng);
        var w2Trained = TrainedWeightD(engine, new[] { Dim1, Dim2 }, rng);
        var w0Cloned = SetFlatTensorD(w0Trained);
        var w1Cloned = SetFlatTensorD(w1Trained);
        var w2Cloned = SetFlatTensorD(w2Trained);
        AssertLogicalEqualD(w0Trained, w0Cloned, "w0");
        AssertLogicalEqualD(w1Trained, w1Cloned, "w1");
        AssertLogicalEqualD(w2Trained, w2Cloned, "w2");

        var input = new Tensor<double>(new[] { 1, Dim0 });
        var inSpan = input.AsWritableSpan();
        for (int i = 0; i < Dim0; i++) inSpan[i] = (rng.NextDouble() - 0.5);

        var tOut = engine.Sigmoid(engine.TensorMatMul(
            engine.Sigmoid(engine.TensorMatMul(
                engine.Sigmoid(engine.TensorMatMul(input, w0Trained)),
                w1Trained)),
            w2Trained));
        var cOut = engine.Sigmoid(engine.TensorMatMul(
            engine.Sigmoid(engine.TensorMatMul(
                engine.Sigmoid(engine.TensorMatMul(input, w0Cloned)),
                w1Cloned)),
            w2Cloned));

        var tS = tOut.AsSpan();
        var cS = cOut.AsSpan();
        double maxDiff = 0; int worstIdx = -1;
        double tWorst = 0, cWorst = 0;
        for (int i = 0; i < tS.Length; i++)
        {
            double d = Math.Abs(tS[i] - cS[i]);
            if (d > maxDiff) { maxDiff = d; worstIdx = i; tWorst = tS[i]; cWorst = cS[i]; }
        }
        _output.WriteLine($"fp64 3-layer forward diff: max={maxDiff:E4} at idx {worstIdx} "
            + $"(trained={tWorst}, cloned={cWorst})");
        Assert.True(maxDiff < 1e-12,
            $"#318 fp64 — 3-layer forward output diverges between trained and cloned weights "
            + $"with byte-identical logical content. Max abs diff: {maxDiff:E4} at idx {worstIdx} "
            + $"(tolerance 1e-12). This is the consumer-side DBM Clone-after-train signature.");
    }

    private static Tensor<float> TrainedWeightF(CpuEngine engine, int[] shape, Random rng)
    {
        int total = 1; foreach (var d in shape) total *= d;
        var oldW = new Tensor<float>(shape);
        var delta = new Tensor<float>(shape);
        var oS = oldW.AsWritableSpan(); var dS = delta.AsWritableSpan();
        for (int i = 0; i < total; i++)
        {
            oS[i] = (float)((rng.NextDouble() - 0.5) * 0.1);
            dS[i] = (float)((rng.NextDouble() - 0.5) * 0.001);
        }
        return engine.TensorSubtract(oldW, delta);
    }

    private static Tensor<double> TrainedWeightD(CpuEngine engine, int[] shape, Random rng)
    {
        int total = 1; foreach (var d in shape) total *= d;
        var oldW = new Tensor<double>(shape);
        var delta = new Tensor<double>(shape);
        var oS = oldW.AsWritableSpan(); var dS = delta.AsWritableSpan();
        for (int i = 0; i < total; i++)
        {
            oS[i] = (rng.NextDouble() - 0.5) * 0.1;
            dS[i] = (rng.NextDouble() - 0.5) * 0.001;
        }
        return engine.TensorSubtract(oldW, delta);
    }

    private static Tensor<float> SetFlatTensorF(Tensor<float> source)
    {
        var dst = new Tensor<float>(source._shape);
        var src = source.AsSpan(); var dstSpan = dst.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dstSpan[i] = src[i];
        return dst;
    }

    private static Tensor<double> SetFlatTensorD(Tensor<double> source)
    {
        var dst = new Tensor<double>(source._shape);
        var src = source.AsSpan(); var dstSpan = dst.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dstSpan[i] = src[i];
        return dst;
    }

    private static void AssertLogicalEqualF(Tensor<float> a, Tensor<float> b, string name)
    {
        var aS = a.AsSpan(); var bS = b.AsSpan();
        Assert.Equal(aS.Length, bS.Length);
        for (int i = 0; i < aS.Length; i++)
            Assert.True(aS[i] == bS[i],
                $"{name}[{i}] differs: trained={aS[i]}, cloned={bS[i]}.");
    }

    private static void AssertLogicalEqualD(Tensor<double> a, Tensor<double> b, string name)
    {
        var aS = a.AsSpan(); var bS = b.AsSpan();
        Assert.Equal(aS.Length, bS.Length);
        for (int i = 0; i < aS.Length; i++)
            Assert.True(aS[i] == bS[i],
                $"{name}[{i}] differs: trained={aS[i]}, cloned={bS[i]}.");
    }
}
