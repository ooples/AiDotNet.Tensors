using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Acceptance tests for issue #167 — multi-dim symbolic shapes.
///
/// Verifies:
///   • The new factory methods (<c>BatchAndSeqDynamic</c>, <c>AllDynamic</c>,
///     <c>From</c>) and <c>params</c> constructor mark the expected dimensions.
///   • <c>ComputeKey</c> distinguishes shapes that the old single-value FNV
///     hash would have collided (e.g. <c>[3, ?]</c> vs <c>[?, 3]</c>).
///   • <c>Matches</c> correctly accepts variable values in every symbolic
///     position simultaneously.
///   • The cache-hit-rate acceptance criterion: 16 random
///     <c>(batch, seq)</c> combos → exactly one compile, fifteen cache hits.
/// </summary>
public class MultiDimSymbolicShapeTests
{
    // ── Factory methods ──────────────────────────────────────────────────────
    [Fact]
    public void BatchAndSeqDynamic_MarksFirstTwoDimensions()
    {
        var sym = SymbolicShape.BatchAndSeqDynamic(new[] { 1, 128, 512 });
        Assert.Equal(new[] { 1, 128, 512 }, sym.ConcreteShape);
        Assert.Equal(new[] { 0, 1 }, sym.SymbolicDimensions);
    }

    [Fact]
    public void BatchAndSeqDynamic_RequiresRankAtLeastTwo()
    {
        Assert.Throws<ArgumentException>(() =>
            SymbolicShape.BatchAndSeqDynamic(new[] { 128 }));
    }

    [Fact]
    public void AllDynamic_MarksEveryDimension()
    {
        var sym = SymbolicShape.AllDynamic(new[] { 2, 3, 4, 5 });
        Assert.Equal(new[] { 0, 1, 2, 3 }, sym.SymbolicDimensions);
    }

    [Fact]
    public void From_AcceptsParamsIndices()
    {
        // Issue calls out From(concrete, params dynamicDims) as an alias.
        var sym = SymbolicShape.From(new[] { 1, 3, 224, 224 }, 0, 2, 3);
        Assert.Equal(new[] { 0, 2, 3 }, sym.SymbolicDimensions);
    }

    [Fact]
    public void Constructor_AcceptsParamsIndices()
    {
        // The unified params ctor replaces the old (int[], int[]?) signature.
        var sym = new SymbolicShape(new[] { 1, 128, 512 }, 0, 1);
        Assert.Equal(new[] { 0, 1 }, sym.SymbolicDimensions);
    }

    [Fact]
    public void Constructor_NoParamsMeansFullyStatic()
    {
        var sym = new SymbolicShape(new[] { 32, 128 });
        Assert.Empty(sym.SymbolicDimensions);
    }

    // ── Matches() across multiple symbolic dims ─────────────────────────────
    [Theory]
    [InlineData(1, 128)]
    [InlineData(8, 256)]
    [InlineData(32, 1024)]
    [InlineData(1024, 4)]
    public void Matches_AcceptsAnyCombinationOfDynamicBatchAndSeq(int batch, int seq)
    {
        var sym = SymbolicShape.BatchAndSeqDynamic(new[] { 1, 128, 512 });
        Assert.True(sym.Matches(new[] { batch, seq, 512 }),
            $"Should match (batch={batch}, seq={seq}, dim=512)");
    }

    [Fact]
    public void Matches_RejectsChangeToStaticDim()
    {
        // dim=512 is static in BatchAndSeqDynamic; changing it must force a miss.
        var sym = SymbolicShape.BatchAndSeqDynamic(new[] { 1, 128, 512 });
        Assert.False(sym.Matches(new[] { 1, 128, 256 }),
            "Change to the static feature dim must not match.");
    }

    [Fact]
    public void Matches_RejectsRankMismatch()
    {
        var sym = SymbolicShape.BatchAndSeqDynamic(new[] { 1, 128, 512 });
        Assert.False(sym.Matches(new[] { 1, 128 }));
        Assert.False(sym.Matches(new[] { 1, 128, 512, 1 }));
    }

    // ── ComputeKey — collision fix ──────────────────────────────────────────
    [Fact]
    public void ComputeKey_DistinguishesSymbolicPositionSwap()
    {
        // Before the rank + position bitmask mix-in, these two shapes both
        // "one symbolic dim, one concrete value of 3" produced the same key.
        // That's a latent cache-collision bug; multi-dim support would make
        // it trigger much more often.
        var a = new SymbolicShape(new[] { 5, 3 }, 0); // [?, 3]
        var b = new SymbolicShape(new[] { 3, 5 }, 1); // [3, ?]
        Assert.NotEqual(a.ComputeKey(), b.ComputeKey());
    }

    [Fact]
    public void ComputeKey_DistinguishesRank()
    {
        var r2 = new SymbolicShape(new[] { 1, 512 }, 0);
        var r3 = new SymbolicShape(new[] { 1, 128, 512 }, 0, 1);
        Assert.NotEqual(r2.ComputeKey(), r3.ComputeKey());
    }

    [Fact]
    public void ComputeKey_SameKeyAcrossDynamicValues()
    {
        // Different values in dynamic positions, identical static dims, same
        // symbolic layout → same key (this is the point of symbolic shapes).
        var a = new SymbolicShape(new[] { 1, 128, 512 }, 0, 1);
        var b = new SymbolicShape(new[] { 32, 1024, 512 }, 0, 1);
        Assert.Equal(a.ComputeKey(), b.ComputeKey());
    }

    [Fact]
    public void ComputeKey_DifferentStaticDimsProduceDifferentKeys()
    {
        var a = new SymbolicShape(new[] { 1, 128, 512 }, 0, 1);
        var b = new SymbolicShape(new[] { 1, 128, 256 }, 0, 1); // different feature dim
        Assert.NotEqual(a.ComputeKey(), b.ComputeKey());
    }

    // ── Cache-hit-rate acceptance criterion ─────────────────────────────────
    [Fact]
    public void CompiledModelCache_MultiDimSymbolic_SixteenShapes_OneCompile()
    {
        // Issue #167 acceptance criterion: "Cache hit rate on a benchmark
        // with 16 random batch×seq combos: 100% after the first call."
        //
        // Strategy: wrap the forward action in a counter. Call
        // GetOrCompileInference 16 times with 16 distinct (batch, seq)
        // tuples that all share a fixed feature dim, passing
        // BatchAndSeqDynamic as the symbolic shape. Count forward invocations
        // — compile triggers exactly one; cache hits do not re-trace.
        var engine = new CpuEngine();
        const int dim = 512;

        using var cache = new CompiledModelCache<float>();

        int forwardInvocations = 0;
        void RegisterForward(int batch, int seq)
        {
            // Rebuild the forward closure for each call — matches how a real
            // serving loop would build fresh tensors per request.
            var input = Tensor<float>.CreateRandom(new[] { batch, seq, dim });
            var weight = Tensor<float>.CreateRandom(new[] { dim, dim });
            Action forward = () =>
            {
                forwardInvocations++;
                var output = engine.TensorMatMul(input, weight);
                _ = engine.ReduceSum(output, null);
            };
            cache.GetOrCompileInference(
                input._shape,
                forward,
                SymbolicShape.BatchAndSeqDynamic(new[] { batch, seq, dim }));
        }

        // 16 distinct (batch, seq) combinations — covers typical serving pools
        // (batch 1–32, seq 64–2048).
        var combos = new (int b, int s)[]
        {
            (1, 128), (1, 512), (2, 64), (2, 256),
            (4, 128), (4, 1024), (8, 64), (8, 2048),
            (16, 128), (16, 512), (32, 64), (32, 256),
            (1, 1024), (3, 512), (5, 128), (7, 256),
        };
        foreach (var (b, s) in combos)
            RegisterForward(b, s);

        Assert.Equal(1, forwardInvocations);
        Assert.Equal(1, cache.InferencePlanCount);
    }

    [Fact]
    public void CompiledModelCache_StaticDimChange_ForcesRecompile()
    {
        // Complementary assertion: if the caller tries to reuse a plan under
        // a DIFFERENT static dim, the symbolic key changes and a new plan
        // is compiled. Guards against over-eager matching that would silently
        // serve a shape-incompatible plan.
        var engine = new CpuEngine();
        using var cache = new CompiledModelCache<float>();

        int forwardInvocations = 0;

        void Compile(int dim)
        {
            var input = Tensor<float>.CreateRandom(new[] { 1, 128, dim });
            var weight = Tensor<float>.CreateRandom(new[] { dim, dim });
            cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    forwardInvocations++;
                    var output = engine.TensorMatMul(input, weight);
                    _ = engine.ReduceSum(output, null);
                },
                SymbolicShape.BatchAndSeqDynamic(new[] { 1, 128, dim }));
        }

        Compile(512);
        Compile(512); // same dim — cache hit
        Compile(256); // different feature dim — recompile
        Compile(256); // same again — cache hit

        Assert.Equal(2, forwardInvocations);
        Assert.Equal(2, cache.InferencePlanCount);
    }
}
