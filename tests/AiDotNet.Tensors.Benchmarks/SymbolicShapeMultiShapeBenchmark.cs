#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks the production-serving throughput advantage of multi-dim symbolic
/// shapes (issue #167) over the naive per-shape-compile behaviour that
/// <c>torch.compile</c> falls back to when its dynamic-shape support doesn't
/// hold the tensor's symbol marking across the trace.
///
/// <para><b>Scenario:</b> a transformer serving 16 distinct <c>(batch, seq)</c>
/// combinations over the same weights. In production, request batches arrive
/// with varying token counts and batch sizes.</para>
///
/// <para><b>Two strategies measured:</b></para>
/// <list type="bullet">
///   <item><c>CompileOncePlusReplays</c> — our approach. Register each shape
///     via <see cref="CompiledModelCache{T}.GetOrCompileInference(int[], Action, SymbolicShape)"/>
///     with <see cref="SymbolicShape.BatchAndSeqDynamic"/>. First call compiles;
///     subsequent 15 calls hit the cache and skip the trace+compile phase.</item>
///   <item><c>RecompilePerShape</c> — the torch.compile gotcha. Use the
///     non-symbolic shape key path so every distinct shape forces a fresh
///     trace and compile.</item>
/// </list>
///
/// <para>The <c>RecompilePerShape/CompileOncePlusReplays</c> ratio is the
/// headline number — how many times faster our approach serves the same mix
/// of shapes compared to torch.compile's recompile-per-shape behaviour.</para>
///
/// <para><b>Why this is a fair comparison to PyTorch:</b> <c>torch.compile</c>'s
/// published behaviour is that shape variation not declared with
/// <c>torch._dynamo.mark_dynamic</c> retraces on every new shape. Even with
/// <c>mark_dynamic</c>, PyTorch's guards frequently specialise and retrace
/// anyway (documented as the library's biggest production gotcha). Our
/// <see cref="SymbolicShape"/> doesn't guard-specialise — the symbolic key
/// is the cache key — so a single compile serves any shape matching the
/// static dims.</para>
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class SymbolicShapeMultiShapeBenchmark
{
    // 16 distinct (batch, seq) combinations covering typical serving pools.
    private static readonly (int batch, int seq)[] Combos = new (int, int)[]
    {
        (1, 128), (1, 512), (2, 64), (2, 256),
        (4, 128), (4, 1024), (8, 64), (8, 2048),
        (16, 128), (16, 512), (32, 64), (32, 256),
        (1, 1024), (3, 512), (5, 128), (7, 256),
    };

    private const int FeatureDim = 512;

    private CpuEngine _engine = null!;
    private Tensor<float>[] _weights = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();
        // One weight matrix reused across all shapes — matches serving a single
        // fine-tuned model against a variable request mix.
        _weights = new[] { Tensor<float>.CreateRandom(new[] { FeatureDim, FeatureDim }) };
    }

    /// <summary>
    /// Our approach: one compile, fifteen cache hits. Every distinct (batch, seq)
    /// tuple lands on the same cached plan because <see cref="SymbolicShape.BatchAndSeqDynamic"/>
    /// keys the plan on the static feature dim alone.
    /// </summary>
    [Benchmark(Baseline = true, Description = "1 compile + 15 cache hits (multi-dim symbolic)")]
    public int CompileOncePlusReplays()
    {
        using var cache = new CompiledModelCache<float>();
        int compiled = 0;
        foreach (var (b, s) in Combos)
        {
            var input = Tensor<float>.CreateRandom(new[] { b, s, FeatureDim });
            cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    compiled++;
                    var output = _engine.TensorMatMul(input, _weights[0]);
                    return _engine.ReduceSum(output, null);
                },
                SymbolicShape.BatchAndSeqDynamic(new[] { b, s, FeatureDim }));
        }
        return compiled; // Should be 1.
    }

    /// <summary>
    /// torch.compile's gotcha, reproduced: the non-symbolic overload keys on
    /// the exact concrete shape, so every distinct (batch, seq) forces a fresh
    /// trace and compile.
    /// </summary>
    [Benchmark(Description = "16 cold compiles (recompile-per-shape — torch.compile gotcha)")]
    public int RecompilePerShape()
    {
        using var cache = new CompiledModelCache<float>();
        int compiled = 0;
        foreach (var (b, s) in Combos)
        {
            var input = Tensor<float>.CreateRandom(new[] { b, s, FeatureDim });
            cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    compiled++;
                    var output = _engine.TensorMatMul(input, _weights[0]);
                    return _engine.ReduceSum(output, null);
                });
        }
        return compiled; // Will be 16.
    }
}
#endif
