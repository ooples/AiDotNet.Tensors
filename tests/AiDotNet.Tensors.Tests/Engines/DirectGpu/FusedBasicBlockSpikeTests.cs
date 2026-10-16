using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// SPIKE (not for merge as is): one ResNet basic block's training forward, x -> conv -> BN -> ReLU -> conv -> BN -> +x
/// -> ReLU, through the engine's separate ops versus the five fused kernels (conv with a BN-statistics epilogue, a
/// finalize, conv with a BN+ReLU prologue and statistics epilogue, a finalize, and BN+add+ReLU). Checks they agree and
/// times both per ResNet-18 stage shape. Opt in with AIDOTNET_FUSED_BLOCK_SPIKE=1.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedBasicBlockSpikeTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public FusedBasicBlockSpikeTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private static float[] Rand(int n, int seed, float scale, float offset = 0f)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => offset + (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    [SkippableTheory]
    [InlineData(8, 64, 32)]
    [InlineData(8, 128, 16)]
    [InlineData(8, 256, 8)]
    [InlineData(8, 512, 4)]
    public void Fused_basic_block_forward_matches_the_engine_and_is_timed(int batch, int channels, int size)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_FUSED_BLOCK_SPIKE") == "1", "Spike: set AIDOTNET_FUSED_BLOCK_SPIKE=1.");
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        Skip.IfNot(cuda!.EnsureFusedBlockModule(), "Fused kernels did not compile.");
        IEngine engine = gpu;
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            int c = channels, s = size, total = batch * c * s * s;
            const float eps = 1e-5f;
            var x = new Tensor<float>(Rand(total, 1, 1f), new[] { batch, c, s, s }).Gpu();
            var w1 = new Tensor<float>(Rand(c * c * 9, 2, 1f / MathF.Sqrt(c * 9)), new[] { c, c, 3, 3 }).Gpu();
            var w2 = new Tensor<float>(Rand(c * c * 9, 3, 1f / MathF.Sqrt(c * 9)), new[] { c, c, 3, 3 }).Gpu();
            var g1 = new Tensor<float>(Rand(c, 4, 0.2f, 1f), new[] { c }).Gpu();
            var b1 = new Tensor<float>(Rand(c, 5, 0.1f), new[] { c }).Gpu();
            var g2 = new Tensor<float>(Rand(c, 6, 0.2f, 1f), new[] { c }).Gpu();
            var b2 = new Tensor<float>(Rand(c, 7, 0.1f), new[] { c }).Gpu();

            Tensor<float> Unfused()
            {
                var y1 = engine.FusedConv2D(x, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
                var a1 = engine.ReLU(engine.BatchNorm(y1, g1, b1, eps, out _, out _));
                var y2 = engine.FusedConv2D(a1, w2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
                return engine.ReLU(engine.TensorAdd(engine.BatchNorm(y2, g2, b2, eps, out _, out _), x));
            }

            // Fused: device buffers straight from the tensors, plus per-channel scratch.
            IGpuBuffer B(Tensor<float> t) => t.TryGetGpuBuffer() ?? throw new InvalidOperationException("not resident");
            using var y1b = cuda.AllocateBuffer(total);
            using var y2b = cuda.AllocateBuffer(total);
            using var outb = cuda.AllocateBuffer(total);
            using var sum1 = cuda.AllocateBuffer(new float[2 * c]);   // c doubles, zeroed
            using var sq1 = cuda.AllocateBuffer(new float[2 * c]);
            using var sum2 = cuda.AllocateBuffer(new float[2 * c]);
            using var sq2 = cuda.AllocateBuffer(new float[2 * c]);
            using var mean1 = cuda.AllocateBuffer(c); using var inv1 = cuda.AllocateBuffer(c);
            using var sc1 = cuda.AllocateBuffer(c); using var sh1 = cuda.AllocateBuffer(c);
            using var mean2 = cuda.AllocateBuffer(c); using var inv2 = cuda.AllocateBuffer(c);
            using var sc2 = cuda.AllocateBuffer(c); using var sh2 = cuda.AllocateBuffer(c);
            var conv1 = new CudaConv2DArgs(B(x), B(w1), y1b, batch, c, s, s, c, s, s, 3, 3, 1, 1, 1, 1, 1, 1);
            long count = (long)batch * s * s;
            void Fused()
            {
                cuda.FusedConvBatchStats(conv1, sum1, sq1);
                cuda.FusedBatchNormFinalize(sum1, sq1, B(g1), B(b1), mean1, inv1, sc1, sh1, c, count, eps);
                var conv2 = new CudaConv2DArgs(y1b, B(w2), y2b, batch, c, s, s, c, s, s, 3, 3, 1, 1, 1, 1, 1, 1);
                cuda.FusedConvBatchStats(conv2, sum2, sq2, sc1, sh1);
                cuda.FusedBatchNormFinalize(sum2, sq2, B(g2), B(b2), mean2, inv2, sc2, sh2, c, count, eps);
                cuda.FusedBatchNormAddRelu(y2b, sc2, sh2, B(x), outb, total, c, s * s);
            }

            var reference = Unfused().ToArray();
            Fused();
            var fused = cuda.DownloadBuffer(outb);
            double maxErr = 0;
            for (int i = 0; i < total; i++)
                maxErr = Math.Max(maxErr, Math.Abs(reference[i] - fused[i]) / Math.Max(1.0, Math.Abs(reference[i])));
            _output.WriteLine($"[{batch}x{c}x{s}x{s}] max rel err fused vs engine: {maxErr:E2}");
            Assert.True(maxErr < 2e-3, $"fused block disagrees with the engine: max rel err {maxErr:E2}");

            double Time(Action body)
            {
                for (int i = 0; i < 20; i++) body();
                gpu.SynchronizeStream();
                var times = new double[7];
                for (int r = 0; r < times.Length; r++)
                {
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < 50; i++) body();
                    gpu.SynchronizeStream();
                    times[r] = sw.Elapsed.TotalMilliseconds * 1000 / 50;
                }
                Array.Sort(times);
                return times[0];
            }
            double unfusedUs = Time(() => Unfused());
            double fusedUs = Time(Fused);
            _output.WriteLine($"[{batch}x{c}x{s}x{s}] block forward: engine {unfusedUs:F1} us, fused {fusedUs:F1} us ({unfusedUs / fusedUs:F2}x)");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
