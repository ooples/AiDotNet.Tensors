using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd
{
    /// <summary>
    /// 4-way comparison on the SAME MLP [32,128] -> 64 -> 10:
    ///   1. Eager + GradientTape (old baseline)
    ///   2. Compiled Training Plan (PR #106 baseline — our fastest proven path)
    ///   3. TensorCodec Phase B — fused multi-layer GEMM (forward + backward)
    ///   4. PyTorch reference from BenchmarkDotNet
    /// </summary>
    public class TensorCodecBaselineBenchmark
    {
        private readonly ITestOutputHelper _output;

        public TensorCodecBaselineBenchmark(ITestOutputHelper output) => _output = output;

        private static float[] MakeRandomArr(int length, int seed)
        {
            var rng = new Random(seed);
            var data = new float[length];
            for (int i = 0; i < length; i++)
                data[i] = (float)(rng.NextDouble() * 2 - 1);
            return data;
        }

        [Fact]
        public void FourWayComparison_MLP_TrainingStep()
        {
            var engine = new CpuEngine();
            int m = 32, k = 128, h = 64, n = 10;

            // Shared raw arrays
            var inputArr = MakeRandomArr(m * k, 42);
            var w1Arr = MakeRandomArr(k * h, 43);
            var w2Arr = MakeRandomArr(h * n, 44);

            // Tensor wrappers for engine paths
            var input = new Tensor<float>(inputArr, new[] { m, k });
            var w1 = new Tensor<float>(w1Arr, new[] { k, h });
            var w2 = new Tensor<float>(w2Arr, new[] { h, n });

            int warmup = 50;
            int iters = 1000;

            // ================================================================
            // PATH 1: Eager + GradientTape (old baseline)
            // ================================================================
            for (int i = 0; i < warmup; i++)
            {
                using (var tape = new GradientTape<float>())
                {
                    var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                    var output = engine.TensorMatMul(h1, w2);
                    var loss = engine.ReduceSum(output, null);
                    tape.ComputeGradients(loss, new[] { w1, w2 });
                }
            }

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                using (var tape = new GradientTape<float>())
                {
                    var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                    var output = engine.TensorMatMul(h1, w2);
                    var loss = engine.ReduceSum(output, null);
                    tape.ComputeGradients(loss, new[] { w1, w2 });
                }
            }
            sw.Stop();
            double eagerTapeMs = sw.Elapsed.TotalMilliseconds / iters;

            // ================================================================
            // PATH 2: Compiled Training Plan (PR #106 — our proven faster path)
            // ================================================================
            CompiledTrainingPlan<float> compiledPlan;
            using (var scope = GraphMode.Enable())
            {
                var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                engine.TensorMatMul(h1, w2);
                compiledPlan = scope.CompileTraining(new[] { w1, w2 });
            }

            for (int i = 0; i < warmup; i++)
                compiledPlan.Step();

            sw.Restart();
            for (int i = 0; i < iters; i++)
                compiledPlan.Step();
            sw.Stop();
            double compiledMs = sw.Elapsed.TotalMilliseconds / iters;

            // ================================================================
            // PATH 3: TensorCodec Phase B (fused forward + fused backward)
            // ================================================================
            Func<float, float> relu = x => x > 0f ? x : 0f;
            var fusedOut = new float[m * n];
            var fusedAct = new float[m * h];
            var gradOutput = new float[m * n];
            // Seed grad with ones (same as tape)
            for (int i = 0; i < gradOutput.Length; i++) gradOutput[i] = 1f;
            var gradW1 = new float[k * h];
            var gradW2 = new float[h * n];
            var gradB1 = new float[0];
            var gradB2 = new float[0];
            var gradInput = new float[m * k];

            for (int i = 0; i < warmup; i++)
            {
                FusedMultiLayerGemm.FusedGemmActivationGemm(inputArr, w1Arr, w2Arr, fusedOut, fusedAct, m, k, h, n, relu);
                FusedMultiLayerBackward.ComputeGradients(gradOutput, inputArr, w1Arr, w2Arr, fusedAct,
                    gradW1, gradW2, gradB1, gradB2, gradInput, m, k, h, n,
                    FusedMultiLayerBackward.ReLUDerivative);
            }

            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                FusedMultiLayerGemm.FusedGemmActivationGemm(inputArr, w1Arr, w2Arr, fusedOut, fusedAct, m, k, h, n, relu);
                FusedMultiLayerBackward.ComputeGradients(gradOutput, inputArr, w1Arr, w2Arr, fusedAct,
                    gradW1, gradW2, gradB1, gradB2, gradInput, m, k, h, n,
                    FusedMultiLayerBackward.ReLUDerivative);
            }
            sw.Stop();
            double codecMs = sw.Elapsed.TotalMilliseconds / iters;

            // ================================================================
            // RESULTS
            // ================================================================
            double pytorchMs = 0.266; // BenchmarkDotNet reference

            _output.WriteLine("========================================================");
            _output.WriteLine("  4-WAY COMPARISON: MLP [32,128] -> 64 -> 10 TRAINING");
            _output.WriteLine("========================================================");
            _output.WriteLine("");
            _output.WriteLine("  1. Eager + GradientTape:    {0:F4}ms", eagerTapeMs);
            _output.WriteLine("  2. Compiled Training Plan:  {0:F4}ms  ({1:F2}x vs eager)", compiledMs, eagerTapeMs / compiledMs);
            _output.WriteLine("  3. TensorCodec Phase B:     {0:F4}ms  ({1:F2}x vs eager)", codecMs, eagerTapeMs / codecMs);
            _output.WriteLine("  4. PyTorch (reference):     {0:F4}ms", pytorchMs);
            _output.WriteLine("");
            _output.WriteLine("  --- vs PyTorch ---");
            _output.WriteLine("  Eager tape:      {0:F2}x {1}", eagerTapeMs / pytorchMs,
                eagerTapeMs < pytorchMs ? "FASTER" : "slower");
            _output.WriteLine("  Compiled plan:   {0:F2}x {1}", compiledMs < pytorchMs ? pytorchMs / compiledMs : compiledMs / pytorchMs,
                compiledMs < pytorchMs ? "FASTER" : "slower");
            _output.WriteLine("  TensorCodec:     {0:F2}x {1}", codecMs < pytorchMs ? pytorchMs / codecMs : codecMs / pytorchMs,
                codecMs < pytorchMs ? "FASTER" : "slower");
            _output.WriteLine("");
            _output.WriteLine("  --- Phase B vs Compiled Plan (our best baseline) ---");
            _output.WriteLine("  Speedup: {0:F2}x", compiledMs / codecMs);
            _output.WriteLine("========================================================");
        }
    }
}
