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
    public class TensorCodecMultiSizeBenchmark
    {
        private readonly ITestOutputHelper _output;

        public TensorCodecMultiSizeBenchmark(ITestOutputHelper output) => _output = output;

        private static double Measure(Action action, int warmup, int iters)
        {
            for (int i = 0; i < warmup; i++) action();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) action();
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds / iters;
        }

        [Theory]
        [InlineData(8, 64, 32, 10, 500)]     // Tiny
        [InlineData(32, 128, 64, 10, 500)]    // Small (main benchmark)
        [InlineData(64, 256, 128, 32, 200)]   // Medium
        [InlineData(128, 512, 256, 64, 100)]  // Large
        public void CompiledPlan_VsEager_VaryingSizes(int m, int k, int h, int n, int iters)
        {
            var engine = new CpuEngine();
            var input = Tensor<float>.CreateRandom(new[] { m, k });
            var w1 = Tensor<float>.CreateRandom(new[] { k, h });
            var w2 = Tensor<float>.CreateRandom(new[] { h, n });
            int warmup = 30;

            // Eager + tape
            double eagerMs = Measure(() =>
            {
                using (var tape = new GradientTape<float>())
                {
                    var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                    var output = engine.TensorMatMul(h1, w2);
                    var loss = engine.ReduceSum(output, null);
                    tape.ComputeGradients(loss, new[] { w1, w2 });
                }
            }, warmup, iters);

            // Compiled plan
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                engine.TensorMatMul(h1, w2);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
            double compiledMs = Measure(() => plan.Step(), warmup, iters);

            // Phase B standalone
            var inputArr = input.GetDataArray();
            var w1Arr = w1.GetDataArray();
            var w2Arr = w2.GetDataArray();
            Func<float, float> relu = x => x > 0f ? x : 0f;
            var fOut = new float[m * n];
            var fAct = new float[m * h];
            var gOut = new float[m * n]; for (int i = 0; i < gOut.Length; i++) gOut[i] = 1f;
            var gW1 = new float[k * h]; var gW2 = new float[h * n];
            var gIn = new float[m * k]; var empty = new float[0];
            var ws = new float[m * h];

            double phaseBMs = Measure(() =>
            {
                FusedMultiLayerGemm.FusedGemmActivationGemm(inputArr, w1Arr, w2Arr, fOut, fAct, m, k, h, n, relu);
                FusedMultiLayerBackward.ComputeGradients(gOut, inputArr, w1Arr, w2Arr, fAct,
                    gW1, gW2, empty, empty, gIn, m, k, h, n,
                    FusedMultiLayerBackward.ReLUDerivative, ws);
            }, warmup, iters);

            double speedupVsEager = eagerMs / compiledMs;
            double phaseBVsEager = eagerMs / phaseBMs;

            _output.WriteLine("[{0}x{1}->{2}->{3}]  Eager: {4:F4}ms  Compiled: {5:F4}ms ({6:F1}x)  PhaseB: {7:F4}ms ({8:F1}x)",
                m, k, h, n, eagerMs, compiledMs, speedupVsEager, phaseBMs, phaseBVsEager);
        }

        [Theory]
        [InlineData(100000, 500)]    // 100K elements
        [InlineData(1000000, 200)]   // 1M elements
        public void ElementwiseOps_CompiledVsEager(int size, int iters)
        {
            var engine = new CpuEngine();
            var a = Tensor<float>.CreateRandom(new[] { size });
            var b = Tensor<float>.CreateRandom(new[] { size });
            int warmup = 30;

            // Eager add
            double eagerAddMs = Measure(() => engine.TensorAdd(a, b), warmup, iters);

            // Eager ReLU
            double eagerReluMs = Measure(() => engine.ReLU(a), warmup, iters);

            // Eager Sigmoid
            double eagerSigMs = Measure(() => engine.Sigmoid(a), warmup, iters);

            // Compiled add
            CompiledInferencePlan<float> addPlan;
            using (var scope = GraphMode.Enable())
            {
                engine.TensorAdd(a, b);
                addPlan = scope.CompileInference<float>();
            }
            double compiledAddMs = Measure(() => addPlan.Execute(), warmup, iters);

            // Compiled ReLU
            CompiledInferencePlan<float> reluPlan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReLU(a);
                reluPlan = scope.CompileInference<float>();
            }
            double compiledReluMs = Measure(() => reluPlan.Execute(), warmup, iters);

            _output.WriteLine("Elementwise [{0}]:", size);
            _output.WriteLine("  Add:     Eager {0:F4}ms  Compiled {1:F4}ms  ({2:F2}x)",
                eagerAddMs, compiledAddMs, eagerAddMs / compiledAddMs);
            _output.WriteLine("  ReLU:    Eager {0:F4}ms  Compiled {1:F4}ms  ({2:F2}x)",
                eagerReluMs, compiledReluMs, eagerReluMs / compiledReluMs);
            _output.WriteLine("  Sigmoid: Eager {0:F4}ms", eagerSigMs);
        }
    }
}
