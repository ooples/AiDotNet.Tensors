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

        [Fact]
        public void AllActivations_CompiledVsEager()
        {
            var engine = new CpuEngine();
            int size = 100000;
            var a = Tensor<float>.CreateRandom(new[] { size });
            int warmup = 30, iters = 500;

            var ops = new (string name, Func<Tensor<float>> eager)[]
            {
                ("ReLU", () => engine.ReLU(a)),
                ("Sigmoid", () => engine.Sigmoid(a)),
                ("Tanh", () => engine.Tanh(a)),
                ("GELU", () => engine.GELU(a)),
                ("Swish", () => engine.Swish(a)),
                ("LeakyReLU", () => engine.LeakyReLU(a, 0.01f)),
                ("Softmax", () => engine.Softmax(a, 0)),
            };

            _output.WriteLine("Activations [{0}] — Compiled vs Eager:", size);
            _output.WriteLine("  {0,-15} {1,10} {2,10} {3,10}", "Op", "Eager(ms)", "Compiled", "Speedup");

            foreach (var (name, eagerFn) in ops)
            {
                double eagerMs = Measure(() => { eagerFn(); }, warmup, iters);

                CompiledInferencePlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    eagerFn();
                    plan = scope.CompileInference<float>();
                }
                double compiledMs = Measure(() => plan.Execute(), warmup, iters);

                _output.WriteLine("  {0,-15} {1,10:F4} {2,10:F4} {3,10:F2}x", name, eagerMs, compiledMs, eagerMs / compiledMs);
            }
        }

        [Fact]
        public void AllBinaryOps_CompiledVsEager()
        {
            var engine = new CpuEngine();
            int size = 100000;
            var a = Tensor<float>.CreateRandom(new[] { size });
            var b = Tensor<float>.CreateRandom(new[] { size });
            int warmup = 30, iters = 500;

            var ops = new (string name, Func<Tensor<float>> eager)[]
            {
                ("Add", () => engine.TensorAdd(a, b)),
                ("Subtract", () => engine.TensorSubtract(a, b)),
                ("Multiply", () => engine.TensorMultiply(a, b)),
                ("Divide", () => engine.TensorDivide(a, b)),
                ("Negate", () => engine.TensorNegate(a)),
            };

            _output.WriteLine("Binary/Unary Ops [{0}] — Compiled vs Eager:", size);
            _output.WriteLine("  {0,-15} {1,10} {2,10} {3,10}", "Op", "Eager(ms)", "Compiled", "Speedup");

            foreach (var (name, eagerFn) in ops)
            {
                double eagerMs = Measure(() => { eagerFn(); }, warmup, iters);

                CompiledInferencePlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    eagerFn();
                    plan = scope.CompileInference<float>();
                }
                double compiledMs = Measure(() => plan.Execute(), warmup, iters);

                _output.WriteLine("  {0,-15} {1,10:F4} {2,10:F4} {3,10:F2}x", name, eagerMs, compiledMs, eagerMs / compiledMs);
            }
        }

        [Fact]
        public void MatMul_VaryingSizes_CompiledVsEager()
        {
            var engine = new CpuEngine();
            int warmup = 30, iters = 500;

            var sizes = new (int m, int k, int n)[]
            {
                (8, 32, 16),
                (32, 128, 64),
                (64, 256, 128),
                (128, 512, 256),
                (256, 1024, 512),
            };

            _output.WriteLine("MatMul Varying Sizes — Compiled vs Eager:");
            _output.WriteLine("  {0,-25} {1,10} {2,10} {3,10}", "Size", "Eager(ms)", "Compiled", "Speedup");

            foreach (var (m, k, n) in sizes)
            {
                var a = Tensor<float>.CreateRandom(new[] { m, k });
                var b = Tensor<float>.CreateRandom(new[] { k, n });

                double eagerMs = Measure(() => engine.TensorMatMul(a, b), warmup, iters);

                CompiledInferencePlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    engine.TensorMatMul(a, b);
                    plan = scope.CompileInference<float>();
                }
                double compiledMs = Measure(() => plan.Execute(), warmup, iters);

                _output.WriteLine("  [{0}x{1}]@[{1}x{2}] {3,14:F4} {4,10:F4} {5,10:F2}x",
                    m, k, n, eagerMs, compiledMs, eagerMs / compiledMs);
            }
        }

        [Fact]
        public void CNN_Conv2D_ReLU_Pool_CompiledVsEager()
        {
            var engine = new CpuEngine();
            // Small CNN: [1, 3, 16, 16] input, [8, 3, 3, 3] kernel, stride=1, pad=1
            var input = Tensor<float>.CreateRandom(new[] { 1, 3, 16, 16 });
            var kernel = Tensor<float>.CreateRandom(new[] { 8, 3, 3, 3 });
            int warmup = 20, iters = 200;

            // Eager: Conv2D → ReLU → MaxPool2D
            double eagerMs = Measure(() =>
            {
                var conv = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
                var relu = engine.ReLU(conv);
                engine.MaxPool2D(relu, poolSize: 2, stride: 2);
            }, warmup, iters);

            // Compiled plan
            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var conv = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
                var relu = engine.ReLU(conv);
                engine.MaxPool2D(relu, poolSize: 2, stride: 2);
                plan = scope.CompileInference<float>();
            }
            double compiledMs = Measure(() => plan.Execute(), warmup, iters);

            _output.WriteLine("CNN [1,3,16,16] Conv3x3→ReLU→Pool2x2:");
            _output.WriteLine("  Eager:    {0:F4}ms", eagerMs);
            _output.WriteLine("  Compiled: {0:F4}ms ({1:F2}x)", compiledMs, eagerMs / compiledMs);
        }

        [Fact]
        public void TransformerBlock_MatMul_Softmax_MatMul_CompiledVsEager()
        {
            var engine = new CpuEngine();
            // Attention-like: Q@K^T → Softmax → @V
            int batch = 4, heads = 4, seqLen = 16, dHead = 16;
            var q = Tensor<float>.CreateRandom(new[] { batch * heads, seqLen, dHead });
            var k = Tensor<float>.CreateRandom(new[] { batch * heads, dHead, seqLen }); // pre-transposed
            var v = Tensor<float>.CreateRandom(new[] { batch * heads, seqLen, dHead });
            int warmup = 20, iters = 200;

            // Eager attention
            double eagerMs = Measure(() =>
            {
                var scores = engine.BatchMatMul(q, k);
                var attn = engine.Softmax(scores, -1);
                engine.BatchMatMul(attn, v);
            }, warmup, iters);

            // Compiled
            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var scores = engine.BatchMatMul(q, k);
                var attn = engine.Softmax(scores, -1);
                engine.BatchMatMul(attn, v);
                plan = scope.CompileInference<float>();
            }
            double compiledMs = Measure(() => plan.Execute(), warmup, iters);

            _output.WriteLine("Attention [{0}heads, seq{1}, d{2}]:", heads, seqLen, dHead);
            _output.WriteLine("  Eager:    {0:F4}ms", eagerMs);
            _output.WriteLine("  Compiled: {0:F4}ms ({1:F2}x)", compiledMs, eagerMs / compiledMs);
        }

        [Theory]
        [InlineData("ReLU")]
        [InlineData("Tanh")]
        [InlineData("GELU")]
        [InlineData("LeakyReLU")]
        public void ThreeLayerMLP_DifferentActivations_CompiledVsEager(string activationName)
        {
            var engine = new CpuEngine();
            int m = 32, d1 = 128, d2 = 64, d3 = 32, dOut = 10;
            var input = Tensor<float>.CreateRandom(new[] { m, d1 });
            var w1 = Tensor<float>.CreateRandom(new[] { d1, d2 });
            var w2 = Tensor<float>.CreateRandom(new[] { d2, d3 });
            var w3 = Tensor<float>.CreateRandom(new[] { d3, dOut });
            int warmup = 30, iters = 500;

            Func<Tensor<float>, Tensor<float>> activation = activationName switch
            {
                "ReLU" => t => engine.ReLU(t),
                "Tanh" => t => engine.Tanh(t),
                "GELU" => t => engine.GELU(t),
                "LeakyReLU" => t => engine.LeakyReLU(t, 0.01f),
                _ => t => engine.ReLU(t)
            };

            // Eager training
            double eagerMs = Measure(() =>
            {
                using (var tape = new GradientTape<float>())
                {
                    var h1 = activation(engine.TensorMatMul(input, w1));
                    var h2 = activation(engine.TensorMatMul(h1, w2));
                    var output = engine.TensorMatMul(h2, w3);
                    var loss = engine.ReduceSum(output, null);
                    tape.ComputeGradients(loss, new[] { w1, w2, w3 });
                }
            }, warmup, iters);

            // Compiled training
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h1 = activation(engine.TensorMatMul(input, w1));
                var h2 = activation(engine.TensorMatMul(h1, w2));
                var output = engine.TensorMatMul(h2, w3);
                plan = scope.CompileTraining(new[] { w1, w2, w3 });
            }
            double compiledMs = Measure(() => plan.Step(), warmup, iters);

            _output.WriteLine("3-Layer MLP [{0}] {1}x{2}->{3}->{4}->{5}:", activationName, m, d1, d2, d3, dOut);
            _output.WriteLine("  Eager:    {0:F4}ms", eagerMs);
            _output.WriteLine("  Compiled: {0:F4}ms ({1:F2}x)", compiledMs, eagerMs / compiledMs);
        }

        [Theory]
        [InlineData(32, 512, 256, 128, 64, 10, 100)]    // Wide MLP
        [InlineData(128, 256, 128, 64, 32, 10, 100)]     // Large batch
        [InlineData(32, 1024, 512, 256, 128, 10, 50)]    // Very deep/wide
        public void DeepMLP_CompiledVsEager(int m, int d1, int d2, int d3, int d4, int dOut, int iters)
        {
            var engine = new CpuEngine();
            var input = Tensor<float>.CreateRandom(new[] { m, d1 });
            var w1 = Tensor<float>.CreateRandom(new[] { d1, d2 });
            var w2 = Tensor<float>.CreateRandom(new[] { d2, d3 });
            var w3 = Tensor<float>.CreateRandom(new[] { d3, d4 });
            var w4 = Tensor<float>.CreateRandom(new[] { d4, dOut });
            int warmup = 20;

            double eagerMs = Measure(() =>
            {
                using (var tape = new GradientTape<float>())
                {
                    var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                    var h2 = engine.ReLU(engine.TensorMatMul(h1, w2));
                    var h3 = engine.ReLU(engine.TensorMatMul(h2, w3));
                    var output = engine.TensorMatMul(h3, w4);
                    var loss = engine.ReduceSum(output, null);
                    tape.ComputeGradients(loss, new[] { w1, w2, w3, w4 });
                }
            }, warmup, iters);

            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                var h2 = engine.ReLU(engine.TensorMatMul(h1, w2));
                var h3 = engine.ReLU(engine.TensorMatMul(h2, w3));
                var output = engine.TensorMatMul(h3, w4);
                plan = scope.CompileTraining(new[] { w1, w2, w3, w4 });
            }
            double compiledMs = Measure(() => plan.Step(), warmup, iters);

            _output.WriteLine("4-Layer MLP [{0}x{1}->{2}->{3}->{4}->{5}]:", m, d1, d2, d3, d4, dOut);
            _output.WriteLine("  Eager: {0:F4}ms, Compiled: {1:F4}ms ({2:F2}x)",
                eagerMs, compiledMs, eagerMs / compiledMs);
        }
    }
}
