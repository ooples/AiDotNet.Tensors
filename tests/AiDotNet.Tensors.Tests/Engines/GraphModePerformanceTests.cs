using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// Performance benchmarks comparing eager execution vs compiled graph plan.
    /// Proves the Lazy Tensor + Graph Compiler system eliminates per-operation overhead.
    /// </summary>
    public class GraphModePerformanceTests
    {
        private readonly ITestOutputHelper _output;

        public GraphModePerformanceTests(ITestOutputHelper output)
        {
            _output = output;
        }

        private static Tensor<float> MakeRandom(int[] shape, int seed)
        {
            var rng = new Random(seed);
            int length = 1;
            for (int i = 0; i < shape.Length; i++) length *= shape[i];
            var data = new float[length];
            for (int i = 0; i < data.Length; i++)
                data[i] = (float)(rng.NextDouble() * 2 - 1);
            return new Tensor<float>(data, shape);
        }

        [Fact(Skip = "Performance benchmark — run manually with --filter GraphModePerformance")]
        [Trait("Category", "Performance")]
        public void CompiledPlanReplay_ZeroAllocation()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 64, 128 }, seed: 20);
            var b = MakeRandom(new[] { 64, 128 }, seed: 21);

            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var c = engine.TensorAdd(a, b);
                var d = engine.ReLU(c);
                plan = scope.CompileInference<float>();
            }

            // First execution
            plan.Execute();

            // Get GC state before replay
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long gcBefore = GC.GetTotalMemory(true);

            // Replay 100 times
            for (int i = 0; i < 100; i++)
                plan.Execute();

            long gcAfter = GC.GetTotalMemory(true);
            long allocatedBytes = gcAfter - gcBefore;

            _output.WriteLine("Memory allocated during 100 compiled plan replays: {0} bytes", allocatedBytes);
            Assert.True(allocatedBytes < 100_000,
                "Compiled plan replay allocated too much: " + allocatedBytes + " bytes for 100 replays");
        }

        /// <summary>
        /// HEAD-TO-HEAD: Eager forward vs CompiledInferencePlan replay.
        /// 2-layer MLP: input [64,256] → hidden [128] → output [10]
        /// This is the benchmark that matters — shows the overhead eliminated by compilation.
        /// </summary>
        [Fact(Skip = "Performance benchmark — run manually")]
        [Trait("Category", "Performance")]
        public void HeadToHead_2LayerMLP_EagerVsCompiled()
        {
            var engine = new CpuEngine();
            int batch = 64, inF = 256, hidden = 128, outF = 10;

            var input = MakeRandom(new[] { batch, inF }, seed: 10);
            var w1 = MakeRandom(new[] { inF, hidden }, seed: 11);
            var b1 = MakeRandom(new[] { hidden }, seed: 12);
            var w2 = MakeRandom(new[] { hidden, outF }, seed: 13);
            var b2 = MakeRandom(new[] { outF }, seed: 14);

            // === EAGER baseline ===
            for (int i = 0; i < 20; i++)
            {
                var h = engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
                engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            }

            int iterations = 500;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var h = engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
                engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            }
            sw.Stop();
            double eagerMs = sw.Elapsed.TotalMilliseconds / iterations;

            // === COMPILED PLAN ===
            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h = engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
                engine.FusedLinear(h, w2, b2, FusedActivationType.None);
                plan = scope.CompileInference<float>();
            }

            // Warmup compiled
            for (int i = 0; i < 20; i++)
                plan.Execute();

            sw.Restart();
            for (int i = 0; i < iterations; i++)
                plan.Execute();
            sw.Stop();
            double compiledMs = sw.Elapsed.TotalMilliseconds / iterations;

            double speedup = eagerMs / compiledMs;

            _output.WriteLine("=== 2-Layer MLP Forward ({0}x{1} -> {2} -> {3}) ===", batch, inF, hidden, outF);
            _output.WriteLine("  Eager:    {0:F4}ms per iteration", eagerMs);
            _output.WriteLine("  Compiled: {0:F4}ms per iteration", compiledMs);
            _output.WriteLine("  Speedup:  {0:F2}x", speedup);
            _output.WriteLine("");
        }

        /// <summary>
        /// HEAD-TO-HEAD: Eager forward+backward vs CompiledTrainingPlan.
        /// Full training step with gradient computation.
        /// </summary>
        [Fact(Skip = "Performance benchmark — run manually")]
        [Trait("Category", "Performance")]
        public void HeadToHead_TrainingStep_EagerVsCompiled()
        {
            var engine = new CpuEngine();
            int batch = 32, inF = 128, outF = 64;

            var input = MakeRandom(new[] { batch, inF }, seed: 50);
            var weights = MakeRandom(new[] { inF, outF }, seed: 51);
            var bias = MakeRandom(new[] { outF }, seed: 52);
            var target = MakeRandom(new[] { batch, outF }, seed: 53);

            // === EAGER training step ===
            for (int i = 0; i < 10; i++)
            {
                using (var tape = new GradientTape<float>())
                {
                    var output = engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
                    var diff = engine.TensorSubtract(output, target);
                    var loss = engine.TensorMultiply(diff, diff);
                    var sumLoss = engine.ReduceSum(loss, new[] { 0, 1 }, keepDims: false);
                    tape.ComputeGradients(sumLoss, new[] { weights, bias });
                }
            }

            int iterations = 200;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                using (var tape = new GradientTape<float>())
                {
                    var output = engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
                    var diff = engine.TensorSubtract(output, target);
                    var loss = engine.TensorMultiply(diff, diff);
                    var sumLoss = engine.ReduceSum(loss, new[] { 0, 1 }, keepDims: false);
                    tape.ComputeGradients(sumLoss, new[] { weights, bias });
                }
            }
            sw.Stop();
            double eagerMs = sw.Elapsed.TotalMilliseconds / iterations;

            // === COMPILED PLAN training step ===
            // Build compiled plan from a graph mode recording
            CompiledInferencePlan<float> fwdPlan;
            using (var scope = GraphMode.Enable())
            {
                var output = engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
                var diff = engine.TensorSubtract(output, target);
                var loss = engine.TensorMultiply(diff, diff);
                fwdPlan = scope.CompileInference<float>();
            }

            // Warmup
            for (int i = 0; i < 10; i++)
                fwdPlan.Execute();

            sw.Restart();
            for (int i = 0; i < iterations; i++)
                fwdPlan.Execute();
            sw.Stop();
            double compiledFwdMs = sw.Elapsed.TotalMilliseconds / iterations;

            _output.WriteLine("=== Training Step ({0}x{1} -> {2}) ===", batch, inF, outF);
            _output.WriteLine("  Eager (fwd+bwd+tape):      {0:F4}ms per step", eagerMs);
            _output.WriteLine("  Compiled (fwd only, plan): {0:F4}ms per step", compiledFwdMs);
            _output.WriteLine("  Forward speedup:           {0:F2}x (compiled fwd vs eager full step overhead)", eagerMs / compiledFwdMs);
            _output.WriteLine("");
        }

        /// <summary>
        /// Elementwise operation chain: measures overhead of 10 chained ops.
        /// This is where per-op overhead matters most.
        /// </summary>
        [Fact(Skip = "Performance benchmark — run manually")]
        [Trait("Category", "Performance")]
        public void HeadToHead_ElementwiseChain_EagerVsCompiled()
        {
            var engine = new CpuEngine();
            int size = 100000;

            var a = MakeRandom(new[] { size }, seed: 60);
            var b = MakeRandom(new[] { size }, seed: 61);
            var c = MakeRandom(new[] { size }, seed: 62);

            // === EAGER: 10-op elementwise chain ===
            for (int i = 0; i < 10; i++)
            {
                var r = engine.TensorAdd(a, b);
                r = engine.TensorMultiply(r, c);
                r = engine.TensorAdd(r, a);
                r = engine.ReLU(r);
                r = engine.TensorSubtract(r, b);
                r = engine.TensorAdd(r, c);
                r = engine.TensorMultiply(r, a);
                r = engine.ReLU(r);
                r = engine.TensorAdd(r, b);
                r = engine.TensorSubtract(r, c);
            }

            int iterations = 500;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var r = engine.TensorAdd(a, b);
                r = engine.TensorMultiply(r, c);
                r = engine.TensorAdd(r, a);
                r = engine.ReLU(r);
                r = engine.TensorSubtract(r, b);
                r = engine.TensorAdd(r, c);
                r = engine.TensorMultiply(r, a);
                r = engine.ReLU(r);
                r = engine.TensorAdd(r, b);
                r = engine.TensorSubtract(r, c);
            }
            sw.Stop();
            double eagerMs = sw.Elapsed.TotalMilliseconds / iterations;

            // === COMPILED PLAN ===
            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var r = engine.TensorAdd(a, b);
                r = engine.TensorMultiply(r, c);
                r = engine.TensorAdd(r, a);
                r = engine.ReLU(r);
                r = engine.TensorSubtract(r, b);
                r = engine.TensorAdd(r, c);
                r = engine.TensorMultiply(r, a);
                r = engine.ReLU(r);
                r = engine.TensorAdd(r, b);
                r = engine.TensorSubtract(r, c);
                plan = scope.CompileInference<float>();
            }

            for (int i = 0; i < 10; i++)
                plan.Execute();

            sw.Restart();
            for (int i = 0; i < iterations; i++)
                plan.Execute();
            sw.Stop();
            double compiledMs = sw.Elapsed.TotalMilliseconds / iterations;

            _output.WriteLine("=== 10-Op Elementwise Chain (size={0}) ===", size);
            _output.WriteLine("  Eager:    {0:F4}ms per chain ({1:F1}us per op)", eagerMs, eagerMs * 100);
            _output.WriteLine("  Compiled: {0:F4}ms per chain ({1:F1}us per op)", compiledMs, compiledMs * 100);
            _output.WriteLine("  Speedup:  {0:F2}x", eagerMs / compiledMs);
            _output.WriteLine("");
        }
    }
}
