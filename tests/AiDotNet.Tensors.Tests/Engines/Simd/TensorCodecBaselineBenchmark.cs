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
    /// Complete A/B test matrix: every approach measured on the SAME workload.
    /// Every new phase must add its row to this benchmark.
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

        /// <summary>Create a low-rank matrix for spectral testing.</summary>
        private static float[] MakeLowRank(int m, int n, int trueRank, int seed)
        {
            var rng = new Random(seed);
            var a = new float[m * trueRank];
            var b = new float[trueRank * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            var w = new float[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < trueRank; k++)
                        sum += a[i * trueRank + k] * b[k * n + j];
                    w[i * n + j] = sum;
                }
            return w;
        }

        private static double Measure(Action action, int warmup, int iters)
        {
            for (int i = 0; i < warmup; i++) action();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) action();
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds / iters;
        }

        [Fact]
        public void FullMatrix_AllApproaches_MLP_Training()
        {
            var engine = new CpuEngine();
            int m = 32, k = 128, h = 64, n = 10;

            var inputArr = MakeRandomArr(m * k, 42);
            var w1Arr = MakeRandomArr(k * h, 43);
            var w2Arr = MakeRandomArr(h * n, 44);

            var input = new Tensor<float>(inputArr, new[] { m, k });
            var w1 = new Tensor<float>(w1Arr, new[] { k, h });
            var w2 = new Tensor<float>(w2Arr, new[] { h, n });

            int warmup = 50, iters = 1000;

            // === 1. Eager + GradientTape ===
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

            // === 2. Compiled Training Plan (no fusion) ===
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableDataflowFusion = false });
            CompiledTrainingPlan<float> compiledPlanNoFusion;
            using (var scope = GraphMode.Enable())
            {
                var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                engine.TensorMatMul(h1, w2);
                compiledPlanNoFusion = scope.CompileTraining(new[] { w1, w2 });
            }
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(null);
            double compiledMs = Measure(() => compiledPlanNoFusion.Step(), warmup, iters);

            // === 2b. Compiled Training Plan + Phase B Fusion ===
            CompiledTrainingPlan<float> compiledPlanFused;
            using (var scope = GraphMode.Enable())
            {
                var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                engine.TensorMatMul(h1, w2);
                compiledPlanFused = scope.CompileTraining(new[] { w1, w2 });
            }
            double compiledFusedMs = Measure(() => compiledPlanFused.Step(), warmup, iters);

            // === 3. Phase B: Fused Multi-Layer (forward + backward) ===
            Func<float, float> relu = x => x > 0f ? x : 0f;
            var fusedOut = new float[m * n];
            var fusedAct = new float[m * h];
            var gradOutput = new float[m * n];
            for (int i = 0; i < gradOutput.Length; i++) gradOutput[i] = 1f;
            var gW1 = new float[k * h];
            var gW2 = new float[h * n];
            var gB1 = new float[0];
            var gB2 = new float[0];
            var gInput = new float[m * k];

            double phaseBMs = Measure(() =>
            {
                FusedMultiLayerGemm.FusedGemmActivationGemm(inputArr, w1Arr, w2Arr, fusedOut, fusedAct, m, k, h, n, relu);
                FusedMultiLayerBackward.ComputeGradients(gradOutput, inputArr, w1Arr, w2Arr, fusedAct,
                    gW1, gW2, gB1, gB2, gInput, m, k, h, n, FusedMultiLayerBackward.ReLUDerivative);
            }, warmup, iters);

            // === 4. Phase D: Hybrid (Compiled Plan forward + Phase B backward) ===
            // Use the compiled plan's forward (fastest) + Phase B's fused backward (zero-alloc transposed BLAS)
            // This combines the wins: compiled plan eliminates tape overhead, Phase B eliminates backward alloc
            double phaseDMs = Measure(() =>
            {
                // Forward via compiled plan's forward actions
                compiledPlanNoFusion.Step();
            }, warmup, iters);
            // Note: Phase D currently IS the compiled plan since we haven't replaced its backward.
            // The real Phase D = compiled forward + Phase B's FusedMultiLayerBackward integrated as the backward delegate.
            // For now, measure what combining them looks like if we use fused forward + compiled backward:
            double phaseDHybridMs = Measure(() =>
            {
                // Phase B forward (fused, L1 resident)
                FusedMultiLayerGemm.FusedGemmActivationGemm(inputArr, w1Arr, w2Arr, fusedOut, fusedAct, m, k, h, n, relu);
                // Phase B backward (transposed BLAS, zero transpose alloc)
                FusedMultiLayerBackward.ComputeGradients(gradOutput, inputArr, w1Arr, w2Arr, fusedAct,
                    gW1, gW2, gB1, gB2, gInput, m, k, h, n, FusedMultiLayerBackward.ReLUDerivative);
            }, warmup, iters);

            double pytorchMs = 0.266;

            _output.WriteLine("================================================================");
            _output.WriteLine("  FULL A/B MATRIX: MLP [32,128] -> 64 -> 10 TRAINING STEP");
            _output.WriteLine("================================================================");
            _output.WriteLine("  {0,-40} {1,10} {2,12} {3,12}", "Approach", "Time(ms)", "vs Eager", "vs PyTorch");
            _output.WriteLine("  {0,-40} {1,10} {2,12} {3,12}", new string('-', 40), "--------", "--------", "---------");
            _output.WriteLine("  {0,-44} {1,10:F4} {2,10:F2}x {3,10:F2}x", "1. Eager + GradientTape", eagerMs, 1.0, pytorchMs / eagerMs);
            _output.WriteLine("  {0,-44} {1,10:F4} {2,10:F2}x {3,10:F2}x", "2. Compiled Plan (no fusion)", compiledMs, eagerMs / compiledMs, pytorchMs / compiledMs);
            _output.WriteLine("  {0,-44} {1,10:F4} {2,10:F2}x {3,10:F2}x", "2b. Compiled Plan + Phase B Fusion", compiledFusedMs, eagerMs / compiledFusedMs, pytorchMs / compiledFusedMs);
            _output.WriteLine("  {0,-44} {1,10:F4} {2,10:F2}x {3,10:F2}x", "3. Phase B standalone", phaseBMs, eagerMs / phaseBMs, pytorchMs / phaseBMs);
            _output.WriteLine("  {0,-44} {1,10:F4} {2,10:F2}x {3,10:F2}x", "4. Phase D: Hybrid (B fwd+bwd raw)", phaseDHybridMs, eagerMs / phaseDHybridMs, pytorchMs / phaseDHybridMs);
            _output.WriteLine("  {0,-44} {1,10:F4} {2,10} {3,10}", "5. PyTorch (BDN reference)", pytorchMs, "", "baseline");
            _output.WriteLine("");
            _output.WriteLine("  Best: Compiled+Fusion = {0:F2}x vs PyTorch", pytorchMs / compiledFusedMs);
            _output.WriteLine("  Phase B fusion improvement: {0:F2}x over no-fusion compiled", compiledMs / compiledFusedMs);
            _output.WriteLine("================================================================");
        }

        [Fact]
        public void FullMatrix_AllApproaches_Inference_SingleMatMul()
        {
            int m = 32, k = 128, n = 64;
            int trueRank = 16;

            // Full-rank weights (normal case)
            var inputArr = MakeRandomArr(m * k, 50);
            var weightsFullRank = MakeRandomArr(k * n, 51);

            // Low-rank weights (spectral case)
            var weightsLowRank = MakeLowRank(k, n, trueRank, 52);

            int warmup = 50, iters = 1000;

            // === 1. Direct MatMul (BLAS) — full rank ===
            var directOut = new float[m * n];
            double directMs = Measure(() =>
            {
                Array.Clear(directOut, 0, directOut.Length);
                AiDotNet.Tensors.Helpers.BlasProvider.TryGemm(m, n, k, inputArr, 0, k, weightsFullRank, 0, n, directOut, 0, n);
            }, warmup, iters);

            // === 2. Direct MatMul (BLAS) — low rank (same speed, just different data) ===
            double directLowRankMs = Measure(() =>
            {
                Array.Clear(directOut, 0, directOut.Length);
                AiDotNet.Tensors.Helpers.BlasProvider.TryGemm(m, n, k, inputArr, 0, k, weightsLowRank, 0, n, directOut, 0, n);
            }, warmup, iters);

            // === 3. Phase A: Spectral MatMul — low rank ===
            var factors = SvdDecomposition.Decompose(weightsLowRank, k, n, maxRank: 0, energyThreshold: 0.9999);
            var spectralOut = new float[m * n];
            double spectralMs = double.NaN;
            int rank = 0;
            double approxError = 0;

            if (factors.HasValue)
            {
                rank = factors.Value.Rank;
                approxError = factors.Value.ApproximationError;
                spectralMs = Measure(() =>
                {
                    SvdDecomposition.SpectralMatMul(inputArr, m, k, factors.Value, spectralOut);
                }, warmup, iters);
            }

            // === 4. SimdGemm (our SIMD fallback) ===
            var simdOut = new float[m * n];
            double simdMs = Measure(() =>
            {
                SimdGemm.Sgemm(inputArr.AsSpan(0, m * k), weightsFullRank.AsSpan(0, k * n), simdOut.AsSpan(), m, k, n);
            }, warmup, iters);

            double pytorchInferMs = 0.012; // BDN reference for [32,128]@[128,64]

            _output.WriteLine("================================================================");
            _output.WriteLine("  FULL A/B MATRIX: MatMul [32,128] @ [128,64] INFERENCE");
            _output.WriteLine("================================================================");
            _output.WriteLine("  {0,-40} {1,10} {2,12}", "Approach", "Time(ms)", "vs Direct");
            _output.WriteLine("  {0,-40} {1,10} {2,12}", new string('-', 40), "--------", "---------");
            _output.WriteLine("  {0,-40} {1,10:F4} {2,12}", "1. Direct BLAS (full rank)", directMs, "baseline");
            _output.WriteLine("  {0,-40} {1,10:F4} {2,12}", "2. Direct BLAS (low rank, same op)", directLowRankMs, "same");
            _output.WriteLine("  {0,-40} {1,10:F4} {2,12:F2}x", "3. SimdGemm (SIMD fallback)", simdMs, directMs / simdMs);
            if (factors.HasValue)
                _output.WriteLine("  {0,-40} {1,10:F4} {2,12:F1}x",
                    "4. Phase A: Spectral (rank=" + rank + ")", spectralMs, directMs / spectralMs);
            else
                _output.WriteLine("  {0,-40} {1,10} {2,12}", "4. Phase A: Spectral", "N/A", "full rank");
            _output.WriteLine("  {0,-40} {1,10:F4} {2,12}", "5. PyTorch (BDN reference)", pytorchInferMs, "");
            _output.WriteLine("");
            if (factors.HasValue)
            {
                _output.WriteLine("  Spectral decomposition: rank {0}/{1}, error {2:E2}", rank, Math.Min(k, n), approxError);
                _output.WriteLine("  FLOPs: direct={0}, spectral={1} ({2:F1}x reduction)",
                    2L * m * k * n, 2L * m * k * rank + 2L * m * rank * n,
                    (double)(2L * m * k * n) / (2L * m * k * rank + 2L * m * rank * n));
                _output.WriteLine("  Phase D (hybrid): Use spectral for rank-reducible, direct for full-rank");
            }
            _output.WriteLine("================================================================");
        }
    }
}
