using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd
{
    public class FusedMultiLayerGemmTests
    {
        private readonly ITestOutputHelper _output;
        private readonly CpuEngine _engine = new CpuEngine();

        public FusedMultiLayerGemmTests(ITestOutputHelper output) => _output = output;

        private static float[] MakeRandom(int length, int seed)
        {
            var rng = new Random(seed);
            var data = new float[length];
            for (int i = 0; i < length; i++)
                data[i] = (float)(rng.NextDouble() * 2 - 1);
            return data;
        }

        [Theory]
        [InlineData(8, 32, 16, 10)]    // Small MLP
        [InlineData(32, 128, 64, 10)]   // Medium MLP (main benchmark shape)
        [InlineData(1, 16, 8, 4)]       // Single sample
        [InlineData(64, 256, 128, 32)]  // Larger
        public void FusedForward_MatchesUnfused(int m, int k, int h, int n)
        {
            var input = MakeRandom(m * k, 42);
            var w1 = MakeRandom(k * h, 43);
            var w2 = MakeRandom(h * n, 44);

            // Unfused: GEMM1 → ReLU → GEMM2
            var hidden = new float[m * h];
            SimdGemm.Sgemm(input.AsSpan(0, m * k), w1.AsSpan(0, k * h), hidden.AsSpan(), m, k, h);
            for (int i = 0; i < m * h; i++)
                hidden[i] = hidden[i] > 0 ? hidden[i] : 0; // ReLU
            var unfusedOutput = new float[m * n];
            SimdGemm.Sgemm(hidden.AsSpan(0, m * h), w2.AsSpan(0, h * n), unfusedOutput.AsSpan(), m, h, n);

            // Fused
            var fusedOutput = new float[m * n];
            var fusedActivated = new float[m * h];
            Func<float, float> relu = x => x > 0f ? x : 0f;
            FusedMultiLayerGemm.FusedGemmActivationGemm(
                input, w1, w2, fusedOutput, fusedActivated, m, k, h, n, relu);

            // Compare outputs
            double maxDiff = 0;
            for (int i = 0; i < m * n; i++)
            {
                double diff = Math.Abs(unfusedOutput[i] - fusedOutput[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            _output.WriteLine("FusedForward [{0}x{1}->{2}->{3}]: max diff = {4:E3}", m, k, h, n, maxDiff);
            Assert.True(maxDiff < 1e-4,
                "Fused output diverged from unfused: max diff = " + maxDiff);

            // Compare activated intermediates
            double maxActDiff = 0;
            for (int i = 0; i < m * h; i++)
            {
                double diff = Math.Abs(hidden[i] - fusedActivated[i]);
                if (diff > maxActDiff) maxActDiff = diff;
            }
            Assert.True(maxActDiff < 1e-4,
                "Fused activated diverged: max diff = " + maxActDiff);
        }

        [Theory]
        [InlineData(8, 32, 16, 10)]
        [InlineData(32, 128, 64, 10)]
        public void FusedBackward_MatchesUnfused(int m, int k, int h, int n)
        {
            var input = MakeRandom(m * k, 50);
            var w1 = MakeRandom(k * h, 51);
            var w2 = MakeRandom(h * n, 52);
            var gradOutput = MakeRandom(m * n, 53);

            // Forward (fused) to get activated intermediate
            var output = new float[m * n];
            var activated = new float[m * h];
            Func<float, float> relu = x => x > 0f ? x : 0f;
            FusedMultiLayerGemm.FusedGemmActivationGemm(
                input, w1, w2, output, activated, m, k, h, n, relu);

            // Fused backward
            var fusedGradW1 = new float[k * h];
            var fusedGradW2 = new float[h * n];
            var fusedGradB1 = new float[h];
            var fusedGradB2 = new float[n];
            var fusedGradInput = new float[m * k];
            FusedMultiLayerBackward.ComputeGradients(
                gradOutput, input, w1, w2, activated,
                fusedGradW1, fusedGradW2, fusedGradB1, fusedGradB2, fusedGradInput,
                m, k, h, n, FusedMultiLayerBackward.ReLUDerivative);

            // Unfused backward (manual)
            // Step 1: gradW2 = activated^T @ gradOutput
            var unfusedGradW2 = new float[h * n];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int row = 0; row < m; row++)
                        sum += activated[row * h + i] * gradOutput[row * n + j];
                    unfusedGradW2[i * n + j] = sum;
                }

            // Step 2: grad_h = gradOutput @ W2^T
            var grad_h = new float[m * h];
            for (int row = 0; row < m; row++)
                for (int j = 0; j < h; j++)
                {
                    float sum = 0;
                    for (int col = 0; col < n; col++)
                        sum += gradOutput[row * n + col] * w2[j * n + col];
                    grad_h[row * h + j] = sum;
                }

            // Step 3: Apply ReLU derivative
            for (int i = 0; i < m * h; i++)
                grad_h[i] *= activated[i] > 0 ? 1f : 0f;

            // Step 4: gradW1 = input^T @ grad_h
            var unfusedGradW1 = new float[k * h];
            for (int i = 0; i < k; i++)
                for (int j = 0; j < h; j++)
                {
                    float sum = 0;
                    for (int row = 0; row < m; row++)
                        sum += input[row * k + i] * grad_h[row * h + j];
                    unfusedGradW1[i * h + j] = sum;
                }

            // Compare
            double maxW1Diff = MaxDiff(unfusedGradW1, fusedGradW1);
            double maxW2Diff = MaxDiff(unfusedGradW2, fusedGradW2);

            _output.WriteLine("FusedBackward [{0}x{1}->{2}->{3}]: gradW1 max diff = {4:E3}, gradW2 max diff = {5:E3}",
                m, k, h, n, maxW1Diff, maxW2Diff);

            Assert.True(maxW1Diff < 1e-3, "gradW1 diverged: " + maxW1Diff);
            Assert.True(maxW2Diff < 1e-3, "gradW2 diverged: " + maxW2Diff);
        }

        [Fact(Skip = "Performance benchmark — run manually")]
        public void FusedForward_Performance_VsUnfused()
        {
            int m = 32, k = 128, h = 64, n = 10;
            var input = MakeRandom(m * k, 60);
            var w1 = MakeRandom(k * h, 61);
            var w2 = MakeRandom(h * n, 62);
            Func<float, float> relu = x => x > 0f ? x : 0f;

            // Warmup
            var output = new float[m * n];
            var activated = new float[m * h];
            for (int i = 0; i < 20; i++)
                FusedMultiLayerGemm.FusedGemmActivationGemm(input, w1, w2, output, activated, m, k, h, n, relu);

            // Benchmark fused
            int iters = 500;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                FusedMultiLayerGemm.FusedGemmActivationGemm(input, w1, w2, output, activated, m, k, h, n, relu);
            sw.Stop();
            double fusedMs = sw.Elapsed.TotalMilliseconds / iters;

            // Benchmark unfused
            var hidden = new float[m * h];
            var unfusedOut = new float[m * n];
            for (int i = 0; i < 20; i++)
            {
                SimdGemm.Sgemm(input.AsSpan(0, m * k), w1.AsSpan(0, k * h), hidden.AsSpan(), m, k, h);
                for (int j = 0; j < m * h; j++) hidden[j] = hidden[j] > 0 ? hidden[j] : 0;
                SimdGemm.Sgemm(hidden.AsSpan(0, m * h), w2.AsSpan(0, h * n), unfusedOut.AsSpan(), m, h, n);
            }

            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                SimdGemm.Sgemm(input.AsSpan(0, m * k), w1.AsSpan(0, k * h), hidden.AsSpan(), m, k, h);
                for (int j = 0; j < m * h; j++) hidden[j] = hidden[j] > 0 ? hidden[j] : 0;
                SimdGemm.Sgemm(hidden.AsSpan(0, m * h), w2.AsSpan(0, h * n), unfusedOut.AsSpan(), m, h, n);
            }
            sw.Stop();
            double unfusedMs = sw.Elapsed.TotalMilliseconds / iters;

            double speedup = unfusedMs / fusedMs;
            _output.WriteLine("Fused: {0:F4}ms, Unfused: {1:F4}ms, Speedup: {2:F2}x", fusedMs, unfusedMs, speedup);
        }

        private static double MaxDiff(float[] a, float[] b)
        {
            double max = 0;
            int len = Math.Min(a.Length, b.Length);
            for (int i = 0; i < len; i++)
            {
                double d = Math.Abs(a[i] - b[i]);
                if (d > max) max = d;
            }
            return max;
        }
    }
}
