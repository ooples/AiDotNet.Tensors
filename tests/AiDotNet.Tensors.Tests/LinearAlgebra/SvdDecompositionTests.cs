using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra
{
    public class SvdDecompositionTests
    {
        private readonly ITestOutputHelper _output;

        public SvdDecompositionTests(ITestOutputHelper output) => _output = output;

        private static float[] MakeRandom(int length, int seed)
        {
            var rng = new Random(seed);
            var data = new float[length];
            for (int i = 0; i < length; i++)
                data[i] = (float)(rng.NextDouble() * 2 - 1);
            return data;
        }

        /// <summary>Create a low-rank matrix: W = A[m,r] @ B[r,n] so it has rank at most r.</summary>
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

        [Fact]
        public void SVD_LowRankMatrix_DetectsRank()
        {
            int m = 64, n = 32, trueRank = 8;
            var w = MakeLowRank(m, n, trueRank, 42);

            var factors = SvdDecomposition.Decompose(w, m, n, maxRank: 0, energyThreshold: 0.999);

            Assert.NotNull(factors);
            _output.WriteLine("True rank: {0}, Detected rank: {1}, Error: {2:E3}",
                trueRank, factors.Value.Rank, factors.Value.ApproximationError);

            // Should detect close to the true rank
            Assert.True(factors.Value.Rank <= trueRank + 2,
                "Rank should be close to true rank " + trueRank + " but got " + factors.Value.Rank);
            Assert.True(factors.Value.ApproximationError < 0.01,
                "Error should be small for low-rank matrix: " + factors.Value.ApproximationError);
        }

        [Fact]
        public void SVD_Reconstruction_AccurateForLowRank()
        {
            int m = 32, n = 16, trueRank = 4;
            var w = MakeLowRank(m, n, trueRank, 43);

            var factors = SvdDecomposition.Decompose(w, m, n, maxRank: trueRank + 2, energyThreshold: 0.9999);
            Assert.NotNull(factors);

            // Reconstruct: W_approx = leftFactor @ rightFactor
            var f = factors.Value;
            var reconstructed = new float[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < f.Rank; k++)
                        sum += f.LeftFactor[i * f.Rank + k] * f.RightFactor[k * n + j];
                    reconstructed[i * n + j] = sum;
                }

            double maxDiff = 0;
            for (int i = 0; i < m * n; i++)
            {
                double diff = Math.Abs(w[i] - reconstructed[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            _output.WriteLine("Rank {0}: reconstruction max diff = {1:E3}", f.Rank, maxDiff);
            Assert.True(maxDiff < 1e-3, "Reconstruction too inaccurate: " + maxDiff);
        }

        [Fact]
        public void SpectralMatMul_MatchesDirect()
        {
            int m = 32, n = 16, trueRank = 4;
            int batchSize = 8;
            var w = MakeLowRank(m, n, trueRank, 44);
            var x = MakeRandom(batchSize * m, 45);

            var factors = SvdDecomposition.Decompose(w, m, n, maxRank: trueRank + 2, energyThreshold: 0.9999);
            Assert.NotNull(factors);

            // Direct: y = x @ W
            var directOutput = new float[batchSize * n];
            for (int i = 0; i < batchSize; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < m; k++)
                        sum += x[i * m + k] * w[k * n + j];
                    directOutput[i * n + j] = sum;
                }

            // Spectral: y = (x @ leftFactor) @ rightFactor
            var spectralOutput = new float[batchSize * n];
            SvdDecomposition.SpectralMatMul(x, batchSize, m, factors.Value, spectralOutput);

            double maxDiff = 0;
            for (int i = 0; i < batchSize * n; i++)
            {
                double diff = Math.Abs(directOutput[i] - spectralOutput[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            _output.WriteLine("SpectralMatMul: max diff = {0:E3}, rank = {1}", maxDiff, factors.Value.Rank);
            Assert.True(maxDiff < 1e-2, "SpectralMatMul too inaccurate: " + maxDiff);
        }

        [Fact(Skip = "Performance benchmark — run manually")]
        public void SpectralMatMul_Performance_VsDirect()
        {
            int m = 128, n = 64, trueRank = 16;
            int batchSize = 32;
            var w = MakeLowRank(m, n, trueRank, 46);
            var x = MakeRandom(batchSize * m, 47);

            var factors = SvdDecomposition.Decompose(w, m, n, maxRank: trueRank + 2, energyThreshold: 0.9999);
            Assert.NotNull(factors);

            var directOutput = new float[batchSize * n];
            var spectralOutput = new float[batchSize * n];

            int warmup = 50, iters = 1000;

            // Warmup
            for (int i = 0; i < warmup; i++)
            {
                AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(x.AsSpan(0, batchSize * m), w.AsSpan(0, m * n),
                    directOutput.AsSpan(), batchSize, m, n);
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(x.AsSpan(0, batchSize * m), w.AsSpan(0, m * n),
                    directOutput.AsSpan(), batchSize, m, n);
            }
            sw.Stop();
            double directMs = sw.Elapsed.TotalMilliseconds / iters;

            // Warmup spectral
            for (int i = 0; i < warmup; i++)
                SvdDecomposition.SpectralMatMul(x, batchSize, m, factors.Value, spectralOutput);

            sw.Restart();
            for (int i = 0; i < iters; i++)
                SvdDecomposition.SpectralMatMul(x, batchSize, m, factors.Value, spectralOutput);
            sw.Stop();
            double spectralMs = sw.Elapsed.TotalMilliseconds / iters;

            _output.WriteLine("[128,64] rank-{0}: Direct={1:F4}ms, Spectral={2:F4}ms, Speedup={3:F2}x",
                factors.Value.Rank, directMs, spectralMs, directMs / spectralMs);
        }

        [Theory]
        [InlineData(256, 128, 16)]
        [InlineData(512, 256, 32)]
        public void SpectralMatMul_LargerSizes_VsDirect(int k, int n, int trueRank)
        {
            int batchSize = 32;
            var w = MakeLowRank(k, n, trueRank, 60 + k);
            var x = MakeRandom(batchSize * k, 61 + k);

            var factors = SvdDecomposition.Decompose(w, k, n, maxRank: 0, energyThreshold: 0.9999);
            Assert.NotNull(factors);

            int warmup = 30, iters = 500;
            var directOut = new float[batchSize * n];
            var spectralOut = new float[batchSize * n];

            for (int i = 0; i < warmup; i++)
                AiDotNet.Tensors.Helpers.BlasProvider.TryGemm(batchSize, n, k, x, 0, k, w, 0, n, directOut, 0, n);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                AiDotNet.Tensors.Helpers.BlasProvider.TryGemm(batchSize, n, k, x, 0, k, w, 0, n, directOut, 0, n);
            sw.Stop();
            double directMs = sw.Elapsed.TotalMilliseconds / iters;

            for (int i = 0; i < warmup; i++)
                SvdDecomposition.SpectralMatMul(x, batchSize, k, factors.Value, spectralOut);
            sw.Restart();
            for (int i = 0; i < iters; i++)
                SvdDecomposition.SpectralMatMul(x, batchSize, k, factors.Value, spectralOut);
            sw.Stop();
            double spectralMs = sw.Elapsed.TotalMilliseconds / iters;

            double flopRedux = (double)(2L * batchSize * k * n) / (2L * batchSize * k * factors.Value.Rank + 2L * batchSize * factors.Value.Rank * n);
            _output.WriteLine("[{0},{1}] rank-{2}: Direct={3:F4}ms, Spectral={4:F4}ms, Speedup={5:F2}x, FLOPs={6:F1}x reduction",
                k, n, factors.Value.Rank, directMs, spectralMs, directMs / spectralMs, flopRedux);
        }

        [Fact]
        public void SVD_FullRankMatrix_ReturnsNull()
        {
            // A full-rank random matrix shouldn't benefit from spectral decomposition
            int m = 32, n = 32;
            var w = MakeRandom(m * n, 48);

            var factors = SvdDecomposition.Decompose(w, m, n, maxRank: 0, energyThreshold: 0.999);

            // Full rank matrix: rank won't be reducible by 2x
            // Decompose returns null when rank >= minDim/2
            _output.WriteLine("Full rank matrix: factors is {0}", factors == null ? "null (correct)" : "not null (rank=" + factors.Value.Rank + ")");
            Assert.Null(factors);
        }
    }
}
