using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The members that used to be compositions of other backend members (TENS0008), each against a host reference: AddRelu,
/// AddSigmoid and AddGelu; Fma with its output aliasing the addend -- the in-place accumulate the old
/// Multiply-into-D-then-Add composition got wrong (2AB instead of AB + C); the split-complex element-wise family; the
/// batched dot product; the CSR SpMM with bias; the base-2/base-10 exponentials and logarithms, expm1 and log1p; the
/// MSE and cross-entropy gradients; and the softplus gradient written over its incoming gradient. Runs on every backend this
/// machine has; each one absent is skipped by name.
/// </summary>
public abstract class FusedElementwiseBackendTestsBase
{
    private readonly string _name;

    protected FusedElementwiseBackendTestsBase(string name) => _name = name;
    private const int Length = 4099;          // a float4 body and a scalar tail
    private const double RelativeTolerance = 2e-6;
    private const double AbsoluteTolerance = 1e-6;

    private static float[] Values(int salt) => Values(salt, Length);

    private static float[] Values(int salt, int length)
    {
        var rng = RandomHelper.CreateSeededRandom(1638 + salt);
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 8 - 4);
        return data;
    }

    private static void AssertClose(double expected, float actual, int index, string op)
        => Assert.True(Math.Abs(expected - actual) <= AbsoluteTolerance + RelativeTolerance * Math.Abs(expected),
            $"{op}[{index}]: expected {expected}, got {actual}");

    [SkippableFact]
    public void AddActivations_EqualTheActivatedSum()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var a = Values(1);
        var b = Values(2);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(Length);

        var cases = new (string Op, Action Run, Func<double, double> Reference)[]
        {
            ("AddRelu", () => backend.AddRelu(bufA, bufB, bufC, Length), x => x > 0 ? x : 0),
            ("AddSigmoid", () => backend.AddSigmoid(bufA, bufB, bufC, Length), x => 1 / (1 + Math.Exp(-x))),
            ("AddGelu", () => backend.AddGelu(bufA, bufB, bufC, Length),
                x => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))),
        };
        foreach (var (op, run, reference) in cases)
        {
            run();
            var c = backend.DownloadBuffer(bufC);
            for (int i = 0; i < Length; i++) AssertClose(reference((double)(a[i] + b[i])), c[i], i, op);
        }
    }

    [SkippableFact]
    public void Fma_IntoItsAddend_IsAProductPlusTheAddend()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var a = Values(3);
        var b = Values(4);
        var c = Values(5);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(c);

        backend.Fma(bufA, bufB, bufC, bufC, Length);   // D is C

        var d = backend.DownloadBuffer(bufC);
        for (int i = 0; i < Length; i++) AssertClose((double)a[i] * b[i] + c[i], d[i], i, "Fma");
    }

    [SkippableFact]
    public void SplitComplexFamily_EqualsTheComplexArithmetic()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var aR = Values(6); var aI = Values(7); var bR = Values(8); var bI = Values(9);
        using var bufAR = backend.AllocateBuffer(aR);
        using var bufAI = backend.AllocateBuffer(aI);
        using var bufBR = backend.AllocateBuffer(bR);
        using var bufBI = backend.AllocateBuffer(bI);
        using var oR = backend.AllocateBuffer(Length);
        using var oI = backend.AllocateBuffer(Length);

        void Check(string op, Func<int, (double Re, double Im)> reference)
        {
            var re = backend.DownloadBuffer(oR);
            var im = backend.DownloadBuffer(oI);
            for (int i = 0; i < Length; i++)
            {
                var (er, ei) = reference(i);
                AssertClose(er, re[i], i, op + ".re");
                AssertClose(ei, im[i], i, op + ".im");
            }
        }

        backend.SplitComplexMultiply(bufAR, bufAI, bufBR, bufBI, oR, oI, Length);
        Check("Multiply", i => ((double)aR[i] * bR[i] - (double)aI[i] * bI[i], (double)aR[i] * bI[i] + (double)aI[i] * bR[i]));
        backend.SplitComplexAdd(bufAR, bufAI, bufBR, bufBI, oR, oI, Length);
        Check("Add", i => ((double)aR[i] + bR[i], (double)aI[i] + bI[i]));
        backend.SplitComplexCrossSpectral(bufAR, bufAI, bufBR, bufBI, oR, oI, Length);
        Check("CrossSpectral", i => ((double)aR[i] * bR[i] + (double)aI[i] * bI[i], (double)aI[i] * bR[i] - (double)aR[i] * bI[i]));
        backend.SplitComplexConjugate(bufAR, bufAI, oR, oI, Length);
        Check("Conjugate", i => (aR[i], -aI[i]));
        backend.SplitComplexScale(bufAR, bufAI, oR, oI, -1.75f, Length);
        Check("Scale", i => (aR[i] * -1.75, aI[i] * -1.75));

        backend.SplitComplexMagnitudeSquared(bufAR, bufAI, oR, Length);
        var mag = backend.DownloadBuffer(oR);
        for (int i = 0; i < Length; i++) AssertClose((double)aR[i] * aR[i] + (double)aI[i] * aI[i], mag[i], i, "MagnitudeSquared");
    }

    [SkippableFact]
    public void BatchedDotProduct_EqualsEachRowsDot()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int rows = 5, width = 1027;   // more columns than a 256-thread group, and not a multiple of it
        var a = Values(10, rows * width); var b = Values(11, rows * width);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var result = backend.AllocateBuffer(rows);

        backend.BatchedDotProduct(bufA, bufB, result, rows, width);

        var r = backend.DownloadBuffer(result);
        for (int row = 0; row < rows; row++)
        {
            double expected = 0;
            for (int i = 0; i < width; i++) expected += (double)a[row * width + i] * b[row * width + i];
            Assert.True(Math.Abs(expected - r[row]) <= 1e-3 * (1 + Math.Abs(expected)), $"row {row}: expected {expected}, got {r[row]}");
        }
    }

    [SkippableFact]
    public void CsrSpMMBias_AddsTheColumnBiasToTheProduct()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        // A [3 x 4] in CSR: row 0 {0: 2, 3: -1}, row 1 empty, row 2 {1: 0.5, 2: 4}; B [4 x 5]; bias [5].
        const int m = 3, k = 4, n = 5;
        var values = new[] { 2f, -1f, 0.5f, 4f };
        var columns = new[] { 0, 3, 1, 2 };
        var rowPointers = new[] { 0, 2, 2, 4 };
        var dense = Values(12);
        var bias = new[] { 0.25f, -0.5f, 1f, 2f, -3f };
        using var bufValues = backend.AllocateBuffer(values);
        using var bufColumns = backend.AllocateIntBuffer(columns);
        using var bufRows = backend.AllocateIntBuffer(rowPointers);
        using var bufDense = backend.AllocateBuffer(dense);
        using var bufBias = backend.AllocateBuffer(bias);
        using var output = backend.AllocateBuffer(m * n);

        backend.CsrSpMMBias(bufValues, bufColumns, bufRows, bufDense, bufBias, output, m, k, n, values.Length);

        var c = backend.DownloadBuffer(output);
        for (int row = 0; row < m; row++)
            for (int col = 0; col < n; col++)
            {
                double expected = bias[col];
                for (int e = rowPointers[row]; e < rowPointers[row + 1]; e++) expected += (double)values[e] * dense[columns[e] * n + col];
                AssertClose(expected, c[row * n + col], row * n + col, "CsrSpMMBias");
            }
    }

    [SkippableFact]
    public void ExponentialsAndLogarithms_EqualTheReference()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var x = Values(13);                                // [-4, 4): exponentials
        var positive = new float[Length];                  // (0, 8]: logarithms
        var small = new float[Length];                     // |x| < 1e-4: where expm1 and log1p differ from exp - 1, log(1 + x)
        for (int i = 0; i < Length; i++)
        {
            positive[i] = Math.Abs(x[i]) * 2 + 1e-3f;
            small[i] = x[i] * 2.5e-5f;
        }
        using var bufX = backend.AllocateBuffer(x);
        using var bufPositive = backend.AllocateBuffer(positive);
        using var bufSmall = backend.AllocateBuffer(small);
        using var output = backend.AllocateBuffer(Length);

        void Check(string op, Action run, float[] input, Func<double, double> reference, double relative)
        {
            run();
            var y = backend.DownloadBuffer(output);
            for (int i = 0; i < Length; i++)
            {
                double expected = reference(input[i]);
                Assert.True(Math.Abs(expected - y[i]) <= 1e-30 + relative * Math.Abs(expected),
                    $"{op}[{i}]({input[i]:R}): expected {expected:R}, got {y[i]:R}");
            }
        }

        Check("Exp2", () => backend.Exp2(bufX, output, Length), x, v => Math.Pow(2, v), 1e-5);
        Check("Exp10", () => backend.Exp10(bufX, output, Length), x, v => Math.Pow(10, v), 1e-5);
        Check("ExpM1", () => backend.ExpM1(bufX, output, Length), x, v => Math.Exp(v) - 1, 1e-5);
        Check("Log2", () => backend.Log2(bufPositive, output, Length), positive, v => Math.Log(v, 2), 1e-5);
        Check("Log10", () => backend.Log10(bufPositive, output, Length), positive, v => Math.Log10(v), 1e-5);
        Check("Log1P", () => backend.Log1P(bufPositive, output, Length), positive, v => Math.Log(1 + v), 1e-5);
        // Near zero the answer is about x itself; exp(x) - 1 or log(1 + x) in float keeps only a few digits of it.
        Check("ExpM1(small)", () => backend.ExpM1(bufSmall, output, Length), small, v => v + v * v / 2, 1e-4);
        Check("Log1P(small)", () => backend.Log1P(bufSmall, output, Length), small, v => v - v * v / 2, 1e-4);
    }

    [SkippableFact]
    public void LossGradients_EqualTheReference()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int batch = 7, classes = 11, n = batch * classes;
        var predictions = Values(14, n);
        var targets = Values(15, n);
        using var bufP = backend.AllocateBuffer(predictions);
        using var bufT = backend.AllocateBuffer(targets);
        using var grad = backend.AllocateBuffer(n);

        backend.MseBackward(bufP, bufT, grad, n);
        var mse = backend.DownloadBuffer(grad);
        for (int i = 0; i < n; i++) AssertClose(2.0 * (predictions[i] - targets[i]) / n, mse[i], i, "MseBackward");

        backend.CrossEntropyBackward(bufP, bufT, grad, batch, classes);
        var ce = backend.DownloadBuffer(grad);
        // Logits and dense targets (the contract CUDA, OpenCL and CpuEngine share): (softmax(row) - target) / batch.
        for (int b = 0; b < batch; b++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < classes; c++) max = Math.Max(max, predictions[b * classes + c]);
            for (int c = 0; c < classes; c++) sum += Math.Exp(predictions[b * classes + c] - max);
            for (int c = 0; c < classes; c++)
            {
                int i = b * classes + c;
                AssertClose((Math.Exp(predictions[i] - max) / sum - targets[i]) / batch, ce[i], i, "CrossEntropyBackward");
            }
        }
    }

    [SkippableFact]
    public void CrossEntropyLoss_IsTheMeanNegativeLogSoftmaxOfTheTargets()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int batch = 7, classes = 11, n = batch * classes;
        var logits = Values(18, n);
        var targets = new float[n];   // dense, soft rows summing to 1
        var rng = RandomHelper.CreateSeededRandom(1657);
        for (int b = 0; b < batch; b++)
        {
            double rowSum = 0;
            for (int c = 0; c < classes; c++) rowSum += targets[b * classes + c] = (float)rng.NextDouble();
            for (int c = 0; c < classes; c++) targets[b * classes + c] = (float)(targets[b * classes + c] / rowSum);
        }
        using var bufLogits = backend.AllocateBuffer(logits);
        using var bufTargets = backend.AllocateBuffer(targets);

        float loss = backend.CrossEntropyLoss(bufLogits, bufTargets, batch, classes);

        double expected = 0;
        for (int b = 0; b < batch; b++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < classes; c++) max = Math.Max(max, logits[b * classes + c]);
            for (int c = 0; c < classes; c++) sum += Math.Exp(logits[b * classes + c] - max);
            double logSum = max + Math.Log(sum);
            for (int c = 0; c < classes; c++) expected -= targets[b * classes + c] * (logits[b * classes + c] - logSum);
        }
        expected /= batch;
        AssertClose(expected, loss, 0, "CrossEntropyLoss");
    }
    [SkippableFact]
    public void SoftplusBackward_IntoItsIncomingGradient_IsTheGradientTimesTheSigmoid()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var gradient = Values(16);
        var input = Values(17);
        for (int i = 0; i < Length; i++) input[i] *= 5;   // [-20, 20): both branches of the stable sigmoid
        using var bufGrad = backend.AllocateBuffer(gradient);
        using var bufInput = backend.AllocateBuffer(input);

        backend.SoftplusBackward(bufGrad, bufInput, bufGrad, Length);   // the result overwrites its own input gradient

        var result = backend.DownloadBuffer(bufGrad);
        for (int i = 0; i < Length; i++) AssertClose(gradient[i] / (1 + Math.Exp(-input[i])), result[i], i, "SoftplusBackward");
    }

    [SkippableFact]
    public void ExpM1AndLog1P_EdgeCases_MatchTheMathLibrary()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        // 86: exp is finite (2.2e37) but um1 * x overflows; 1e38: log(u) * x overflows; -1 and -inf are log1p's edge
        // and outside its domain.
        var x = new[] { 86f, 1e38f, -1f, float.NegativeInfinity, -2f, 0f, -0.5f };
        using var input = backend.AllocateBuffer(x);
        using var output = backend.AllocateBuffer(x.Length);

        backend.ExpM1(input, output, 1);
        float expm1At86 = backend.DownloadBuffer(output)[0];
        Assert.True(!float.IsInfinity(expm1At86) && Math.Abs(expm1At86 / Math.Exp(86) - 1) < 1e-5, $"ExpM1(86) = {expm1At86}");

        backend.Log1P(input, output, x.Length);
        var y = backend.DownloadBuffer(output);
        AssertClose(Math.Log(1 + 1e38), y[1], 1, "Log1P(1e38)");
        Assert.True(float.IsNegativeInfinity(y[2]), $"Log1P(-1) = {y[2]}, expected -inf");
        Assert.True(float.IsNaN(y[3]), $"Log1P(-inf) = {y[3]}, expected NaN");
        Assert.True(float.IsNaN(y[4]), $"Log1P(-2) = {y[4]}, expected NaN");
        Assert.Equal(0f, y[5]);
        AssertClose(Math.Log(0.5), y[6], 6, "Log1P(-0.5)");
    }

    [SkippableFact]
    public void CrossEntropyLoss_IsInvariantToAHugeCommonLogit()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int batch = 3, classes = 4, n = batch * classes;
        var logits = new float[n];
        var targets = new float[n];
        for (int i = 0; i < n; i++)
        {
            logits[i] = 1e8f;                                  // equal logits: softmax is uniform whatever the offset
            targets[i] = (i % classes == i / classes) ? 1f : 0f;
        }
        using var bufLogits = backend.AllocateBuffer(logits);
        using var bufTargets = backend.AllocateBuffer(targets);

        float loss = backend.CrossEntropyLoss(bufLogits, bufTargets, batch, classes);

        AssertClose(Math.Log(classes), loss, 0, "CrossEntropyLoss(1e8 logits)");
    }

    [SkippableTheory]
    [InlineData(32)]
    [InlineData(160)]   // wider than the HIP attention kernels hold in registers (128)
    public void AttentionBackward_IntoUsedDestinations_EqualsTheReference(int headDim)
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int batch = 1, heads = 2, seqQ = 4, seqK = 5;
        float scale = 1f / (float)Math.Sqrt(headDim);
        int bh = batch * heads;
        var q = Values(20, bh * seqQ * headDim);
        var k = Values(21, bh * seqK * headDim);
        var v = Values(22, bh * seqK * headDim);
        var dO = Values(23, bh * seqQ * headDim);
        var p = new float[bh * seqQ * seqK];                   // softmax(q k^T * scale), the forward's weights
        for (int h = 0; h < bh; h++)
            for (int i = 0; i < seqQ; i++)
            {
                double max = double.NegativeInfinity;
                var s = new double[seqK];
                for (int j = 0; j < seqK; j++)
                {
                    for (int d = 0; d < headDim; d++) s[j] += (double)q[(h * seqQ + i) * headDim + d] * k[(h * seqK + j) * headDim + d];
                    s[j] *= scale;
                    max = Math.Max(max, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < seqK; j++) sum += Math.Exp(s[j] - max);
                for (int j = 0; j < seqK; j++) p[(h * seqQ + i) * seqK + j] = (float)(Math.Exp(s[j] - max) / sum);
            }

        // Reference: dV = P^T dO; dS = P o (dO V^T - rowsum(P o dO V^T)) * scale; dQ = dS K; dK = dS^T Q.
        var dQ = new double[q.Length];
        var dK = new double[k.Length];
        var dV = new double[v.Length];
        for (int h = 0; h < bh; h++)
            for (int i = 0; i < seqQ; i++)
            {
                var dp = new double[seqK];
                double dot = 0;
                for (int j = 0; j < seqK; j++)
                {
                    for (int d = 0; d < headDim; d++) dp[j] += (double)dO[(h * seqQ + i) * headDim + d] * v[(h * seqK + j) * headDim + d];
                    dot += dp[j] * p[(h * seqQ + i) * seqK + j];
                }
                for (int j = 0; j < seqK; j++)
                {
                    double pij = p[(h * seqQ + i) * seqK + j];
                    double ds = pij * (dp[j] - dot) * scale;
                    for (int d = 0; d < headDim; d++)
                    {
                        dQ[(h * seqQ + i) * headDim + d] += ds * k[(h * seqK + j) * headDim + d];
                        dK[(h * seqK + j) * headDim + d] += ds * q[(h * seqQ + i) * headDim + d];
                        dV[(h * seqK + j) * headDim + d] += pij * dO[(h * seqQ + i) * headDim + d];
                    }
                }
            }

        using var bQ = backend.AllocateBuffer(q);
        using var bK = backend.AllocateBuffer(k);
        using var bV = backend.AllocateBuffer(v);
        using var bDO = backend.AllocateBuffer(dO);
        using var bP = backend.AllocateBuffer(p);
        // Destinations already holding values: a backward must overwrite them, not add to them.
        var stale = new float[q.Length + k.Length];
        for (int i = 0; i < stale.Length; i++) stale[i] = 7f;
        using var gQ = backend.AllocateBuffer(stale.AsSpan(0, q.Length).ToArray());
        using var gK = backend.AllocateBuffer(stale.AsSpan(0, k.Length).ToArray());
        using var gV = backend.AllocateBuffer(stale.AsSpan(0, v.Length).ToArray());

        backend.ScaledDotProductAttentionBackward(bDO, bQ, bK, bV, bP, gQ, gK, gV, batch, heads, seqQ, seqK, headDim, scale, false);

        void Check(string name, double[] expected, float[] actual)
        {
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 + 1e-3 * Math.Abs(expected[i]),
                    $"{name}[{i}] (headDim {headDim}): expected {expected[i]}, got {actual[i]}");
        }
        Check("dQ", dQ, backend.DownloadBuffer(gQ));
        Check("dK", dK, backend.DownloadBuffer(gK));
        Check("dV", dV, backend.DownloadBuffer(gV));
    }

    [SkippableFact]
    public void ApplyMelFilterbank_IsThePowerSpectrumTimesTheFilterbankTransposed()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int frames = 6, freqs = 33, mels = 10;
        var power = Values(40, frames * freqs);
        var filters = Values(41, mels * freqs);
        using var bufPower = backend.AllocateBuffer(power);
        using var bufFilters = backend.AllocateBuffer(filters);
        using var mel = backend.AllocateBuffer(frames * mels);

        backend.ApplyMelFilterbank(bufPower, bufFilters, mel, frames, freqs, mels);

        var y = backend.DownloadBuffer(mel);
        for (int b = 0; b < frames; b++)
            for (int m = 0; m < mels; m++)
            {
                double expected = 0;
                for (int f = 0; f < freqs; f++) expected += (double)power[b * freqs + f] * filters[m * freqs + f];
                Assert.True(Math.Abs(expected - y[b * mels + m]) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"mel[{b},{m}]: expected {expected}, got {y[b * mels + m]}");
            }
    }
    [SkippableTheory]
    [InlineData(1f, 0f)]
    [InlineData(0.5f, 2f)]
    public void SparseGemm24_EqualsThePrunedDenseProduct(float alpha, float beta)
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int M = 5, K = 12, N = 7;
        var dense = Values(50, M * K);
        var b = Values(51, K * N);
        var c0 = Values(52, M * N);
        var pruned = PruneTwoOfFour(dense, M, K);
        using var bufDense = backend.AllocateBuffer(dense);
        using var values = backend.AllocateBuffer(M * K / 2);
        using var indices = backend.AllocateByteBuffer(M * K / 4);   // the contract: one byte of offsets per block of four
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(c0);
        backend.Enforce2x4Sparsity(bufDense, values, indices, M, K);

        backend.SparseGemm(values, indices, bufB, bufC, M, N, K, alpha, beta);

        var y = backend.DownloadBuffer(bufC);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double expected = alpha * Dot(pruned, b, i, j, K, N) + beta * c0[i * N + j];
                Assert.True(Math.Abs(expected - y[i * N + j]) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"C[{i},{j}]: expected {expected}, got {y[i * N + j]}");
            }

        var biasValues = Values(53, N);
        using var bias = backend.AllocateBuffer(biasValues);
        using var relu = backend.SparseGemmBiasRelu(values, indices, bufB, bias, M, N, K);
        var r = backend.DownloadBuffer(relu);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double expected = Math.Max(0, Dot(pruned, b, i, j, K, N) + biasValues[j]);
                Assert.True(Math.Abs(expected - r[i * N + j]) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"relu[{i},{j}]: expected {expected}, got {r[i * N + j]}");
            }
    }

    private static float[] PruneTwoOfFour(float[] dense, int rows, int cols)
    {
        var pruned = new float[dense.Length];
        for (int block = 0; block < rows * cols / 4; block++)
        {
            int at = block * 4;
            var order = new[] { 0, 1, 2, 3 };
            Array.Sort(order, (x, y) => Math.Abs(dense[at + y]).CompareTo(Math.Abs(dense[at + x])));
            pruned[at + order[0]] = dense[at + order[0]];
            pruned[at + order[1]] = dense[at + order[1]];
        }
        return pruned;
    }

    private static double Dot(float[] a, float[] b, int i, int j, int k, int n)
    {
        double sum = 0;
        for (int t = 0; t < k; t++) sum += (double)a[i * k + t] * b[t * n + j];
        return sum;
    }

    [SkippableFact]
    public void GemmBiasEpilogues_EqualTheActivatedProduct()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int M = 9, K = 13, N = 11;
        var a = Values(60, M * K);
        var b = Values(61, K * N);
        var bias = Values(62, N);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufBias = backend.AllocateBuffer(bias);
        var epilogues = new (string Name, Func<IGpuBuffer> Run, Func<double, double> Activation)[]
        {
            ("GemmBias", () => backend.GemmBias(bufA, bufB, bufBias, M, N, K), x => x),
            ("GemmBiasRelu", () => backend.GemmBiasRelu(bufA, bufB, bufBias, M, N, K), x => Math.Max(0, x)),
            ("GemmBiasGelu", () => backend.GemmBiasGelu(bufA, bufB, bufBias, M, N, K),
                x => 0.5 * x * (1 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))),
            ("GemmBiasSigmoid", () => backend.GemmBiasSigmoid(bufA, bufB, bufBias, M, N, K), x => 1 / (1 + Math.Exp(-x))),
            ("GemmBiasTanh", () => backend.GemmBiasTanh(bufA, bufB, bufBias, M, N, K), Math.Tanh),
            ("GemmBiasSwish", () => backend.GemmBiasSwish(bufA, bufB, bufBias, M, N, K), x => x / (1 + Math.Exp(-x))),
            ("GemmBiasLeakyRelu", () => backend.GemmBiasLeakyRelu(bufA, bufB, bufBias, M, N, K, 0.2f), x => x > 0 ? x : 0.2 * x),
        };
        foreach (var (name, run, activation) in epilogues)
        {
            using var output = run();
            var y = backend.DownloadBuffer(output);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    double expected = activation(Dot(a, b, i, j, K, N) + bias[j]);
                    Assert.True(Math.Abs(expected - y[i * N + j]) <= 1e-4 * (1 + Math.Abs(expected)),
                        $"{name}[{i},{j}]: expected {expected}, got {y[i * N + j]}");
                }
        }
    }

    [SkippableTheory]
    [InlineData(16)]
    [InlineData(64)]
    public void Rfft_WritesOnlyTheNonNegativeFrequencyBins(int n)
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var x = Values(70, n);
        const float Sentinel = 7f;
        using var input = backend.AllocateBuffer(x);
        // n slots, so a transform that writes past the n / 2 + 1 bins of the contract shows up in the slots after them.
        using var re = backend.AllocateBuffer(Filled(n, Sentinel));
        using var im = backend.AllocateBuffer(Filled(n, Sentinel));

        backend.RFFT(input, re, im, n);

        var yr = backend.DownloadBuffer(re);
        var yi = backend.DownloadBuffer(im);
        for (int k = 0; k <= n / 2; k++)
        {
            double sr = 0, si = 0;
            for (int j = 0; j < n; j++)
            {
                double angle = -2 * Math.PI * ((long)k * j % n) / n;
                sr += x[j] * Math.Cos(angle);
                si += x[j] * Math.Sin(angle);
            }
            Assert.True(Math.Abs(sr - yr[k]) <= 1e-3 * (1 + Math.Abs(sr)), $"re[{k}]: expected {sr}, got {yr[k]}");
            Assert.True(Math.Abs(si - yi[k]) <= 1e-3 * (1 + Math.Abs(si)), $"im[{k}]: expected {si}, got {yi[k]}");
        }
        for (int k = n / 2 + 1; k < n; k++)
        {
            Assert.Equal(Sentinel, yr[k]);
            Assert.Equal(Sentinel, yi[k]);
        }
    }

    private static float[] Filled(int length, float value)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = value;
        return data;
    }

    [SkippableTheory]
    [InlineData(8, 16, false, false)]
    [InlineData(8, 16, true, false)]
    [InlineData(8, 16, false, true)]   // in place
    [InlineData(6, 10, false, false)]  // not a power of two
    [InlineData(6, 10, true, true)]
    public void Fft2D_EqualsTheTwoDimensionalDft(int height, int width, bool inverse, bool inPlace)
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        int count = height * width;
        var xr = Values(80, count);
        var xi = Values(81, count);
        using var inRe = backend.AllocateBuffer(xr);
        using var inIm = backend.AllocateBuffer(xi);
        using var outRe = backend.AllocateBuffer(count);
        using var outIm = backend.AllocateBuffer(count);
        var re = inPlace ? inRe : outRe;
        var im = inPlace ? inIm : outIm;

        backend.FFT2D(inRe, inIm, re, im, height, width, inverse);

        var yr = backend.DownloadBuffer(re);
        var yi = backend.DownloadBuffer(im);
        double sign = inverse ? 1 : -1, scale = inverse ? 1.0 / count : 1.0;
        for (int ky = 0; ky < height; ky++)
            for (int kx = 0; kx < width; kx++)
            {
                double sr = 0, si = 0;
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        double angle = sign * 2 * Math.PI * ((double)(ky * y % height) / height + (double)(kx * x % width) / width);
                        double c = Math.Cos(angle), s = Math.Sin(angle);
                        sr += xr[y * width + x] * c - xi[y * width + x] * s;
                        si += xr[y * width + x] * s + xi[y * width + x] * c;
                    }
                sr *= scale;
                si *= scale;
                int at = ky * width + kx;
                Assert.True(Math.Abs(sr - yr[at]) <= 1e-3 * (1 + Math.Abs(sr)), $"re[{ky},{kx}]: expected {sr}, got {yr[at]}");
                Assert.True(Math.Abs(si - yi[at]) <= 1e-3 * (1 + Math.Abs(si)), $"im[{ky},{kx}]: expected {si}, got {yi[at]}");
            }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void HalfGemm_RespectsAccumulationPrecision(bool accumulateFp32)
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var half = backend as IGpuHalfPrecisionBackend;
        Skip.If(half is null || !half.SupportsHgemm, _name + " has no half-precision GEMM.");
        const int M = 19, K = 23, N = 17;   // not multiples of a tile
        float[] Inputs(int seed, int count)
        {
            if (accumulateFp32) return Values(seed, count);
            // HGEMM promises FP16 accumulation, not a single rounding after an FP32 dot.
            // Integers in [-4,4] keep every product and partial sum exactly representable
            // in FP16 (at most 23 * 16), independently of the kernel's reduction order.
            var random = new Random(seed);
            var values = new float[count];
            for (int index = 0; index < count; index++) values[index] = random.Next(-4, 5);
            return values;
        }
        using var a32 = backend.AllocateBuffer(Inputs(90, M * K));
        using var b32 = backend.AllocateBuffer(Inputs(91, K * N));
        using var a16 = backend.AllocateByteBuffer(M * K * 2);
        using var b16 = backend.AllocateByteBuffer(K * N * 2);
        using var c16 = backend.AllocateByteBuffer(M * N * 2);
        using var c32 = backend.AllocateBuffer(M * N);
        using var aBack = backend.AllocateBuffer(M * K);
        using var bBack = backend.AllocateBuffer(K * N);
        backend.ConvertToFp16(a32, a16, M * K);
        backend.ConvertToFp16(b32, b16, K * N);
        backend.ConvertToFp32(a16, aBack, M * K);
        backend.ConvertToFp32(b16, bBack, K * N);
        var ah = backend.DownloadBuffer(aBack);
        var bh = backend.DownloadBuffer(bBack);

        if (accumulateFp32)
            half.GemmFp16In32fOut(a16, b16, c32, M, N, K);
        else
        {
            half.Hgemm(a16, b16, c16, M, N, K);
            backend.ConvertToFp32(c16, c32, M * N);
        }

        var y = backend.DownloadBuffer(c32);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double expected = Dot(ah, bh, i, j, K, N);
                if (accumulateFp32)
                    Assert.True(Math.Abs(expected - y[i * N + j]) <= 1e-5 * Math.Abs(expected) + 1e-5,
                        $"C[{i},{j}]: expected {expected}, got {y[i * N + j]}");
                else
                    Assert.Equal((float)expected, y[i * N + j]);
            }
    }

    // The backend, or null when unavailable; the returned disposable disposes a backend the test created.
    private static IDisposable? Create(string name, out IDirectGpuBackend? backend)
    {
        backend = null;
        try
        {
            IDirectGpuBackend candidate = name switch
            {
                "cuda" => new CudaBackend(),
                "hip" => new HipBackend(),
                "opencl" => new OpenClBackend(),
                "vulkan" => VulkanBackend.Instance,   // the process-wide instance; never disposed here
                "metal" => new MetalBackend(),
#if NET7_0_OR_GREATER
                "webgpu" => DawnHostedWebGpu.BackendOrNull()
                            ?? throw new PlatformNotSupportedException("Dawn is not set up (AIDOTNET_WEBGPU_NODE_DIR)."),
#endif
                _ => throw new PlatformNotSupportedException(name),
            };
            if (candidate is VulkanBackend vulkan && !vulkan.IsAvailable) vulkan.Initialize();
            bool shared = name is "vulkan" or "webgpu";
            if (!candidate.IsAvailable)
            {
                if (!shared) (candidate as IDisposable)?.Dispose();
                return null;
            }
            backend = candidate;
            return shared ? null : candidate as IDisposable;
        }
        catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
        {
            return null;   // no driver for this backend on this machine
        }
    }
}

// One class per backend, so each backend's CI lane selects its own (the Metal lane runs FullyQualifiedName~Metal).
[Collection("DirectGpuSerial")]
public sealed class CudaFusedElementwiseTests : FusedElementwiseBackendTestsBase { public CudaFusedElementwiseTests() : base("cuda") { } }

[Collection("DirectGpuSerial")]
public sealed class HipFusedElementwiseTests : FusedElementwiseBackendTestsBase { public HipFusedElementwiseTests() : base("hip") { } }

[Collection("DirectGpuSerial")]
public sealed class OpenClFusedElementwiseTests : FusedElementwiseBackendTestsBase { public OpenClFusedElementwiseTests() : base("opencl") { } }

[Collection("DirectGpuSerial")]
public sealed class VulkanFusedElementwiseTests : FusedElementwiseBackendTestsBase { public VulkanFusedElementwiseTests() : base("vulkan") { } }

[Collection("DirectGpuSerial")]
public sealed class MetalFusedElementwiseTests : FusedElementwiseBackendTestsBase { public MetalFusedElementwiseTests() : base("metal") { } }

[Collection("DirectGpuSerial")]
public sealed class WebGpuFusedElementwiseTests : FusedElementwiseBackendTestsBase { public WebGpuFusedElementwiseTests() : base("webgpu") { } }
