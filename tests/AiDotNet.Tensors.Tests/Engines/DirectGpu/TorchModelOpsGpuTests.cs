using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The 17 model-level ops of <c>CpuEngine.TorchModelOps</c> on the GPU engine, GPU against CPU: the forward and, under a
/// tape, every input's gradient, on each backend this machine has. Every op must launch device kernels, record no host
/// fallback, and (with device-resident inputs) keep its forward and backward on the device: no device-to-host
/// download inside the step. The activations are fed the edge values their kernels special-case (signed zeros,
/// magnitudes past float's square root, infinities, NaN) alongside random values.
/// </summary>
public abstract class TorchModelOpsGpuTestsBase
{
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected TorchModelOpsGpuTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchModelOpsBackend && gpu.GetBackend() is ITorchParityElementwiseBackend,
            "This backend has no model-op device kernels.");
        return gpu;
    }

    private static Tensor<float> Rand(int seed, double scale, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static Tensor<float> Copy(Tensor<float> t) => new Tensor<float>(t.ToArray(), (int[])t._shape.Clone());

    private static void AssertClose(string what, float[] cpu, float[] gpu, double tolerance)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{what}[{i}]: cpu {c:R} gpu {g:R}"); continue; }
            if (float.IsInfinity(c) || float.IsInfinity(g)) { Assert.True(c == g, $"{what}[{i}]: cpu {c:R} gpu {g:R}"); continue; }
            Assert.True(Math.Abs(c - g) <= tolerance * Math.Max(1.0, Math.Abs(c)), $"{what}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    /// <summary>
    /// Runs <paramref name="op"/> on the CPU and on the GPU (inputs device-resident there), each under its own tape with
    /// loss = sum(y * r), and compares y and every input's gradient. On the GPU the step must launch kernels, record no
    /// fallback and download nothing before the results are read.
    /// </summary>
    private static void CheckForwardAndGradients(DirectGpuTensorEngine gpu, string name, Func<IEngine, Tensor<float>[], Tensor<float>> op,
        Tensor<float>[] inputs, double forwardTolerance, double gradientTolerance, bool differentiable = true)
    {
        var cpu = new CpuEngine();
        var cpuInputs = inputs.Select(Copy).ToArray();
        float[] cpuY;
        float[][] cpuGrads;
        using (var tape = new GradientTape<float>())
        {
            var y = op(cpu, cpuInputs);
            cpuY = y.ToArray();
            var r = Rand(97, 1.0, (int[])y._shape.Clone());
            var loss = cpu.ReduceSum(cpu.TensorMultiply(y, r), Enumerable.Range(0, y.Rank).ToArray(), keepDims: false);
            if (differentiable)
            {
                var grads = tape.ComputeGradients(loss, cpuInputs);
                cpuGrads = cpuInputs.Select(x => grads[x].ToArray()).ToArray();
            }
            else cpuGrads = Array.Empty<float[]>();
        }

        // Device-resident inputs: the product with 1 leaves each input's values on the device.
        var gpuInputs = inputs.Select(x => gpu.TensorMultiplyScalar(Copy(x), 1f)).ToArray();
        long launches = GpuLaunchProbe.Count;
        long fallbacks = TorchParityElementwiseDeviceTestsBase.FallbackCount(name);
        Tensor<float> gy;
        Dictionary<Tensor<float>, Tensor<float>>? gpuGrads = null;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
        using (var tape = new GradientTape<float>())
        {
            gy = op(gpu, gpuInputs);
            Assert.True(gy.IsGpuResident || gy.HasPendingGpuData, $"{name} ran on the host");
            var r = Rand(97, 1.0, (int[])gy._shape.Clone());
            var loss = gpu.ReduceSum(gpu.TensorMultiply(gy, r), Enumerable.Range(0, gy.Rank).ToArray(), keepDims: false);
            if (differentiable) gpuGrads = tape.ComputeGradients(loss, gpuInputs);
        }
        Assert.True(GpuLaunchProbe.Count > launches, $"{name}: no kernel launched");
        Assert.Equal(fallbacks, TorchParityElementwiseDeviceTestsBase.FallbackCount(name));
        Assert.True(scope.Downloads == 0,
            $"{name} read device data back to the host: " + string.Join(", ", scope.Events
                .Where(e => e.Kind == GpuTransferKind.DeviceToHost).Select(e => $"{e.Bytes} B {e.Operation}")));

        AssertClose(name, cpuY, gy.ToArray(), forwardTolerance);
        if (!differentiable) return;
        for (int i = 0; i < gpuInputs.Length; i++)
        {
            Assert.True(gpuGrads!.ContainsKey(gpuInputs[i]), $"{name}: no gradient for input {i}");
            AssertClose($"{name} d input {i}", cpuGrads[i], gpuGrads[gpuInputs[i]].ToArray(), gradientTolerance);
        }
    }

    private static readonly float[] Edges =
    {
        float.NegativeInfinity, -1e30f, -1e20f, -88f, -20f, -5f, -1.5f, -1f, -0.7f, -0.3f, -1e-7f, -0f, 0f, 1e-7f, 0.29999998f,
        0.3f, 0.30000004f, 0.5f, 0.62f, 0.63f, 1f, 1.5f, 5f, 8.9f, 9.1f, 20f, 88f, 1e20f, 1e30f, float.PositiveInfinity, float.NaN,
    };

    private static Tensor<float> ActivationInput(int seed, bool withEdges)
    {
        var rng = new Random(seed);
        int n = (withEdges ? Edges.Length : 0) + 1500;
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++)
            t[i] = withEdges && i < Edges.Length ? Edges[i] : (float)((rng.NextDouble() - 0.5) * 12);
        return t;
    }

    public static TheoryData<string> Activations => new()
    {
        "TensorLisht", "TensorBentIdentity", "TensorGaussian", "TensorScaledTanh", "TensorThresholdedRelu", "TensorIsru",
        "TensorSqrbf", "TensorSurrogateSpike",
    };

    private static Tensor<float> Activation(IEngine e, string name, Tensor<float> x) => name switch
    {
        "TensorLisht" => e.TensorLisht(x),
        "TensorBentIdentity" => e.TensorBentIdentity(x),
        "TensorGaussian" => e.TensorGaussian(x),
        "TensorScaledTanh" => e.TensorScaledTanh(x, 1.7, 0.6),
        "TensorThresholdedRelu" => e.TensorThresholdedRelu(x, 0.5),
        "TensorIsru" => e.TensorIsru(x, 0.8),
        "TensorSqrbf" => e.TensorSqrbf(x, 1.3),
        "TensorSurrogateSpike" => e.TensorSurrogateSpike(x, 0.3, 4.0),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [SkippableTheory]
    [MemberData(nameof(Activations))]
    public void Activation_EdgeValues_MatchCpu(string name)
    {
        var gpu = GpuOrSkip();
        var x = ActivationInput(5, withEdges: true);
        var cpuY = Activation(new CpuEngine(), name, Copy(x)).ToArray();
        long fallbacks = TorchParityElementwiseDeviceTestsBase.FallbackCount(name);
        var gy = Activation(gpu, name, Copy(x));
        AssertClose(name, cpuY, gy.ToArray(), 2e-6);
        Assert.Equal(fallbacks, TorchParityElementwiseDeviceTestsBase.FallbackCount(name));
    }

    [SkippableTheory]
    [MemberData(nameof(Activations))]
    public void Activation_ForwardAndGradient_MatchCpu_OnDevice(string name)
    {
        var gpu = GpuOrSkip();
        CheckForwardAndGradients(gpu, name, (e, v) => Activation(e, name, v[0]), new[] { ActivationInput(6, withEdges: false) }, 2e-6, 1e-5);
    }

    [SkippableFact]
    public void FakeQuantize_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        var x = Rand(11, 3.0, 4, 37);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0]), new[] { x }, 1e-6, 0);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 4, null, null, symmetric: false), new[] { x }, 1e-6, 0);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 3, 0.25, 1, symmetric: false), new[] { x }, 1e-6, 0);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 5, null, 2, symmetric: true), new[] { x }, 1e-6, 0);
        // Over 1024 values, so the min / max reduction folds several values per slot.
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 6, null, null, symmetric: false),
            new[] { Rand(12, 2.0, 3000) }, 1e-6, 0);
    }

    public static TheoryData<int> QuantizationWidths
    {
        get
        {
            var data = new TheoryData<int>();
            for (int bits = 1; bits <= 30; bits++) data.Add(bits);
            return data;
        }
    }

    // Values whose x / step lands on, just inside, just outside and halfway around both clamp bounds (from 25 bits on
    // these exceed float's 24-bit significand, where a float qMax or a float quotient rounds), plus random values,
    // signed zeros, subnormals, infinities and NaN.
    private static Tensor<float> QuantizationInput(int seed, long qMin, long qMax, long zp, double step)
    {
        var values = new List<float> { 0f, -0f, 1e-45f, -1e-45f, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
        foreach (long bound in new[] { qMin - zp, qMax - zp, (qMax - zp) / 2, (qMin - zp) / 2 })
            for (int k = -4; k <= 4; k++)
                foreach (double frac in new[] { 0.0, 0.5, 0.25, -0.25 })
                {
                    float v = (float)((bound + k + frac) * step);
                    values.Add(v);
                    values.Add(NextBits(v, 1));
                    values.Add(NextBits(v, -1));
                }
        var rng = new Random(seed);
        double span = (qMax - qMin + 4) * step;
        for (int i = 0; i < 1200; i++) values.Add((float)((rng.NextDouble() - 0.5) * span));
        return new Tensor<float>(values.ToArray(), new[] { values.Count });
    }

    // The float whose bit pattern is v's plus delta (an adjacent float for finite v).
    private static float NextBits(float v, int delta)
        => BitConverter.ToSingle(BitConverter.GetBytes(BitConverter.ToInt32(BitConverter.GetBytes(v), 0) + delta), 0);

    private static Tensor<float> Finite(Tensor<float> t)
    {
        var kept = t.ToArray().Where(v => !float.IsInfinity(v)).ToArray();
        return new Tensor<float>(kept, new[] { kept.Length });
    }

    /// <summary>
    /// Fake quantization at every width the op takes (1 to 30 bits), symmetric and asymmetric, with derived and fixed
    /// steps and zero points: the GPU's values and pass-through mask (the gradient) equal the CPU's bit for bit. From
    /// 25 bits qMax and the quantized values no longer fit a float, which once made the device clamp differently.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(QuantizationWidths))]
    public void FakeQuantize_EveryWidth_MatchesCpuBitForBit(int numBits)
    {
        var gpu = GpuOrSkip();
        foreach (bool symmetric in new[] { true, false })
        {
            var (qMin, qMax) = PortableTorchModelOps.FakeQuantizeRange(numBits, symmetric);
            // Derived step and zero point: the data's extremes map onto the bounds, and +-max/2 onto halfway points.
            // (NaN is skipped by the min / max; an infinity makes the range infinite and the step 1.)
            var full = QuantizationInput(numBits, qMin, qMax, 0, 3.7 / Math.Max(1, qMax));
            foreach (var data in new[] { Finite(full), full })
            {
                CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], numBits, null, null, symmetric),
                    new[] { data }, 0, 0);
                CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], numBits, 0, 3, symmetric),
                    new[] { data }, 0, 0);
            }
            // All-negative and all-positive data: the asymmetric zero point lands on a bound.
            foreach (float sign in new[] { -1f, 1f })
            {
                var oneSided = new Tensor<float>(Finite(full).ToArray().Select(v => sign * (Math.Abs(v) + 0.5f)).ToArray(), new[] { Finite(full).Length });
                CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], numBits, null, null, symmetric),
                    new[] { oneSided }, 0, 0);
            }

            // Fixed steps: dyadic (x / step exact), non-dyadic (x / step rounds in double), tiny and huge.
            foreach (double step in new[] { 1.0, 0.25, 1.0 / 3, 0.1, 1e-30, 1e30 })
                foreach (int? zp in new int?[] { null, symmetric ? -7 : 11 })
                {
                    var x = QuantizationInput(numBits * 31 + 7, qMin, qMax, zp ?? 0, step);
                    CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], numBits, step, zp, symmetric),
                        new[] { x }, 0, 0);
                }
        }
    }

    /// <summary>
    /// The widest symmetric case the review found: at 26 bits qMax = 2^25 - 1, which a float rounds up to 2^25, so the
    /// device passed x = 2^25 (scale 1) that the CPU clamps.
    /// </summary>
    [SkippableFact]
    public void FakeQuantize_ValueJustPastQMax_IsClampedAsOnTheCpu()
    {
        var gpu = GpuOrSkip();
        var x = new Tensor<float>(new[] { 33554432f, 33554430f, -33554432f, -33554436f }, new[] { 4 });
        var cpu = new CpuEngine();
        foreach (var engine in new IEngine[] { cpu, gpu })
        {
            using var tape = new GradientTape<float>();
            var input = engine == gpu ? gpu.TensorMultiplyScalar(Copy(x), 1f) : Copy(x);
            var y = engine.TensorFakeQuantize(input, 26, 1.0, null, symmetric: true);
            var loss = engine.ReduceSum(y, new[] { 0 }, keepDims: false);
            var grad = tape.ComputeGradients(loss, new[] { input })[input].ToArray();
            // The clamped 2^25 - 1 is a float tie, rounded to even 2^25; the mask is what tells it from a pass.
            Assert.Equal(new[] { 33554432f, 33554430f, -33554432f, -33554432f }, y.ToArray());
            Assert.Equal(new[] { 0f, 1f, 1f, 0f }, grad);
        }
    }

    /// <summary>A zero point so far out that the clamp bounds leave 32-bit integers: the GPU engine computes it on the host.</summary>
    [SkippableFact]
    public void FakeQuantize_HugeZeroPoint_MatchesCpu()
    {
        var gpu = GpuOrSkip();
        var x = QuantizationInput(5, -(1L << 29), (1L << 29) - 1, 0, 1.0);
        foreach (int zp in new[] { int.MaxValue, int.MinValue, int.MaxValue - (1 << 29) })
            foreach (bool symmetric in new[] { true, false })
            {
                var expected = new CpuEngine().TensorFakeQuantize(Copy(x), 30, 1.0, zp, symmetric).ToArray();
                AssertClose($"zp {zp}", expected, gpu.TensorFakeQuantize(Copy(x), 30, 1.0, zp, symmetric).ToArray(), 0);
            }
    }

    /// <summary>
    /// A CRF whose forward-backward state (batch * seqLen * numTags + 2 * batch * (seqLen + 1)) outgrows an int: the GPU
    /// engine throws the CPU's ArgumentException, message and parameter included, before allocating or launching
    /// anything (the emissions are a stride-0 view, so nothing large exists), with and without a tape.
    /// </summary>
    [SkippableFact]
    public void CrfForward_OversizedState_ThrowsTheCpusArgumentException()
    {
        var gpu = GpuOrSkip();
        foreach (var shape in new[] { new[] { 1000, 1_000_000, 1 }, new[] { 2, 400_000_000, 2 }, new[] { 1, 1_073_741_823, 1 } })
        {
            var emissions = new Tensor<float>(new[] { 1, 1, 1 }).ExpandTo(shape);
            var transitions = new Tensor<float>(new[] { shape[2], shape[2] });
            Assert.True((long)shape[0] * shape[1] * shape[2] <= int.MaxValue);
            var cpuError = Assert.Throws<ArgumentException>(() => new CpuEngine().TensorCrfForward(emissions, transitions));
            long launches = GpuLaunchProbe.Count;
            var gpuError = Assert.Throws<ArgumentException>(() => gpu.TensorCrfForward(emissions, transitions));
            Assert.Equal(cpuError.Message, gpuError.Message);
            Assert.Equal("emissions", gpuError.ParamName);
            Assert.Contains($"batch={shape[0]}, seqLen={shape[1]}, numTags={shape[2]}", gpuError.Message);
            using (new GradientTape<float>())
                Assert.Equal(cpuError.Message, Assert.Throws<ArgumentException>(() => gpu.TensorCrfForward(emissions, transitions)).Message);
            Assert.Equal(launches, GpuLaunchProbe.Count);
        }
    }

    [SkippableFact]
    public void TopKSoftmax_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 3), new[] { Rand(21, 4.0, 5, 8) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 1), new[] { Rand(22, 4.0, 2, 3, 6) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 64), new[] { Rand(23, 4.0, 3, 64) }, 2e-6, 1e-5);

        // Vocabulary-sized rows take the chunk-selection path: small k, a large k, and rows with heavy ties (values
        // rounded to a few levels, so the lower-index rule decides most of the selection).
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 8), new[] { Rand(24, 6.0, 2, 50000) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 300), new[] { Rand(25, 6.0, 2, 5000) }, 2e-6, 1e-5);
        var ties = Rand(26, 3.0, 3, 4099);
        for (int i = 0; i < ties.Length; i++) ties[i] = (float)Math.Round(ties[i]);
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 37), new[] { ties }, 2e-6, 1e-5);

        // Ties go to the lower index; NaN sorts below every number (and is selected only when k reaches it).
        var tied = new Tensor<float>(new[] { 1f, 3f, 3f, 2f, 3f, float.NaN, 0f, 5f, 2f, 2f, float.NaN, 1f }, new[] { 3, 4 });
        foreach (int k in new[] { 1, 2, 3, 4 })
        {
            var cpuY = new CpuEngine().TensorTopKSoftmax(Copy(tied), k).ToArray();
            AssertClose($"TensorTopKSoftmax ties k={k}", cpuY, gpu.TensorTopKSoftmax(Copy(tied), k).ToArray(), 2e-6);
        }
    }

    /// <summary>
    /// Rows of a vocabulary (width 50,000) built to stress a large-k selection: (0) 21 tied levels with signed zeros,
    /// NaN and -inf sprinkled in, so the k-th key falls inside a tie group and the lower-index rule decides it; (1) almost
    /// all +0 / -0 (one tie group), a few +-1 and NaN; (2) 3,000 numbers and NaN after them, so k = 8192 reaches the NaN
    /// entries; (3) continuous values. Values stay within [-5, 5], so every selected exp(x - top) is a normal float and a
    /// zero output means "not selected" on both engines.
    /// </summary>
    internal static Tensor<float> TopKStressRows(int seed, int width)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { 4, width });
        for (int i = 0; i < width; i++)
        {
            float level = (rng.Next(21) - 10) * 0.5f;
            if (level == 0f && rng.Next(2) == 0) level = -0f;
            if (i % 97 == 5) level = float.NaN;
            else if (i % 101 == 7) level = float.NegativeInfinity;
            t[0, i] = level;

            int pick = rng.Next(100);
            t[1, i] = pick < 3 ? 1f : pick < 6 ? -1f : pick == 6 ? float.NaN : rng.Next(2) == 0 ? -0f : 0f;
            t[2, i] = i < 3000 ? (i % 50 == 0 ? (i % 100 == 0 ? -0f : 0f) : (float)(rng.NextDouble() * 10 - 5)) : float.NaN;
            t[3, i] = (float)(rng.NextDouble() * 10 - 5);
        }
        return t;
    }

    /// <summary>
    /// A large k over vocabulary-sized rows (the one-work-group-per-row path): the forward and gradients against the
    /// CPU, and the exact selection (which entries are zero, which NaN) entry by entry.
    /// </summary>
    [SkippableFact]
    public void TopKSoftmax_LargeK_VocabularyRows_MatchCpuSelectionExactly()
    {
        var gpu = GpuOrSkip();
        var x = TopKStressRows(31, 50000);
        foreach (int k in new[] { 1024, 8192 })
        {
            CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], k), new[] { x }, 2e-6, 1e-5);
            var cpuY = new CpuEngine().TensorTopKSoftmax(Copy(x), k).ToArray();
            var gpuY = gpu.TensorTopKSoftmax(gpu.TensorMultiplyScalar(Copy(x), 1f), k).ToArray();
            int selected = 0;
            for (int i = 0; i < cpuY.Length; i++)
            {
                Assert.True(float.IsNaN(cpuY[i]) == float.IsNaN(gpuY[i]) && (cpuY[i] == 0f) == (gpuY[i] == 0f),
                    $"k={k} row {i / 50000} index {i % 50000} (x {x[i]:R}): cpu {cpuY[i]:R} gpu {gpuY[i]:R}");
                if (cpuY[i] != 0f) selected++;
            }
            // Every row selects k entries; row 0's -inf entries never reach the top k, so each selected output is nonzero.
            Assert.Equal(4 * k, selected);
        }
    }

    [SkippableFact]
    public void SinusoidalTimeEmbedding_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        // Diffusion-style integer timesteps up to 1000 (angles up to 1000 rad) and fractional ones; an odd dim pads a 0.
        var t = new Tensor<float>(new[] { 0f, 1f, 7f, 49.5f, 250f, 999f, 1000f, 0.001f, 123.456f }, new[] { 9 });
        CheckForwardAndGradients(gpu, "TensorSinusoidalTimeEmbedding", (e, v) => e.TensorSinusoidalTimeEmbedding(v[0], 64), new[] { t }, 2e-6, 2e-4);
        CheckForwardAndGradients(gpu, "TensorSinusoidalTimeEmbedding", (e, v) => e.TensorSinusoidalTimeEmbedding(v[0], 7),
            new[] { new Tensor<float>(new[] { 0.5f, 1.7f, 3.2f }, new[] { 3, 1 }) }, 2e-6, 1e-5);
    }

    [SkippableFact]
    public void CrfForward_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        int b = 3, s = 6, k = 5;
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1], v[2], v[3]),
            new[] { Rand(31, 2.0, b, s, k), Rand(32, 1.0, k, k), Rand(33, 1.0, k), Rand(34, 1.0, k) }, 2e-5, 2e-5);
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1]),
            new[] { Rand(35, 2.0, 7, 4), Rand(36, 1.0, 4, 4) }, 2e-5, 2e-5);
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1], v[2]),
            new[] { Rand(37, 2.0, 2, 1, 3), Rand(38, 1.0, 3, 3), Rand(39, 1.0, 3) }, 2e-5, 2e-5);

        // More tags than a work group has lanes (each lane covers several tags), and a long sequence.
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1], v[2], v[3]),
            new[] { Rand(42, 2.0, 2, 3, 300), Rand(43, 1.0, 300, 300), Rand(44, 1.0, 300), Rand(45, 1.0, 300) }, 2e-5, 2e-5);
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1]),
            new[] { Rand(46, 2.0, 3, 400, 6), Rand(47, 1.0, 6, 6) }, 2e-5, 2e-5);   // log alpha reaches ~1e3: the offset form keeps full precision

        // Forbidden transitions (-inf, a constraint mask): log-sum-exp over -inf terms and a fully forbidden target.
        var masked = Rand(40, 1.0, 4, 4);
        masked[0, 1] = float.NegativeInfinity;
        masked[2, 3] = float.NegativeInfinity;
        for (int i = 0; i < 4; i++) masked[i, 2] = float.NegativeInfinity;
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1]),
            new[] { Rand(41, 2.0, 2, 5, 4), masked }, 2e-5, 2e-5);
    }

    /// <summary>
    /// The four host-computed ops compile into a GPU inference plan (no host-boundary refusal) and each replay runs the
    /// device kernels on the plan's live input.
    /// </summary>
    [SkippableFact]
    public void HostKernelOps_CompileIntoInferencePlans_AndReplayOnLiveInputs()
    {
        var gpu = GpuOrSkip();
        var cases = new (string Name, int[] Shape, Func<IEngine, Tensor<float>, Tensor<float>> Op)[]
        {
            ("TensorFakeQuantize", new[] { 3, 20 }, (e, v) => e.TensorFakeQuantize(v, 6, null, null, symmetric: false)),
            ("TensorTopKSoftmax", new[] { 3, 20 }, (e, v) => e.TensorTopKSoftmax(v, 4)),
            ("TensorSinusoidalTimeEmbedding", new[] { 5 }, (e, v) => e.TensorSinusoidalTimeEmbedding(e.TensorMultiplyScalar(v, 100f), 16)),
            ("TensorCrfForward", new[] { 2, 4, 3 }, (e, v) => e.TensorCrfForward(v, new Tensor<float>(Enumerable.Range(0, 9).Select(i => 0.1f * i).ToArray(), new[] { 3, 3 }))),
        };
        foreach (var (name, shape, op) in cases)
        {
            var x = Rand(70, 2.0, shape);
            AiDotNet.Tensors.Engines.Compilation.CompiledInferencePlan<float> plan;
            using (var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.EnableInference())
            {
                var y = op(gpu, x);
                plan = scope.CompileInference(y, new[] { x });
            }
            using (plan)
            {
                for (int round = 0; round < 2; round++)
                {
                    Rand(71 + round, 2.0, shape).ToArray().AsSpan().CopyTo(x.AsWritableSpan());
                    var expected = op(new CpuEngine(), Copy(x)).ToArray();
                    long launches = GpuLaunchProbe.Count;
                    var replayed = plan.Execute().ToArray();
                    Assert.True(GpuLaunchProbe.Count > launches, $"{name}: the compiled replay launched no kernel");
                    AssertClose($"{name} compiled replay {round}", expected, replayed, 2e-5);
                }
            }
        }
    }

    [SkippableFact]
    public void ComposedOps_MatchCpu_AndStayOnDevice()
    {
        var gpu = GpuOrSkip();
        CheckForwardAndGradients(gpu, "TensorLogSoftmin", (e, v) => e.TensorLogSoftmin(v[0], 1), new[] { Rand(51, 3.0, 4, 9) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorLogSoftmin", (e, v) => e.TensorLogSoftmin(v[0], 0), new[] { Rand(52, 3.0, 4, 9) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorHyperbolicLinear", (e, v) => e.TensorHyperbolicLinear(v[0], v[1], v[2], 0.8),
            new[] { Rand(53, 0.3, 5, 6), Rand(54, 0.5, 4, 6), Rand(55, 0.2, 4) }, 2e-5, 1e-4);
        CheckForwardAndGradients(gpu, "TensorHyperbolicLinear", (e, v) => e.TensorHyperbolicLinear(v[0], v[1]),
            new[] { Rand(56, 0.3, 2, 3, 6), Rand(57, 0.5, 4, 6) }, 2e-5, 1e-4);
        CheckForwardAndGradients(gpu, "TensorSoftSplit", (e, v) => e.TensorSoftSplit(v[0], v[1], v[2], 2, 0.1, 0.7),
            new[] { Rand(58, 1.0, 5), Rand(59, 1.0, 3, 2), Rand(60, 1.0, 3, 2) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorSoftKnn", (e, v) => e.TensorSoftKnn(v[0], v[1], v[2], 0.9),
            new[] { Rand(61, 1.0, 3, 4), Rand(62, 1.0, 6, 4), Rand(63, 1.0, 6, 2) }, 2e-5, 1e-4);
        CheckForwardAndGradients(gpu, "TensorSoftLocallyWeighted", (e, v) => e.TensorSoftLocallyWeighted(v[0], v[1], v[2], 1.5),
            new[] { Rand(64, 1.0, 4), Rand(65, 1.0, 5, 4), Rand(66, 1.0, 5) }, 2e-5, 1e-4);
    }
}

/// <summary>An engine over the OpenCL backend, which the default order (CUDA first) does not pick on an NVIDIA machine.</summary>
public sealed class OpenClTensorEngineTestFixture : IDirectGpuTensorEngineFixture, IDisposable
{
    private readonly OpenClBackend? _backend;
    public DirectGpuTensorEngine? Engine { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public OpenClTensorEngineTestFixture()
    {
        try
        {
            _backend = new OpenClBackend();
            if (!_backend.IsAvailable)
            {
                _backend.Dispose();
                _backend = null;
                return;
            }
            Engine = new DirectGpuTensorEngine(new DirectGpuEngine(_backend));
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            _backend?.Dispose();
            _backend = null;
            Engine = null;
        }
    }

    public void Dispose()
    {
        Engine?.Dispose();
        _backend?.Dispose();
    }
}

/// <summary>An engine over the CUDA backend, which a machine's default backend order need not pick.</summary>
public sealed class CudaTensorEngineTestFixture : IDirectGpuTensorEngineFixture, IDisposable
{
    private readonly AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend? _backend;
    public DirectGpuTensorEngine? Engine { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public CudaTensorEngineTestFixture()
    {
        try
        {
            _backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
            if (!_backend.IsAvailable)
            {
                _backend.Dispose();
                _backend = null;
                return;
            }
            Engine = new DirectGpuTensorEngine(new DirectGpuEngine(_backend));
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException or EntryPointNotFoundException)
        {
            _backend?.Dispose();
            _backend = null;
            Engine = null;
        }
    }

    public void Dispose()
    {
        Engine?.Dispose();
        _backend?.Dispose();
    }
}

[Collection("DirectGpuSerial")]
public class TorchModelOpsCudaTests : TorchModelOpsGpuTestsBase, IClassFixture<CudaTensorEngineTestFixture>
{
    public TorchModelOpsCudaTests(CudaTensorEngineTestFixture fixture) : base(fixture) { }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class TorchModelOpsGpuTests : TorchModelOpsGpuTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public TorchModelOpsGpuTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("DirectGpuSerial")]
public class TorchModelOpsOpenClTests : TorchModelOpsGpuTestsBase, IClassFixture<OpenClTensorEngineTestFixture>
{
    public TorchModelOpsOpenClTests(OpenClTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("VulkanGlobalState")]
public class TorchModelOpsVulkanTests : TorchModelOpsGpuTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public TorchModelOpsVulkanTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("DirectGpuSerial")]
public class TorchModelOpsHipTests : TorchModelOpsGpuTestsBase, IClassFixture<HipTensorEngineTestFixture>
{
    public TorchModelOpsHipTests(HipTensorEngineTestFixture fixture) : base(fixture) { }
}

#if NET7_0_OR_GREATER
/// <summary>The FP32 kernels Metal and WebGPU run, as WGSL on Dawn.</summary>
[Collection("DirectGpuSerial")]
public class TorchModelOpsDawnTests : TorchModelOpsGpuTestsBase, IClassFixture<DawnTorchParityFixture>
{
    public TorchModelOpsDawnTests(DawnTorchParityFixture fixture) : base(fixture) { }
}
#endif

/// <summary>Source-level checks of the shared model-op kernels on every dialect (no device needed).</summary>
public sealed class TorchModelOpsKernelSourceTests
{
    [Fact]
    public void EveryKernel_IsTranslated_ForOpenClAndGlsl()
    {
        foreach (string name in TorchModelOpsKernels.KernelNames)
        {
            Assert.Contains("__kernel void " + name + "(", TorchModelOpsTranslation.OpenCl);
            Assert.True(TorchModelOpsTranslation.Glsl.ContainsKey(name), $"no GLSL shader for {name}");
        }
        Assert.Equal(TorchModelOpsKernels.KernelNames.Length, TorchModelOpsTranslation.Glsl.Count);
    }

    [Fact]
    public void EveryKernel_IsInThePortableSource_AndTranslatesToWgsl()
    {
        var modules = AiDotNet.Tensors.Engines.DirectGpu.WebGpu.CudaDialectToWgsl.Translate(TorchParityF32Kernels.GetSource());
        foreach (string name in TorchModelOpsKernels.KernelNames) Assert.True(modules.ContainsKey(name), $"no WGSL module for {name}");
    }

    [Fact]
    public void FakeQuantizeStepSplit_IsExact_IncludingSubnormals()
    {
        foreach (double step in new[] { 1.0, 1.0 / 3, 0.1, 1e-30, 1e30, double.Epsilon, 3 * double.Epsilon, double.MaxValue, 2.2250738585072014E-308 })
        {
            var (lo, hi, exponent) = PortableTorchModelOps.SplitStep(step);
            long significand = ((long)hi << 32) | (uint)lo;
            Assert.InRange(significand, 1L << 52, (1L << 53) - 1);
            // Scaled one binade at a time (exact for these values), so neither 2^exponent nor a partial product leaves
            // double's range.
            double value = significand;
            for (int e = exponent; e > 0; e--) value *= 2;
            for (int e = exponent; e < 0; e++) value /= 2;
            Assert.Equal(step, value);
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => PortableTorchModelOps.SplitStep(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PortableTorchModelOps.SplitStep(double.PositiveInfinity));
    }

    [Fact]
    public void FakeQuantizeRange_IsTheCpusRange_AndHugeZeroPointsStayOnTheHost()
    {
        Assert.Equal((-(1L << 29), (1L << 29) - 1), PortableTorchModelOps.FakeQuantizeRange(30, true));
        Assert.Equal((0L, (1L << 30) - 1), PortableTorchModelOps.FakeQuantizeRange(30, false));
        Assert.True(PortableTorchModelOps.FakeQuantizeFitsDevice(30, false, null));
        Assert.True(PortableTorchModelOps.FakeQuantizeFitsDevice(30, true, 1 << 30));
        Assert.False(PortableTorchModelOps.FakeQuantizeFitsDevice(30, true, int.MaxValue));
        Assert.False(PortableTorchModelOps.FakeQuantizeFitsDevice(30, false, int.MinValue));
        Assert.False(PortableTorchModelOps.FakeQuantizeFitsDevice(31, true, null));
    }

    [Fact]
    public void CrfStateLengths_ReportAnArgumentError_NotAnOverflow()
    {
        Assert.Equal(2 * 3 * 4 + 2 * 2 * 4, PortableTorchModelOps.CrfAlphaLength(2, 3, 4));
        Assert.Equal(2 * 3 * 4 + 2 * 2 * 3, PortableTorchModelOps.CrfBetaLength(2, 3, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => PortableTorchModelOps.CrfAlphaLength(1000, 1_000_000, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PortableTorchModelOps.CrfBetaLength(1000, 1_000_000, 1));
        Assert.Equal(3_000_002_000L, PortableTorchModelOps.CrfAlphaLengthLong(1000, 1_000_000, 1));
    }

    [Fact]
    public void CrfForward_OversizedState_ThrowsArgumentException_OnTheCpu_BeforeAllocating()
    {
        var emissions = new Tensor<float>(new[] { 1, 1, 1 }).ExpandTo(new[] { 1000, 1_000_000, 1 });
        var transitions = new Tensor<float>(new[] { 0.5f }, new[] { 1, 1 });
        var error = Assert.Throws<ArgumentException>(() => new CpuEngine().TensorCrfForward(emissions, transitions));
        Assert.Equal("emissions", error.ParamName);
        Assert.Contains("batch=1000, seqLen=1000000, numTags=1", error.Message);
        Assert.Contains("3000002000", error.Message);
    }

    [Fact]
    public void FrequencySplit_IsExact_AndThresholdRoundsDown()
    {
        var into = new float[3];
        foreach (double f in new[] { 1.0, 0.7498942093324559, 1e-4, 0.36517412725483767 })
        {
            TorchModelOpsCodes.SplitFrequency(f, into, 0);
            Assert.True(Math.Abs((double)into[0] + into[1] + into[2] - f) <= 1e-15 * f, $"split of {f}");
            Assert.True((BitConverter.ToInt32(BitConverter.GetBytes(into[0]), 0) & 0xFFF) == 0);
            Assert.True((BitConverter.ToInt32(BitConverter.GetBytes(into[1]), 0) & 0xFFF) == 0);
        }
        Assert.Equal(0.29999998f, TorchModelOpsCodes.FloatThresholdBelow(0.3));
        Assert.Equal(0.5f, TorchModelOpsCodes.FloatThresholdBelow(0.5));
        Assert.Equal(-0.3f, TorchModelOpsCodes.FloatThresholdBelow(-0.3));   // -0.3f lies below -0.3
        Assert.Equal(float.MaxValue, TorchModelOpsCodes.FloatThresholdBelow(1e300));
    }
}
