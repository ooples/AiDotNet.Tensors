using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Gradient parity, CPU vs GPU, for every op whose forward stopped bailing on an active tape.
///
/// WHY THE SOURCE AUDIT IS NOT ENOUGH: TapeBailAuditTests proves a recording EXISTS. It cannot prove the
/// recording is CORRECT — wrong saved state, the wrong backward, or the wrong overload all leave the source
/// looking right while gradients come out silently wrong. TensorMax/TensorMin are the standing example: the
/// GPU implements (tensor, scalar) while CpuEngine records only (tensor, tensor), so reusing that backward
/// would attribute gradient to an operand that does not exist. Only running backprop catches that class.
///
/// GUARDS AGAINST A VACUOUS PASS (each has already produced a false green in this codebase):
///   1. Calls go through IEngine. DirectGpuTensorEngine uses EXPLICIT interface implementations, so
///      gpu.Op(x) resolves to the inherited CpuEngine method and compares CPU against itself.
///   2. IsGpuAvailable is asserted — constructing the engine succeeds with no backend, and the ops fall
///      back to base silently by design.
///   3. maxAbs > 0.0 is asserted — two independent float32 implementations never agree to the last bit, so
///      an exact zero delta IS the fallback signature. A first run passed 6/6 entirely on CPU and the
///      exact 0.000E+000 was the only tell.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuTapeGradientParityTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuTapeGradientParityTests(ITestOutputHelper output) => _out = output;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static bool TryGpu(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable) { candidate.Dispose(); engine = null; return false; }
            engine = candidate;
            return true;
        }
        catch (Exception) { engine = null; return false; }
    }

    private static Tensor<float> Rand(int[] shape, int seed, double lo = -1.0, double hi = 1.0)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(lo + rng.NextDouble() * (hi - lo));
        return t;
    }

    /// <summary>
    /// Runs forward+backward on one engine and returns the flattened input gradient.
    /// A non-uniform output weighting keeps the upstream gradient from being constant, which is what makes
    /// a wrong backward observable rather than accidentally matching.
    /// </summary>
    private static float[] GradientOf(IEngine engine, Tensor<float> x, Func<IEngine, Tensor<float>, Tensor<float>> op)
    {
        AiDotNetEngine.Current = engine;

        var input = new Tensor<float>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++) input[i] = x[i];

        using var tape = new GradientTape<float>();
        var y = op(engine, input);

        var weight = new Tensor<float>(y.Shape.ToArray());
        for (int i = 0; i < weight.Length; i++) weight[i] = 0.13f + 0.017f * (i % 11);
        var loss = engine.ReduceSum(engine.TensorMultiply(y, weight), null);

        var grads = tape.ComputeGradients(loss, new[] { input });
        Assert.True(grads.TryGetValue(input, out var g) && g is not null,
            "no gradient reached the input — the forward recorded no tape node");

        var flat = new float[g.Length];
        for (int i = 0; i < g.Length; i++) flat[i] = g[i];
        return flat;
    }

    /// <summary>
    /// Whether bit-identical CPU/GPU results are expected for this op, so the divergence guard must not
    /// be used as the engagement probe.
    /// </summary>
    /// <remarks>
    /// The maxAbs &gt; 0 guard assumes two independent float32 implementations cannot agree to the last bit.
    /// That holds for anything doing arithmetic, but NOT for ops that only move data: Upsample3D replicates
    /// elements in the forward and its backward sums gradOutput over each block — exact float adds of the
    /// same values on both engines, so bit-identity is the CORRECT answer and the guard false-positives.
    /// Those ops use the deferred-materialisation counter instead, which observes GPU residency directly.
    /// </remarks>
    private enum Engagement { ExpectDivergence, UseResidencyCounter }

    private void AssertGradientParity(string opName, Tensor<float> x,
        Func<IEngine, Tensor<float>, Tensor<float>> op, double tol = 1e-4,
        Engagement probe = Engagement.ExpectDivergence)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null,
            $"GPU backend did not resolve, so {opName} would have been compared against itself. Copy the "
            + "CUDA natives into the test output directory AFTER building.");

        using (gpu!)
        {
            var cpuGrad = GradientOf(new CpuEngine(), x, op);

            // Count deferred GPU->host materialisations across the GPU run. A non-zero delta means results
            // were GPU-resident and had to be downloaded — direct evidence the device path ran, independent
            // of whether the numbers happen to match the CPU exactly.
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
            var gpuGrad = GradientOf(gpu, x, op);
            long materialisations = AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount;

            Assert.Equal(cpuGrad.Length, gpuGrad.Length);

            double maxAbs = 0, maxRel = 0;
            for (int i = 0; i < cpuGrad.Length; i++)
            {
                double d = Math.Abs(cpuGrad[i] - gpuGrad[i]);
                maxAbs = Math.Max(maxAbs, d);
                maxRel = Math.Max(maxRel, d / Math.Max(1.0, Math.Abs(cpuGrad[i])));
            }

            _out.WriteLine($"{opName,-14} maxAbs={maxAbs:E3}  maxRel={maxRel:E3}  materialisations={materialisations}");

            if (probe == Engagement.ExpectDivergence)
            {
                Assert.True(maxAbs > 0.0,
                    $"{opName}: CPU and GPU gradients were BIT-IDENTICAL, so the GPU path did not run and "
                    + "this compared the CPU implementation with itself.");
            }
            else
            {
                Assert.True(materialisations > 0,
                    $"{opName}: no deferred GPU->host materialisation occurred, so nothing was ever "
                    + "GPU-resident and the device path did not run. (This op is exact on both engines, so "
                    + "bit-identical output cannot be used as the engagement signal.)");
            }
            Assert.True(maxRel <= tol,
                $"{opName}: GPU gradient diverged from CPU (maxRel={maxRel:E3}). A gross mismatch means the "
                + "recorded backward, its saved state, or the overload is wrong — not float rounding.");
        }
    }

    /// <summary>
    /// The FORWARD computed while a tape records matches CpuEngine's. Gradient parity alone cannot see a wrong taped
    /// forward whenever the backward reads saved state (a pre-activation) rather than the output, so ops with a
    /// separate taped forward path need this too.
    /// </summary>
    private void AssertTapedForwardMatchesCpu(string opName, Tensor<float> x,
        Func<IEngine, Tensor<float>, Tensor<float>> op, double tol = 1e-4)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu!)
        {
            float[] ForwardUnderTape(IEngine engine)
            {
                AiDotNetEngine.Current = engine;
                using var tape = new GradientTape<float>();
                var y = op(engine, x);
                var values = new float[y.Length];
                for (int i = 0; i < values.Length; i++) values[i] = y[i];
                return values;
            }

            var cpu = ForwardUnderTape(new CpuEngine());
            var device = ForwardUnderTape(gpu);
            Assert.Equal(cpu.Length, device.Length);
            double maxRel = 0;
            for (int i = 0; i < cpu.Length; i++)
                maxRel = Math.Max(maxRel, Math.Abs(cpu[i] - device[i]) / Math.Max(1.0, Math.Abs(cpu[i])));
            _out.WriteLine($"{opName,-14} taped forward maxRel={maxRel:E3}");
            Assert.True(maxRel <= tol, $"{opName}: taped GPU forward diverged from CPU (maxRel={maxRel:E3}).");
        }
    }

    [SkippableFact]
    public void TensorCosh_gradients_match_cpu() =>
        AssertGradientParity("TensorCosh", Rand([4, 16], seed: 21, lo: -2.0, hi: 2.0),
            static (e, t) => e.TensorCosh(t));

    /// <summary>
    /// Transpose only moves data, so its gradient is exact on both engines and the divergence probe cannot
    /// show the device ran; the residency counter checks the gradient, and the next test pins the engagement.
    /// </summary>
    [SkippableFact]
    public void TensorTranspose_gradients_match_cpu() =>
        AssertGradientParity("TensorTranspose", Rand([6, 10], seed: 24),
            static (e, t) => e.TensorTranspose(t),
            probe: Engagement.UseResidencyCounter);

    /// <summary>
    /// Under a recording tape each of these must launch its kernel and download nothing. Every one used to bail
    /// to CpuEngine whenever a tape was active, so every training-time call went through the host (seen in
    /// AutoformerModel training: DirectGpuTensorEngine.TensorTranspose -> CpuEngine.TensorTranspose). Their
    /// gradients are exact on both engines, so this residency check is what proves the device path ran.
    /// </summary>
    private void AssertStaysOnDeviceUnderTape(string opName, Func<IEngine, Tensor<float>, Tensor<float>> op,
        Action<Tensor<float>, Tensor<float>> checkResult)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu!)
        {
            IEngine engine = gpu!;
            AiDotNetEngine.Current = engine;
            var x = Rand([6, 10], seed: 25);
            using var tape = new GradientTape<float>();
            _ = engine.TensorAdd(x, x);                       // upload x and warm the path outside the count
            AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();

            var y = op(engine, x);

            long launches = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count;
            long readbacks = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Readbacks;
            _out.WriteLine($"{opName} under tape: launches={launches} readbacks={readbacks}");
            Assert.True(launches >= 1, $"{opName} launched no GPU kernel while a tape was recording — it ran on the CPU.");
            Assert.Equal(0, readbacks);
            checkResult(x, y);
        }
    }

    [SkippableFact]
    public void PixelShuffle_gradients_match_cpu() =>
        // A pure permutation is exact on both engines, so engagement is shown by device residency, not divergence.
        AssertGradientParity("PixelShuffle", Rand([2, 8, 3, 4], seed: 41),
            static (e, t) => e.PixelShuffle(t, 2), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void PixelShuffle_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("PixelShuffle",
            static (e, t) => e.PixelShuffle(e.Reshape(t, new[] { 1, 4, 3, 5 }), 2),
            static (_, y) => Assert.Equal(new[] { 1, 1, 6, 10 }, y.Shape.ToArray()));

    private static readonly Tensor<float> StackOther = Rand([6, 10], seed: 42);

    [SkippableFact]
    public void TensorStack_gradients_match_cpu() =>
        // Stacking copies; exact on both engines, so engagement is shown by device residency.
        AssertGradientParity("TensorStack", Rand([6, 10], seed: 43),
            static (e, t) => e.TensorStack(new[] { t, StackOther, t }, -1), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorStack_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorStack", static (e, t) => e.TensorStack(new[] { t, t }, 0),
            static (_, y) => Assert.Equal(new[] { 2, 6, 10 }, y.Shape.ToArray()));

    [SkippableFact]
    public void TensorDiagonal_gradients_match_cpu() =>
        // Rectangular on purpose: the diagonal is min(rows, cols) long and the gradient keeps the input shape.
        AssertGradientParity("TensorDiagonal", Rand([6, 10], seed: 44),
            static (e, t) => e.TensorDiagonal(t), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorDiagonal_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorDiagonal", static (e, t) => e.TensorDiagonal(t), static (x, y) =>
        {
            Assert.Equal(new[] { 6 }, y.Shape.ToArray());
            for (int i = 0; i < 6; i++) Assert.Equal(x[i, i], y[i]);
        });

    [SkippableFact]
    public void Upsample_through_IEngine_gradients_match_cpu() =>
        // Called through IEngine on purpose: that explicit implementation is the one that used to bail.
        AssertGradientParity("Upsample", Rand([1, 2, 3, 4], seed: 45),
            static (e, t) => ((IEngine)e).Upsample(t, 2, 2), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void Upsample_through_IEngine_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("Upsample",
            static (e, t) => ((IEngine)e).Upsample(e.Reshape(t, new[] { 1, 1, 6, 10 }), 2, 2),
            static (_, y) => Assert.Equal(new[] { 1, 1, 12, 20 }, y.Shape.ToArray()));

    // 2-D indices on purpose: the backward's device ScatterAdd mis-shaped an unflattened index set.
    private static readonly Tensor<int> GatherIndices = new(new[] { 0, 5, 2, 2, 4, 1 }, new[] { 2, 3 });

    [SkippableFact]
    public void TensorGather_gradients_match_cpu() =>
        AssertGradientParity("TensorGather", Rand([6, 10], seed: 46),
            static (e, t) => e.TensorGather(t, GatherIndices, 0), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorGather_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorGather", static (e, t) => e.TensorGather(t, GatherIndices, 0),
            static (x, y) =>
            {
                Assert.Equal(new[] { 6, 10 }, y.Shape.ToArray());          // CpuEngine's fast-path shape
                for (int c = 0; c < 10; c++) Assert.Equal(x[5, c], y[1, c]);
            });

    private static Tensor<bool> BoolMask()
    {
        var mask = new Tensor<bool>(new[] { 6, 10 });
        for (int i = 0; i < mask.Length; i++) mask[i] = i % 3 == 0;
        return mask;
    }

    private static Tensor<Bit> BitMask()
    {
        var mask = new Tensor<Bit>(new[] { 6, 10 });
        for (int i = 0; i < mask.Length; i++) mask[i] = i % 3 == 0;
        return mask;
    }

    [SkippableFact]
    public void TensorMaskedFill_bool_mask_gradients_match_cpu() =>
        AssertGradientParity("MaskedFill(bool)", Rand([6, 10], seed: 47),
            static (e, t) => e.TensorMaskedFill(t, BoolMask(), -2f), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorMaskedFill_bit_mask_gradients_match_cpu() =>
        AssertGradientParity("MaskedFill(Bit)", Rand([6, 10], seed: 48),
            static (e, t) => e.TensorMaskedFill(t, BitMask(), -2f), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorMaskedFill_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorMaskedFill", static (e, t) => e.TensorMaskedFill(t, BoolMask(), -2f),
            static (x, y) => { Assert.Equal(-2f, y[0, 0]); Assert.Equal(x[0, 1], y[0, 1]); });

    // GatherIndices repeats row 2, so the backward must ACCUMULATE into a table row, not overwrite it.
    [SkippableFact]
    public void Embedding_gradients_match_cpu() =>
        AssertGradientParity("Embedding", Rand([6, 10], seed: 83),
            static (e, t) => e.Embedding(GatherIndices, t), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void Embedding_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("Embedding", static (e, t) => e.Embedding(GatherIndices, t),
            static (x, y) => { Assert.Equal(new[] { 2, 3, 10 }, y.Shape.ToArray()); Assert.Equal(x[5, 3], y[0, 1, 3]); });
    // 6 -> 4 rows and 5 -> 3 columns give OVERLAPPING windows of unequal length, which is where a separable
    // device backward could drift from the host loop's per-window accumulation.
    [SkippableFact]
    public void AdaptiveAvgPool2D_gradients_match_cpu() =>
        AssertGradientParity("AdaptiveAvgPool2D", Rand([6, 10], seed: 89),
            static (e, t) => e.AdaptiveAvgPool2D(t.Reshape(new[] { 1, 2, 6, 5 }), 4, 3),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void AdaptiveAvgPool2D_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("AdaptiveAvgPool2D",
            static (e, t) => e.AdaptiveAvgPool2D(t.Reshape(new[] { 1, 2, 6, 5 }), 4, 3),
            static (x, y) => Assert.Equal(new[] { 1, 2, 4, 3 }, y.Shape.ToArray()));
    // Constant mode drops the fill positions; Reflect and Circular fold several output cells onto one input cell,
    // so the device scatter-add must accumulate them exactly as the host loop does.
    [SkippableTheory]
    [InlineData(PadMode.Constant)]
    [InlineData(PadMode.Reflect)]
    [InlineData(PadMode.Replicate)]
    [InlineData(PadMode.Circular)]
    public void PadNd_gradients_match_cpu(PadMode mode) =>
        AssertGradientParity($"PadNd({mode})", Rand([6, 10], seed: 103),
            (e, t) => e.PadNd(t, new[] { 3, 2, 1, 4 }, mode, 0.5f), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void PadNd_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("PadNd", static (e, t) => e.PadNd(t, new[] { 3, 2, 1, 4 }, PadMode.Constant, 0.5f),
            static (x, y) => { Assert.Equal(new[] { 11, 15 }, y.Shape.ToArray()); Assert.Equal(0.5f, y[0, 0]); Assert.Equal(x[0, 0], y[1, 3]); });
    private static readonly Tensor<int> ColumnIndices = new(new[] { 1, 4, 7 }, new[] { 3 });

    [SkippableFact]
    public void TensorIndexCopy_destination_gradients_match_cpu() =>
        AssertGradientParity("IndexCopy(dest)", Rand([6, 10], seed: 107),
            static (e, t) => e.TensorIndexCopy(t, 1, ColumnIndices, Rand([6, 3], seed: 109)),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorIndexCopy_source_gradients_match_cpu() =>
        AssertGradientParity("IndexCopy(source)", Rand([6, 3], seed: 113),
            static (e, t) => e.TensorIndexCopy(Rand([6, 10], seed: 127), 1, ColumnIndices, t),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorIndexFill_gradients_match_cpu() =>
        AssertGradientParity("IndexFill", Rand([6, 10], seed: 131),
            static (e, t) => e.TensorIndexFill(t, 1, ColumnIndices, -3f), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorIndexCopy_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorIndexCopy",
            static (e, t) => e.TensorIndexCopy(t, 1, ColumnIndices, Rand([6, 3], seed: 137)),
            static (x, y) => Assert.Equal(x[0, 0], y[0, 0]));

    [SkippableFact]
    public void TensorIndexFill_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorIndexFill", static (e, t) => e.TensorIndexFill(t, 1, ColumnIndices, -3f),
            static (x, y) => { Assert.Equal(-3f, y[0, 1]); Assert.Equal(x[0, 0], y[0, 0]); });
    // BitMask has 20 trues; a 25-element source leaves a tail the forward never reads, whose gradient must be 0.
    [SkippableFact]
    public void TensorMaskedScatter_destination_gradients_match_cpu() =>
        AssertGradientParity("MaskedScatter(dest)", Rand([6, 10], seed: 139),
            static (e, t) => e.TensorMaskedScatter(t, BitMask(), Rand([25], seed: 149)),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorMaskedScatter_source_gradients_match_cpu() =>
        AssertGradientParity("MaskedScatter(source)", Rand([25], seed: 151),
            static (e, t) => e.TensorMaskedScatter(Rand([6, 10], seed: 157), BitMask(), t),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorMaskedScatter_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorMaskedScatter",
            static (e, t) => e.TensorMaskedScatter(t, BitMask(), Rand([25], seed: 163)),
            static (x, y) => Assert.Equal(x[0, 1], y[0, 1]));
    [SkippableFact]
    public void TensorSetSlice_destination_gradients_match_cpu() =>
        AssertGradientParity("SetSlice(dest)", Rand([6, 10], seed: 167),
            static (e, t) => e.TensorSetSlice(t, Rand([2, 4], seed: 173), new[] { 3, 5 }),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorSetSlice_source_gradients_match_cpu() =>
        AssertGradientParity("SetSlice(source)", Rand([2, 4], seed: 179),
            static (e, t) => e.TensorSetSlice(Rand([6, 10], seed: 181), t, new[] { 3, 5 }),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorSetSlice_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorSetSlice",
            static (e, t) => e.TensorSetSlice(t, Rand([2, 4], seed: 191), new[] { 3, 5 }),
            static (x, y) => Assert.Equal(x[0, 0], y[0, 0]));
    private static readonly Tensor<float> LinearWeight = Rand([10, 4], seed: 193);
    private static readonly Tensor<float> LinearBias = Rand([4], seed: 197);

    public enum FusedLinearVariant { Generic, ReLU, Sigmoid, Tanh, GELU, Swish }

    private static Tensor<float> RunFusedLinear(IEngine e, Tensor<float> x, FusedLinearVariant variant) => variant switch
    {
        FusedLinearVariant.Generic => e.FusedLinear(x, LinearWeight, LinearBias, FusedActivationType.ReLU),
        FusedLinearVariant.ReLU => e.FusedLinearReLU(x, LinearWeight, LinearBias),
        FusedLinearVariant.Sigmoid => e.FusedLinearSigmoid(x, LinearWeight, LinearBias),
        FusedLinearVariant.Tanh => e.FusedLinearTanh(x, LinearWeight, LinearBias),
        FusedLinearVariant.GELU => e.FusedLinearGELU(x, LinearWeight, LinearBias),
        _ => e.FusedLinearSwish(x, LinearWeight, LinearBias),
    };

    [SkippableTheory]
    [InlineData(FusedLinearVariant.Generic)]
    [InlineData(FusedLinearVariant.ReLU)]
    [InlineData(FusedLinearVariant.Sigmoid)]
    [InlineData(FusedLinearVariant.Tanh)]
    [InlineData(FusedLinearVariant.GELU)]
    [InlineData(FusedLinearVariant.Swish)]
    public void FusedLinear_gradients_match_cpu(FusedLinearVariant variant) =>
        AssertGradientParity($"FusedLinear({variant})", Rand([6, 10], seed: 199),
            (e, t) => RunFusedLinear(e, t, variant), probe: Engagement.UseResidencyCounter);

    [SkippableTheory]
    [InlineData(FusedLinearVariant.Generic)]
    [InlineData(FusedLinearVariant.ReLU)]
    [InlineData(FusedLinearVariant.Sigmoid)]
    [InlineData(FusedLinearVariant.Tanh)]
    [InlineData(FusedLinearVariant.GELU)]
    [InlineData(FusedLinearVariant.Swish)]
    public void FusedLinear_stays_on_the_device_while_a_tape_records(FusedLinearVariant variant) =>
        AssertStaysOnDeviceUnderTape($"FusedLinear({variant})", (e, t) => RunFusedLinear(e, t, variant),
            static (x, y) => Assert.Equal(new[] { 6, 4 }, y.Shape.ToArray()));
    // The generic FusedLinear across every activation it takes on the device under a tape.
    [SkippableTheory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.Sigmoid)]
    [InlineData(FusedActivationType.Tanh)]
    [InlineData(FusedActivationType.GELU)]
    [InlineData(FusedActivationType.Swish)]
    [InlineData(FusedActivationType.LeakyReLU)]   // no CPU-matching kernel: goes through ActivationRegistry
    [InlineData(FusedActivationType.Mish)]
    public void FusedLinear_generic_activation_gradients_match_cpu(FusedActivationType activation) =>
        AssertGradientParity($"FusedLinear({activation})", Rand([6, 10], seed: 211),
            (e, t) => e.FusedLinear(t, LinearWeight, LinearBias, activation), probe: Engagement.UseResidencyCounter);

    [SkippableTheory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.Sigmoid)]
    [InlineData(FusedActivationType.Tanh)]
    [InlineData(FusedActivationType.GELU)]
    [InlineData(FusedActivationType.Swish)]
    [InlineData(FusedActivationType.LeakyReLU)]
    [InlineData(FusedActivationType.Mish)]
    public void FusedLinear_generic_activation_taped_forward_matches_cpu(FusedActivationType activation) =>
        AssertTapedForwardMatchesCpu($"FusedLinear({activation})", Rand([6, 10], seed: 223),
            (e, t) => e.FusedLinear(t, LinearWeight, LinearBias, activation));

    [SkippableTheory]
    [InlineData(FusedLinearVariant.ReLU)]
    [InlineData(FusedLinearVariant.Sigmoid)]
    [InlineData(FusedLinearVariant.Tanh)]
    [InlineData(FusedLinearVariant.GELU)]
    [InlineData(FusedLinearVariant.Swish)]
    public void FusedLinear_named_variant_taped_forward_matches_cpu(FusedLinearVariant variant) =>
        AssertTapedForwardMatchesCpu($"FusedLinear({variant})", Rand([6, 10], seed: 227),
            (e, t) => RunFusedLinear(e, t, variant));

    [SkippableTheory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.Sigmoid)]
    [InlineData(FusedActivationType.GELU)]
    public void FusedLinear_generic_activation_stays_on_the_device_while_a_tape_records(FusedActivationType activation) =>
        AssertStaysOnDeviceUnderTape($"FusedLinear({activation})",
            (e, t) => e.FusedLinear(t, LinearWeight, LinearBias, activation),
            static (x, y) => Assert.Equal(new[] { 6, 4 }, y.Shape.ToArray()));
    // [6,10] viewed as [1,2,5,6]; pool 3 stride 2 with padding exercises windows that cover 1, 2, 4 and 6 real cells.
    [SkippableTheory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    public void AvgPool2D_gradients_match_cpu(int padding, bool countIncludePad) =>
        AssertGradientParity($"AvgPool2D(p{padding},{countIncludePad})", Rand([6, 10], seed: 229),
            (e, t) => e.AvgPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), 3, 2, padding, countIncludePad),
            probe: Engagement.UseResidencyCounter);

    [SkippableTheory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    public void AvgPool2D_taped_forward_matches_cpu(int padding, bool countIncludePad) =>
        AssertTapedForwardMatchesCpu($"AvgPool2D(p{padding},{countIncludePad})", Rand([6, 10], seed: 233),
            (e, t) => e.AvgPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), 3, 2, padding, countIncludePad));

    [SkippableFact]
    public void AvgPool2D_int_array_overload_gradients_match_cpu() =>
        AssertGradientParity("AvgPool2D(int[])", Rand([6, 10], seed: 239),
            static (e, t) => e.AvgPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), new[] { 2, 3 }, new[] { 1, 2 }),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void AvgPool2D_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("AvgPool2D",
            static (e, t) => e.AvgPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), 3, 2, 1),
            static (x, y) => Assert.Equal(new[] { 1, 2, 3, 3 }, y.Shape.ToArray()));
    // Rand is continuous, so no window has a tie and the winner (hence the routed gradient) is unambiguous.
    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void MaxPool2D_gradients_match_cpu(int padding) =>
        AssertGradientParity($"MaxPool2D(p{padding})", Rand([6, 10], seed: 241),
            (e, t) => e.MaxPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), 3, 2, padding), probe: Engagement.UseResidencyCounter);

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void MaxPool2D_taped_forward_matches_cpu(int padding) =>
        AssertTapedForwardMatchesCpu($"MaxPool2D(p{padding})", Rand([6, 10], seed: 251),
            (e, t) => e.MaxPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), 3, 2, padding));

    [SkippableFact]
    public void MaxPool2D_padded_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("MaxPool2D",
            static (e, t) => e.MaxPool2D(t.Reshape(new[] { 1, 2, 5, 6 }), 3, 2, 1),
            static (x, y) => Assert.Equal(new[] { 1, 2, 3, 3 }, y.Shape.ToArray()));
    [SkippableFact]
    public void TensorClampMin_gradients_match_cpu() =>
        AssertGradientParity("ClampMin", Rand([6, 10], seed: 73),
            static (e, t) => e.TensorClampMin(t, 0.1f), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorClampMax_gradients_match_cpu() =>
        AssertGradientParity("ClampMax", Rand([6, 10], seed: 79),
            static (e, t) => e.TensorClampMax(t, 0.1f), probe: Engagement.UseResidencyCounter);

    // Inputs EXACTLY at the bound: the host keeps the gradient there (>= / <=), so the device predicate's
    // equality term is what these pin. Random inputs never land on 0.1f, so the tests above cannot see it.
    [SkippableFact]
    public void TensorClampMin_passes_the_gradient_at_the_bound() =>
        AssertGradientParity("ClampMin(ties)", WithBoundTies(Rand([6, 10], seed: 97)),
            static (e, t) => e.TensorClampMin(t, 0.1f), probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorClampMax_passes_the_gradient_at_the_bound() =>
        AssertGradientParity("ClampMax(ties)", WithBoundTies(Rand([6, 10], seed: 101)),
            static (e, t) => e.TensorClampMax(t, 0.1f), probe: Engagement.UseResidencyCounter);

    private static Tensor<float> WithBoundTies(Tensor<float> x)
    {
        for (int i = 0; i < x.Length; i += 7) x[i] = 0.1f;
        return x;
    }
    [SkippableFact]
    public void TensorClampMin_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorClampMin", static (e, t) => e.TensorClampMin(t, 0.1f),
            static (x, y) => Assert.Equal(Math.Max(x[0, 0], 0.1f), y[0, 0]));

    [SkippableFact]
    public void TensorClampMax_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorClampMax", static (e, t) => e.TensorClampMax(t, 0.1f),
            static (x, y) => Assert.Equal(Math.Min(x[0, 0], 0.1f), y[0, 0]));
    [SkippableFact]
    public void TensorWhere_bool_condition_gradients_match_cpu() =>
        AssertGradientParity("Where(bool)", Rand([6, 10], seed: 53),
            static (e, t) => ((IEngine)e).TensorWhere(BoolMask(), t, Rand([6, 10], seed: 59)),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorWhere_bit_condition_gradients_match_cpu() =>
        AssertGradientParity("Where(Bit)", Rand([6, 10], seed: 61),
            static (e, t) => e.TensorWhere(BitMask(), Rand([6, 10], seed: 67), t),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorWhere_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorWhere",
            static (e, t) => ((IEngine)e).TensorWhere(BoolMask(), t, Rand([6, 10], seed: 71)),
            static (x, y) => Assert.Equal(x[0, 0], y[0, 0]));

    [SkippableFact]
    public void TensorTranspose_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorTranspose", static (e, t) => e.TensorTranspose(t), static (x, y) =>
        {
            Assert.Equal(new[] { 10, 6 }, y.Shape.ToArray());
            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 10; c++)
                    Assert.Equal(x[r, c], y[c, r]);
        });

    [SkippableFact]
    public void TensorAddScalar_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorAddScalar", static (e, t) => e.TensorAddScalar(t, 0.75f),
            static (x, y) => { for (int i = 0; i < x.Length; i++) Assert.Equal(x[i] + 0.75f, y[i], 6); });

    [SkippableFact]
    public void TensorSubtractScalar_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorSubtractScalar", static (e, t) => e.TensorSubtractScalar(t, 0.75f),
            static (x, y) => { for (int i = 0; i < x.Length; i++) Assert.Equal(x[i] - 0.75f, y[i], 6); });

    [SkippableFact]
    public void TensorDivideScalar_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("TensorDivideScalar", static (e, t) => e.TensorDivideScalar(t, 1.7f),
            static (x, y) => { for (int i = 0; i < x.Length; i++) Assert.Equal(x[i] / 1.7f, y[i], 5); });

    /// <summary>
    /// NarrowBackward on the GPU engine must build the input gradient on the device from a resident gradOutput:
    /// no readback, at least one launch, and the same values CpuEngine produces. It used to read gradOutput
    /// through a host span and return a host tensor (seen per sample in AutoformerModel training).
    /// </summary>
    [SkippableFact]
    public void NarrowBackward_builds_the_input_gradient_on_the_device()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu!)
        {
            IEngine engine = gpu!;
            AiDotNetEngine.Current = engine;
            var input = Rand([3, 7, 4], seed: 29);
            var hostGrad = Rand([3, 2, 4], seed: 30);
            var residentGrad = engine.TensorAddScalar(hostGrad, 0f);   // a device-resident gradOutput
            var saved = new object[] { 1, 3, 2 };                        // dim 1, start 3, length 2

            var cpuGrads = new Dictionary<Tensor<float>, Tensor<float>>();
            BackwardFunctions<float>.NarrowBackward(hostGrad, [input], hostGrad, saved, new CpuEngine(), cpuGrads);

            AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();
            var gpuGrads = new Dictionary<Tensor<float>, Tensor<float>>();
            BackwardFunctions<float>.NarrowBackward(residentGrad, [input], residentGrad, saved, engine, gpuGrads);
            long launches = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count;
            long readbacks = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Readbacks;
            _out.WriteLine($"NarrowBackward on GPU: launches={launches} readbacks={readbacks}");

            Assert.True(launches >= 1, "NarrowBackward launched nothing on the GPU engine — it built the gradient on the host.");
            Assert.Equal(0, readbacks);
            var expected = cpuGrads[input];
            var actual = gpuGrads[input];
            Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
        }
    }

    private static readonly Tensor<float> LnGamma = Rand([10], seed: 31, lo: 0.5, hi: 1.5);
    private static readonly Tensor<float> LnBeta = Rand([10], seed: 32);

    /// <summary>
    /// LayerNorm's backward consumes the forward's per-row mean and variance, so a wrong saved state (the GPU
    /// kernel's variance slot holds INVERSE std until converted) shows up as a gradient mismatch, not rounding.
    /// </summary>
    [SkippableFact]
    public void LayerNorm_gradients_match_cpu() =>
        AssertGradientParity("LayerNorm", Rand([6, 10], seed: 33, lo: -2.0, hi: 2.0),
            static (e, t) => e.LayerNorm(t, LnGamma, LnBeta, 1e-5, out _, out _));

    [SkippableFact]
    public void LayerNorm_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("LayerNorm", static (e, t) => e.LayerNorm(t, LnGamma, LnBeta, 1e-5, out _, out _),
            static (x, y) =>
            {
                var expected = new CpuEngine().LayerNorm(x, LnGamma, LnBeta, 1e-5, out _, out _);
                for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], y[i], 4);
            });

    private static readonly Tensor<float> ConcatTail = Rand([6, 3], seed: 34);

    /// <summary>Concat only moves data (exact on both engines), so the residency counter proves the device ran.</summary>
    [SkippableFact]
    public void Concat_gradients_match_cpu() =>
        AssertGradientParity("Concat", Rand([6, 10], seed: 35),
            static (e, t) => e.Concat(new[] { t, ConcatTail }, -1),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void Concat_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("Concat", static (e, t) => e.Concat(new[] { t, ConcatTail }, 1), static (x, y) =>
        {
            Assert.Equal(new[] { 6, 13 }, y.Shape.ToArray());
            for (int r = 0; r < 6; r++)
            {
                for (int c = 0; c < 10; c++) Assert.Equal(x[r, c], y[r, c]);
                for (int c = 0; c < 3; c++) Assert.Equal(ConcatTail[r, c], y[r, 10 + c]);
            }
        });

    /// <summary>
    /// ReduceSum's gradient is a broadcast of gradOutput (exact on both engines), so a DOUBLE recording — the
    /// general path composing permute/reshape that also recorded — would show up as exactly 2x here.
    /// </summary>
    [SkippableTheory]
    [InlineData(0)]    // general path (non-innermost axis)
    [InlineData(1)]    // IEngine innermost-axis kernel
    [InlineData(-1)]   // full reduction
    public void ReduceSum_gradients_match_cpu(int axis) =>
        AssertGradientParity($"ReduceSum[{axis}]", Rand([6, 10], seed: 36),
            (e, t) => axis < 0 ? e.ReduceSum(t, null, keepDims: true) : e.ReduceSum(t, new[] { axis }, keepDims: true),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void ReduceSum_stays_on_the_device_while_a_tape_records() =>
        AssertStaysOnDeviceUnderTape("ReduceSum", static (e, t) => e.ReduceSum(t, new[] { 0 }, keepDims: false), static (x, y) =>
        {
            Assert.Equal(new[] { 10 }, y.Shape.ToArray());
            for (int c = 0; c < 10; c++)
            {
                float sum = 0; for (int r = 0; r < 6; r++) sum += x[r, c];
                Assert.Equal(sum, y[c], 4);
            }
        });

    /// <summary>
    /// Mean and MSE-loss backwards scaled by <c>gradOutput[0]</c> — a blocking readback of the resident upstream
    /// gradient in every backward pass. On the GPU engine they must now read nothing back and still match CPU.
    /// </summary>
    [SkippableTheory]
    [InlineData("MeanBackward")]
    [InlineData("MSELossBackward")]
    public void Scalar_scaled_backwards_do_not_read_the_upstream_gradient_back(string backward)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu!)
        {
            IEngine engine = gpu!;
            AiDotNetEngine.Current = engine;
            var predictions = Rand([5, 8], seed: 37);
            var targets = Rand([5, 8], seed: 38);
            var hostUpstream = new Tensor<float>([1]); hostUpstream[0] = 0.83f;
            var residentUpstream = engine.TensorAddScalar(hostUpstream, 0f);
            Tensor<float>[] inputs = backward == "MeanBackward" ? [predictions] : [predictions, targets];
            BackwardFunction<float> fn = backward == "MeanBackward"
                ? BackwardFunctions<float>.MeanBackward
                : BackwardFunctions<float>.MSELossBackward;
            _ = engine.TensorAdd(predictions, targets);          // upload both inputs outside the count

            var cpuGrads = new Dictionary<Tensor<float>, Tensor<float>>();
            fn(hostUpstream, inputs, hostUpstream, [], new CpuEngine(), cpuGrads);

            AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();
            var gpuGrads = new Dictionary<Tensor<float>, Tensor<float>>();
            fn(residentUpstream, inputs, residentUpstream, [], engine, gpuGrads);
            long launches = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count;
            long readbacks = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Readbacks;
            _out.WriteLine($"{backward} on GPU: launches={launches} readbacks={readbacks}");

            Assert.True(launches >= 1, $"{backward} launched nothing on the GPU engine.");
            Assert.Equal(0, readbacks);
            var expected = cpuGrads[predictions];
            var actual = gpuGrads[predictions];
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
        }
    }

    /// <summary>
    /// ELU's derivative has a kink at 0 when alpha != 1. The tape backward (like PyTorch) takes the alpha branch at
    /// exactly 0; the CPU engine op and the CUDA/HIP/WebGPU kernels took 1 while OpenCL took alpha. All must agree
    /// now, at the one point random data never lands on — and the tape backward must run on the device.
    /// </summary>
    [SkippableFact]
    public void ELU_backward_agrees_at_zero_across_engines_and_stays_on_the_device()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu!)
        {
            IEngine engine = gpu!;
            AiDotNetEngine.Current = engine;
            const double alpha = 0.5;
            var x = new Tensor<float>([2, 4]);
            float[] values = [-2f, -0.5f, 0f, 0f, 0.25f, 1f, -1e-3f, 3f];
            for (int i = 0; i < values.Length; i++) x[i] = values[i];
            var cpu = new CpuEngine();
            var y = cpu.ELU(x, alpha);
            var g = Rand([2, 4], seed: 39, lo: 0.5, hi: 1.5);

            var cpuGrads = new Dictionary<Tensor<float>, Tensor<float>>();
            BackwardFunctions<float>.ELUBackward(g, [x], y, [alpha], cpu, cpuGrads);
            var cpuOp = cpu.EluBackward(g, x, y, alpha);

            var residentG = engine.TensorAddScalar(g, 0f);
            _ = engine.TensorAdd(x, y);                                   // upload x and y outside the count
            AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();
            var gpuGrads = new Dictionary<Tensor<float>, Tensor<float>>();
            BackwardFunctions<float>.ELUBackward(residentG, [x], y, [alpha], engine, gpuGrads);
            long launches = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Count;
            long readbacks = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Readbacks;
            _out.WriteLine($"ELUBackward on GPU: launches={launches} readbacks={readbacks}");
            Assert.True(launches >= 1, "ELUBackward launched nothing on the GPU engine.");
            Assert.Equal(0, readbacks);

            var gpuOp = engine.EluBackward(g, x, y, alpha);
            for (int i = 0; i < values.Length; i++)
            {
                float expected = values[i] > 0 ? g[i] : g[i] * (y[i] + (float)alpha);   // PyTorch convention
                Assert.Equal(expected, cpuGrads[x][i], 5);
                Assert.Equal(expected, gpuGrads[x][i], 5);
                Assert.Equal(expected, cpuOp[i], 5);
                Assert.Equal(expected, gpuOp[i], 5);
            }
        }
    }

    [SkippableFact]
    public void TensorAddScalar_gradients_match_cpu() =>
        AssertGradientParity("TensorAddScalar", Rand([4, 16], seed: 26), static (e, t) => e.TensorAddScalar(t, 0.75f),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorSubtractScalar_gradients_match_cpu() =>
        AssertGradientParity("TensorSubtractScalar", Rand([4, 16], seed: 27), static (e, t) => e.TensorSubtractScalar(t, 0.75f),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorDivideScalar_gradients_match_cpu() =>
        AssertGradientParity("TensorDivideScalar", Rand([4, 16], seed: 28), static (e, t) => e.TensorDivideScalar(t, 1.7f),
            probe: Engagement.UseResidencyCounter);

    [SkippableFact]
    public void TensorSinh_gradients_match_cpu() =>
        AssertGradientParity("TensorSinh", Rand([4, 16], seed: 22, lo: -2.0, hi: 2.0),
            static (e, t) => e.TensorSinh(t));

    /// <summary>Rank-5 [N,C,D,H,W] is the only shape the GPU Upsample3D kernel accepts.</summary>
    [SkippableFact]
    public void Upsample3D_gradients_match_cpu() =>
        AssertGradientParity("Upsample3D", Rand([2, 2, 3, 4, 4], seed: 23),
            static (e, t) => e.Upsample3D(t, 2, 2, 2),
            probe: Engagement.UseResidencyCounter);

    /// <summary>
    /// AffineGrid: theta is [batch, 2, 3], grid is [batch, H, W, 2].
    /// </summary>
    /// <remarks>
    /// Uses the RESIDENCY probe, not divergence. d(grid)/d(theta) is the normalized coordinate grid, fixed
    /// entirely by the output geometry (H, W) — it does NOT depend on any value the forward produced. So
    /// forward precision never reaches the gradient and CPU/GPU agree to the last bit, exactly as for
    /// Upsample3D. Measured: maxAbs 0.000E+000 with 4 materialisations, i.e. the device path ran and the
    /// exact agreement is correct.
    ///
    /// The general rule this establishes: the divergence probe is only valid when the BACKWARD consumes
    /// values the FORWARD computed. "Computes rather than selects" is not sufficient — that was the
    /// distinction I first reasoned from, and it predicted a non-zero delta here, wrongly.
    /// </remarks>
    [SkippableFact]
    public void AffineGrid_gradients_match_cpu() =>
        AssertGradientParity("AffineGrid", Rand([2, 2, 3], seed: 41),
            static (e, t) => e.AffineGrid(t, 5, 7),
            probe: Engagement.UseResidencyCounter);

    /// <summary>
    /// RBFKernel records THREE differentiable inputs (input, centers, epsilons), so all three are checked.
    /// Scatter showed one operand can be exactly right while another is corrupt.
    /// </summary>
    /// <remarks>
    /// EPSILONS ARE DELIBERATELY UNIFORM. CpuEngine saves only epsilons[0] as backward state, so
    /// RBFKernelBackward treats the width as shared across all centers — with varying epsilons the CPU
    /// reference is itself wrong for centers 1..n, and agreeing with a wrong reference would prove nothing.
    /// Uniform epsilons keep the reference correct so this test measures the GPU, not the shared defect.
    /// </remarks>
    [SkippableFact]
    public void RBFKernel_gradients_match_cpu_for_all_three_operands()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null,
            "GPU backend did not resolve — RBFKernel would have been compared against itself.");

        using (gpu!)
        {
            const int batch = 4, features = 3, centers = 5;
            var inputSeed = Rand([batch, features], seed: 51);
            var centerSeed = Rand([centers, features], seed: 52);

            (float[] gi, float[] gc, float[] ge, long mat) Run(IEngine engine)
            {
                AiDotNetEngine.Current = engine;
                var x = new Tensor<float>([batch, features]);
                for (int i = 0; i < x.Length; i++) x[i] = inputSeed[i];
                var c = new Tensor<float>([centers, features]);
                for (int i = 0; i < c.Length; i++) c[i] = centerSeed[i];
                var eps = new Tensor<float>([centers]);
                for (int i = 0; i < centers; i++) eps[i] = 0.75f;   // uniform — see remarks

                AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
                using var tape = new GradientTape<float>();
                var y = engine.RBFKernel(x, c, eps);

                var w = new Tensor<float>(y.Shape.ToArray());
                for (int i = 0; i < w.Length; i++) w[i] = 0.17f + 0.011f * (i % 7);
                var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);

                var grads = tape.ComputeGradients(loss, new[] { x, c, eps });
                Assert.True(grads.TryGetValue(x, out var g1) && g1 is not null, "no gradient to input");
                Assert.True(grads.TryGetValue(c, out var g2) && g2 is not null, "no gradient to centers");
                Assert.True(grads.TryGetValue(eps, out var g3) && g3 is not null, "no gradient to epsilons");

                float[] Flat(Tensor<float> t) { var a = new float[t.Length]; for (int i = 0; i < a.Length; i++) a[i] = t[i]; return a; }
                return (Flat(g1), Flat(g2), Flat(g3),
                        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount);
            }

            var (cI, cC, cE, _) = Run(new CpuEngine());
            var (gI, gC, gE, mat) = Run(gpu);

            double dI = 0, dC = 0, dE = 0;
            for (int i = 0; i < cI.Length; i++) dI = Math.Max(dI, Math.Abs(cI[i] - gI[i]));
            for (int i = 0; i < cC.Length; i++) dC = Math.Max(dC, Math.Abs(cC[i] - gC[i]));
            for (int i = 0; i < cE.Length; i++) dE = Math.Max(dE, Math.Abs(cE[i] - gE[i]));

            _out.WriteLine($"RBFKernel      d(input)={dI:E3}  d(centers)={dC:E3}  d(epsilons)={dE:E3}  materialisations={mat}");

            Assert.True(mat > 0, "RBFKernel: nothing was GPU-resident, so the device path did not run.");
            Assert.True(dI <= 1e-4, $"RBFKernel input gradient diverged: {dI:E3}");
            Assert.True(dC <= 1e-4, $"RBFKernel centers gradient diverged: {dC:E3}");
            Assert.True(dE <= 1e-4, $"RBFKernel epsilons gradient diverged: {dE:E3}");
        }
    }

    /// <summary>
    /// MaxPool3DWithIndices: gradient routes only to the pooled maxima, so this verifies the index decode
    /// as much as the recording.
    /// </summary>
    /// <remarks>
    /// TIE-FREE INPUT IS REQUIRED. Continuous random values make exact ties measure-zero. With ties, CPU
    /// and GPU could select different argmax positions — the forward output would be IDENTICAL (same max
    /// value) while gradients landed on different elements, which reads as a defect but is really a
    /// test-design flaw.
    ///
    /// Residency probe: the backward scatters gradOutput to the selected positions with no arithmetic on
    /// forward values, so CPU and GPU agree bit-for-bit when the indices agree.
    /// </remarks>
    [SkippableFact]
    public void MaxPool3DWithIndices_gradients_match_cpu() =>
        AssertGradientParity("MaxPool3D", Rand([2, 2, 4, 4, 4], seed: 61, lo: -5.0, hi: 5.0),
            static (e, t) => e.MaxPool3DWithIndices(t, new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, out _),
            probe: Engagement.UseResidencyCounter);

    // Scatter has NO test here: its bail was RESTORED because this very test caught a real defect —
    // recording CpuEngine's ScatterBackward on the GPU result gave d(values) exactly right but d(input)
    // wrong by 3.14e-01. Scatter overwrites input at the scattered positions, so d/d(input) must be zero
    // there, and the reused recording did not reproduce that mask. Restore this test together with a
    // correct GPU-side backward. Keeping it while the op bails would only assert CPU against CPU.

    // Sparsemax has NO test here on purpose: its bail was restored because the GPU path throws
    // InvalidOperationException("CUDA kernel not found: where_select"). Add the test back together with
    // the where_select kernel.
}
