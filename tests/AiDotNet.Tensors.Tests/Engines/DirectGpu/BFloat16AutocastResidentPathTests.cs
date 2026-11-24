using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// BFloat16 autocast has no typed elementwise kernels, so the allowlisted elementwise ops run FP32 under it. The
/// resident FP32 paths used to be gated off for EVERY autocast scope, which sent a BF16 training step's in-place
/// updates, divides and activations to host round trips (DCGAN BF16: about 20 device-to-host reads per step, 111 ms
/// against 12 ms in FP32). Under BF16 these ops must stay on the device; under FP16 they keep the typed route.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class BFloat16AutocastResidentPathTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public BFloat16AutocastResidentPathTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static (Tensor<float> A, Tensor<float> B) Operands(DirectGpuTensorEngine engine)
    {
        var source = new Tensor<float>(Enumerable.Range(1, 4096).Select(i => i * 0.25f).ToArray(), new[] { 64, 64 });
        var b = new Tensor<float>(Enumerable.Range(1, 4096).Select(i => 1f + (i % 7)).ToArray(), new[] { 64, 64 });
        engine.EnsureResidentInput(source);
        engine.EnsureResidentInput(b);
        // The in-place target must be device-owned, as an activation or optimizer state in a training step is: an
        // external input stays host-authoritative and its in-place update rightly writes the host (FP32 does too).
        var a = engine.TensorMultiplyScalar(source, 1f);
        return (a, b);
    }

    [SkippableFact]
    public void InPlaceAdd_and_Divide_stay_on_the_device_under_bfloat16_autocast()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var engine = _fixture.Engine!;
        var (a, b) = Operands(engine);
        var expectedSum = a.ToArray().Zip(b.ToArray(), (x, y) => x + y).ToArray();
        var expectedQuotient = expectedSum.Zip(b.ToArray(), (x, y) => x / y).ToArray();

        Tensor<float> quotient;
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        GpuLaunchProbe.CaptureReadbackSites = true;
        try
        {
            using (new AutocastScope(PrecisionMode.BFloat16))
            {
                // Warm up on throwaway operands: the first sight of a shape lets the tuned-kernel registry verify its
                // candidates, which reads their outputs back once. The assertion is about the steady state.
                var (warmA, warmB) = Operands(engine);
                engine.TensorAddInPlace(warmA, warmB);
                _ = engine.TensorDivide(warmA, warmB);

                GpuLaunchProbe.Reset();
                engine.TensorAddInPlace(a, b);
                quotient = engine.TensorDivide(a, b);
                Assert.True(GpuLaunchProbe.Count > 0, "No GPU work was launched.");
                Assert.True(GpuLaunchProbe.Readbacks == 0,
                    $"BF16 autocast elementwise ops made {GpuLaunchProbe.Readbacks} device-to-host transfers: "
                    + string.Join("; ", GpuLaunchProbe.ReadbackSites));
            }
        }
        finally { GpuLaunchProbe.CaptureReadbackSites = savedCapture; }

        Assert.Equal(expectedSum, a.ToArray());
        // The device divide is IEEE-faithful within an ulp or two (CUDA's default division is not correctly rounded),
        // so the quotient is compared to two ulps rather than bitwise; the sum above is exact.
        var actualQuotient = quotient.ToArray();
        for (int i = 0; i < expectedQuotient.Length; i++)
            Assert.True(System.Math.Abs(actualQuotient[i] - expectedQuotient[i]) <= 2.5e-7f * System.Math.Abs(expectedQuotient[i]),
                $"quotient[{i}] = {actualQuotient[i]:R}, expected {expectedQuotient[i]:R}");
    }

    // The DCGAN BF16 regression lived in the compiled step: its resident kernels (in-place optimizer adds, divides,
    // activations) were gated off for every autocast scope and took host round trips on every step.
    [SkippableFact]
    public void Compiled_training_step_under_bfloat16_autocast_makes_no_device_to_host_reads()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var engine = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        try
        {
            var random = new System.Random(5);
            Tensor<float> Filled(int rows, int cols) => new Tensor<float>(
                Enumerable.Range(0, rows * cols).Select(_ => (float)(random.NextDouble() * 0.2 - 0.1)).ToArray(), new[] { rows, cols });
            var input = Filled(32, 64);
            var scale = new Tensor<float>(Enumerable.Range(0, 32 * 64).Select(i => 1.5f + (i % 3)).ToArray(), new[] { 32, 64 });
            var w1 = Filled(64, 64).Gpu();
            var w2 = Filled(64, 64).Gpu();
            ICompiledTrainingPlan<float> plan;
            using (var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.Enable())
            {
                var hidden = engine.TensorDivide(engine.Tanh(engine.TensorMatMul(input, w1)), scale);
                var output = engine.TensorMatMul(hidden, w2);
                engine.ReduceSum(engine.TensorMultiply(output, output), null);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
            AiDotNet.Tensors.Engines.Compilation.GraphMode.SetCurrent(null);
            using (plan)
            using (new AutocastScope(PrecisionMode.BFloat16))
            {
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3f);
                float first = plan.Step().ToArray()[0];
                for (int i = 0; i < 3; i++) plan.Step();   // warmup: tuning, residency and capture settle
                GpuLaunchProbe.CaptureReadbackSites = true;
                GpuLaunchProbe.Reset();
                Tensor<float> loss = plan.Step();
                loss = plan.Step();
                long readbacks = GpuLaunchProbe.Readbacks;
                string sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                Assert.True(readbacks == 0, $"BF16 compiled steps made {readbacks} device-to-host reads: {sites}");
                float last = loss.ToArray()[0];
                Assert.True(last < first, $"the BF16 compiled plan must train: first={first}, last={last}.");
            }
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    public void Float16_autocast_still_routes_the_add_through_the_precision_policy()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var engine = _fixture.Engine!;
        Skip.IfNot(engine.TestBackend is IGpuNarrowFloatElementwise halves
                && halves.SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation.Add, GpuElementType.Float16),
            "Native FP16 elementwise support is unavailable.");
        var (a, b) = Operands(engine);
        int adds = 0;
        void Count(GpuComputePlan plan) { if (plan.OperationKind == GpuPrecisionOperation.Add) adds++; }
        GpuPrecisionDiagnostics.PlanExecuted += Count;
        try
        {
            using var autocast = new AutocastScope(PrecisionMode.Float16);
            _ = engine.TensorAdd(a, b);
        }
        finally { GpuPrecisionDiagnostics.PlanExecuted -= Count; }
        Assert.Equal(1, adds);
    }
}
