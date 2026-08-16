using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Driver correctness for the FP32 fused Adam update: constructs the kernel, launches it, and
/// compares all three mutated buffers against a CPU reference.
/// </summary>
/// <remarks>
/// <para>
/// The existing Adam suite inspects emitted PTX and admission guards only — it never constructs or
/// launches the kernel. A wrong Driver API argument order, or a device arithmetic sequence that
/// does not match the intended update, passes that suite completely. Adam mutates
/// <c>param</c>, <c>m</c> and <c>v</c> in place, so a defect here corrupts optimizer state across
/// steps rather than returning a visibly wrong value once.
/// </para>
/// <para>
/// Skipped unless a validated SM86 device is present, using the exact admission predicate rather
/// than the Ampere family check: the constructor admits SM86 only, so a broad family check would
/// reach it on SM80 and throw instead of skipping.
/// </para>
/// </remarks>
public class DirectPtxAdamUpdateDriverTests
{
    private const float LearningRate = 1e-3f;
    private const float Beta1 = 0.9f;
    private const float Beta2 = 0.999f;
    private const float Epsilon = 1e-8f;

    /// <summary>
    /// Weight decay off and on, at more than one step, because the step drives both bias
    /// corrections and the decay presence changes the emitted body.
    /// </summary>
    public static TheoryData<float, int> Cases => new()
    {
        { 0f, 1 },
        { 0f, 7 },
        { 0.01f, 1 },
        { 0.01f, 7 },
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void AdamUpdate_MatchesCpuReference_OnValidatedDevice(float weightDecay, int step)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(
            DirectPtxArchitecture.HasValidatedAdamUpdate(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in Adam update specialization is admitted only on SM86.");

        const int size = 65_536;
        var hyperparameters = new DirectPtxAdamHyperparameters(
            LearningRate, Beta1, Beta2, Epsilon, weightDecay, step);

        using var kernel = new PtxFusedAdamUpdateF32Kernel(runtime, size, hyperparameters);

        var random = RandomHelper.CreateSeededRandom(20260722 + step);
        float[] param = Sample(random, size, 1.0);
        float[] gradient = Sample(random, size, 0.5);
        float[] firstMoment = Sample(random, size, 0.25);
        float[] secondMoment = Sample(random, size, 0.25).Select(Math.Abs).ToArray();

        // CPU reference, in the same folded form the kernel uses: the two bias corrections are
        // precomputed on the host precisely so the kernel multiplies instead of dividing.
        var expectedParam = new float[size];
        var expectedFirst = new float[size];
        var expectedSecond = new float[size];
        float lrOverBc1 = hyperparameters.LearningRateOverBiasCorrection1;
        float rsqrtBc2 = hyperparameters.ReciprocalSqrtBiasCorrection2;
        for (int i = 0; i < size; i++)
        {
            float g = gradient[i] + weightDecay * param[i];
            float m = Beta1 * firstMoment[i] + (1f - Beta1) * g;
            float v = Beta2 * secondMoment[i] + (1f - Beta2) * g * g;
            expectedFirst[i] = m;
            expectedSecond[i] = v;
            expectedParam[i] = param[i] - lrOverBc1 * m / ((float)Math.Sqrt(v) * rsqrtBc2 + Epsilon);
        }

        using var paramBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var gradientBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var firstBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var secondBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        paramBuffer.Upload<float>(param);
        gradientBuffer.Upload<float>(gradient);
        firstBuffer.Upload<float>(firstMoment);
        secondBuffer.Upload<float>(secondMoment);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(paramBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(gradientBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(firstBuffer, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(secondBuffer, kernel.Blueprint.Tensors[3]),
            hyperparameters);
        runtime.Synchronize();

        var actualParam = new float[size];
        var actualFirst = new float[size];
        var actualSecond = new float[size];
        paramBuffer.Download<float>(actualParam);
        firstBuffer.Download<float>(actualFirst);
        secondBuffer.Download<float>(actualSecond);

        // All THREE mutated buffers. Checking param alone would miss a moment update that is wrong
        // in a way this step happens to mask but the next step would compound.
        AssertClose(expectedFirst, actualFirst, "m");
        AssertClose(expectedSecond, actualSecond, "v");
        AssertClose(expectedParam, actualParam, "param");
    }

    /// <summary>
    /// Aliased buffers must be refused rather than silently corrupting optimizer state.
    /// </summary>
    /// <remarks>
    /// The blueprint declares param, m and v mutually disjoint. The gradient is loaded with
    /// <c>ld.global.nc.v4.f32</c>, which asserts non-aliasing to the compiler, so an aliased launch
    /// is undefined behaviour rather than merely wrong — and the in-place stores have no defined
    /// ordering between threads.
    /// </remarks>
    [SkippableFact]
    public void AdamUpdate_RefusesAliasedBuffers()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(
            DirectPtxArchitecture.HasValidatedAdamUpdate(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in Adam update specialization is admitted only on SM86.");

        const int size = 65_536;
        var hyperparameters = new DirectPtxAdamHyperparameters(
            LearningRate, Beta1, Beta2, Epsilon, 0f, 1);
        using var kernel = new PtxFusedAdamUpdateF32Kernel(runtime, size, hyperparameters);

        using var paramBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var gradientBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var firstBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);

        // v aliases m.
        Assert.Throws<ArgumentException>(() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(paramBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(gradientBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(firstBuffer, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(firstBuffer, kernel.Blueprint.Tensors[3]),
            hyperparameters));
    }

    private static float[] Sample(Random random, int size, double scale) =>
        Enumerable.Range(0, size)
            .Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * scale))
            .ToArray();

    private static void AssertClose(float[] expected, float[] actual, string name)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            // Relative tolerance: nvcc may contract multiply-adds into FMAs, so this is not
            // bit-exact by design. A wrong argument order or a wrong operation is off by far more.
            float tolerance = 1e-5f * Math.Max(1f, Math.Abs(expected[i]));
            Assert.True(
                Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{name}[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }
}
