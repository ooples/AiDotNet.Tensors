using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The capsule-squash direct-PTX specialization (#833). The emitter, shape-domain and gate assertions run without a GPU.
/// The driver assertions skip unless an SM86 (GA10x) NVIDIA GPU is present; they are the verification the family needs
/// before <see cref="DirectPtxFeatureGate.CapsuleSquashPromoted"/> may be set (see that constant).
/// </summary>
public class DirectPtxSquashTests
{
    public static TheoryData<int, int> Shapes => new()
    {
        { 1152, 8 }, { 9216, 8 }, { 1280, 16 }, { 10240, 16 },
    };

    [Fact]
    public void ForwardEmitter_IsThreadResidentPointerOnlyAndFp64()
    {
        string ptx = PtxFusedSquashF32Kernel.EmitPtx(8, 6, SquashDirection.Forward, 1152, 8);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape capsules=1152 dim=8 block=128", ptx);
        Assert.Contains($".visible .entry {PtxFusedSquashF32Kernel.ForwardEntryPoint}(", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.v4.f32"));   // 8 floats = two 16-byte loads
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(8, Count(ptx, "fma.rn.f64"));            // r2, one per element
        Assert.Equal(1, Count(ptx, "sqrt.rn.f64"));
        Assert.Equal(1, Count(ptx, "div.rn.f64"));
        Assert.Equal(8, Count(ptx, "cvt.rn.f32.f64"));        // one rounding per element, at the store
        // eps is the FP32 1e-8 widened to FP64, as the CUDA source's (double)epsilon.
        Assert.Contains("0d" + BitConverter.DoubleToInt64Bits((double)1e-8f).ToString("X16"), ptx);
        Assert.Equal(4, Count(PtxFusedSquashF32Kernel.EmitPtx(8, 6, SquashDirection.Forward, 1280, 16),
            "ld.global.nc.v4.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .f32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void BackwardEmitter_ReadsBothInputsOnceAndWritesOnce()
    {
        string ptx = PtxFusedSquashF32Kernel.EmitPtx(8, 6, SquashDirection.Backward, 1280, 16);
        Assert.Contains($".visible .entry {PtxFusedSquashF32Kernel.BackwardEntryPoint}(", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(8, Count(ptx, "ld.global.nc.v4.f32"));   // x and g, 16 floats each
        Assert.Equal(4, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(2, Count(ptx, "div.rn.f64"));            // scale and coefficient
        Assert.Equal(16, Count(ptx, "cvt.rn.f32.f64"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromoted()
    {
        foreach (var (n, d) in new[] { (1152, 8), (9216, 8), (1280, 16), (10240, 16) })
        {
            Assert.True(PtxFusedSquashF32Kernel.IsSupportedShape(n, d));
            Assert.False(PtxFusedSquashF32Kernel.IsPromotedShape(n, d));
        }
        Assert.False(PtxFusedSquashF32Kernel.IsSupportedShape(1152, 16));
        Assert.False(PtxFusedSquashF32Kernel.IsSupportedShape(100, 8));
        Assert.False(DirectPtxFeatureGate.CapsuleSquashPromoted);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedSquashF32Kernel.EmitPtx(8, 6, SquashDirection.Forward, 1000, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedSquashF32Kernel.EmitPtx(8, 6, SquashDirection.Forward, 1152, 8, blockThreads: 96));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void DriverOnlySquashForward_MatchesFp64ReferenceWithZeroLocalBytes(int numCapsules, int capsuleDim)
    {
        using var runtime = OpenValidatedRuntime();
        using var kernel = new PtxFusedSquashF32Kernel(runtime, SquashDirection.Forward, numCapsules, capsuleDim);
        AssertResidentAudit(kernel);

        int elements = numCapsules * capsuleDim;
        var random = RandomHelper.CreateSeededRandom(20261024);
        float[] input = Enumerable.Range(0, elements).Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 3.0)).ToArray();
        ZeroFirstCapsule(input, capsuleDim);   // the zero-denominator branch
        var expected = ForwardReference(input, numCapsules, capsuleDim);

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        kernel.LaunchForward(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[elements];
        outputBuffer.Download<float>(actual);
        AssertWithinOneUlp(expected, actual);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void DriverOnlySquashBackward_MatchesFp64ReferenceWithZeroLocalBytes(int numCapsules, int capsuleDim)
    {
        using var runtime = OpenValidatedRuntime();
        using var kernel = new PtxFusedSquashF32Kernel(runtime, SquashDirection.Backward, numCapsules, capsuleDim);
        AssertResidentAudit(kernel);

        int elements = numCapsules * capsuleDim;
        var random = RandomHelper.CreateSeededRandom(20261025);
        float[] input = Enumerable.Range(0, elements).Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 3.0)).ToArray();
        float[] grad = Enumerable.Range(0, elements).Select(_ => (float)(random.NextDouble() * 2.0 - 1.0)).ToArray();
        ZeroFirstCapsule(input, capsuleDim);
        var expected = BackwardReference(grad, input, numCapsules, capsuleDim);

        using var gradBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        gradBuffer.Upload<float>(grad);
        inputBuffer.Upload<float>(input);
        kernel.LaunchBackward(
            DirectPtxTensorView.CreateOwned(gradBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[elements];
        outputBuffer.Download<float>(actual);
        AssertWithinOneUlp(expected, actual);
    }

    /// <summary>
    /// The routed path against the NVRTC kernel it replaces, through the public backend entry, with the experiment
    /// override on: the two must agree to within one FP32 ulp (the PTX keeps the CUDA source's FP64 order and
    /// contractions, so they are expected to be bit-identical; one ulp leaves room for a contraction nvcc chose
    /// differently). The dispatch counter proves the PTX ran.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void DriverOnlyRoutedSquash_MatchesTheNvrtcKernel(int numCapsules, int capsuleDim)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = CudaBackend.CreateOrThrow();
        int elements = numCapsules * capsuleDim;
        var random = RandomHelper.CreateSeededRandom(20261026);
        float[] input = Enumerable.Range(0, elements).Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 3.0)).ToArray();
        float[] grad = Enumerable.Range(0, elements).Select(_ => (float)(random.NextDouble() * 2.0 - 1.0)).ToArray();

        using var x = backend.AllocateBuffer(input);
        using var g = backend.AllocateBuffer(grad);
        using var nvrtcY = backend.AllocateBuffer(elements);
        using var ptxY = backend.AllocateBuffer(elements);
        using var nvrtcDx = backend.AllocateBuffer(elements);
        using var ptxDx = backend.AllocateBuffer(elements);

        bool? savedTest = DirectPtxFeatureGate.TestOverride;
        bool savedExperiment = DirectPtxFeatureGate.CapsuleExperimentOverride;
        try
        {
            DirectPtxFeatureGate.TestOverride = false;
            backend.Squash(x, nvrtcY, numCapsules, capsuleDim, PtxFusedSquashF32Kernel.Epsilon);
            backend.SquashBackward(g, x, nvrtcDx, numCapsules, capsuleDim, PtxFusedSquashF32Kernel.Epsilon);

            DirectPtxFeatureGate.TestOverride = true;
            DirectPtxFeatureGate.CapsuleExperimentOverride = true;
            long before = backend.DirectPtxSquashDispatchCount;
            backend.Squash(x, ptxY, numCapsules, capsuleDim, PtxFusedSquashF32Kernel.Epsilon);
            backend.SquashBackward(g, x, ptxDx, numCapsules, capsuleDim, PtxFusedSquashF32Kernel.Epsilon);
            Skip.If(backend.DirectPtxSquashDispatchCount == before,
                "The direct-PTX squash did not dispatch on this device: " + backend.DirectPtxLastError);
            Assert.Equal(before + 2, backend.DirectPtxSquashDispatchCount);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = savedTest;
            DirectPtxFeatureGate.CapsuleExperimentOverride = savedExperiment;
        }
        AssertWithinOneUlp(backend.DownloadBuffer(nvrtcY), backend.DownloadBuffer(ptxY));
        AssertWithinOneUlp(backend.DownloadBuffer(nvrtcDx), backend.DownloadBuffer(ptxDx));
    }

    private static DirectPtxRuntime OpenValidatedRuntime()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasValidatedCapsuleSquash(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            runtime.Dispose();
            Skip.If(true, "The capsule-squash specialization targets GA10x/SM86.");
        }
        return runtime;
    }

    private static void AssertResidentAudit(PtxFusedSquashF32Kernel kernel)
    {
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.Function.RegistersPerThread <= 64,
            $"{kernel.Audit.Function.RegistersPerThread} registers per thread exceed the budget of 64.");
    }

    private static void ZeroFirstCapsule(float[] values, int capsuleDim)
    {
        for (int j = 0; j < capsuleDim; j++) values[j] = 0f;
    }

    // The CUDA kernel's arithmetic in managed FP64: the same sums, the same zero-denominator rule.
    private static float[] ForwardReference(float[] input, int numCapsules, int capsuleDim)
    {
        var output = new float[input.Length];
        double eps = PtxFusedSquashF32Kernel.Epsilon;
        for (int c = 0; c < numCapsules; c++)
        {
            int off = c * capsuleDim;
            double r2 = 0;
            for (int j = 0; j < capsuleDim; j++) r2 += (double)input[off + j] * input[off + j];
            double norm = Math.Sqrt(r2);
            double denominator = (1.0 + r2) * (norm + eps);
            double scale = denominator != 0.0 ? r2 / denominator : 0.0;
            for (int j = 0; j < capsuleDim; j++) output[off + j] = (float)(scale * input[off + j]);
        }
        return output;
    }

    private static float[] BackwardReference(float[] grad, float[] input, int numCapsules, int capsuleDim)
    {
        var output = new float[input.Length];
        double eps = PtxFusedSquashF32Kernel.Epsilon;
        for (int c = 0; c < numCapsules; c++)
        {
            int off = c * capsuleDim;
            double r2 = 0, dot = 0;
            for (int j = 0; j < capsuleDim; j++)
            {
                r2 += (double)input[off + j] * input[off + j];
                dot += (double)input[off + j] * grad[off + j];
            }
            double norm = Math.Sqrt(r2);
            double denominator = (1.0 + r2) * (norm + eps);
            double scale = denominator != 0.0 ? r2 / denominator : 0.0;
            double coefficient = denominator != 0.0 ? (norm + 2.0 * eps - r2 * norm) / (denominator * denominator) : 0.0;
            for (int j = 0; j < capsuleDim; j++)
                output[off + j] = (float)(scale * grad[off + j] + coefficient * input[off + j] * dot);
        }
        return output;
    }

    // FP64 throughout and one FP32 rounding per element: any difference beyond one ulp is a real defect.
    private static void AssertWithinOneUlp(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            if (e == a) continue;
            int ulps = Math.Abs(PtxCompat.SingleToInt32Bits(e) - PtxCompat.SingleToInt32Bits(a));
            Assert.True(ulps <= 1, $"index {i}: actual {a:G9}, expected {e:G9} ({ulps} ulps apart).");
        }
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
