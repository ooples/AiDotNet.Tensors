using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class HipFusedCodegenHardwareTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_HIP_CODEGEN_PROOF";
    private readonly ITestOutputHelper _output;

    public HipFusedCodegenHardwareTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "HardwareProof")]
    public async Task AmdHipRtc_ExecutesOneFusedGfx1012CodeObjectFasterThanThreeLaunches()
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable(ProofEnvironmentVariable),
                "1",
                StringComparison.Ordinal))
        {
            _output.WriteLine($"Set {ProofEnvironmentVariable}=1 to run the AMD HIP code-object proof.");
            return;
        }

        const int elementCount = 1 << 20;
        using var backend = new HipBackend(deviceIndex: 0);
        Assert.True(backend.IsAvailable, "The HIP backend did not initialize on the selected AMD GPU.");
        Assert.Equal(AmdGpuArchitecture.RDNA, backend.Architecture);

        GpuSourceKernel fused = EmitChain(
            backend, CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt);
        GpuSourceKernel negate = EmitChain(backend, CodegenOpKind.Negate);
        GpuSourceKernel exp = EmitChain(backend, CodegenOpKind.Exp);
        GpuSourceKernel sqrt = EmitChain(backend, CodegenOpKind.Sqrt);
        float[] input = Enumerable.Range(0, elementCount)
            .Select(index => (index % 257 - 128) / 128f)
            .ToArray();

        using IGpuBuffer deviceInput = backend.AllocateBuffer(input);
        using IGpuBuffer fusedOutput = backend.AllocateBuffer(elementCount);
        using IGpuBuffer intermediateA = backend.AllocateBuffer(elementCount);
        using IGpuBuffer intermediateB = backend.AllocateBuffer(elementCount);
        using IGpuBuffer composedOutput = backend.AllocateBuffer(elementCount);

        CodegenGraph productionGraph = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt },
            new[] { elementCount });
        CodegenEmitResult production = await backend.EmitAndExecuteCodegenKernelAsync(
            productionGraph,
            CodegenElementType.Float32,
            new[] { deviceInput },
            new[] { fusedOutput },
            elementCount);
        Assert.False(production.Declined, production.DeclineReason);
        backend.Synchronize();

        HipFusedKernelExecutionEvidence first = Profile(
            backend, fused, deviceInput, fusedOutput, elementCount);
        Assert.Equal(KernelTuningBackend.Hip, first.Backend);
        Assert.Equal(CodegenNativeCompilationKind.HipRtc, first.Compilation);
        Assert.Equal(CodegenNativeArtifactKind.HipCodeObject, first.NativeArtifact);
        Assert.Equal(AmdGpuArchitecture.RDNA, first.Architecture);
        Assert.Contains("Radeon RX 5500 XT", first.DeviceName, StringComparison.OrdinalIgnoreCase);
        Assert.True(first.NativeArtifactSizeBytes > 0);
        Assert.Equal(1, first.NativeLaunchCount);
        Assert.InRange(first.BlockSize, 1, 1024);
        Assert.True(first.DeviceDuration > TimeSpan.Zero);

        Profile(backend, negate, deviceInput, intermediateA, elementCount);
        Profile(backend, exp, intermediateA, intermediateB, elementCount);
        Profile(backend, sqrt, intermediateB, composedOutput, elementCount);

        var fusedSamples = new List<TimeSpan>();
        var composedSamples = new List<TimeSpan>();
        for (int sample = 0; sample < 11; sample++)
        {
            fusedSamples.Add(Profile(
                backend, fused, deviceInput, fusedOutput, elementCount).DeviceDuration);
            composedSamples.Add(
                Profile(backend, negate, deviceInput, intermediateA, elementCount).DeviceDuration +
                Profile(backend, exp, intermediateA, intermediateB, elementCount).DeviceDuration +
                Profile(backend, sqrt, intermediateB, composedOutput, elementCount).DeviceDuration);
        }

        TimeSpan fusedMedian = Median(fusedSamples);
        TimeSpan composedMedian = Median(composedSamples);
        Assert.True(
            fusedMedian < composedMedian,
            $"HIP fused device median {fusedMedian.TotalMilliseconds:F4} ms was not below " +
            $"the three-launch median {composedMedian.TotalMilliseconds:F4} ms.");

        float[] actual = backend.DownloadBuffer(fusedOutput);
        for (int i = 0; i < actual.Length; i++)
        {
            float expected = MathF.Sqrt(MathF.Exp(-input[i]));
            Assert.InRange(MathF.Abs(actual[i] - expected), 0f, 2e-5f);
        }

        double speedup = composedMedian.TotalSeconds / fusedMedian.TotalSeconds;
        _output.WriteLine($"AMD HIP device: {first.DeviceName} ({first.Architecture})");
        _output.WriteLine($"HIPRTC AMD code object size: {first.NativeArtifactSizeBytes} bytes");
        _output.WriteLine($"AMD HIP fused median: {fusedMedian.TotalMilliseconds:F4} ms");
        _output.WriteLine($"AMD HIP three-launch median: {composedMedian.TotalMilliseconds:F4} ms");
        _output.WriteLine($"Measured HIP device-time speedup: {speedup:F3}x");
        _output.WriteLine("One generated graph executed as one gfx1012 AMD code-object launch.");
    }

    private static HipFusedKernelExecutionEvidence Profile(
        HipBackend backend,
        GpuSourceKernel kernel,
        IGpuBuffer input,
        IGpuBuffer output,
        int elementCount) => backend.ExecuteFusedPointwiseProfiled(
            kernel, new[] { input }, new[] { output }, elementCount);

    private static GpuSourceKernel EmitChain(
        INativeGpuCodegenExecutor backend,
        params CodegenOpKind[] operations)
    {
        CodegenGraph graph = CodegenLowering.LowerUnaryChain<float>(operations, new[] { 1 << 20 });
        CodegenEmitResult emitted = backend.EmitCodegenKernel(graph, CodegenElementType.Float32);
        Assert.False(emitted.Declined, emitted.DeclineReason);
        Assert.Equal(backend.NativeCodegenBackend, emitted.Kernel?.RequiredRuntime.Backend);
        return Assert.IsType<GpuSourceKernel>(emitted.Kernel);
    }

    private static TimeSpan Median(IReadOnlyList<TimeSpan> samples)
    {
        long[] ticks = samples.Select(sample => sample.Ticks).OrderBy(value => value).ToArray();
        return TimeSpan.FromTicks(ticks[ticks.Length / 2]);
    }
}
