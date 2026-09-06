using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class OpenClFusedCodegenHardwareTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_OPENCL_TUNING_PROOF";
    private readonly ITestOutputHelper _output;

    public OpenClFusedCodegenHardwareTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "HardwareProof")]
    public async Task AmdDriver_ExecutesFusedGraphFasterAndReplaysNativeBinary()
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable(ProofEnvironmentVariable),
                "1",
                StringComparison.Ordinal))
        {
            _output.WriteLine($"Set {ProofEnvironmentVariable}=1 to run the AMD OpenCL hardware proof.");
            return;
        }

        const int elementCount = 1 << 20;
        string cacheDirectory = Path.Combine(
            Path.GetTempPath(), "aidotnet-opencl-codegen-proof-" + Guid.NewGuid().ToString("N"));
        string? previousCacheDirectory = Environment.GetEnvironmentVariable("AIDOTNET_KERNEL_CACHE_DIR");
        bool previousCacheEnabled = DirectOpenClProgram.EnableBinaryCache;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_KERNEL_CACHE_DIR", cacheDirectory);
            DirectOpenClProgram.EnableBinaryCache = true;

            GpuSourceKernel fused;
            GpuSourceKernel negate;
            GpuSourceKernel exp;
            GpuSourceKernel sqrt;
            float[] input = Enumerable.Range(0, elementCount)
                .Select(index => (index % 257 - 128) / 128f)
                .ToArray();

            TimeSpan fusedMedian;
            TimeSpan composedMedian;
            long nativeArtifactSizeBytes;
            string deviceDescription;
            using (var backend = new OpenClBackend(deviceIndex: 0))
            {
                Assert.True(backend.IsAvailable);
                Assert.Equal(KernelTuningDeviceKind.AmdGpu, backend.TuningDeviceKind);
                Assert.True(backend.IsProfilingEnabled);
                fused = EmitChain(backend,
                    CodegenOpKind.Negate,
                    CodegenOpKind.Exp,
                    CodegenOpKind.Sqrt);
                negate = EmitChain(backend, CodegenOpKind.Negate);
                exp = EmitChain(backend, CodegenOpKind.Exp);
                sqrt = EmitChain(backend, CodegenOpKind.Sqrt);

                using IGpuBuffer deviceInput = backend.AllocateBuffer(input);
                using IGpuBuffer fusedOutput = backend.AllocateBuffer(elementCount);
                using IGpuBuffer intermediateA = backend.AllocateBuffer(elementCount);
                using IGpuBuffer intermediateB = backend.AllocateBuffer(elementCount);
                using IGpuBuffer composedOutput = backend.AllocateBuffer(elementCount);

                CodegenGraph productionGraph = CodegenLowering.LowerUnaryChain<float>(
                    new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt },
                    new[] { elementCount });
                CodegenEmitResult productionResult = await backend.EmitAndExecuteCodegenKernelAsync(
                    productionGraph,
                    CodegenElementType.Float32,
                    new[] { deviceInput },
                    new[] { fusedOutput },
                    elementCount);
                Assert.False(productionResult.Declined, productionResult.DeclineReason);
                backend.Synchronize();

                OpenClFusedKernelExecutionEvidence first = backend.ExecuteFusedPointwiseProfiled(
                    fused, new[] { deviceInput }, new[] { fusedOutput }, elementCount);
                Assert.Equal(KernelTuningBackend.OpenCl, first.Backend);
                Assert.Equal(CodegenNativeCompilationKind.OpenClDriverCompiler, first.Compilation);
                Assert.Equal(CodegenNativeArtifactKind.OpenClDeviceBinary, first.NativeArtifact);
                Assert.Equal(KernelTuningDeviceKind.AmdGpu, first.Device.Kind);
                Assert.False(string.IsNullOrWhiteSpace(first.Device.LocalKey));
                Assert.True(first.NativeArtifactSizeBytes > 0);
                Assert.Equal(OpenClProgramBuildOrigin.SourceCompilation, first.ProgramOrigin);
                Assert.Equal(OpenClProgramBinaryType.Executable, first.ProgramBinaryType);
                Assert.Equal(1, first.NativeLaunchCount);
                Assert.InRange(first.LocalWorkgroupSize, 1, 256);
                Assert.True(first.DeviceDuration > TimeSpan.Zero);
                nativeArtifactSizeBytes = first.NativeArtifactSizeBytes;
                deviceDescription = $"{backend.DeviceName} ({backend.DeviceVendor})";

                // Compile and warm the equivalent three-launch path before comparing device time.
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

                fusedMedian = Median(fusedSamples);
                composedMedian = Median(composedSamples);
                Assert.True(
                    fusedMedian < composedMedian,
                    $"Fused device median {fusedMedian.TotalMilliseconds:F4} ms was not below " +
                    $"the three-kernel median {composedMedian.TotalMilliseconds:F4} ms.");

                float[] actual = backend.DownloadBuffer(fusedOutput);
                for (int i = 0; i < actual.Length; i++)
                {
                    float expected = MathF.Sqrt(MathF.Exp(-input[i]));
                    Assert.InRange(MathF.Abs(actual[i] - expected), 0f, 2e-5f);
                }
            }

            string[] nativeArtifacts = Directory.GetFiles(cacheDirectory, "*.clbin");
            Assert.NotEmpty(nativeArtifacts);
            Assert.All(nativeArtifacts, path => Assert.True(new FileInfo(path).Length > 0));

            // A new backend/process-level compiler cache must restore the device-native artifact,
            // not recompile the OpenCL C source or merely reuse an in-memory kernel object.
            using (var replayBackend = new OpenClBackend(deviceIndex: 0))
            {
                Assert.True(replayBackend.IsAvailable);
                using IGpuBuffer replayInput = replayBackend.AllocateBuffer(input);
                using IGpuBuffer replayOutput = replayBackend.AllocateBuffer(elementCount);
                OpenClFusedKernelExecutionEvidence replay = replayBackend.ExecuteFusedPointwiseProfiled(
                    fused, new[] { replayInput }, new[] { replayOutput }, elementCount);
                Assert.Equal(OpenClProgramBuildOrigin.NativeBinaryCache, replay.ProgramOrigin);
                Assert.Equal(OpenClProgramBinaryType.Executable, replay.ProgramBinaryType);
                Assert.Equal(KernelTuningDeviceKind.AmdGpu, replay.Device.Kind);
                Assert.True(replay.NativeArtifactSizeBytes > 0);
                Assert.Equal(1, replay.NativeLaunchCount);
                Assert.InRange(replay.LocalWorkgroupSize, 1, 256);
                Assert.True(replay.DeviceDuration > TimeSpan.Zero);

                using var foreignBackend = new OpenClBackend(deviceIndex: 0);
                using IGpuBuffer foreignInput = foreignBackend.AllocateBuffer(input);
                Assert.Throws<ArgumentException>(() => replayBackend.ExecuteFusedPointwise(
                    fused, new[] { foreignInput }, new[] { replayOutput }, elementCount));
            }

            double speedup = composedMedian.TotalSeconds / fusedMedian.TotalSeconds;
            _output.WriteLine($"AMD device: {deviceDescription}");
            _output.WriteLine($"Driver-native executable size: {nativeArtifactSizeBytes} bytes");
            _output.WriteLine($"AMD OpenCL fused median: {fusedMedian.TotalMilliseconds:F4} ms");
            _output.WriteLine($"AMD OpenCL three-kernel median: {composedMedian.TotalMilliseconds:F4} ms");
            _output.WriteLine($"Measured device-time speedup: {speedup:F3}x");
            _output.WriteLine("The backend selected device-valid OpenCL workgroup geometry.");
            _output.WriteLine("Second backend loaded the driver-native OpenCL binary cache entry.");
        }
        finally
        {
            DirectOpenClProgram.EnableBinaryCache = previousCacheEnabled;
            Environment.SetEnvironmentVariable("AIDOTNET_KERNEL_CACHE_DIR", previousCacheDirectory);
            if (Directory.Exists(cacheDirectory)) Directory.Delete(cacheDirectory, recursive: true);
        }
    }

    private static OpenClFusedKernelExecutionEvidence Profile(
        OpenClBackend backend,
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
