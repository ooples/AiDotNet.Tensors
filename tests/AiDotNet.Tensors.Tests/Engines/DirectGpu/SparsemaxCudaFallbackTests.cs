using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("EngineCurrentGlobalState")]
public class SparsemaxCudaFallbackTests
{
    [Fact]
    public void Sparsemax_FallsBackToCpu_WhenCudaWhereSelectKernelIsUnavailable()
    {
        string? priorBackends = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
        DirectGpuEngine directGpu;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "none");
            directGpu = new DirectGpuEngine();
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", priorBackends);
        }

#pragma warning disable SYSLIB0050 // Exercise capability fallback without constructing CUDA hardware state.
        var cuda = (CudaBackend)FormatterServices.GetUninitializedObject(typeof(CudaBackend));
#pragma warning restore SYSLIB0050
        GC.SuppressFinalize(cuda);
        SetField(cuda, "_kernelCache", new ConcurrentDictionary<string, IntPtr>(StringComparer.Ordinal));
        SetField(directGpu, "_backend", cuda);
        SetField(directGpu, "_isAvailable", true);

        var engine = new DirectGpuTensorEngine(directGpu);
        try
        {
            using var input = new Tensor<float>(new[] { 1f, 2f, -1f, 0.5f }, new[] { 1, 4 });
            using var expected = new CpuEngine().Sparsemax(input, -1);
            using var actual = ((IEngine)engine).Sparsemax(input, -1);

            Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f,
                    $"Sparsemax mismatch at {i}: expected {expected[i]}, actual {actual[i]}.");
        }
        finally
        {
            engine.Dispose();
            SetField(directGpu, "_backend", null);
            directGpu.Dispose();
        }
    }

    private static void SetField(object target, string name, object? value)
    {
        FieldInfo field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field not found: {name}");
        field.SetValue(target, value);
    }
}
