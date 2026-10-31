#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorBroadcastAddInPlace with a bias over the trailing axes ([..., D] += [D]) on a tensor whose value is on the
/// device: added in place there (MultiHeadAttentionLayer's fused inference adds its output bias this way), with the
/// same result as the CPU; a host tensor takes the host add.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuTrailingBiasAddInPlaceTests
{
    private static float[] Values(int n, int seed)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((((i + seed) * 7919) % 53) - 26) * 0.05f;
        return v;
    }

    [SkippableFact]
    public void DeviceTensor_TrailingBias_MatchesCpu()
    {
        DirectGpuTensorEngine? engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        Skip.If(engine is null || !engine.IsGpuAvailable, "No DirectGpu backend available.");
        using (engine)
        {
            IEngine gpu = engine;
            IEngine cpu = new CpuEngine();
            foreach (var biasShape in new[] { new[] { 12 }, new[] { 1, 1, 12 }, new[] { 5, 12 } })
            {
                var x = new Tensor<float>(Values(3 * 5 * 12, 1), new[] { 3, 5, 12 });
                var y = new Tensor<float>(Values(3 * 5 * 12, 2), new[] { 3, 5, 12 });
                int biasLength = 1;
                foreach (int d in biasShape) biasLength *= d;
                var bias = new Tensor<float>(Values(biasLength, 3), biasShape);

                var expected = cpu.TensorAdd(x, y);
                cpu.TensorBroadcastAddInPlace(expected, bias);

                var onDevice = gpu.TensorAdd(x, y);   // a device result
                gpu.TensorBroadcastAddInPlace(onDevice, bias);
                var actual = onDevice.ToArray();
                var want = expected.ToArray();
                for (int i = 0; i < want.Length; i++)
                    Assert.True(Math.Abs(want[i] - actual[i]) <= 1e-5f * (1 + Math.Abs(want[i])),
                        $"bias [{string.Join(",", biasShape)}] [{i}]: expected {want[i]}, got {actual[i]}");

                // A host tensor: the host add, same result.
                var host = new Tensor<float>(Values(3 * 5 * 12, 1), new[] { 3, 5, 12 });
                var hostExpected = new Tensor<float>(Values(3 * 5 * 12, 1), new[] { 3, 5, 12 });
                gpu.TensorBroadcastAddInPlace(host, bias);
                cpu.TensorBroadcastAddInPlace(hostExpected, bias);
                Assert.Equal(hostExpected.ToArray(), host.ToArray());
            }
        }
    }
}
#endif