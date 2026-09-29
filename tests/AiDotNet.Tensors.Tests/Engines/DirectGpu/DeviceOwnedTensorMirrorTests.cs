using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Pins the contract for a tensor moved to the device with <see cref="Tensor{T}.Gpu()"/> — a GPU-resident training
/// parameter: it is placed on the dispatcher's backend, and a device-side write reaches host reads, each aliased view
/// into its own slice. (A host-side write followed by InvalidateResidentWeightBuffer detaches the buffer and returns the
/// tensor to the CPU - see ResidentWeightInvalidationDeviceStateTests.)
/// </summary>
/// <remarks>
/// Measured failure these guard (plain MLP, NeuralNetwork.Train on the GPU engine): the fused optimizer updated
/// the resident buffers, then the post-step invalidation dropped that pending download and detached the buffers,
/// so every later update landed in an orphaned buffer and the weights never changed again after step 1.
/// </remarks>
[Collection("DirectGpuSerial")]
public class DeviceOwnedTensorMirrorTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;

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

    private static Tensor<float> Filled(int[] shape, float value)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value;
        return t;
    }

    [SkippableFact]
    public void A_device_write_to_a_moved_tensor_reaches_host_reads()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            var t = Filled([64], 1f).Gpu();
            var backend = gpu.GetBackend()!;
            var buffer = t.TryGetGpuBuffer()!;
            Assert.True(t.IsGpuResident);

            // What a fused on-device optimizer does: write the resident buffer, then re-arm the host download.
            backend.Fill(buffer, 3f, 64);
            gpu.BindResidentBuffer(t, buffer, backend);

            Assert.All(t.ToArray(), v => Assert.Equal(3f, v));
            Assert.Same(buffer, t.TryGetGpuBuffer());
        }
    }

    [SkippableFact]
    public void Aliased_views_each_download_into_their_own_slice()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            // Two parameters that are views into ONE shared buffer, as AiDotNet's ParameterBuffer lays them out.
            var shared = Filled([2, 32], 1f);
            var a = shared.Slice(0).Gpu();
            var b = shared.Slice(1).Gpu();
            var backend = gpu.GetBackend()!;

            backend.Fill(a.TryGetGpuBuffer()!, 5f, 32);
            gpu.BindResidentBuffer(a, a.TryGetGpuBuffer()!, backend);
            backend.Fill(b.TryGetGpuBuffer()!, 7f, 32);
            gpu.BindResidentBuffer(b, b.TryGetGpuBuffer()!, backend);

            // b's registration must not displace a's pending download, and neither may land on the other's slice.
            Assert.All(a.ToArray(), v => Assert.Equal(5f, v));
            Assert.All(b.ToArray(), v => Assert.Equal(7f, v));
        }
    }

    [SkippableFact]
    public void Gpu_places_the_tensor_on_the_dispatchers_backend()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            var t = Filled([16], 2f).Gpu();
            // Engine.DirectGpu is a separate lazily-created engine with its own backend; placing there meant the
            // dispatcher's kernels and fused optimizers were not touching the buffer the tensor pointed at.
            Assert.Same(gpu.PlacementBackend, t._gpuBackend);
        }
    }
}
