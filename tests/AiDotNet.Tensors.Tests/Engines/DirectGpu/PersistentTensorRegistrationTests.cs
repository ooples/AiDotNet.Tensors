using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Registering a persistent tensor must not upload it.
/// </summary>
/// <remarks>
/// Registration used to convert the tensor to float and allocate a GPU buffer immediately, which
/// made cloning quadratic in memory: a cloned layer re-registers every parameter, so each clone
/// re-uploaded a whole model. On a 31.5M-parameter decoder that was ~120 MB of float conversion per
/// clone, and it exhausted memory across ten large models in one test process.
///
/// The read path already allocates on a cache miss and re-uploads when the host Version moves, so
/// the upload belongs there and nowhere else. This test states that as an invariant rather than
/// leaving it to be re-broken: it measures managed allocation across a registration, which is where
/// the float conversion showed up.
/// </remarks>
public class PersistentTensorRegistrationTests
{
#if NET6_0_OR_GREATER   // GetAllocatedBytesForCurrentThread is .NET Core 3.0+; this project also targets net471.
    [Fact]
    public void RegisterPersistentTensor_DoesNotMaterialiseTheTensor()
    {
        // The engine under test must be the GPU one. AiDotNetEngine.Current starts as CpuEngine and
        // stays there when GPU detection is off or unavailable, and CpuEngine.RegisterPersistentTensor
        // is ALREADY a no-op -- so this would have passed without ever running the changed override.
        using var engine = new DirectGpuTensorEngine();
        if (!engine.IsGpuAvailable)
        {
            // No backend means TryGetBackend early-outs and there is nothing to assert about uploads.
            return;
        }

        var tensor = new Tensor<double>(new[] { 512, 512 });   // 2 MB of doubles
        for (int i = 0; i < tensor.Length; i += 512) tensor[i] = i * 0.5;

        long payload = (long)tensor.Length * sizeof(double);

        // Warm any one-time engine state so the measurement is of registration alone.
        engine.RegisterPersistentTensor(tensor, PersistentTensorRole.Weights);
        engine.UnregisterPersistentTensor(tensor);

        // THREAD-LOCAL, not process-wide: GetTotalAllocatedBytes counts every thread, so an
        // unrelated test running concurrently could push this over the threshold and fail it.
        long before = System.GC.GetAllocatedBytesForCurrentThread();
        engine.RegisterPersistentTensor(tensor, PersistentTensorRole.Weights);
        long allocated = System.GC.GetAllocatedBytesForCurrentThread() - before;

        engine.UnregisterPersistentTensor(tensor);

        // A float copy of the payload would be half its size; anything approaching that means the
        // eager upload is back. Bookkeeping is allowed to allocate a little.
        long floatCopy = payload / 2;
        Assert.True(
            allocated < floatCopy / 4,
            $"registration allocated {allocated:N0} bytes for a {payload:N0}-byte tensor. A float "
                + $"copy would be about {floatCopy:N0}; the upload belongs on the read path, which "
                + "allocates on a cache miss and re-uploads when the host Version moves.");
    }
#endif
}
