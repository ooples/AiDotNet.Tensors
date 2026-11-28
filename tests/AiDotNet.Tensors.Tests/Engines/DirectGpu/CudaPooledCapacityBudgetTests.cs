// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Tests.Engines.DevicePrimitives;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Pool bucket padding must never be the reason a GpuMemoryBudget refuses an allocation whose requested size fits:
/// under a budget the CUDA backend allocates the requested size, not the rounded bucket capacity.
/// </summary>
[Collection(GpuMemoryStatsGlobalStateCollection.Name)]
public class CudaPooledCapacityBudgetTests
{
    [SkippableFact]
    public void PooledCapacity_IsPaddedWithoutABudget_AndExactUnderOne()
    {
        CudaBackend backend;
        try { backend = new CudaBackend(); }
        catch (Exception ex) { Skip.If(true, "CUDA backend unavailable: " + ex.Message); return; }

        using (backend)
        {
            Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
            string device = ((IDeviceMemoryAccounted)backend).MemoryAccount.Device;

            // 17 MiB of floats: a size the pool's size classes round up (to the 24 MiB class in the review example).
            int requested = 17 * 1024 * 1024 / sizeof(float);
            GpuMemoryBudget.ClearLimit(device);
            int unbudgeted = backend.ResolvePooledAllocationCapacity(requested, sizeof(float));
            Skip.If(unbudgeted == requested, "This pool configuration does not pad the request (exact-size buckets).");
            Assert.True(unbudgeted > requested);

            GpuMemoryBudget.SetMemoryLimit(100L * 1024 * 1024, device);
            try
            {
                Assert.Equal(requested, backend.ResolvePooledAllocationCapacity(requested, sizeof(float)));
            }
            finally
            {
                GpuMemoryBudget.ClearLimit(device);
            }
        }
    }
}
