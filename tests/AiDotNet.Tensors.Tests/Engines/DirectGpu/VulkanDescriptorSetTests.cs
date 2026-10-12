using System;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A dispatch recorded into a batch reads its descriptor set when the batch is submitted, not when it is recorded.
/// Each recorded dispatch therefore binds a set of its own: the same kernel used several times in one batch, on
/// different buffers, must give every use its own result, in one submission.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanDescriptorSetTests
{
    // The backend is a process-wide singleton other test classes share, so nothing here disposes it.
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        return backend!;
    }

    private const string Source = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; uint marker; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] + 1190.0;
}";

    private static VulkanComputePipeline Pipeline(VulkanBackend backend)
    {
        var method = typeof(VulkanBackend).GetMethod("GetOrCreateGlslPipeline", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("GetOrCreateGlslPipeline not found");
        return (VulkanComputePipeline?)method.Invoke(backend, new object[] { Source, 2, 8u })
            ?? throw new InvalidOperationException("pipeline was not built");
    }

    [SkippableFact]
    public void OneKernelUsedSeveralTimesInABatch_OnDifferentBuffers_GivesEachUseItsOwnResult_InOneSubmission()
    {
        var backend = BackendOrSkip();
        const int uses = 5, size = 300;
        var outputs = new AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer[uses];
        try
        {
            for (int u = 0; u < uses; u++)
            {
                outputs[u] = backend.AllocateBuffer(size);
                backend.Fill(outputs[u], float.NaN, size);
            }
            backend.BeginBatch();
            try
            {
                // The same kernel five times: had a later use rewritten the set an earlier one bound, every output
                // would hold the last use's values (or the batch would have had to be submitted in between).
                for (int u = 0; u < uses; u++)
                    backend.LinspaceKernel(outputs[u], start: 100f * u, step: u + 1, size);
                Assert.Equal(uses, backend.BatchDispatchCount);
            }
            finally
            {
                backend.EndBatch();
            }
            for (int u = 0; u < uses; u++)
            {
                float[] values = backend.DownloadBuffer(outputs[u]);
                for (int i = 0; i < size; i++)
                    Assert.True(Math.Abs(values[i] - (100f * u + (u + 1) * i)) <= 1e-3f * Math.Max(1f, Math.Abs(values[i])),
                        $"use {u}, element {i}: {values[i]}");
            }
        }
        finally
        {
            foreach (var o in outputs) o?.Dispose();
        }
    }

    [SkippableFact]
    public void ARetainedSet_IsNotRewritten_AndComesBackForReuse()
    {
        var backend = BackendOrSkip();
        var pipeline = Pipeline(backend);
        using var a = (VulkanGpuBuffer)backend.AllocateBuffer(64);
        using var b = (VulkanGpuBuffer)backend.AllocateBuffer(64);

        pipeline.UpdateDescriptorSet(a.Storage, b.Storage);
        IntPtr first = pipeline.DescriptorSet;
        // Nothing pending: the next dispatch reuses the same set.
        pipeline.UpdateDescriptorSet(b.Storage, a.Storage);
        Assert.Equal(first, pipeline.DescriptorSet);

        IntPtr retained = pipeline.RetainDescriptorSet();
        Assert.Equal(first, retained);
        pipeline.UpdateDescriptorSet(a.Storage, b.Storage);
        IntPtr second = pipeline.DescriptorSet;
        Assert.NotEqual(retained, second);
        int allocated = pipeline.AllocatedDescriptorSetCount;

        // Many more pending dispatches than one pool holds.
        var pending = new IntPtr[40];
        for (int i = 0; i < pending.Length; i++)
        {
            pending[i] = pipeline.RetainDescriptorSet();
            pipeline.UpdateDescriptorSet(a.Storage, b.Storage);
            Assert.DoesNotContain(pipeline.DescriptorSet, pending.AsSpan(0, i + 1).ToArray());
            Assert.NotEqual(retained, pipeline.DescriptorSet);
        }
        Assert.True(pipeline.AllocatedDescriptorSetCount > allocated);

        // Handed back, they serve the next pending dispatches without another pool.
        pipeline.ReleaseDescriptorSet(retained);
        foreach (IntPtr set in pending) pipeline.ReleaseDescriptorSet(set);
        int settled = pipeline.AllocatedDescriptorSetCount;
        var again = new IntPtr[40];
        for (int i = 0; i < again.Length; i++)
        {
            again[i] = pipeline.RetainDescriptorSet();
            pipeline.UpdateDescriptorSet(a.Storage, b.Storage);
        }
        Assert.Equal(settled, pipeline.AllocatedDescriptorSetCount);
        foreach (IntPtr set in again) pipeline.ReleaseDescriptorSet(set);
    }

    [SkippableFact]
    public void TwoBatchesInARow_ReuseTheFirstBatchsSets()
    {
        var backend = BackendOrSkip();
        const int size = 64;
        using var o1 = backend.AllocateBuffer(size);
        using var o2 = backend.AllocateBuffer(size);
        using var o3 = backend.AllocateBuffer(size);
        void Batch(float start)
        {
            backend.BeginBatch();
            try
            {
                backend.TriangularMask(o1, 8, 8, 0, start);
                backend.TriangularMask(o2, 8, 8, 1, start + 1);
                backend.TriangularMask(o3, 8, 8, -1, start + 2);
            }
            finally
            {
                backend.EndBatch();
            }
        }
        Batch(5f);
        float[] first = backend.DownloadBuffer(o2);
        Batch(50f);
        float[] second = backend.DownloadBuffer(o2);
        // The mask value differs between the batches, so the second batch's dispatch really ran on o2.
        Assert.NotEqual(first, second);
        Assert.Contains(51f, second);
        Assert.DoesNotContain(6f, second);
    }
}
