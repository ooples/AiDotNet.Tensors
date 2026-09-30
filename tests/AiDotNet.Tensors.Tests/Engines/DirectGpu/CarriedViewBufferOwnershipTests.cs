// Copyright (c) AiDotNet. All rights reserved.

using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Inside a resident compiled step, ExpandDims of a strided (permuted) input copies it into a NEW backing array in the
/// view's element order, then carries the input's resident device buffer to the result. For a strided input that
/// buffer is its BASE, in the base's order and owned by the base's cache entry: aliasing it gave the result
/// wrongly-ordered device data and a pending download that read a buffer the base's eviction frees.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CarriedViewBufferOwnershipTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CarriedViewBufferOwnershipTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void ExpandDimsOfAStridedResidentInput_CarriesAnOwnedCorrectlyOrderedBuffer()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var cpu = new CpuEngine();
        var rng = new Random(11);
        var x = new Tensor<float>(new[] { 4, 6 });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        var expected = cpu.TensorExpandDims(cpu.TensorPermute(cpu.TensorTanh(x), new[] { 1, 0 }), 0).ToArray();
        var expectedScaled = new float[expected.Length];
        for (int i = 0; i < expected.Length; i++) expectedScaled[i] = expected[i] * 2f;

        var baseResult = gpu.TensorTanh(x);                        // device-only, row-major [4, 6]
        var strided = gpu.TensorPermute(baseResult, new[] { 1, 0 });  // [6, 4] view of it
        Tensor<float> expanded;
        Tensor<float> scaled;
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
            {
                expanded = gpu.TensorExpandDims(strided, 0);
                // Consumed on the device: reads the carried buffer, so a wrongly-ordered buffer shows up here.
                scaled = gpu.TensorMultiplyScalar(expanded, 2f);
            }
        }
        finally
        {
            gpu.ResumeActivationEviction();
        }

        // An op result owns its buffer through the data vector's shared device state, not the activation cache.
        var baseBuffer = baseResult.VectorDeviceBuffer ?? CachedBuffer(gpu, (object?)baseResult.GetBackingArrayForCacheLookupUnsafe() ?? baseResult.DataVector);
        var carried = expanded._gpuBuffer;
        // Positive control: the carry path ran, so the checks below exercise it.
        Assert.NotNull(baseBuffer);
        Assert.NotNull(carried);
        Assert.False(ReferenceEquals(baseBuffer, carried), "the expanded result aliased its strided input's base buffer");

        var gotScaled = scaled.ToArray();
        for (int i = 0; i < gotScaled.Length; i++)
            Assert.True(Math.Abs(gotScaled[i] - expectedScaled[i]) < 1e-5f, $"device read [{i}] gpu {gotScaled[i]} cpu {expectedScaled[i]}");

        baseBuffer.Dispose();   // what evicting the base's cache entry does
        var got = expanded.ToArray();
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-5f, $"host read [{i}] gpu {got[i]} cpu {expected[i]}");
    }
    // The base result owns its buffer through its activation-cache entry (an op result is not bound to _gpuBuffer).
    private static IGpuBuffer? CachedBuffer(DirectGpuTensorEngine engine, object key)
    {
        var cacheField = typeof(DirectGpuTensorEngine).GetField("_activationCache", BindingFlags.NonPublic | BindingFlags.Instance);
        var cache = cacheField?.GetValue(engine);
        if (cache is null) return null;
        var args = new object?[] { key, null };
        var tryGet = cache.GetType().GetMethod("TryGetValue");
        if (tryGet is null || tryGet.Invoke(cache, args) is not true || args[1] is not { } entry) return null;
        var bufferMember = entry.GetType().GetProperty("Buffer", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (bufferMember is not null) return bufferMember.GetValue(entry) as IGpuBuffer;
        return entry.GetType().GetField("Buffer", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(entry) as IGpuBuffer;
    }
}