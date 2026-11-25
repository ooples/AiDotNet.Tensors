// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The maximum and the minimum of 32-bit floats are NaN when any element is, wherever it sits, as for NumPy, PyTorch
/// and the 64-bit kernels. A maximum built from a compare (or from x86 MAXPS) skips a NaN instead, so the result
/// depended on the NaN's position and on the length.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FloatExtremaNaNPropagationTests
{
    private static readonly int[] Sizes = { 1, 2, 7, 8, 9, 31, 32, 33, 40, 257, 4097, 70_001 };

    private static float[] Values(int size, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var values = new float[size];
        for (int i = 0; i < size; i++) values[i] = (float)((random.NextDouble() - 0.5) * 200.0);
        return values;
    }

    // First, last, around the vector widths, the middle and the scalar tail.
    private static int[] Positions(int size)
    {
        var positions = new System.Collections.Generic.SortedSet<int> { 0, size - 1, size / 2 };
        foreach (int p in new[] { 1, 7, 8, 15, 16, 31, 32, 33, size - 2, size - 8, size - 9 })
            if (p >= 0 && p < size) positions.Add(p);
        var list = new int[positions.Count];
        positions.CopyTo(list);
        return list;
    }

    private static void Check(Func<float[], (float Max, float Min)> extrema, string where)
    {
        foreach (int size in Sizes)
        {
            var values = Values(size, 100 + size);
            float max = float.NegativeInfinity, min = float.PositiveInfinity;
            foreach (float v in values)
            {
                max = Math.Max(max, v);
                min = Math.Min(min, v);
            }

            var clean = extrema(values);
            Assert.True(clean.Max == max, $"{where}: max of {size} finite elements is {clean.Max:R}, expected {max:R}.");
            Assert.True(clean.Min == min, $"{where}: min of {size} finite elements is {clean.Min:R}, expected {min:R}.");

            foreach (int position in Positions(size))
            {
                var poisoned = (float[])values.Clone();
                poisoned[position] = float.NaN;
                var result = extrema(poisoned);
                Assert.True(float.IsNaN(result.Max), $"{where}: max of {size} elements with a NaN at {position} is {result.Max:R}.");
                Assert.True(float.IsNaN(result.Min), $"{where}: min of {size} elements with a NaN at {position} is {result.Min:R}.");
            }

            // Infinities are ordinary values: only a NaN is contagious.
            var infinite = (float[])values.Clone();
            infinite[0] = float.NegativeInfinity;
            infinite[size - 1] = size > 1 ? float.PositiveInfinity : float.NegativeInfinity;
            var bounds = extrema(infinite);
            Assert.True(bounds.Min == float.NegativeInfinity, $"{where}: min of {size} elements with -inf is {bounds.Min:R}.");
            if (size > 1) Assert.True(bounds.Max == float.PositiveInfinity, $"{where}: max of {size} elements with +inf is {bounds.Max:R}.");
        }
    }

    [Fact]
    public void Cpu_engine_max_and_min_are_nan_when_any_element_is()
    {
        var engine = new CpuEngine();
        Check(values =>
        {
            var tensor = new Tensor<float>(new[] { values.Length }, new Vector<float>(values));
            return (engine.TensorMaxValue(tensor), engine.TensorMinValue(tensor));
        }, "CpuEngine");
    }

    [Fact]
    public void Numeric_operations_max_and_min_are_nan_when_any_element_is()
    {
        var operations = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>();
        Check(values => (operations.Max(values), operations.Min(values)), "FloatOperations");
    }

    public enum Device
    {
        OpenCl,
        Vulkan,
        WebGpu,
    }

    private static IDirectGpuBackend? Create(Device device, out IDisposable? owned)
    {
        owned = null;
        try
        {
            if (device == Device.WebGpu)
            {
#if NET7_0_OR_GREATER
                return DawnHostedWebGpu.BackendOrNull();
#else
                return null;
#endif
            }

            if (device == Device.Vulkan)
            {
                var vulkan = VulkanBackend.Instance;
                return vulkan.Initialize() && vulkan.IsGlslCompilerAvailable ? vulkan : null;
            }

            var openCl = new OpenClBackend();
            if (!openCl.IsAvailable)
            {
                openCl.Dispose();
                return null;
            }

            owned = openCl;
            return openCl;
        }
        catch (Exception unavailable) when (unavailable is DllNotFoundException or InvalidOperationException
            or NotSupportedException or TypeInitializationException)
        {
            System.Diagnostics.Trace.TraceInformation("{0} is not available on this host: {1}", device, unavailable);
            return null;
        }
    }

    [SkippableTheory]
    [InlineData(Device.OpenCl)]
    [InlineData(Device.Vulkan)]
    [InlineData(Device.WebGpu)]
    public void Device_max_and_min_are_nan_when_any_element_is(Device device)
    {
        var backend = Create(device, out var owned);
        Skip.If(backend is null, $"{device} is not available on this host.");
        using (owned)
        {
            Check(values =>
            {
                using var buffer = backend.AllocateBuffer(values);
                return (backend.Max(buffer, values.Length), backend.Min(buffer, values.Length));
            }, device.ToString());
        }
    }

    [SkippableFact]
    public void Gpu_engine_max_and_min_are_nan_when_any_element_is()
    {
        Skip.IfNot(AiDotNetEngine.Current is DirectGpuTensorEngine, "The process engine is not a GPU engine.");
        var engine = AiDotNetEngine.Current;
        Check(values =>
        {
            var tensor = new Tensor<float>(new[] { values.Length }, new Vector<float>(values));
            return (engine.TensorMaxValue(tensor), engine.TensorMinValue(tensor));
        }, "DirectGpuTensorEngine");
    }
}
