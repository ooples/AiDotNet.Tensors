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

    private static float FromBits(int bits) => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);

    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    // Two NaNs that are neither float.NaN nor each other, to tell which one a result carries.
    private static readonly float FirstNaN = FromBits(0x7FC00A01);
    private static readonly float LaterNaN = FromBits(0x7FC00B02);

    // With canonicalNaN the range is poisoned with a NaN that is not float.NaN and the result must be float.NaN's
    // bit pattern: the device full reductions' contract (ExactFloatKernels).
    private static void Check(Func<float[], (float Max, float Min)> extrema, string where, bool canonicalNaN = false)
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
                poisoned[position] = canonicalNaN ? FirstNaN : float.NaN;
                var result = extrema(poisoned);
                Assert.True(float.IsNaN(result.Max), $"{where}: max of {size} elements with a NaN at {position} is {result.Max:R}.");
                Assert.True(float.IsNaN(result.Min), $"{where}: min of {size} elements with a NaN at {position} is {result.Min:R}.");
                if (canonicalNaN)
                {
                    Assert.True(Bits(result.Max) == Bits(float.NaN), $"{where}: max of {size} elements with a NaN at {position} has bits 0x{Bits(result.Max):X8}.");
                    Assert.True(Bits(result.Min) == Bits(float.NaN), $"{where}: min of {size} elements with a NaN at {position} has bits 0x{Bits(result.Min):X8}.");
                }
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
        Exception? initializationFailure = null;
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
            if (openCl.IsAvailable)
            {
                owned = openCl;
                return openCl;
            }

            // A device that is there but whose backend failed to come up (a kernel that no longer compiles) is a
            // failure of the code under test; only an absent device is a reason to skip.
            if (openCl.InitializationState == GpuBackendInitializationState.Failed)
                initializationFailure = openCl.InitializationException
                    ?? new InvalidOperationException("The OpenCL backend failed to initialize.");
            openCl.Dispose();
        }
        catch (Exception unavailable) when (unavailable is DllNotFoundException or InvalidOperationException
            or NotSupportedException or TypeInitializationException)
        {
            System.Diagnostics.Trace.TraceInformation("{0} is not available on this host: {1}", device, unavailable);
            return null;
        }

        // Outside the try: its filter would turn this into a skip.
        if (initializationFailure is not null)
            throw new InvalidOperationException("An OpenCL device was detected, but the OpenCL backend failed to initialize.", initializationFailure);
        return null;
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
            }, device.ToString(), canonicalNaN: true);
        }
    }

    // Rows of several widths with a NaN first, in the middle, last, nowhere, and everywhere.
    private static float[] Rows(int rows, int width, out bool[] poisoned, out float[] cleanMax)
    {
        var data = Values(rows * width, 7 * rows + width);
        poisoned = new bool[rows];
        cleanMax = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            float max = float.NegativeInfinity;
            for (int c = 0; c < width; c++) max = Math.Max(max, data[r * width + c]);
            cleanMax[r] = max;
            switch (r % 5)
            {
                case 0:
                    data[r * width + width - 1] = LaterNaN;
                    data[r * width] = FirstNaN;
                    poisoned[r] = true;
                    break;
                case 1:
                    data[r * width + width - 1] = LaterNaN;
                    data[r * width + width / 2] = FirstNaN;
                    poisoned[r] = true;
                    break;
                case 2: data[r * width + width - 1] = FirstNaN; poisoned[r] = true; break;
                case 3: break;
                default:
                    for (int c = 0; c < width; c++) data[r * width + c] = c == 0 ? FirstNaN : LaterNaN;
                    poisoned[r] = true;
                    break;
            }
        }

        return data;
    }

    private static void CheckRows(float[] actual, bool[] poisoned, float[] cleanMax, string where, int width)
    {
        for (int r = 0; r < poisoned.Length; r++)
        {
            if (poisoned[r])
            {
                Assert.True(float.IsNaN(actual[r]), $"{where}: row {r} of width {width} holds a NaN but its max is {actual[r]:R}.");
                // The row's first NaN, bits included: not a canonical NaN, and not a later one.
                Assert.True(Bits(actual[r]) == Bits(FirstNaN),
                    $"{where}: row {r} of width {width} has NaN bits 0x{Bits(actual[r]):X8}, expected the first NaN's 0x{Bits(FirstNaN):X8}.");
            }
            else
                Assert.True(actual[r] == cleanMax[r], $"{where}: row {r} of width {width} is {actual[r]:R}, expected {cleanMax[r]:R}.");
        }
    }

    [Theory]
    [InlineData(10, 1)]
    [InlineData(10, 7)]
    [InlineData(10, 300)]
    [InlineData(5, 4097)]
    public void Cpu_engine_row_max_is_nan_for_a_row_that_holds_one(int rows, int width)
    {
        var engine = new CpuEngine();
        var data = Rows(rows, width, out var poisoned, out var cleanMax);

        var single = new Tensor<float>(new[] { rows, width }, new Vector<float>(data));
        CheckRows(engine.ReduceMax(single, new[] { 1 }, keepDims: false).AsSpan().ToArray(), poisoned, cleanMax, "CpuEngine float", width);

        var wide = new double[data.Length];
        for (int i = 0; i < data.Length; i++) wide[i] = data[i];
        var doubles = engine.ReduceMax(new Tensor<double>(new[] { rows, width }, new Vector<double>(wide)), new[] { 1 }, keepDims: false).AsSpan();
        var narrowed = new float[rows];
        for (int r = 0; r < rows; r++) narrowed[r] = (float)doubles[r];
        CheckRows(narrowed, poisoned, cleanMax, "CpuEngine double", width);

        // The reduced axis first: every column that holds a NaN is NaN.
        var transposed = new float[data.Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < width; c++)
                transposed[c * rows + r] = data[r * width + c];
        var columns = new Tensor<float>(new[] { width, rows }, new Vector<float>(transposed));
        CheckRows(engine.ReduceMax(columns, new[] { 0 }, keepDims: false).AsSpan().ToArray(), poisoned, cleanMax, "CpuEngine axis 0", width);

        // The index a NaN result reports is that of the row's first NaN.
        engine.ReduceMax(single, new[] { 1 }, keepDims: false, out int[] indices);
        for (int r = 0; r < rows; r++)
        {
            if (!poisoned[r]) continue;
            int first = -1;
            for (int c = 0; c < width && first < 0; c++) if (float.IsNaN(data[r * width + c])) first = r * width + c;
            Assert.Equal(first, indices[r]);
        }
    }

    [SkippableTheory]
    [InlineData(Device.OpenCl, 10, 1)]
    [InlineData(Device.OpenCl, 10, 7)]
    [InlineData(Device.OpenCl, 10, 300)]
    [InlineData(Device.OpenCl, 5, 4097)]
    [InlineData(Device.Vulkan, 10, 7)]
    [InlineData(Device.Vulkan, 10, 300)]
    [InlineData(Device.Vulkan, 5, 4097)]
    [InlineData(Device.WebGpu, 10, 7)]
    [InlineData(Device.WebGpu, 10, 300)]
    [InlineData(Device.WebGpu, 5, 4097)]
    public void Device_row_max_is_nan_for_a_row_that_holds_one(Device device, int rows, int width)
    {
        var backend = Create(device, out var owned);
        Skip.If(backend is null, $"{device} is not available on this host.");
        using (owned)
        {
            var data = Rows(rows, width, out var poisoned, out var cleanMax);
            using var input = backend.AllocateBuffer(data);
            using var output = backend.AllocateBuffer(rows);
            backend.MaxAxis(input, output, rows, width);
            var actual = new float[rows];
            backend.DownloadBuffer(output, actual);
            CheckRows(actual, poisoned, cleanMax, device.ToString(), width);
        }
    }

    [SkippableTheory]
    [InlineData(10, 7)]
    [InlineData(10, 300)]
    [InlineData(5, 4097)]
    public void Gpu_engine_row_max_is_nan_for_a_row_that_holds_one(int rows, int width)
    {
        Skip.IfNot(AiDotNetEngine.Current is DirectGpuTensorEngine, "The process engine is not a GPU engine.");
        var data = Rows(rows, width, out var poisoned, out var cleanMax);
        var tensor = new Tensor<float>(new[] { rows, width }, new Vector<float>(data));
        var actual = AiDotNetEngine.Current.ReduceMax(tensor, new[] { 1 }, keepDims: false).AsSpan().ToArray();
        CheckRows(actual, poisoned, cleanMax, "DirectGpuTensorEngine", width);
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
