using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The shapes that made two CUDA kernels serial: max over long rows (a thread per row read each 49k-wide row alone) and
/// the sorted scatter's run starts for few items into many destinations (a thread per item filled every empty run up to
/// its key). Both must match a CPU reference exactly.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaLongRowKernelTests
{
    private static (DirectGpuTensorEngine Gpu, CudaBackend Cuda) Cuda()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        if (gpu is null || !gpu.IsGpuAvailable || gpu.GetBackend() is not CudaBackend cuda)
        {
            gpu?.Dispose();
            Skip.If(true, "CUDA backend did not resolve.");
            throw new InvalidOperationException("Unreachable after Skip.If.");
        }
        return (gpu, cuda);
    }

    [SkippableTheory]
    [InlineData(600, 49152)]
    [InlineData(9, 300)]
    [InlineData(7, 256)]
    [InlineData(1000, 17)]   // short rows: the thread-per-row kernel
    public void MaxAxis_MatchesTheCpu_IncludingNaNAndInfinities(int rows, int width)
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var rng = new Random(rows * 31 + width);
            var data = new float[rows * width];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 200 - 100);
            // The CPU ReduceMax contract (ReduceMaxSeed): scan from -inf, take a value only when strictly greater, so a
            // NaN never wins wherever it sits, an all-NaN row gives -inf, and of equal maxima the first is kept.
            data[0] = float.NaN;                                          // row 0: a leading NaN is skipped
            if (rows > 1) data[width + width / 2] = float.NaN;            // row 1: so is a later one
            if (rows > 2) for (int i = 0; i < width; i++) data[2 * width + i] = float.NegativeInfinity;
            if (rows > 3) data[3 * width + width - 1] = float.PositiveInfinity;
            if (rows > 4) for (int i = 0; i < width; i++) data[4 * width + i] = float.NaN;   // all NaN: -inf
            if (rows > 5)
            {
                // A -0/+0 tie for the maximum, -0 first: -0 is kept, its bits included.
                for (int i = 0; i < width; i++) data[5 * width + i] = -1f - i;
                data[5 * width + 1] = -0f;
                data[5 * width + width - 1] = 0f;
            }
            if (rows > 6)
            {
                // Subnormals: never flushed to zero under --use_fast_math.
                for (int i = 0; i < width; i++) data[6 * width + i] = -float.Epsilon;
                data[6 * width + width - 1] = float.Epsilon;
            }

            var expected = new CpuEngine()
                .ReduceMax(new Tensor<float>(data, new[] { rows, width }), new[] { 1 }, keepDims: false)
                .ToArray();

            using var input = cuda.AllocateBuffer(data);
            using var output = cuda.AllocateBuffer(rows);
            cuda.MaxAxis(input, output, rows, width);
            var actual = cuda.DownloadBuffer(output);
            for (int r = 0; r < rows; r++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[r]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[r]),
                    $"row {r}: gpu {actual[r]} cpu {expected[r]}");
        }
    }

    [SkippableFact]
    public unsafe void MaxAxisRows_ScansARowWithinABlockOfIntMaxValue()
    {
        // A row within blockDim.x of int.MaxValue elements stepped an int index past int.MaxValue, where it wrapped
        // negative and still passed the bound: the kernel read 8 GiB below the row. The row is 8 GiB of address space
        // over one small physical allocation mapped again and again (CUDA virtual memory), so it runs on any device.
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            const int axis = int.MaxValue - 100;   // the last thread's next index passes int.MaxValue
            Check(CudaNativeBindings.cuCtxPushCurrent(cuda.CudaContextHandle), "cuCtxPushCurrent");
            IntPtr reserved = IntPtr.Zero;
            ulong handle = 0, granularity = 0, span = 0;
            int mapped = 0;
            var gate = AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.DirectPtxFeatureGate.TestOverride;
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.DirectPtxFeatureGate.TestOverride = false;   // the NVRTC kernel
            try
            {
                Check(CudaNativeBindings.cuCtxGetDevice(out int device), "cuCtxGetDevice");
                const int VirtualMemoryManagementSupported = 102;
                Skip.If(AiDotNet.Tensors.Engines.CuBlasNative.cuDeviceGetAttribute(out int vmm, VirtualMemoryManagementSupported, device)
                    != CudaResult.Success || vmm == 0, "the device has no virtual memory management");
                var prop = new CudaMemAllocationProp { Type = 1, LocationType = 1, LocationId = device };
                Check(CudaNativeBindings.cuMemGetAllocationGranularity(out granularity, ref prop, 0), "cuMemGetAllocationGranularity");
                ulong rowBytes = (ulong)axis * sizeof(float);
                span = (rowBytes + granularity - 1) / granularity * granularity;
                Check(CudaNativeBindings.cuMemCreate(out handle, granularity, ref prop, 0), "cuMemCreate");
                Check(CudaNativeBindings.cuMemAddressReserve(out reserved, span, granularity, IntPtr.Zero, 0), "cuMemAddressReserve");
                for (ulong at = 0; at < span; at += granularity, mapped++)
                    Check(CudaNativeBindings.cuMemMap(new IntPtr(reserved.ToInt64() + (long)at), granularity, 0, handle, 0), "cuMemMap");
                var access = new CudaMemAccessDesc { LocationType = 1, LocationId = device, Flags = 3 };
                Check(CudaNativeBindings.cuMemSetAccess(reserved, span, ref access, 1), "cuMemSetAccess");

                // Every granule is the same memory: -1 everywhere but one 5, so the row's maximum is 5.
                int perGranule = (int)(granularity / sizeof(float));
                var host = Enumerable.Repeat(-1f, perGranule).ToArray();
                host[perGranule / 3] = 5f;
                fixed (float* src = host)
                    Check(AiDotNet.Tensors.Engines.CuBlasNative.cuMemcpyHtoD(reserved, (IntPtr)src, granularity), "cuMemcpyHtoD");

                var row = new CudaBackend.CudaGpuBuffer(IntPtr.Zero, reserved, axis);
                GC.SuppressFinalize(row);   // the mapping is released below, never through the pool
                using var output = cuda.AllocateBuffer(1);
                cuda.MaxAxis(row, output, 1, axis);
                Assert.Equal(5f, cuda.DownloadBuffer(output)[0]);
            }
            finally
            {
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.DirectPtxFeatureGate.TestOverride = gate;
                if (reserved != IntPtr.Zero)
                {
                    for (int m = 0; m < mapped; m++)
                        CudaNativeBindings.cuMemUnmap(new IntPtr(reserved.ToInt64() + (long)((ulong)m * granularity)), granularity);
                    CudaNativeBindings.cuMemAddressFree(reserved, span);
                }
                if (handle != 0) CudaNativeBindings.cuMemRelease(handle);
                CudaNativeBindings.cuCtxPopCurrent(out _);
            }
        }
    }

    private static void Check(CudaResult result, string call)
    {
        if (result != CudaResult.Success) throw new InvalidOperationException(call + " failed: " + result);
    }

    [SkippableTheory]
    [InlineData(600, 49152, 8)]     // few items, many destinations: long empty runs
    [InlineData(5000, 64, 3)]       // many items per destination
    [InlineData(1, 100000, 2)]      // one item
    public void SortedScatterAdd_ManyDestinations_MatchesTheCpu(int count, int destinations, int features)
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var rng = new Random(count + destinations);
            var indices = Enumerable.Range(0, count).Select(i => i % 7 == 3 ? -1 : rng.Next(destinations + 2)).ToArray();
            var source = Enumerable.Range(0, count * features).Select(i => (float)Math.Sin(i * 0.37)).ToArray();
            var expected = new float[destinations * features];
            for (int i = 0; i < count; i++)
            {
                int d = indices[i];
                if (d < 0 || d >= destinations) continue;
                for (int f = 0; f < features; f++) expected[d * features + f] += source[i * features + f];
            }

            using var src = cuda.AllocateBuffer(source);
            using var idx = cuda.AllocateIntBuffer(indices);
            using var output = cuda.AllocateBuffer(destinations * features);
            cuda.SortedScatterSwitch = true;
            cuda.SortedScatterIgnoresCostModel = true;   // the one-item case is one the cost model never sends here
            bool ran = cuda.TrySortedScatterAdd(src, idx, output, count, destinations, features, accumulate: false);
            Assert.True(ran, "the sorted scatter did not run (kernels missing?)");
            var actual = cuda.DownloadBuffer(output);
            // Bit for bit: the scatter is stable and each destination sums its sources in increasing source order from
            // +0, as the reference does, so even a zero's sign must agree.
            for (int i = 0; i < expected.Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[i]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                    $"output[{i / features}, {i % features}]: gpu {actual[i]} cpu {expected[i]}");
        }
    }
}
