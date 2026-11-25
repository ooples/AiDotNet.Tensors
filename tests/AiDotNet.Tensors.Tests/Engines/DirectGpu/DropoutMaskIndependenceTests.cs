// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A backend's dropout draws its mask from the seed: masks of different seeds must be independent draws, and a mask's
/// elements independent of their neighbours. The engine seeds consecutive dropout calls with consecutive counters, so
/// a generator whose masks for nearby seeds coincide drops the same units step after step.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DropoutMaskIndependenceTests
{
    private const int Size = 1 << 16;
    private const float Rate = 0.3f;

    public enum Device
    {
        OpenCl,
        Vulkan,
        WebGpu,
        Cuda,
    }

    private static IDirectGpuBackend? Create(Device device, out IDisposable? owned, out Exception? initializationFailure)
    {
        owned = null;
        initializationFailure = null;
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

            if (device == Device.Cuda)
            {
                if (!CudaBackend.IsCudaAvailable) return null;
                // The driver and NVRTC are present, so a backend that is not available failed to initialize.
                var cuda = new CudaBackend();
                if (!cuda.IsAvailable)
                {
                    cuda.Dispose();
                    initializationFailure = new InvalidOperationException(
                        "CUDA was detected, but backend initialization failed.");
                    return null;
                }

                owned = cuda;
                return cuda;
            }

            var openCl = new OpenClBackend();
            // A device whose kernels did not compile is a failure, not an absent device: skipping it would hide a
            // compile error in the dropout kernel itself.
            if (openCl.InitializationState == GpuBackendInitializationState.Failed)
            {
                initializationFailure = new InvalidOperationException(
                    "OpenCL was detected, but backend initialization failed.", openCl.InitializationException);
                openCl.Dispose();
                return null;
            }

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

    private static bool[] Kept(IDirectGpuBackend backend, ulong seed)
    {
        var ones = new float[Size];
        for (int i = 0; i < Size; i++) ones[i] = 1f;
        using var input = backend.AllocateBuffer(ones);
        using var output = backend.AllocateBuffer(Size);
        using var mask = backend.AllocateBuffer(Size);
        backend.Dropout(input, output, mask, Size, Rate, seed, training: true);
        var host = new float[Size];
        backend.DownloadBuffer(output, host);
        var kept = new bool[Size];
        float scale = 1f / (1f - Rate);
        for (int i = 0; i < Size; i++)
        {
            Assert.True(host[i] == 0f || Math.Abs(host[i] - scale) <= 1e-5f, $"element {i} is {host[i]}, neither 0 nor 1/keep.");
            kept[i] = host[i] != 0f;
        }

        return kept;
    }

    [SkippableTheory]
    [InlineData(Device.OpenCl)]
    [InlineData(Device.Vulkan)]
    [InlineData(Device.WebGpu)]
    [InlineData(Device.Cuda)]
    public void Masks_are_independent_across_seeds_and_along_the_buffer(Device device)
    {
        var backend = Create(device, out var owned, out var initializationFailure);
        if (initializationFailure is not null)
            throw new InvalidOperationException($"{device} is present but did not initialize.", initializationFailure);
        Skip.If(backend is null, $"{device} is not available on this host.");
        using (owned)
        {
            double keep = 1.0 - Rate;
            // Six standard deviations of a fraction of Size independent Bernoulli draws with success probability p.
            static double Band(double p) => 6.0 * Math.Sqrt(p * (1.0 - p) / Size);
            ulong[] seeds = { 1, 2, 3, 4, 1000, 1001, 0x9E3779B97F4A7C15UL, 0x9E3779B97F4A7C16UL, ulong.MaxValue };
            var masks = new List<bool[]>();
            foreach (ulong seed in seeds)
            {
                var mask = Kept(backend, seed);
                masks.Add(mask);
                int count = 0;
                for (int i = 0; i < Size; i++) if (mask[i]) count++;
                Assert.True(Math.Abs(((double)count / Size) - keep) <= Band(keep),
                    $"{device} seed {seed}: kept {(double)count / Size:F4} of the elements, expected {keep:F4}.");

                // Neighbours at lags 1 to 16: both kept with probability keep squared.
                for (int lag = 1; lag <= 16; lag++)
                {
                    int both = 0;
                    for (int i = 0; i + lag < Size; i++) if (mask[i] && mask[i + lag]) both++;
                    double fraction = (double)both / (Size - lag);
                    Assert.True(Math.Abs(fraction - (keep * keep)) <= Band(keep * keep),
                        $"{device} seed {seed}: elements {lag} apart are both kept {fraction:F4} of the time, expected {keep * keep:F4}.");
                }
            }

            // The same seed is the same mask.
            var again = Kept(backend, seeds[0]);
            for (int i = 0; i < Size; i++) Assert.True(again[i] == masks[0][i], $"{device}: seed {seeds[0]} drew two masks (element {i}).");

            // Two seeds agree where independent draws would: keep^2 + drop^2 of the elements.
            double agree = (keep * keep) + (Rate * Rate);
            for (int a = 0; a < masks.Count; a++)
                for (int b = a + 1; b < masks.Count; b++)
                {
                    int same = 0;
                    for (int i = 0; i < Size; i++) if (masks[a][i] == masks[b][i]) same++;
                    double fraction = (double)same / Size;
                    Assert.True(Math.Abs(fraction - agree) <= Band(agree),
                        $"{device}: the masks of seeds {seeds[a]} and {seeds[b]} agree on {fraction:F4} of the elements, expected {agree:F4}.");
                }
        }
    }
}
