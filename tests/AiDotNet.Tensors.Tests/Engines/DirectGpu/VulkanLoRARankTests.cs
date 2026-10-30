// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="VulkanBackend.FusedLoRAForward"/> at every rank (#1230). The single-dispatch kernel keeps a row's
/// projection in work-group shared memory, which holds <see cref="VulkanBackend.LoRASharedMemoryRank"/> entries;
/// a larger rank used to be refused and now runs as two dispatches. Both sides of that limit are checked against
/// the same CPU reference: output = base + scaling * (input . A) . B.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanLoRARankTests
{
    private readonly VulkanBackend _backend;
    private readonly bool _ready;

    public VulkanLoRARankTests()
    {
        try
        {
            _backend = VulkanBackend.Instance;
            _ready = _backend.Initialize() && _backend.IsGlslCompilerAvailable;
        }
        catch
        {
            _ready = false;
        }
    }

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the Vulkan LoRA kernels were unavailable (Vulkan device + libshaderc required).");
        return false;
    }

    private static float[] Random(int n, int seed, double scale)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return d;
    }

    [SkippableTheory]
    [InlineData(3, 16, 4, 12)]
    [InlineData(2, 40, 255, 24)]
    [InlineData(2, 40, 256, 24)]     // the largest rank the single-dispatch kernel takes
    [InlineData(2, 40, 257, 24)]     // the first rank that needs the projection buffer
    [InlineData(5, 33, 300, 17)]
    [InlineData(1, 8, 1024, 300)]    // batch * rank and batch * outputFeatures both span several work-groups
    [InlineData(300, 8, 259, 3)]
    public void FusedLoRAForward_AnyRank_MatchesCpu(int batch, int inF, int rank, int outF)
    {
        Skip.If(!EnsureReady(), "Vulkan LoRA kernels not available.");
        const float scaling = 0.37f;
        var input = Random(batch * inF, 1 + rank, 1.0);
        var baseOutput = Random(batch * outF, 2 + rank, 2.0);
        var loraA = Random(inF * rank, 3 + rank, 0.5);
        var loraB = Random(rank * outF, 4 + rank, 0.5);

        var expected = new float[batch * outF];
        var proj = new double[rank];
        double largest = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < rank; r++)
            {
                double acc = 0;
                for (int i = 0; i < inF; i++) acc += (double)input[b * inF + i] * loraA[i * rank + r];
                proj[r] = acc;
            }
            for (int o = 0; o < outF; o++)
            {
                double delta = 0;
                for (int r = 0; r < rank; r++) delta += proj[r] * loraB[r * outF + o];
                expected[b * outF + o] = (float)(baseOutput[b * outF + o] + scaling * delta);
                largest = Math.Max(largest, Math.Abs(scaling * delta));
            }
        }
        // The update has to be large enough that an output equal to the base would fail.
        Assert.True(largest > 0.05, $"degenerate case: the largest update is {largest}");

        using var inputB = _backend.AllocateBuffer(input);
        using var baseB = _backend.AllocateBuffer(baseOutput);
        using var aB = _backend.AllocateBuffer(loraA);
        using var bB = _backend.AllocateBuffer(loraB);
        using var outB = _backend.AllocateBuffer(batch * outF);
        _backend.FusedLoRAForward(inputB, baseB, aB, bB, outB, batch, inF, rank, outF, scaling);
        var actual = _backend.DownloadBuffer(outB);

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 + 1e-4 * Math.Abs(expected[i]),
                $"[{i}] expected {expected[i]}, got {actual[i]} (batch {batch}, in {inF}, rank {rank}, out {outF})");
        // The base output is an input and must not be written.
        Assert.Equal(baseOutput, _backend.DownloadBuffer(baseB).AsSpan(0, baseOutput.Length).ToArray());
    }
}

#endif
