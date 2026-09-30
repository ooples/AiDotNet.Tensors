using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Vulkan dispatches whose shader declares more push constants than the dispatch supplied read the rest as garbage:
/// masked fill never saw its fill value, and index select, the per-row losses and the batched dot product never saw
/// their second dimension. Each op is checked against the definition its CUDA kernel implements.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanPushConstantParityTests
{
    private const float Tolerance = 1e-5f;
    private const float LogFloor = 1e-7f;   // the CUDA and Vulkan loss kernels clamp predictions to this before the log
    private const int Rows = 5, Width = 7;

    [SkippableFact]
    public void MaskedFill_IndexSelect_Losses_AndBatchDot_UseEveryPushConstant()
    {
        VulkanBackend? vulkan = null;
        bool available;
        try
        {
            vulkan = VulkanBackend.Instance;
            available = vulkan.Initialize() && vulkan.IsGlslCompilerAvailable;
        }
        catch (Exception)
        {
            available = false;
        }
        Skip.If(!available || vulkan is null, "Vulkan with its runtime GLSL compiler (libshaderc) is not available.");
        var backend = vulkan!;
        IGpuBatchExecution batch = backend;

        var values = Filled(Rows * Width, 1);
        var probabilities = new float[Rows * Width];
        var targets = new float[Rows * Width];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Width; c++)
            {
                probabilities[r * Width + c] = (c + 1f) / (Width * (Width + 1) / 2f);
                targets[r * Width + c] = c == r % Width ? 1f : 0f;
            }
        var mask = new float[Rows * Width];
        for (int i = 0; i < mask.Length; i++) mask[i] = i % 3 == 0 ? 1f : 0f;
        float[] rowIds = { 4, 0, 2 };
        const float Fill = -2.5f;

        var expectedFill = new float[values.Length];
        for (int i = 0; i < values.Length; i++) expectedFill[i] = mask[i] != 0f ? Fill : values[i];
        var expectedSelect = new float[rowIds.Length * Width];
        for (int k = 0; k < rowIds.Length; k++)
            for (int c = 0; c < Width; c++) expectedSelect[k * Width + c] = values[(int)rowIds[k] * Width + c];
        var expectedCrossEntropy = new float[Rows];
        var expectedMse = new float[Rows];
        var expectedDot = new float[Rows];
        for (int r = 0; r < Rows; r++)
        {
            double ce = 0, sq = 0, dot = 0;
            for (int c = 0; c < Width; c++)
            {
                int i = r * Width + c;
                if (targets[i] > 0f) ce -= targets[i] * Math.Log(Math.Max(probabilities[i], LogFloor));
                double d = values[i] - targets[i];
                sq += d * d;
                dot += values[i] * probabilities[i];
            }
            expectedCrossEntropy[r] = (float)ce;
            expectedMse[r] = (float)(sq / Width);
            expectedDot[r] = (float)dot;
        }

        using var valueBuffer = backend.AllocateBuffer(values);
        using var maskBuffer = backend.AllocateBuffer(mask);
        using var idBuffer = backend.AllocateBuffer(rowIds);
        using var probabilityBuffer = backend.AllocateBuffer(probabilities);
        using var targetBuffer = backend.AllocateBuffer(targets);
        using var filled = backend.AllocateBuffer(values.Length);
        using var selected = backend.AllocateBuffer(rowIds.Length * Width);
        using var crossEntropy = backend.AllocateBuffer(Rows);
        using var mse = backend.AllocateBuffer(Rows);
        using var dots = backend.AllocateBuffer(Rows);

        backend.MaskedFillKernel(valueBuffer, maskBuffer, filled, Fill, values.Length);
        batch.IndexSelect(valueBuffer, idBuffer, selected, rowIds.Length, Width);
        batch.CrossEntropyLoss(probabilityBuffer, targetBuffer, crossEntropy, Rows, Width);
        batch.MseLoss(valueBuffer, targetBuffer, mse, Rows, Width);
        backend.BatchDotProduct(valueBuffer, probabilityBuffer, dots, Rows, Width);

        Compare("MaskedFillKernel", expectedFill, backend.DownloadBuffer(filled));
        Compare("IndexSelect", expectedSelect, backend.DownloadBuffer(selected));
        Compare("CrossEntropyLoss", expectedCrossEntropy, backend.DownloadBuffer(crossEntropy));
        Compare("MseLoss", expectedMse, backend.DownloadBuffer(mse));
        Compare("BatchDotProduct", expectedDot, backend.DownloadBuffer(dots));
    }

    private static void Compare(string op, float[] expected, float[] actual)
    {
        Assert.True(actual.Length >= expected.Length, $"{op}: Vulkan returned {actual.Length} values, expected {expected.Length}.");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"{op}[{i}]: Vulkan {actual[i]}, expected {expected[i]}");
    }

    private static float[] Filled(int length, int seed)
    {
        var rng = new Random(seed);
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }
}
