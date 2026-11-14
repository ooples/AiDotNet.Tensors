// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// A convolution kernel registered for streaming with a quantized inference store (bf16, int8, int4) is read-only.
/// CpuEngine.Conv2D and Conv2DInto read the double kernel through the writable accessor, so every inference
/// convolution over such a weight threw "Cannot mutate a streaming tensor stored with a quantized inference
/// encoding" (OpenVocabSAM in AiDotNet). Serialized against other registry tests (global state).
/// </summary>
[Collection("WeightRegistry")]
public class StreamingQuantizedConvKernelTests
{
    private const int InChannels = 2, OutChannels = 3, Size = 8, KernelSize = 3;

    [Theory]
    [InlineData(StreamingStoreDtype.Bf16)]
    [InlineData(StreamingStoreDtype.Int8)]
    public void Conv2D_ReadsAQuantizedStreamingKernel_WithoutMutatingIt(StreamingStoreDtype dtype)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-qconv-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = dtype,
        });
        WeightRegistry.SetStreamingExecutionTraining(false); // inference
        try
        {
            var kernel = new Tensor<double>(new[] { OutChannels, InChannels, KernelSize, KernelSize });
            for (int i = 0; i < kernel.Length; i++) kernel[i] = Math.Sin(i * 0.37) * 0.5;
            kernel.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(kernel);
            Assert.NotEqual(StreamingEncoding.Native, kernel.StreamingStoreEncoding);
            Assert.NotEqual(StreamingEncoding.Lossless, kernel.StreamingStoreEncoding);

            var input = new Tensor<double>(new[] { 1, InChannels, Size, Size });
            for (int i = 0; i < input.Length; i++) input[i] = Math.Cos(i * 0.21);

            // The kernel's stored (dequantized) values before either convolution: the reference is computed from
            // this copy, and the kernel is compared with it after each call, so an engine that altered the weights
            // cannot also move the reference.
            var stored = new double[kernel.Length];
            for (int i = 0; i < stored.Length; i++) stored[i] = kernel[i];

            var engine = new CpuEngine();
            var viaConv2D = engine.Conv2D(input, kernel, stride: 1, padding: 1);
            for (int i = 0; i < stored.Length; i++)
                Assert.True(stored[i] == kernel[i], $"Conv2D changed kernel[{i}]: {stored[i]:R} -> {kernel[i]:R}");
            var viaInto = new Tensor<double>(new[] { 1, OutChannels, Size, Size });
            engine.Conv2DInto(viaInto, input, kernel, stride: 1, padding: 1);
            for (int i = 0; i < stored.Length; i++)
                Assert.True(stored[i] == kernel[i], $"Conv2DInto changed kernel[{i}]: {stored[i]:R} -> {kernel[i]:R}");

            // Reference: a direct 3x3, padding 1 convolution over the stored values.
            for (int oc = 0; oc < OutChannels; oc++)
            for (int oy = 0; oy < Size; oy++)
            for (int ox = 0; ox < Size; ox++)
            {
                double expected = 0;
                for (int ic = 0; ic < InChannels; ic++)
                for (int ky = 0; ky < KernelSize; ky++)
                for (int kx = 0; kx < KernelSize; kx++)
                {
                    int iy = oy + ky - 1, ix = ox + kx - 1;
                    if (iy < 0 || iy >= Size || ix < 0 || ix >= Size) continue;
                    expected += input[(ic * Size + iy) * Size + ix]
                        * stored[((oc * InChannels + ic) * KernelSize + ky) * KernelSize + kx];
                }
                int o = (oc * Size + oy) * Size + ox;
                Assert.True(Math.Abs(viaConv2D[o] - expected) <= 1e-12 * Math.Max(1, Math.Abs(expected)),
                    $"Conv2D[{oc},{oy},{ox}]: expected {expected:R}, got {viaConv2D[o]:R}");
                Assert.True(Math.Abs(viaInto[o] - expected) <= 1e-12 * Math.Max(1, Math.Abs(expected)),
                    $"Conv2DInto[{oc},{oy},{ox}]: expected {expected:R}, got {viaInto[o]:R}");
            }
        }
        finally
        {
            WeightRegistry.SetStreamingExecutionTraining(null);
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
