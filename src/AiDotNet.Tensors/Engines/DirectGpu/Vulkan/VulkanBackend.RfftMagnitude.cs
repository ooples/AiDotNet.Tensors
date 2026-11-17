// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static partial class VulkanBatchExecutionKernels
{
    internal static class RfftMagnitudeShaders
    {
        internal static readonly string Magnitude = Shader(new[] { "r:spectrum", "w:resultValues" },
            "int rows; int n; int spectrumStride; float epsilon;", @"
void main() {
    int i = bx_gid(); int bins = n / 2 + 1; if (i >= rows * bins) return;
    int source = 2 * ((i / bins) * spectrumStride + i % bins);
    precise float re = spectrum[source]; precise float im = spectrum[source + 1];
    precise float squared = re * re + im * im;
    resultValues[i] = sqrt(squared + epsilon);
}
");
        internal static readonly string Seed = Shader(new[] { "r:spectrum", "r:upstream", "w:seed" },
            "int rows; int n; int spectrumStride; float epsilon;", @"
void main() {
    int i = bx_gid(); if (i >= rows * n) return;
    int k = i % n; int bins = n / 2 + 1;
    precise float re = 0.0; precise float im = 0.0;
    if (k < bins) {
        int source = 2 * ((i / n) * spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        precise float squared = re * re + im * im;
        float scale = upstream[(i / n) * bins + k] / sqrt(squared + epsilon);
        re = re * scale; im = im * scale;
    }
    seed[2 * i] = re; seed[2 * i + 1] = im;
}
");
        internal static readonly string Gradient = Shader(new[] { "r:complexValues", "rw:resultValues" },
            "int count; float scale; int accumulate;", @"
void main() {
    int i = bx_gid(); if (i >= count) return;
    precise float value = complexValues[2 * i] * scale;
    resultValues[i] = accumulate != 0 ? resultValues[i] + value : value;
}
");
    }
}

public sealed partial class VulkanBackend : IRfftMagnitudeKernels
{
    IRfftNativePlan? IRfftMagnitudeKernels.GetOptimizedRealFftPlan(int n, int rows, bool inverse) => null;
    void IRfftMagnitudeKernels.RfftMagnitude(IGpuBuffer spectrum, IGpuBuffer output, int rows, int n, int spectrumStride, float epsilon)
        => GlslNaryOp(VulkanBatchExecutionKernels.RfftMagnitudeShaders.Magnitude, new[] { spectrum, output },
            checked(rows * (n / 2 + 1)), new[] { Bits(rows), Bits(n), Bits(spectrumStride), FloatBits(epsilon) });
    void IRfftMagnitudeKernels.RfftMagnitudeSeed(IGpuBuffer spectrum, IGpuBuffer upstream, IGpuBuffer seed, int rows, int n, int spectrumStride, float epsilon)
        => GlslNaryOp(VulkanBatchExecutionKernels.RfftMagnitudeShaders.Seed, new[] { spectrum, upstream, seed },
            checked(rows * n), new[] { Bits(rows), Bits(n), Bits(spectrumStride), FloatBits(epsilon) });
    void IRfftMagnitudeKernels.RfftRealGradient(IGpuBuffer complex, IGpuBuffer output, int count, float scale, bool accumulate)
        => GlslNaryOp(VulkanBatchExecutionKernels.RfftMagnitudeShaders.Gradient, new[] { complex, output },
            count, new[] { Bits(count), FloatBits(scale), Bits(accumulate ? 1 : 0) });
}
