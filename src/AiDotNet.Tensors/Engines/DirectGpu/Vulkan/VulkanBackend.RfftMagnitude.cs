// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static partial class VulkanBatchExecutionKernels
{
    internal static class RfftMagnitudeShaders
    {
        private const string Norm = @"
float bx_rfft_norm(float re, float im, float epsilonValue) {
    float peak = max(abs(re), abs(im));
    if (peak > 1e19 || (peak > 0.0 && peak < 1e-18) || epsilonValue > 1e37) {
        peak = max(peak, sqrt(epsilonValue));
        precise vec3 scaled = vec3(re, im, sqrt(epsilonValue)) / peak;
        precise float sum = scaled.x * scaled.x + scaled.y * scaled.y + scaled.z * scaled.z;
        return peak * sqrt(sum);
    }
    precise float squared = re * re + im * im;
    return sqrt(squared + epsilonValue);
}
";
        internal static readonly string Magnitude = Shader(new[] { "r:spectrum", "w:resultValues" },
            "int rows; int n; int spectrumStride; float epsilon;", Norm + @"
void main() {
    int i = bx_gid(); int bins = n / 2 + 1; if (i >= rows * bins) return;
    int source = 2 * ((i / bins) * spectrumStride + i % bins);
    precise float re = spectrum[source]; precise float im = spectrum[source + 1];
    resultValues[i] = bx_rfft_norm(re, im, epsilon);
}
");
        internal static readonly string Seed = Shader(new[] { "r:spectrum", "r:upstream", "w:seed" },
            "int rows; int n; int spectrumStride; float epsilon;", Norm + @"
void main() {
    int i = bx_gid(); if (i >= rows * n) return;
    int k = i % n; int bins = n / 2 + 1;
    precise float re = 0.0; precise float im = 0.0;
    if (k < bins) {
        int source = 2 * ((i / n) * spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        float magnitude = bx_rfft_norm(re, im, epsilon);
        float grad = upstream[(i / n) * bins + k];
        re = (re / magnitude) * grad; im = (im / magnitude) * grad;
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
