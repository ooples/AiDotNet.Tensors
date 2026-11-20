#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TensorLayout = AiDotNet.Tensors.LinearAlgebra.TensorLayout;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    private static bool s_waveformConvBiasObserved;

    private bool TryWaveformConvBiasInto(Tensor<float> output, Tensor<float> input, Tensor<float> kernel,
        Tensor<float> bias, int[] stride, int[] padding, int[] dilation)
    {
        int width = input._shape[3], channels = input._shape[1], filters = kernel._shape[0], taps = kernel._shape[3];
        if (GetType() != typeof(CpuEngine) || !Avx2.IsSupported || !Fma.IsSupported
            || input.Layout != TensorLayout.Nchw || output.Layout != TensorLayout.Nchw
            || !output.IsContiguous || !input.IsContiguous || !kernel.IsContiguous || !bias.IsContiguous
            || input.HasPendingGpuData || kernel.HasPendingGpuData || bias.HasPendingGpuData || output.HasPendingGpuData
            || input._shape[2] != 1 || kernel._shape[2] != 1 || output._shape[2] != 1
            || stride[0] != 1 || stride[1] != 1 || padding[0] != 0 || dilation[0] != 1 || dilation[1] <= 0
            || width < 64 || width % 32 != 0 || output._shape[3] != width || channels < 8 || channels % 8 != 0
            || filters < 8 || taps < 3 || taps > 15 || taps % 2 == 0 || padding[1] != (long)(taps / 2) * dilation[1])
            return false;
        using var keepInput = new KeepAliveScope(input);
        using var keepKernel = new KeepAliveScope(kernel);
        using var keepBias = new KeepAliveScope(bias);
        using var keepOutput = new KeepAliveScope(output);
        var x = input.ReadOnlyDataUnmarked;
        var k = kernel.ReadOnlyDataUnmarked;
        var b = bias.ReadOnlyDataUnmarked;
        var y = output.DataUnmarked;
        if (x.Span.Overlaps(y.Span) || k.Span.Overlaps(y.Span) || b.Span.Overlaps(y.Span)) return false;
        int spacing = dilation[1], pad = padding[1];
        // Parallel rows have disjoint outputs. Each vector holds adjacent waveform positions.
        CpuParallelSettings.ParallelForOrSerial(0, input._shape[0] * filters, (long)output.Length * channels * taps, row =>
        {
            int filter = row % filters, image = row / filters;
            WaveformConvBiasRow(x.Span.Slice(image * channels * width, channels * width),
                k.Span.Slice(filter * channels * taps, channels * taps), b.Span[filter],
                y.Span.Slice(row * width, width), channels, width, taps, spacing, pad);
        }, deterministicSafe: true);
        output.IncrementVersion();
        if (!s_waveformConvBiasObserved && Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
        {
            s_waveformConvBiasObserved = true;
            AiDotNetEngine.LogDiagnostic("[WAVEFORM-CONV-BIAS] executed");
        }
        return true;
    }

    private static unsafe void WaveformConvBiasRow(ReadOnlySpan<float> input, ReadOnlySpan<float> kernel,
        float bias, Span<float> output, int channels, int width, int taps, int spacing, int pad)
    {
        fixed (float* x = input, k = kernel, y = output)
        {
            for (int t = 0; t < width; t += 32)
            {
                var a = Vector256.Create(bias);
                var b = a;
                var c = a;
                var d = a;
                for (int group = 0; group < channels; group += 8)
                    for (int tap = 0; tap < taps; tap++)
                    {
                        int pos = t + tap * spacing - pad;
                        for (int ch = group; ch < group + 8; ch++)
                        {
                            var weight = Vector256.Create(k[ch * taps + tap]);
                            float* src = x + ch * width;
                            if (pos >= 0 && pos <= width - 32)
                            {
                                a = Fma.MultiplyAdd(Avx.LoadVector256(src + pos), weight, a);
                                b = Fma.MultiplyAdd(Avx.LoadVector256(src + pos + 8), weight, b);
                                c = Fma.MultiplyAdd(Avx.LoadVector256(src + pos + 16), weight, c);
                                d = Fma.MultiplyAdd(Avx.LoadVector256(src + pos + 24), weight, d);
                            }
                            else
                            {
                                a = Fma.MultiplyAdd(PaddedWaveformLoad(src, pos, width), weight, a);
                                b = Fma.MultiplyAdd(PaddedWaveformLoad(src, pos + 8, width), weight, b);
                                c = Fma.MultiplyAdd(PaddedWaveformLoad(src, pos + 16, width), weight, c);
                                d = Fma.MultiplyAdd(PaddedWaveformLoad(src, pos + 24, width), weight, d);
                            }
                        }
                    }
                Avx.Store(y + t, a);
                Avx.Store(y + t + 8, b);
                Avx.Store(y + t + 16, c);
                Avx.Store(y + t + 24, d);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> PaddedWaveformLoad(float* x, int pos, int width)
    {
        if (pos >= 0 && pos <= width - 8) return Avx.LoadVector256(x + pos);
        if (pos <= -8 || pos >= width) return Vector256<float>.Zero;
        float* edge = stackalloc float[8];
        for (int i = 0; i < 8; i++) edge[i] = (uint)(pos + i) < (uint)width ? x[pos + i] : 0f;
        return Avx.LoadVector256(edge);
    }
}
#endif
