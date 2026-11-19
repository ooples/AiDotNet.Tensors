using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TensorLayout = AiDotNet.Tensors.LinearAlgebra.TensorLayout;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    internal static bool SingleOutputConvBiasEnabled = Environment.GetEnvironmentVariable("AIDOTNET_CPU_FUSED_OUTPUT_BIAS") == "1";
    [ThreadStatic] internal static int SingleOutputConvBiasRuns;

    // Long, narrow waveform projections do not benefit from a general im2col/GEMM
    // followed by a channel epilogue. Keep eight spatial positions in each vector,
    // seed the accumulator with bias, and traverse eight-channel kernel blocks.
    internal unsafe bool TrySingleOutputConvBiasInto(Tensor<float> output, Tensor<float> input,
        Tensor<float> kernel, Tensor<float> bias, int[] stride, int[] padding, int[] dilation)
    {
#if NET5_0_OR_GREATER
        if (GetType() != typeof(CpuEngine) || !Avx2.IsSupported || !Fma.IsSupported
            || input.Rank != 4 || kernel.Rank != 4 || output.Rank != 4 || bias.Rank != 1
            || !input.IsContiguous || !kernel.IsContiguous || !bias.IsContiguous || !output.IsContiguous
            || input.Layout != TensorLayout.Nchw || output.Layout != TensorLayout.Nchw
            || input.HasPendingGpuData || kernel.HasPendingGpuData || bias.HasPendingGpuData || output.HasPendingGpuData
            || stride.Length != 2 || padding.Length != 2 || dilation.Length != 2
            || stride[0] != 1 || stride[1] != 1 || dilation[0] != 1 || dilation[1] != 1 || padding[0] != 0
            || input._shape[0] != 1 || input._shape[2] != 1 || kernel._shape[0] != 1 || kernel._shape[2] != 1
            || input._shape[1] < 8 || input._shape[1] > 64 || input._shape[1] % 8 != 0
            || input._shape[3] < 256 || kernel._shape[1] != input._shape[1]
            || kernel._shape[3] < 1 || kernel._shape[3] > 15 || kernel._shape[3] % 2 == 0
            || padding[1] != kernel._shape[3] / 2 || bias.Length != 1
            || output._shape[0] != 1 || output._shape[1] != 1 || output._shape[2] != 1
            || output._shape[3] != input._shape[3]) return false;
        using var keepInput = new KeepAliveScope(input);
        using var keepKernel = new KeepAliveScope(kernel);
        using var keepBias = new KeepAliveScope(bias);
        using var keepOutput = new KeepAliveScope(output);
        var x = input.ReadOnlyDataUnmarked.Span;
        var w = kernel.ReadOnlyDataUnmarked.Span;
        var b = bias.ReadOnlyDataUnmarked.Span;
        var y = output.DataUnmarked.Span;
        if (x.Overlaps(y) || w.Overlaps(y) || b.Overlaps(y)) return false;
        int width = input._shape[3], channels = input._shape[1], taps = kernel._shape[3], pad = padding[1];
        float* edge = stackalloc float[8];
        fixed (float* px = x, pw = w, py = y)
        {
            for (int position = 0; position < width; position += 8)
            {
                var accumulator = Vector256.Create(b[0]);
                for (int group = 0; group < channels; group += 8)
                    for (int tap = 0; tap < taps; tap++)
                        for (int channel = group; channel < group + 8; channel++)
                        {
                            int source = position + tap - pad;
                            Vector256<float> values;
                            if (source >= 0 && source + 8 <= width)
                                values = Avx.LoadVector256(px + channel * width + source);
                            else
                            {
                                for (int lane = 0; lane < 8; lane++)
                                {
                                    int index = source + lane;
                                    edge[lane] = index >= 0 && index < width ? px[channel * width + index] : 0f;
                                }
                                values = Avx.LoadVector256(edge);
                            }
                            accumulator = Fma.MultiplyAdd(values, Vector256.Create(pw[channel * taps + tap]), accumulator);
                        }
                if (position + 8 <= width) Avx.Store(py + position, accumulator);
                else
                {
                    Avx.Store(edge, accumulator);
                    for (int lane = 0; lane < width - position; lane++) py[position + lane] = edge[lane];
                }
            }
        }
        output.IncrementVersion();
        SingleOutputConvBiasRuns++;
        if (SingleOutputConvBiasRuns == 1 && Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
            AiDotNetEngine.LogDiagnostic("[FUSED-OUTPUT-BIAS] executed width=" + width + " channels=" + channels);
        return true;
#else
        return false;
#endif
    }
}
