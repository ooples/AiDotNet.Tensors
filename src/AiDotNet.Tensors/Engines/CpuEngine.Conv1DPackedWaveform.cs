#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // The FP32 Goto microkernel computes six output channels and sixteen adjacent
    // waveform positions. Packing directly from NCHW removes the intermediate
    // im2col matrix, its second pack, and the full product panel.
    private const int WaveformMicroRows = 6;
    private const int WaveformMicroColumns = 16;
    private const int WaveformTileColumns = 128;

    private static unsafe bool TryConv1DPackedWaveform(float[] input, int inputOffset, float[] kernel,
        int kernelOffset, float[] output, int outputOffset, int batch, int channels, int width, int outChannels,
        int taps, int stride, int padding, int dilation, int outputWidth)
    {
        int reduction = checked(channels * taps);
        // One K panel preserves the existing FMA reduction without an extra
        // rounded partial sum. The long-waveform envelope amortizes filter packing.
        if (channels != outChannels || channels < 32 || channels > 128 || channels % 16 != 0
            || reduction > 512 || stride != 1 || outputWidth != width || width < 16384
            || width % WaveformTileColumns != 0 || padding > width / WaveformMicroColumns
            || 2L * padding != (long)(taps - 1) * dilation
            || !Avx2.IsSupported || !Fma.IsSupported || BlasManaged.BlasManaged.s_disableGotoGemm
            || (BlasProvider.IsDeterministicMode && BlasManaged.BlasManaged.s_disableDeterministicGoto)
            || !GotoGemmFp32.IsAvailable)
            return false;

        int fullRows = channels / WaveformMicroRows * WaveformMicroRows;
        int tailRows = channels - fullRows;
        int weightCount = checked((fullRows + (tailRows == 0 ? 0 : WaveformMicroRows)) * reduction);
        float[] weights = CpuStorageAllocationDiagnostics.RentWorkspace<float>(weightCount, "workspace.conv1d_waveform_packed_filter");
        try
        {
            int[] offsets = CpuStorageAllocationDiagnostics.RentWorkspace<int>(reduction * 2, "workspace.conv1d_waveform_offsets");
            try
            {
                PackWaveformFilter(kernel, kernelOffset, weights, offsets, channels, width, taps, padding, dilation, fullRows);
                int tilesPerImage = width / WaveformTileColumns;
                int tiles = checked(batch * tilesPerImage);
                int workers = Math.Min(tiles, Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism));
                int panelLength = reduction * WaveformTileColumns;
                int scratchLength = panelLength + (tailRows > 0 ? WaveformMicroRows * WaveformTileColumns : 0);
                int imageElements = checked(channels * width);
                fixed (float* source = input, filter = weights, destination = output)
                {
                    nint inputAddress = (nint)(source + inputOffset), weightAddress = (nint)filter;
                    nint outputAddress = (nint)(destination + outputOffset);
                    CpuParallelSettings.ParallelForOrSerial(0, workers, (long)channels * width * reduction * batch, worker =>
                    {
                        float[] scratch = CpuStorageAllocationDiagnostics.RentWorkspace<float>(scratchLength, "workspace.conv1d_waveform_microtiles");
                        try
                        {
                            fixed (float* panel = scratch)
                                for (int tile = worker; tile < tiles; tile += workers)
                                {
                                    int image = tile / tilesPerImage, first = (tile % tilesPerImage) * WaveformTileColumns;
                                    float* x = (float*)inputAddress + image * imageElements;
                                    float* y = (float*)outputAddress + image * imageElements;
                                    PackWaveformInput(x, panel, offsets, reduction, width, padding, first);
                                    // Both blocks consist of complete microtiles. The scalar-tail
                                    // operands are unused, so no dense input or product is needed.
                                    GotoGemmFp32.RunBlockPackedAB(null, 0, null, 0, y, width, 0, first,
                                        fullRows, WaveformTileColumns, reduction, reduction, (float*)weightAddress, panel, 0, 1);
                                    if (tailRows > 0)
                                    {
                                        float* tail = panel + panelLength;
                                        GotoGemmFp32.RunBlockPackedAB(null, 0, null, 0, tail, WaveformTileColumns, 0, 0,
                                            WaveformMicroRows, WaveformTileColumns, reduction, reduction,
                                            (float*)weightAddress + fullRows * reduction, panel, 0, 1);
                                        for (int row = 0; row < tailRows; row++)
                                            new ReadOnlySpan<float>(tail + row * WaveformTileColumns, WaveformTileColumns)
                                                .CopyTo(new Span<float>(y + (fullRows + row) * width + first, WaveformTileColumns));
                                    }
                                }
                        }
                        finally { ArrayPool<float>.Shared.Return(scratch); }
                    }, deterministicSafe: true);
                }
            }
            finally { ArrayPool<int>.Shared.Return(offsets); }
        }
        finally { ArrayPool<float>.Shared.Return(weights); }
        return true;
    }

    private static void PackWaveformFilter(float[] kernel, int kernelOffset, float[] weights, int[] offsets,
        int channels, int width, int taps, int padding, int dilation, int fullRows)
    {
        int reduction = channels * taps;
        int tailRows = channels - fullRows;
        // Preserve the original channel-block, tap, channel order. The second
        // half of offsets contains only the displacement within a source plane.
        for (int block = 0, k = 0; block < channels; block += 16)
            for (int tap = 0; tap < taps; tap++)
                for (int channel = block; channel < block + 16; channel++, k++)
                {
                    int shift = tap * dilation - padding;
                    offsets[k] = channel * width + shift;
                    offsets[reduction + k] = shift;
                    for (int row = 0; row < fullRows; row += WaveformMicroRows)
                        for (int lane = 0; lane < WaveformMicroRows; lane++)
                            weights[row * reduction + k * WaveformMicroRows + lane] =
                                kernel[kernelOffset + ((row + lane) * channels + channel) * taps + tap];
                    if (tailRows > 0)
                    {
                        int target = fullRows * reduction + k * WaveformMicroRows;
                        for (int lane = 0; lane < tailRows; lane++)
                            weights[target + lane] = kernel[kernelOffset + ((fullRows + lane) * channels + channel) * taps + tap];
                        for (int lane = tailRows; lane < WaveformMicroRows; lane++) weights[target + lane] = 0;
                    }
                }
    }

    private static unsafe void PackWaveformInput(float* input, float* panel, int[] offsets, int reduction,
        int width, int padding, int first)
    {
        for (int column = 0; column < WaveformTileColumns; column += WaveformMicroColumns)
        {
            int time = first + column;
            float* packed = panel + column * reduction;
            if (time >= padding && time + WaveformMicroColumns <= width - padding)
            {
                for (int k = 0; k < reduction; k++, packed += WaveformMicroColumns)
                {
                    float* source = input + offsets[k] + time;
                    Avx.Store(packed, Avx.LoadVector256(source));
                    Avx.Store(packed + 8, Avx.LoadVector256(source + 8));
                }
            }
            else
            {
                for (int k = 0; k < reduction; k++, packed += WaveformMicroColumns)
                {
                    int index = time + offsets[reduction + k];
                    float* source = input + offsets[k] + time;
                    for (int lane = 0; lane < WaveformMicroColumns; lane++)
                        packed[lane] = (uint)(index + lane) < (uint)width ? source[lane] : 0;
                }
            }
        }
    }
}
#endif
