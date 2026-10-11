using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Memory layout for tensor data. Determines how multi-dimensional data is stored contiguously.
/// </summary>
public enum MemoryLayout
{
    /// <summary>
    /// Channels-first layout: [batch, channels, height, width].
    /// Default for PyTorch. Each channel is stored contiguously.
    /// Better for depthwise convolutions and per-channel operations.
    /// </summary>
    NCHW,

    /// <summary>
    /// Channels-last layout: [batch, height, width, channels].
    /// Default for TensorFlow. Spatial pixels are stored contiguously with all channels.
    /// Better for standard convolutions (contiguous channel access enables SIMD vectorization).
    /// </summary>
    NHWC
}

/// <summary>
/// Utilities for converting between NCHW and NHWC tensor layouts.
/// </summary>
public static class TensorLayout
{
    /// <summary>
    /// Converts a 4D tensor from NCHW to NHWC layout.
    /// [batch, channels, height, width] -> [batch, height, width, channels]
    /// </summary>
    public static Tensor<T> ToNHWC<T>(Tensor<T> nchw)
    {
        if (nchw.Rank != 4)
            throw new ArgumentException($"Expected 4D tensor [N,C,H,W] but got rank {nchw.Rank}.");

        int n = nchw._shape[0];
        int c = nchw._shape[1];
        int h = nchw._shape[2];
        int w = nchw._shape[3];

        var nhwc = new Tensor<T>(new[] { n, h, w, c });
        var src = (nchw.IsContiguous ? nchw : nchw.Contiguous()).AsSpan();
        var dst = nhwc.AsWritableSpan();

        for (int batch = 0; batch < n; batch++)
        {
            int srcBatchOffset = batch * c * h * w;
            int dstBatchOffset = batch * h * w * c;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dstPixel = dstBatchOffset + (y * w + x) * c;
                    for (int ch = 0; ch < c; ch++)
                    {
                        int srcIdx = srcBatchOffset + ch * h * w + y * w + x;
                        dst[dstPixel + ch] = src[srcIdx];
                    }
                }
            }
        }

        return nhwc;
    }

    /// <summary>
    /// Converts a 4D tensor from NHWC to NCHW layout.
    /// [batch, height, width, channels] -> [batch, channels, height, width]
    /// </summary>
    public static Tensor<T> ToNCHW<T>(Tensor<T> nhwc)
    {
        if (nhwc.Rank != 4)
            throw new ArgumentException($"Expected 4D tensor [N,H,W,C] but got rank {nhwc.Rank}.");

        int n = nhwc._shape[0];
        int h = nhwc._shape[1];
        int w = nhwc._shape[2];
        int c = nhwc._shape[3];

        var nchw = new Tensor<T>(new[] { n, c, h, w });
        var src = (nhwc.IsContiguous ? nhwc : nhwc.Contiguous()).AsSpan();
        var dst = nchw.AsWritableSpan();

        for (int batch = 0; batch < n; batch++)
        {
            int srcBatchOffset = batch * h * w * c;
            int dstBatchOffset = batch * c * h * w;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int srcPixel = srcBatchOffset + (y * w + x) * c;
                    for (int ch = 0; ch < c; ch++)
                    {
                        int dstIdx = dstBatchOffset + ch * h * w + y * w + x;
                        dst[dstIdx] = src[srcPixel + ch];
                    }
                }
            }
        }

        return nchw;
    }

    /// <summary>
    /// Converts a 4D tensor from NCHW to NHWC layout directly into a pre-allocated destination.
    /// Zero allocation.
    /// </summary>
    public static void ToNHWCInto<T>(Tensor<T> destination, Tensor<T> nchw)
    {
        if (nchw.Rank != 4)
            throw new ArgumentException($"Expected 4D tensor [N,C,H,W] but got rank {nchw.Rank}.");
        if (destination.Rank != 4)
            throw new ArgumentException($"Expected 4D destination [N,H,W,C] but got rank {destination.Rank}.");
        if (destination.Length != nchw.Length)
            throw new ArgumentException($"Destination length ({destination.Length}) must match source length ({nchw.Length}).");
        // Validate destination shape matches expected NHWC layout
        int n = nchw._shape[0], c = nchw._shape[1], h = nchw._shape[2], w = nchw._shape[3];
        if (destination._shape[0] != n || destination._shape[1] != h || destination._shape[2] != w || destination._shape[3] != c)
            throw new ArgumentException($"Destination shape [{string.Join(",", destination._shape)}] must be [N={n},H={h},W={w},C={c}].");

        var src = (nchw.IsContiguous ? nchw : nchw.Contiguous()).AsSpan();
        var dst = destination.AsWritableSpan();

        for (int batch = 0; batch < n; batch++)
        {
            int srcBatchOffset = batch * c * h * w;
            int dstBatchOffset = batch * h * w * c;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dstPixel = dstBatchOffset + (y * w + x) * c;
                    for (int ch = 0; ch < c; ch++)
                    {
                        dst[dstPixel + ch] = src[srcBatchOffset + ch * h * w + y * w + x];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Converts a 4D tensor from NHWC to NCHW layout directly into a pre-allocated destination.
    /// Zero allocation.
    /// </summary>
    public static void ToNCHWInto<T>(Tensor<T> destination, Tensor<T> nhwc)
    {
        if (nhwc.Rank != 4)
            throw new ArgumentException($"Expected 4D tensor [N,H,W,C] but got rank {nhwc.Rank}.");
        if (destination.Rank != 4)
            throw new ArgumentException($"Expected 4D destination [N,C,H,W] but got rank {destination.Rank}.");
        if (destination.Length != nhwc.Length)
            throw new ArgumentException($"Destination length ({destination.Length}) must match source length ({nhwc.Length}).");

        int n = nhwc._shape[0];
        int h = nhwc._shape[1];
        int w = nhwc._shape[2];
        int c = nhwc._shape[3];

        var src = (nhwc.IsContiguous ? nhwc : nhwc.Contiguous()).AsSpan();
        var dst = destination.AsWritableSpan();

        for (int batch = 0; batch < n; batch++)
        {
            int srcBatchOffset = batch * h * w * c;
            int dstBatchOffset = batch * c * h * w;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int srcPixel = srcBatchOffset + (y * w + x) * c;
                    for (int ch = 0; ch < c; ch++)
                    {
                        dst[dstBatchOffset + ch * h * w + y * w + x] = src[srcPixel + ch];
                    }
                }
            }
        }
    }
}
