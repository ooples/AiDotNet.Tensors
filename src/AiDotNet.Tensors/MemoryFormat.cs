namespace AiDotNet.Tensors;

/// <summary>
/// Specifies the memory layout format for tensors.
///
/// Memory format affects performance of convolution, pooling, and normalization operations.
/// NHWC (channels-last) is faster on modern CPUs with AVX2/AVX-512 because it enables
/// better vectorization of per-channel operations.
///
/// Usage:
///   var input = tensor.ToFormat(MemoryFormat.ChannelsLast);
///   var output = engine.Conv2D(input, kernel); // uses NHWC path
///
/// PyTorch equivalent: torch.channels_last
/// </summary>
public enum MemoryFormat
{
    /// <summary>
    /// NCHW: batch × channels × height × width (default).
    /// Standard layout used by most deep learning frameworks.
    /// Each channel's spatial data is contiguous in memory.
    /// </summary>
    Contiguous = 0,

    /// <summary>
    /// NHWC: batch × height × width × channels (channels-last).
    /// Faster for CPU convolution with AVX2/AVX-512 because all channels
    /// at a spatial position are contiguous, enabling SIMD vectorization
    /// over channel dimension.
    /// </summary>
    ChannelsLast = 1,

    /// <summary>
    /// NDHWC: batch × depth × height × width × channels (3D channels-last).
    /// Extension of ChannelsLast for 3D convolutions.
    /// </summary>
    ChannelsLast3D = 2
}
