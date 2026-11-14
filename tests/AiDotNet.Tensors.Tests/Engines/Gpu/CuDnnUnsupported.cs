#if !NET462
using System;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Recognises a cuDNN "not supported" failure in the message <c>CuDnnContext.CheckStatus</c> throws. That message
/// carries <c>cudnnGetErrorString</c>'s text when the library provides it (cuDNN 9: <c>CUDNN_STATUS_NOT_SUPPORTED</c>,
/// <c>CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH</c>) and the status enum's name otherwise (<c>NotSupported</c>,
/// <c>ArchMismatch</c>). Matching only the enum names missed cuDNN 9's text, so a BF16 convolution on an sm_75 GPU
/// failed instead of skipping.
/// </summary>
internal static class CuDnnUnsupported
{
    internal static bool Is(Exception? ex) =>
        ex is InvalidOperationException
        && (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
            || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)
            || ex.Message.Contains("NOT_SUPPORTED", StringComparison.Ordinal)
            || ex.Message.Contains("ARCH_MISMATCH", StringComparison.Ordinal));
}
#endif
