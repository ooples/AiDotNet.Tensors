// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability: raw data straight into a float32 device tensor, behind <see cref="CpuEngine.TensorFromBytes{T}"/>
/// (torch.frombuffer + .to(float32)) and <see cref="CpuEngine.TensorParse{T}(System.ReadOnlySpan{byte}, byte, int[], System.Globalization.NumberStyles, System.IFormatProvider, T)"/>
/// (numeric text): the bytes upload once and every conversion runs on the device.
/// </summary>
/// <remarks>
/// Every backend runs <see cref="DataDecodeKernelTemplate"/>'s kernels (WGSL: WebGpu/WebGpuDataDecodeKernels) through
/// <see cref="PortableDataDecode"/>. Every buffer holds 32-bit integers: <c>words</c> is the byte data packed four per
/// word, little-endian; float outputs are written as their bits. A false return launches nothing and the caller runs
/// the host op.
/// </remarks>
internal interface IDataDecodeBackend
{
    /// <summary><c>output[i]</c> = element i of <paramref name="words"/> as float32, for i below
    /// <paramref name="count"/>; every element of output is written.</summary>
    bool TryDecodeBytes(IGpuBuffer words, IGpuBuffer output, int count, NumericStorageType sourceType, bool littleEndian);

    /// <summary><c>segmentCounts[s]</c> = separators (<paramref name="delimiter"/> or a line feed) in text bytes
    /// <c>[64 s, 64 s + 64)</c> below <paramref name="length"/>.</summary>
    bool TryCountSeparators(IGpuBuffer words, IGpuBuffer segmentCounts, int length, byte delimiter);

    /// <summary>Replaces <paramref name="values"/>[0, count) by its exclusive prefix sum and writes the sum of all of
    /// them to <c>total[0]</c>.</summary>
    bool TryExclusiveScan(IGpuBuffer values, IGpuBuffer total, int count);

    /// <summary>
    /// <c>bounds[2 f]</c>, <c>bounds[2 f + 1]</c> = the [start, end) byte range of field f, from each segment's first
    /// field index (the scanned <see cref="TryCountSeparators"/> counts). A carriage return before a line feed, or at
    /// the end of a text whose final newline the caller trimmed (<paramref name="trailingNewline"/>), is left out.
    /// </summary>
    bool TryScatterFieldBounds(IGpuBuffer words, IGpuBuffer segmentOffsets, IGpuBuffer bounds, int length, byte delimiter,
        bool trailingNewline);

    /// <summary>
    /// <c>output[f]</c> = field f parsed to float32 (an empty field: <paramref name="missingValue"/>), and
    /// <c>status[f]</c> = 0, or 1 for a field the kernel does not decide (its output is then missingValue and the host
    /// parses it). <paramref name="styleMask"/> holds the <see cref="DataDecodeKernelTemplate"/> style bits.
    /// </summary>
    bool TryParseFields(IGpuBuffer words, IGpuBuffer bounds, IGpuBuffer output, IGpuBuffer status, int fields,
        int styleMask, float missingValue);
}
