using System.IO;
using System.Text;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Minimal GGUF-format reader — parses the v3 header, metadata key-
/// value pairs, and tensor descriptors from a stream. Payload bytes
/// for each tensor are exposed as byte-ranges the caller can route
/// through <see cref="QuantizationHelpers.DequantizeInt4"/> etc. based
/// on the reported <see cref="GgufTensorInfo.Type"/>.
///
/// <para>GGUF is the ggml/llama.cpp container for quantized LLMs —
/// magic <c>0x46554747</c> ("GGUF" LE), a u32 version, u64 counts, and
/// a stream of metadata + tensor descriptors. Format reference:
/// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md.</para>
///
/// <para><b>Scope:</b> this reader parses the structure and exposes
/// each tensor's type + shape + byte-offset so a consumer can map the
/// payload into the right packed type. Type-specific dequantization
/// (Q4_0 / Q4_K / Q8_0 / F16 / ...) is outside this file — it belongs
/// wherever the downstream pipeline lives. The Q4_0 block layout
/// matches our <see cref="PackedInt4"/> so consumers wiring the
/// straightforward <c>Q4_0</c> case can copy byte-for-byte.</para>
/// </summary>
public static class GgufReader
{
    private const uint GgufMagic = 0x46554747u; // "GGUF" little-endian

    /// <summary>Supported GGUF versions. v2 + v3 use the same on-disk
    /// structure for header + metadata + tensor info that this reader
    /// cares about.</summary>
    public static readonly int[] SupportedVersions = { 2, 3 };

    /// <summary>Parse the full header, metadata, and tensor index from
    /// <paramref name="stream"/>. Leaves the stream positioned at the
    /// first byte of tensor data.</summary>
    public static GgufFile Read(Stream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(stream));

        using var br = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        uint magic = br.ReadUInt32();
        if (magic != GgufMagic)
            throw new InvalidDataException(
                $"Not a GGUF file: magic 0x{magic:X8} (expected 0x{GgufMagic:X8}).");

        uint version = br.ReadUInt32();
        if (Array.IndexOf(SupportedVersions, (int)version) < 0)
            throw new NotSupportedException(
                $"GGUF version {version} is not supported. Supported: {string.Join(", ", SupportedVersions)}.");

        ulong tensorCount = br.ReadUInt64();
        ulong metadataCount = br.ReadUInt64();

        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        for (ulong i = 0; i < metadataCount; i++)
        {
            string key = ReadString(br);
            object value = ReadValue(br);
            metadata[key] = value;
        }

        var tensors = new List<GgufTensorInfo>((int)Math.Min(tensorCount, int.MaxValue));
        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString(br);
            uint nDims = br.ReadUInt32();
            if (nDims > 8)
                throw new InvalidDataException($"Tensor {name} has implausible dim count {nDims}.");
            var dims = new long[nDims];
            for (int d = 0; d < nDims; d++) dims[d] = (long)br.ReadUInt64();
            uint typeCode = br.ReadUInt32();
            ulong offset = br.ReadUInt64();
            tensors.Add(new GgufTensorInfo(name, (GgufType)typeCode, dims, offset));
        }

        return new GgufFile((int)version, metadata, tensors);
    }

    private static string ReadString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        if (len > int.MaxValue)
            throw new InvalidDataException($"GGUF string length {len} exceeds 2 GB.");
        var bytes = br.ReadBytes((int)len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object ReadValue(BinaryReader br)
    {
        // GGUF metadata values are tagged with a u32 type code then the
        // payload. Types: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32,
        // 5=int32, 6=float32, 7=bool, 8=string, 9=array, 10=uint64,
        // 11=int64, 12=float64.
        uint type = br.ReadUInt32();
        return type switch
        {
            0 => (object)br.ReadByte(),
            1 => br.ReadSByte(),
            2 => br.ReadUInt16(),
            3 => br.ReadInt16(),
            4 => br.ReadUInt32(),
            5 => br.ReadInt32(),
            6 => br.ReadSingle(),
            7 => br.ReadByte() != 0,
            8 => ReadString(br),
            9 => ReadArray(br),
            10 => br.ReadUInt64(),
            11 => br.ReadInt64(),
            12 => br.ReadDouble(),
            _ => throw new InvalidDataException($"Unknown GGUF metadata type code {type}."),
        };
    }

    private static object[] ReadArray(BinaryReader br)
    {
        uint elemType = br.ReadUInt32();
        ulong count = br.ReadUInt64();
        if (count > (ulong)int.MaxValue)
            throw new InvalidDataException($"GGUF array length {count} exceeds int32.");
        var arr = new object[(int)count];
        // Mirror ReadValue's switch for consistency, but type is fixed
        // per-array so we avoid per-element branch misprediction.
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = elemType switch
            {
                0 => (object)br.ReadByte(),
                1 => br.ReadSByte(),
                2 => br.ReadUInt16(),
                3 => br.ReadInt16(),
                4 => br.ReadUInt32(),
                5 => br.ReadInt32(),
                6 => br.ReadSingle(),
                7 => br.ReadByte() != 0,
                8 => ReadString(br),
                // Type 9 = nested array. The spec is recursive — array
                // elements may themselves be arrays of any valid type —
                // so a container-level file that embeds e.g. a list of
                // token-merge pairs reaches this branch. Omitting it
                // would throw InvalidDataException on valid GGUF input.
                9 => ReadArray(br),
                10 => br.ReadUInt64(),
                11 => br.ReadInt64(),
                12 => br.ReadDouble(),
                _ => throw new InvalidDataException($"Unknown GGUF array element type {elemType}."),
            };
        }
        return arr;
    }
}

/// <summary>Header + metadata + tensor index parsed from a GGUF file.</summary>
public sealed class GgufFile
{
    public int Version { get; }
    public IReadOnlyDictionary<string, object> Metadata { get; }
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    public GgufFile(int version, Dictionary<string, object> metadata, List<GgufTensorInfo> tensors)
    {
        Version = version;
        Metadata = metadata;
        Tensors = tensors;
    }
}

/// <summary>
/// Descriptor for a single GGUF tensor — no payload bytes, just the
/// shape + dtype + stream-offset so the caller can mmap or read the
/// body on demand.
/// </summary>
public sealed class GgufTensorInfo
{
    public string Name { get; }
    public GgufType Type { get; }
    public long[] Dimensions { get; }
    public ulong PayloadOffset { get; }

    public GgufTensorInfo(string name, GgufType type, long[] dimensions, ulong payloadOffset)
    {
        Name = name;
        Type = type;
        Dimensions = dimensions;
        PayloadOffset = payloadOffset;
    }
}

/// <summary>
/// GGUF tensor type codes. Matches the <c>ggml_type</c> enum from
/// upstream ggml; only the commonly-used codes are enumerated. Values
/// we don't enumerate remain legal to read as raw bytes.
/// </summary>
public enum GgufType : uint
{
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
}
