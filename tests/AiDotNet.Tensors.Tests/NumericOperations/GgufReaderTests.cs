using System.IO;
using System.Text;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests for issue #207 B1 — <see cref="GgufReader"/>. Builds synthetic
/// GGUF v3 byte streams and verifies structural + metadata parsing.
/// </summary>
public class GgufReaderTests
{
    [Fact]
    public void Read_BadMagic_Throws()
    {
        var bytes = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0 };
        Assert.Throws<InvalidDataException>(() => GgufReader.Read(new MemoryStream(bytes)));
    }

    [Fact]
    public void Read_UnsupportedVersion_Throws()
    {
        var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u); // magic
            w.Write(999u);        // bogus version
            w.Write((ulong)0);    // tensor count
            w.Write((ulong)0);    // meta count
        }
        ms.Position = 0;
        Assert.Throws<NotSupportedException>(() => GgufReader.Read(ms));
    }

    [Fact]
    public void Read_HeaderOnly_ParsesCountsCorrectly()
    {
        var ms = BuildMinimalGguf(
            version: 3,
            tensors: Array.Empty<(string, GgufType, long[])>(),
            metadata: new Dictionary<string, object>());
        var file = GgufReader.Read(ms);
        Assert.Equal(3, file.Version);
        Assert.Empty(file.Tensors);
        Assert.Empty(file.Metadata);
    }

    [Fact]
    public void Read_SingleTensor_CapturesNameTypeShape()
    {
        var ms = BuildMinimalGguf(
            version: 3,
            tensors: new[] { ("weights.0", GgufType.Q4_0, new long[] { 32, 128 }) },
            metadata: new Dictionary<string, object>());
        var file = GgufReader.Read(ms);
        Assert.Single(file.Tensors);
        var t = file.Tensors[0];
        Assert.Equal("weights.0", t.Name);
        Assert.Equal(GgufType.Q4_0, t.Type);
        Assert.Equal(new long[] { 32, 128 }, t.Dimensions);
    }

    [Fact]
    public void Read_Metadata_TypedValuesDecoded()
    {
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "llama",
            ["general.file_type"]    = (uint)4,
            ["layer.count"]          = 32,
            ["use_mmap"]             = true,
            ["dropout"]              = 0.1f,
        };
        var ms = BuildMinimalGguf(3, Array.Empty<(string, GgufType, long[])>(), meta);
        var file = GgufReader.Read(ms);
        Assert.Equal("llama",    file.Metadata["general.architecture"]);
        Assert.Equal((uint)4,    file.Metadata["general.file_type"]);
        Assert.Equal(32,         file.Metadata["layer.count"]);
        Assert.Equal(true,       file.Metadata["use_mmap"]);
        Assert.Equal(0.1f,       file.Metadata["dropout"]);
    }

    [Fact]
    public void Read_NullStream_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => GgufReader.Read(null!));
    }

    // ──────────── helpers ────────────

    /// <summary>
    /// Hand-assemble a GGUF v2/v3 byte stream. Metadata values encode
    /// the types this test expects (string, uint32, int32, bool, fp32).
    /// </summary>
    private static MemoryStream BuildMinimalGguf(
        int version,
        (string name, GgufType type, long[] dims)[] tensors,
        Dictionary<string, object> metadata)
    {
        var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);           // magic "GGUF"
            w.Write((uint)version);
            w.Write((ulong)tensors.Length);
            w.Write((ulong)metadata.Count);

            foreach (var kv in metadata)
            {
                WriteString(w, kv.Key);
                WriteValue(w, kv.Value);
            }
            foreach (var (name, type, dims) in tensors)
            {
                WriteString(w, name);
                w.Write((uint)dims.Length);
                foreach (var d in dims) w.Write((ulong)d);
                w.Write((uint)type);
                w.Write((ulong)0); // payload offset (mock — no body in this test)
            }
        }
        ms.Position = 0;
        return ms;
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void WriteValue(BinaryWriter w, object v)
    {
        switch (v)
        {
            case string s:      w.Write(8u); WriteString(w, s); break;
            case bool b:        w.Write(7u); w.Write((byte)(b ? 1 : 0)); break;
            case float f:       w.Write(6u); w.Write(f); break;
            case int i:         w.Write(5u); w.Write(i); break;
            case uint u:        w.Write(4u); w.Write(u); break;
            case long l:        w.Write(11u); w.Write(l); break;
            case ulong ul:      w.Write(10u); w.Write(ul); break;
            default: throw new NotSupportedException($"test helper: unsupported type {v.GetType()}");
        }
    }
}
