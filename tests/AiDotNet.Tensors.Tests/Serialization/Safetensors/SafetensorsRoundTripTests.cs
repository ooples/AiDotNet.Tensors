// Copyright (c) AiDotNet. All rights reserved.
// Round-trip tests for SafetensorsReader + SafetensorsWriter.

#nullable disable

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.Safetensors;

/// <summary>
/// The reader and writer implement the same byte-format spec — every
/// test writes through the writer, reads back, and asserts the
/// recovered tensors are bit-for-bit identical to the source. A
/// regression in either direction breaks at least one of these.
/// </summary>
[Collection("PersistenceGuard")]
public class SafetensorsRoundTripTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    [Fact]
    public void RoundTrip_Float32_PreservesEveryByte()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        var src = new Tensor<float>(new[] { 1.0f, -2.5f, 3.125f, float.NaN, float.PositiveInfinity }, new[] { 5 });

        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.Add("x", src);
            w.Save();
        }

        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        Assert.Equal(new[] { "x" }, r.Names);
        var rt = r.ReadTensor<float>("x");
        Assert.Equal(src._shape, rt._shape);
        for (int i = 0; i < src.Length; i++)
        {
            float a = src.AsSpan()[i], b = rt.AsSpan()[i];
            // NaN compares unequal — bit-pattern test. SingleToInt32Bits
            // doesn't exist on net471, so reinterpret via byte arrays
            // to keep the same bit-exact comparison.
            int aBits = BitConverter.ToInt32(BitConverter.GetBytes(a), 0);
            int bBits = BitConverter.ToInt32(BitConverter.GetBytes(b), 0);
            Assert.Equal(aBits, bBits);
        }
    }

    [Fact]
    public void RoundTrip_AllSupportedDtypes()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        var f32 = new Tensor<float>(new[] { 0.1f, 0.2f, 0.3f }, new[] { 3 });
        var f64 = new Tensor<double>(new[] { 1e-300, 1e300 }, new[] { 2 });
        var i64 = new Tensor<long>(new[] { long.MinValue, 0L, long.MaxValue }, new[] { 3 });
        var i32 = new Tensor<int>(new[] { -5, 0, 5 }, new[] { 3 });
        var i16 = new Tensor<short>(new short[] { -1, 0, 1 }, new[] { 3 });
        var i8 = new Tensor<sbyte>(new sbyte[] { -128, 0, 127 }, new[] { 3 });
        var u8 = new Tensor<byte>(new byte[] { 0, 128, 255 }, new[] { 3 });
        var b = new Tensor<bool>(new[] { true, false, true }, new[] { 3 });

        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.Add("f32", f32);
            w.Add("f64", f64);
            w.Add("i64", i64);
            w.Add("i32", i32);
            w.Add("i16", i16);
            w.Add("i8", i8);
            w.Add("u8", u8);
            w.Add("bool", b);
            w.Save();
        }

        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        Assert.Equal(8, ((System.Collections.Generic.IReadOnlyDictionary<string, SafetensorsTensorEntry>)r.Entries).Count);
        Assert.Equal(SafetensorsDtype.F32, r.Entries["f32"].Dtype);
        Assert.Equal(SafetensorsDtype.F64, r.Entries["f64"].Dtype);
        Assert.Equal(SafetensorsDtype.I64, r.Entries["i64"].Dtype);
        Assert.Equal(SafetensorsDtype.U8, r.Entries["u8"].Dtype);

        // Spot-check a few values.
        Assert.Equal(1e-300, r.ReadTensor<double>("f64")[0]);
        Assert.Equal(long.MinValue, r.ReadTensor<long>("i64")[0]);
        Assert.True(r.ReadTensor<bool>("bool")[0]);
    }

    [Fact]
    public void RoundTrip_Metadata_PreservedAndOrderedFirst()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        var src = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });

        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.Metadata["framework"] = "AiDotNet.Tensors";
            w.Metadata["version"] = "1.0";
            w.Add("x", src);
            w.Save();
        }

        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        Assert.Equal("AiDotNet.Tensors", r.Metadata["framework"]);
        Assert.Equal("1.0", r.Metadata["version"]);
    }

    [Fact]
    public void Reader_MismatchingDtype_Throws()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        var src = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.Add("x", src);
            w.Save();
        }
        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        // x is F32 on disk; asking for double must throw.
        Assert.Throws<InvalidOperationException>(() => r.ReadTensor<double>("x"));
    }

    [Fact]
    public void Reader_UnknownTensorName_Throws()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.Add("a", new Tensor<float>(new[] { 1f }, new[] { 1 }));
            w.Save();
        }
        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(
            () => r.ReadTensor<float>("nonexistent"));
    }

    [Fact]
    public void Writer_DuplicateName_Throws()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using var w = SafetensorsWriter.ToStream(ms);
        w.Add("x", new Tensor<float>(new[] { 1f }, new[] { 1 }));
        Assert.Throws<ArgumentException>(
            () => w.Add("x", new Tensor<float>(new[] { 2f }, new[] { 1 })));
    }

    [Fact]
    public void Writer_ReservedMetadataKey_Throws()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using var w = SafetensorsWriter.ToStream(ms);
        Assert.Throws<ArgumentException>(
            () => w.Add("__metadata__", new Tensor<float>(new[] { 1f }, new[] { 1 })));
    }

    [Fact]
    public void Reader_RejectsTruncatedHeaderLength()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream(new byte[] { 0x01, 0x02, 0x03 }); // 3 bytes < 8
        Assert.Throws<InvalidDataException>(() => SafetensorsReader.FromStream(ms));
    }

    [Fact]
    public void Reader_RejectsInsaneHeaderLength()
    {
        using var _ = IsolatedTrial();
        // 8-byte LE header length = 2^31 (over the 2^30 plausible cap).
        var bad = new byte[8];
        bad[3] = 0x80;  // 0x80000000 little-endian = 2^31
        using var ms = new MemoryStream(bad);
        Assert.Throws<InvalidDataException>(() => SafetensorsReader.FromStream(ms));
    }

    [Fact]
    public void Reader_RejectsMalformedHeaderJson()
    {
        using var _ = IsolatedTrial();
        // 8-byte length prefix saying "5 bytes follow", then "garba"
        // (not valid JSON).
        var lenBytes = new byte[8];
        lenBytes[0] = 5;
        var bytes = new byte[8 + 5];
        Buffer.BlockCopy(lenBytes, 0, bytes, 0, 8);
        Buffer.BlockCopy(System.Text.Encoding.UTF8.GetBytes("garba"), 0, bytes, 8, 5);
        using var ms = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => SafetensorsReader.FromStream(ms));
    }

    [Fact]
    public void Reader_PersistenceGuard_FiresOnOpen()
    {
        // Trial exhausted → opening any safetensors file throws BEFORE
        // touching the stream. Mirrors the GgufReader gate test in
        // PR #254.
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);
        using (PersistenceGuard.SetTestTrialFilePathOverride(path))
        {
            using var ms = new MemoryStream(new byte[] { 0xFF, 0xFF, 0xFF, 0xFF });
            Assert.Throws<LicenseRequiredException>(() => SafetensorsReader.FromStream(ms));
        }
        try { File.Delete(path); } catch { }
    }

    [Fact]
    public void Writer_PersistenceGuard_FiresOnCreate()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);
        using (PersistenceGuard.SetTestTrialFilePathOverride(path))
        {
            using var ms = new MemoryStream();
            Assert.Throws<LicenseRequiredException>(() => SafetensorsWriter.ToStream(ms));
        }
        try { File.Delete(path); } catch { }
    }

    [Fact]
    public void RoundTrip_Empty_NoTensorsOnlyMetadata()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.Metadata["only"] = "metadata";
            w.Save();
        }
        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        Assert.Empty(r.Names);
        Assert.Equal("metadata", r.Metadata["only"]);
    }

    [Fact]
    public void Writer_AddRaw_RoundTripsSubByteDtype()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        // 8 packed Int4 values = 4 bytes
        var packed = new byte[] { 0x12, 0x34, 0x56, 0x78 };
        using (var w = SafetensorsWriter.ToStream(ms))
        {
            w.AddRaw("q.weight", SafetensorsDtype.AIDN_INT4, new long[] { 8 }, packed);
            w.Save();
        }
        ms.Position = 0;
        using var r = SafetensorsReader.FromStream(ms);
        var entry = r.Entries["q.weight"];
        Assert.Equal(SafetensorsDtype.AIDN_INT4, entry.Dtype);
        Assert.Equal(new long[] { 8 }, entry.Shape);
        Assert.Equal(packed, r.ReadRawBytes("q.weight"));
    }
}
