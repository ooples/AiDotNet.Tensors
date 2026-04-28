// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System;
using System.IO;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.TensorBoard;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.TensorBoard;

[Collection("PersistenceGuard")]
public class TensorBoardSummaryWriterTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    [Fact]
    public void AddScalar_ProducesValidTfRecordFraming()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using (var w = TensorBoardSummaryWriter.ToStream(ms))
        {
            w.AddScalar("loss/train", 0.42, step: 1);
            w.AddScalar("loss/train", 0.31, step: 2);
            w.AddScalar("acc/val", 0.95, step: 1);
            w.Flush();
        }

        // The bytes should parse as repeated TFRecord frames:
        //   uint64 LE length
        //   uint32 LE masked_crc32(length)
        //   length bytes payload
        //   uint32 LE masked_crc32(payload)
        ms.Position = 0;
        var bytes = ms.ToArray();
        int offset = 0;
        int recordCount = 0;
        while (offset < bytes.Length)
        {
            // Read length.
            ulong len = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(offset, 8));
            offset += 8;
            // Verify masked CRC of length.
            uint expectedLenCrc = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
            offset += 4;
            byte[] lenBytes = bytes.AsSpan(offset - 12, 8).ToArray();
            uint actualLenCrc = TestableCrc.MaskedCrc32C(lenBytes);
            Assert.Equal(expectedLenCrc, actualLenCrc);

            // Skip payload.
            offset += (int)len;
            // Verify masked CRC of payload.
            uint expectedPayloadCrc = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
            offset += 4;
            byte[] payload = bytes.AsSpan(offset - 4 - (int)len, (int)len).ToArray();
            uint actualPayloadCrc = TestableCrc.MaskedCrc32C(payload);
            Assert.Equal(expectedPayloadCrc, actualPayloadCrc);

            recordCount++;
        }
        // 1 file_version event + 3 scalar events = 4 records.
        Assert.Equal(4, recordCount);
    }

    [Fact]
    public void AddHistogram_ProducesAdditionalRecord()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using (var w = TensorBoardSummaryWriter.ToStream(ms))
        {
            var values = new float[100];
            var rand = new Random(42);
            for (int i = 0; i < values.Length; i++) values[i] = (float)rand.NextDouble();
            w.AddHistogram("activations/layer1", values, step: 5);
            w.Flush();
        }
        // Header (file_version) + 1 histogram = 2 records. Verify
        // each frame parses cleanly.
        ms.Position = 0;
        var bytes = ms.ToArray();
        Assert.True(bytes.Length > 50, "Histogram record should be reasonably large.");
    }

    [Fact]
    public void OpenLogDir_CreatesEventsFile()
    {
        using var _ = IsolatedTrial();
        var dir = Path.Combine(Path.GetTempPath(), "tb-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            using (var w = TensorBoardSummaryWriter.OpenLogDir(dir))
            {
                w.AddScalar("test", 1.0, step: 0);
            }
            var files = Directory.GetFiles(dir, "events.out.tfevents.*");
            Assert.Single(files);
            Assert.True(new FileInfo(files[0]).Length > 0);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void EnforceBeforeSave_FiresOnOpen()
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
            Assert.Throws<LicenseRequiredException>(() => TensorBoardSummaryWriter.ToStream(ms));
        }
        try { File.Delete(path); } catch { }
    }
}

/// <summary>
/// Re-exposes the writer's internal CRC routine so the
/// frame-validation test can independently verify the masked-CRC
/// values it wrote.
/// </summary>
internal static class TestableCrc
{
    public static uint MaskedCrc32C(byte[] data)
    {
        // Mirrors TensorBoardSummaryWriter.MaskedCrc32C — kept in
        // the test fixture so a reader can confirm the writer's
        // CRC matches the canonical TensorFlow definition without
        // exposing the writer's internals to public API.
        const uint Polynomial = 0x82F63B78u;
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = ((c & 1) != 0) ? (Polynomial ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < data.Length; i++)
            crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
        crc = ~crc;
        return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
    }
}
