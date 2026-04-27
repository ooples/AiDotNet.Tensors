// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Pickle;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.Pickle;

/// <summary>
/// PtReader is hard to test against real PyTorch output without
/// pulling Python into CI. Instead we hand-craft minimal pickle
/// streams that exercise the same opcodes torch.save emits — that
/// way the interpreter's coverage is verified without a Python
/// build dependency.
/// </summary>
[Collection("PersistenceGuard")]
public class PtReaderTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    [Fact]
    public void Reader_RejectsTruncatedHeader()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream(new byte[] { 0x80 });   // PROTO opcode but nothing after
        Assert.Throws<InvalidDataException>(() => PtReader.FromStream(ms));
    }

    [Fact]
    public void Reader_RejectsNonSeekableStream()
    {
        using var _ = IsolatedTrial();
        // Build a non-seekable wrapper around a small valid byte
        // sequence — the reader needs Seek to detect zip vs legacy.
        // Reader sniffs 4 bytes first, so the stream must hold ≥ 4
        // before the seekable-check fires.
        var data = new byte[] { 0x80, 0x02, 0x7D, 0x2E, 0x00 };  // PROTO 2 EMPTY_DICT STOP plus a pad byte
        using var ms = new MemoryStream(data);
        var nonSeekable = new NonSeekableStream(ms);
        Assert.Throws<InvalidOperationException>(() => PtReader.FromStream(nonSeekable));
    }

    [Fact]
    public void Reader_DetectsZipFormatMagic()
    {
        using var _ = IsolatedTrial();
        // Build a minimal zip with a data.pkl entry containing a
        // valid pickle stream that produces an empty dict.
        using var zipMs = new MemoryStream();
        using (var archive = new ZipArchive(zipMs, ZipArchiveMode.Create, leaveOpen: true))
        {
            var entry = archive.CreateEntry("model/data.pkl");
            using var es = entry.Open();
            // pickle: PROTO 2, EMPTY_DICT, STOP
            es.WriteByte(0x80); es.WriteByte(0x02);
            es.WriteByte(0x7D);  // EMPTY_DICT '}'
            es.WriteByte(0x2E);  // STOP '.'
        }
        zipMs.Position = 0;
        var reader = PtReader.FromStream(zipMs);
        Assert.Empty(reader.Tensors);
    }

    [Fact]
    public void EnforceBeforeLoad_FiresOnOpen()
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
            using var ms = new MemoryStream(new byte[] { 0x80, 0x02, 0x7D, 0x2E });
            Assert.Throws<LicenseRequiredException>(() => PtReader.FromStream(ms));
        }
        try { File.Delete(path); } catch { }
    }

    [Fact]
    public void PickleInterpreter_ParsesShortBinUnicode()
    {
        // PROTO 2, SHORT_BINUNICODE ('hello'), STOP — top of stack
        // should be the string "hello".
        var bytes = new byte[] { 0x80, 0x02, 0x8C, 0x05, (byte)'h', (byte)'e', (byte)'l', (byte)'l', (byte)'o', 0x2E };
        var interp = new PickleInterpreterTestProxy(bytes);
        Assert.Equal("hello", interp.Run());
    }

    [Fact]
    public void PickleInterpreter_ParsesEmptyDict()
    {
        // PROTO 2, EMPTY_DICT, STOP
        var bytes = new byte[] { 0x80, 0x02, 0x7D, 0x2E };
        var interp = new PickleInterpreterTestProxy(bytes);
        var r = interp.Run();
        Assert.IsType<System.Collections.Generic.Dictionary<object, object>>(r);
    }

    [Fact]
    public void PickleInterpreter_ParsesIntegers()
    {
        // PROTO 2, BININT1 (0x07), STOP
        var bytes = new byte[] { 0x80, 0x02, 0x4B, 0x07, 0x2E };
        var interp = new PickleInterpreterTestProxy(bytes);
        Assert.Equal(7L, interp.Run());
    }

    [Fact]
    public void PickleInterpreter_ParsesNone()
    {
        var bytes = new byte[] { 0x80, 0x02, 0x4E, 0x2E };  // PROTO 2, NONE, STOP
        var interp = new PickleInterpreterTestProxy(bytes);
        Assert.Null(interp.Run());
    }

    [Fact]
    public void PickleInterpreter_ParsesEmptyTuple()
    {
        var bytes = new byte[] { 0x80, 0x02, 0x29, 0x2E };  // EMPTY_TUPLE
        var interp = new PickleInterpreterTestProxy(bytes);
        var r = interp.Run();
        Assert.NotNull(r);
        Assert.IsType<object[]>(r);
        Assert.Empty((object[])r);
    }

    [Fact]
    public void PickleInterpreter_ParsesBoolean()
    {
        // NEWTRUE (0x88) and NEWFALSE (0x89)
        var trueBytes = new byte[] { 0x80, 0x02, 0x88, 0x2E };
        var falseBytes = new byte[] { 0x80, 0x02, 0x89, 0x2E };
        Assert.Equal(true, new PickleInterpreterTestProxy(trueBytes).Run());
        Assert.Equal(false, new PickleInterpreterTestProxy(falseBytes).Run());
    }

    /// <summary>
    /// Test-only wrapper that uses InternalsVisibleTo to instantiate
    /// the internal <see cref="PickleInterpreter"/>.
    /// </summary>
    private sealed class PickleInterpreterTestProxy
    {
        private readonly PickleInterpreter _interp;
        public PickleInterpreterTestProxy(byte[] bytes)
        {
            _interp = new PickleInterpreter(new MemoryStream(bytes));
        }
        public object Run() => _interp.Load();
    }

    /// <summary>Wraps a stream to deny seek — used to test the seekable-stream guard.</summary>
    private sealed class NonSeekableStream : Stream
    {
        private readonly Stream _inner;
        public NonSeekableStream(Stream inner) { _inner = inner; }
        public override bool CanRead => _inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => _inner.Position; set => throw new NotSupportedException(); }
        public override void Flush() => _inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
