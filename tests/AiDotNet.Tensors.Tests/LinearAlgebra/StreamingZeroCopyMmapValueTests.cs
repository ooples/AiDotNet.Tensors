// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// MEASURES the value of lever 5 (zero-copy mmap residency) BEFORE committing to the
/// risky tensor-aliases-mmap change. Today a rehydrate, for a hot (page-cache-resident)
/// weight, allocates a fresh byte[] and copies the bytes into it (then again into the
/// tensor) before the GEMM reads them. Zero-copy would let the GEMM read directly from
/// the mmap'd page-cache pages — eliminating that alloc + copy. This quantifies exactly
/// that saving: "access via copy" (alloc + memcpy + touch) vs "access via mmap span"
/// (touch only), at realistic weight sizes, with the mmap mapped once (as the pool would).
/// If the copy is a large fraction of access time, zero-copy is worth the change; if the
/// downstream compute dominates, it isn't.
/// </summary>
public class StreamingZeroCopyMmapValueTests
{
    private readonly ITestOutputHelper _out;
    public StreamingZeroCopyMmapValueTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData(1 << 20)]   // 1 MiB weight
    [InlineData(8 << 20)]   // 8 MiB weight
    [InlineData(64 << 20)]  // 64 MiB weight
    public unsafe void ZeroCopyVsCopy_PerAccessSaving(int bytes)
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-zc-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "backing.bin");
        try
        {
            // Lay down a weight-sized blob and map it once (read-only), as the pool would.
            var blob = new byte[bytes];
            for (int i = 0; i < bytes; i += 64) blob[i] = (byte)(i & 0xFF);
            File.WriteAllBytes(path, blob);

            using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var view = mmf.CreateViewAccessor(0, bytes, MemoryMappedFileAccess.Read);
            byte* basePtr = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            try
            {
                // Warm the page cache (first touch pulls pages in).
                long warm = 0; for (int i = 0; i < bytes; i += 4096) warm += basePtr[i];
                Assert.True(warm >= 0);

                const int iters = 40;
                long sinkA = 0, sinkB = 0;

                // (A) CURRENT: rehydrate = alloc byte[] + copy from the mmap into it
                // (ReadFromBacking), then the GEMM reads it. Touch = read pass.
                double Copy()
                {
                    double best = double.MaxValue;
                    for (int r = 0; r < iters; r++)
                    {
                        var sw = Stopwatch.StartNew();
                        var buf = new byte[bytes];
                        new ReadOnlySpan<byte>(basePtr, bytes).CopyTo(buf); // the copy zero-copy removes
                        long s = 0; for (int i = 0; i < bytes; i += 64) s += buf[i]; // GEMM read
                        sw.Stop();
                        sinkA += s;
                        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                    }
                    return best;
                }

                // (B) ZERO-COPY: the GEMM reads straight from the mmap span — no alloc/copy.
                double ZeroCopy()
                {
                    double best = double.MaxValue;
                    for (int r = 0; r < iters; r++)
                    {
                        var sw = Stopwatch.StartNew();
                        var span = new ReadOnlySpan<byte>(basePtr, bytes);
                        long s = 0; for (int i = 0; i < bytes; i += 64) s += span[i]; // GEMM read
                        sw.Stop();
                        sinkB += s;
                        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                    }
                    return best;
                }

                Copy(); ZeroCopy(); // warm
                double copyMs = Copy();
                double zcMs = ZeroCopy();
                Assert.Equal(sinkA > 0, sinkB > 0);

                double savedMs = copyMs - zcMs;
                double savedPct = copyMs > 0 ? savedMs / copyMs * 100 : 0;
                _out.WriteLine($"weight {bytes / (1024.0 * 1024):F0} MiB  per-access: copy {copyMs:F3} ms, zero-copy {zcMs:F3} ms " +
                               $"→ saves {savedMs:F3} ms ({savedPct:F0}%)  [the alloc+memcpy zero-copy removes]");
            }
            finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }
}
