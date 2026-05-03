// Copyright (c) AiDotNet. All rights reserved.
// Issue #285 — chunker integration tests using DispatchProxy mock.
// Gated to non-Framework targets: DispatchProxy (.NET Core 5+) isn't
// on .NET Framework, and the mock backend depends on it.

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("BufferSizeCapTests")] // shares isolation with the cap tests
public class ChunkerWithMockBackendTests
{
    // ───────────────────────────────────────────────────────────────────
    // TryRunUnaryChunked
    // ───────────────────────────────────────────────────────────────────

    [Fact]
    public void TryRunUnaryChunked_SplitsAcrossCap_ProducesCorrectResult()
    {
        // Cap = 16 bytes (4 floats per chunk). Tensor of 12 elements →
        // 3 chunks. Op: y = x * 2. Verify the chunker dispatches 3 times
        // and the concatenated output equals the elementwise op applied
        // to the whole input.
        var state = new MockBackendState { MaxBufferAllocBytes = 16 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine(); // backend not used for chunker — we pass mock directly

        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new[] { 12 });
        var ex = new GpuBufferTooLargeException("Mock", requestedBytes: 48, deviceMaxAllocBytes: 16, deviceName: "MockDevice");

        // Op: y[i] = x[i] * 2. The mock's MockGpuBuffer wraps the host
        // float[] directly, so we read in/write out via Data.
        var result = engine.TryRunUnaryChunked<float>(backend, input,
            (b, bIn, bOut, len) =>
            {
                state.UnaryOpCalls++;
                var bufferIn = (MockGpuBuffer)bIn;
                var bufferOut = (MockGpuBuffer)bOut;
                for (int i = 0; i < len; i++) bufferOut.Data[i] = bufferIn.Data[i] * 2;
            },
            ex,
            out bool emitted);

        Assert.NotNull(result);
        Assert.Equal(12, result!.Length);
        for (int i = 0; i < 12; i++)
            Assert.Equal((i + 1) * 2f, result[i]);
        Assert.Equal(3, state.UnaryOpCalls); // 12 elements / 4 per chunk
        Assert.True(emitted, "chunker should emit a chunked_N event when it dispatches");
    }

    [Fact]
    public void TryRunUnaryChunked_ChunkCountExceedsMax_ReturnsNullAndEmits()
    {
        // Cap = 4 bytes (1 element per chunk), 100 elements → 100 chunks.
        // EffectiveMaxChunkCount = 4 → chunker bails to CPU.
        var state = new MockBackendState { MaxBufferAllocBytes = 4 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var input = new Tensor<float>(new float[100], new[] { 100 });
        var ex = new GpuBufferTooLargeException("Mock", 400, 4, "MockDevice");

        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = new GpuFallbackOptions { MaxChunkCount = 4 };

            var result = engine.TryRunUnaryChunked<float>(backend, input,
                (b, bIn, bOut, len) => state.UnaryOpCalls++,
                ex,
                out bool emitted);

            Assert.Null(result);                        // CPU fallback
            Assert.Equal(0, state.UnaryOpCalls);        // op never dispatched
            Assert.True(emitted, "chunker emits a cpu_fallback_chunks_N_exceeds_M event before bailing");
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    [Fact]
    public void TryRunUnaryChunked_ScalarTensor_ReturnsNullWithoutEmitting()
    {
        // 1-element tensors can't be chunked — bail to CPU silently.
        var state = new MockBackendState { MaxBufferAllocBytes = 4 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var input = new Tensor<float>(new float[] { 42 }, new[] { 1 });
        var ex = new GpuBufferTooLargeException("Mock", 4, 4, "MockDevice");

        var result = engine.TryRunUnaryChunked<float>(backend, input,
            (b, bIn, bOut, len) => state.UnaryOpCalls++,
            ex,
            out bool emitted);

        Assert.Null(result);
        Assert.False(emitted, "scalar input bails before any event is recorded");
    }

    [Fact]
    public void TryRunUnaryChunked_CapBelowSingleFloat_ReturnsNullWithoutEmitting()
    {
        // Cap < sizeof(float) → can't even fit a single element. Don't
        // retry chunking; bail to CPU silently.
        var state = new MockBackendState { MaxBufferAllocBytes = 3 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var ex = new GpuBufferTooLargeException("Mock", 16, 3, "MockDevice");

        var result = engine.TryRunUnaryChunked<float>(backend, input,
            (b, bIn, bOut, len) => state.UnaryOpCalls++,
            ex,
            out bool emitted);

        Assert.Null(result);
        Assert.False(emitted);
    }

    [Fact]
    public void TryRunUnaryChunked_CapZero_ReturnsNullWithoutEmitting()
    {
        // capBytes <= 0 means the device cap is unknown; chunker can't
        // make a sizing decision and bails.
        var state = new MockBackendState { MaxBufferAllocBytes = 0 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var ex = new GpuBufferTooLargeException("Mock", 16, 0, "MockDevice");

        var result = engine.TryRunUnaryChunked<float>(backend, input,
            (b, bIn, bOut, len) => state.UnaryOpCalls++,
            ex,
            out bool emitted);

        Assert.Null(result);
        Assert.False(emitted);
    }

    // ───────────────────────────────────────────────────────────────────
    // TryRunBinaryChunked
    // ───────────────────────────────────────────────────────────────────

    [Fact]
    public void TryRunBinaryChunked_SplitsAcrossCap_ProducesCorrectResult()
    {
        // Cap = 16 bytes (4 floats per chunk). Two 8-element tensors →
        // 2 chunks. Op: c = a + b.
        var state = new MockBackendState { MaxBufferAllocBytes = 16 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var left = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 8 });
        var right = new Tensor<float>(new float[] { 10, 20, 30, 40, 50, 60, 70, 80 }, new[] { 8 });
        var ex = new GpuBufferTooLargeException("Mock", 32, 16, "MockDevice");

        var result = engine.TryRunBinaryChunked<float>(backend, left, right,
            (b, bA, bB, bC, len) =>
            {
                state.BinaryOpCalls++;
                var ba = (MockGpuBuffer)bA;
                var bb = (MockGpuBuffer)bB;
                var bc = (MockGpuBuffer)bC;
                for (int i = 0; i < len; i++) bc.Data[i] = ba.Data[i] + bb.Data[i];
            },
            ex,
            out bool emitted);

        Assert.NotNull(result);
        Assert.Equal(8, result!.Length);
        for (int i = 0; i < 8; i++)
            Assert.Equal(left.GetFlat(i) + right.GetFlat(i), result[i]);
        Assert.Equal(2, state.BinaryOpCalls);
        Assert.True(emitted);
    }

    [Fact]
    public void TryRunBinaryChunked_ChunkCountExceedsMax_ReturnsNullAndEmits()
    {
        var state = new MockBackendState { MaxBufferAllocBytes = 4 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var left = new Tensor<float>(new float[100], new[] { 100 });
        var right = new Tensor<float>(new float[100], new[] { 100 });
        var ex = new GpuBufferTooLargeException("Mock", 400, 4, "MockDevice");

        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = new GpuFallbackOptions { MaxChunkCount = 4 };

            var result = engine.TryRunBinaryChunked<float>(backend, left, right,
                (b, bA, bB, bC, len) => state.BinaryOpCalls++,
                ex,
                out bool emitted);

            Assert.Null(result);
            Assert.Equal(0, state.BinaryOpCalls);
            Assert.True(emitted);
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    [Fact]
    public void TryRunUnaryChunked_OpThrows_FallsThroughToCpu()
    {
        // Per PR #288 review: a per-chunk dispatch failure (kernel error,
        // download error, OOM-during-loop, etc.) should fall through to
        // CPU rather than propagate and crash the training step. The op
        // delegate throws on the second chunk; the chunker catches and
        // returns null + emits a cpu_fallback_chunk_dispatch_error event.
        var state = new MockBackendState { MaxBufferAllocBytes = 16 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 8 });
        var ex = new GpuBufferTooLargeException("Mock", 32, 16, "MockDevice");

        var result = engine.TryRunUnaryChunked<float>(backend, input,
            (b, bIn, bOut, len) =>
            {
                state.UnaryOpCalls++;
                if (state.UnaryOpCalls == 2)
                    throw new InvalidOperationException("simulated kernel dispatch error");
            },
            ex,
            out bool emitted);

        Assert.Null(result);
        Assert.Equal(2, state.UnaryOpCalls); // first chunk ran, second threw
        Assert.True(emitted, "chunker emits an event before+after the failure");
    }

    [Fact]
    public void TryRunBinaryChunked_OpThrows_FallsThroughToCpu()
    {
        var state = new MockBackendState { MaxBufferAllocBytes = 16 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var left = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 8 });
        var right = new Tensor<float>(new float[] { 10, 20, 30, 40, 50, 60, 70, 80 }, new[] { 8 });
        var ex = new GpuBufferTooLargeException("Mock", 32, 16, "MockDevice");

        var result = engine.TryRunBinaryChunked<float>(backend, left, right,
            (b, bA, bB, bC, len) =>
            {
                state.BinaryOpCalls++;
                if (state.BinaryOpCalls == 1)
                    throw new InvalidOperationException("simulated dispatch error");
            },
            ex,
            out bool emitted);

        Assert.Null(result);
        Assert.True(emitted);
    }

    [Fact]
    public void TryRunUnaryChunked_FailFastPolicy_RethrowsSubChunkException()
    {
        // Edge case: with NeverChunk_FailFast, a sub-chunk that ALSO can't
        // fit (GpuBufferTooLargeException raised during the chunk loop)
        // should re-throw rather than be swallowed. Verifies the catch
        // separates "buffer too large" (rethrow) from "dispatch error"
        // (swallow → CPU).
        var state = new MockBackendState { MaxBufferAllocBytes = 16 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 8 });
        var ex = new GpuBufferTooLargeException("Mock", 32, 16, "MockDevice");

        // Op delegate throws GpuBufferTooLargeException on first chunk —
        // simulating a sub-allocation that's still too big somehow.
        Assert.Throws<GpuBufferTooLargeException>(() =>
            engine.TryRunUnaryChunked<float>(backend, input,
                (b, bIn, bOut, len) =>
                {
                    state.UnaryOpCalls++;
                    throw new GpuBufferTooLargeException("Mock", 999, 16, "MockDevice");
                },
                ex,
                out bool _));
    }

    [Fact]
    public void TryRunBinaryChunked_ScalarTensor_ReturnsNullWithoutEmitting()
    {
        var state = new MockBackendState { MaxBufferAllocBytes = 4 };
        var backend = MockDirectGpuBackend.Create(state);
        var engine = new DirectGpuTensorEngine();

        var left = new Tensor<float>(new float[] { 1 }, new[] { 1 });
        var right = new Tensor<float>(new float[] { 2 }, new[] { 1 });
        var ex = new GpuBufferTooLargeException("Mock", 4, 4, "MockDevice");

        var result = engine.TryRunBinaryChunked<float>(backend, left, right,
            (b, bA, bB, bC, len) => state.BinaryOpCalls++,
            ex,
            out bool emitted);

        Assert.Null(result);
        Assert.False(emitted);
    }
}
#endif
