// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the OpenCL weight-only fused dequant-GEMM (P0): the quantized
// LLM-serving decode hot path. Validates OpenClBackend.DequantGemmInt8 against a CPU
// reference implementing the EXACT contract of FusedDequantMatmulKernels.Q8MatMul.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Proves the OpenCL int8 weight-only dequant-GEMM kernel is numerically correct (not just
/// non-crashing) by comparing it to a scalar CPU reference computed from identical inputs.
/// Skips when no OpenCL device is available, unless AIDOTNET_REQUIRE_GPU_TESTS=1.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class QuantGemmOpenClTests : IDisposable
{
    private readonly OpenClBackend? _backend;
    private readonly bool _ready;
    private readonly Exception? _initException;

    public QuantGemmOpenClTests()
    {
        try
        {
            _backend = new OpenClBackend();
            _ready = _backend.IsAvailable;
        }
        catch (Exception ex)
        {
            _initException = ex;
            _ready = false;
        }
    }

    public void Dispose() => _backend?.Dispose();

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required but the OpenCL backend was unavailable.", _initException);
        return false;
    }

    // Scalar CPU reference == the documented contract of FusedDequantMatmulKernels.Q8MatMul.
    private static float[] Reference(float[] act, sbyte[] w, float[] scales, int M, int K, int N, int groupSize, int scaleCount)
    {
        var outp = new float[M * N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                if (scaleCount == 1)
                {
                    float s = scales[0];
                    for (int k = 0; k < K; k++) acc += act[i * K + k] * w[k * N + j];
                    acc *= s;
                }
                else
                {
                    for (int k = 0; k < K; k++)
                    {
                        int flat = k * N + j;
                        acc += act[i * K + k] * w[flat] * scales[flat / groupSize];
                    }
                }
                outp[i * N + j] = acc;
            }
        }
        return outp;
    }

    [Theory]
    [InlineData(0)]    // per-tensor (scaleCount == 1)
    [InlineData(64)]   // per-group over flattened K*N
    [InlineData(128)]  // per-group, different group size
    public void DequantGemmInt8_MatchesCpuOracle(int groupSize)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;

        const int M = 8, K = 128, N = 64;
        var rng = new Random(20260717);

        var act = new float[M * K];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var w = new sbyte[K * N];
        for (int i = 0; i < w.Length; i++) w[i] = (sbyte)rng.Next(-127, 128);

        bool perTensor = groupSize <= 0;
        int scaleCount;
        float[] scales;
        int gsArg;
        if (perTensor)
        {
            scaleCount = 1;
            scales = new[] { 0.03f };
            gsArg = K * N; // unused by the kernel when scaleCount == 1
        }
        else
        {
            int totalGroups = (K * N + groupSize - 1) / groupSize;
            scaleCount = totalGroups;
            scales = new float[totalGroups];
            for (int g = 0; g < totalGroups; g++) scales[g] = 0.01f + (float)(rng.NextDouble() * 0.05);
            gsArg = groupSize;
        }

        var expected = Reference(act, w, scales, M, K, N, gsArg, scaleCount);

        // Upload: activations + scales as float buffers; weights as a byte buffer (sbyte bit-reinterpret).
        var wbytes = new byte[w.Length];
        for (int i = 0; i < w.Length; i++) wbytes[i] = unchecked((byte)w[i]);

        float[] actual;
        IGpuBuffer? actBuf = null, scaleBuf = null, wBuf = null, outBuf = null;
        try
        {
            actBuf = backend.AllocateBuffer(act);
            scaleBuf = backend.AllocateBuffer(scales);
            wBuf = backend.AllocateByteBuffer(w.Length);
            backend.UploadByteBuffer(wBuf, wbytes);
            outBuf = backend.DequantGemmInt8(actBuf, wBuf, scaleBuf, M, K, N, gsArg, scaleCount);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            actBuf?.Dispose();
            scaleBuf?.Dispose();
            wBuf?.Dispose();
            outBuf?.Dispose();
        }

        Assert.Equal(M * N, actual.Length);
        for (int idx = 0; idx < expected.Length; idx++)
        {
            float e = expected[idx];
            float a = actual[idx];
            float tol = 1e-2f + 1e-3f * Math.Abs(e); // fast-relaxed-math + FMA reordering on GPU
            Assert.True(Math.Abs(e - a) <= tol,
                $"Mismatch at {idx}: expected {e}, got {a} (tol {tol}, groupSize {groupSize})");
        }
    }

    [Theory]
    [InlineData(0)]    // per-tensor (scaleCount == 1)
    [InlineData(64)]   // per-group over flattened K*N
    [InlineData(128)]  // per-group, different group size
    public void DequantGemmInt4_MatchesCpuOracle(int groupSize)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;

        const int M = 8, K = 128, N = 64;
        const int kn = K * N;
        var rng = new Random(0x4B4B4B);

        var act = new float[M * K];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // int4 values in [-8,7]: keep an unpacked sbyte[] for the reference, and a
        // 2-per-byte packed byte[] (low nibble = even element) for the GPU — matching PackedInt4.
        var w = new sbyte[kn];
        for (int i = 0; i < kn; i++) w[i] = (sbyte)rng.Next(-8, 8);
        var packed = new byte[(kn + 1) / 2];
        for (int idx = 0; idx < kn; idx++)
        {
            int lo = w[idx] & 0x0F;
            if ((idx & 1) == 0) packed[idx >> 1] = (byte)((packed[idx >> 1] & 0xF0) | lo);
            else packed[idx >> 1] = (byte)((packed[idx >> 1] & 0x0F) | (lo << 4));
        }

        bool perTensor = groupSize <= 0;
        int scaleCount;
        float[] scales;
        int gsArg;
        if (perTensor)
        {
            scaleCount = 1;
            scales = new[] { 0.02f };
            gsArg = kn;
        }
        else
        {
            int totalGroups = (kn + groupSize - 1) / groupSize;
            scaleCount = totalGroups;
            scales = new float[totalGroups];
            for (int g = 0; g < totalGroups; g++) scales[g] = 0.01f + (float)(rng.NextDouble() * 0.05);
            gsArg = groupSize;
        }

        var expected = Reference(act, w, scales, M, K, N, gsArg, scaleCount);

        float[] actual;
        IGpuBuffer? actBuf = null, scaleBuf = null, wBuf = null, outBuf = null;
        try
        {
            actBuf = backend.AllocateBuffer(act);
            scaleBuf = backend.AllocateBuffer(scales);
            wBuf = backend.AllocateByteBuffer(packed.Length);
            backend.UploadByteBuffer(wBuf, packed);
            outBuf = backend.DequantGemmInt4(actBuf, wBuf, scaleBuf, M, K, N, gsArg, scaleCount);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            actBuf?.Dispose();
            scaleBuf?.Dispose();
            wBuf?.Dispose();
            outBuf?.Dispose();
        }

        Assert.Equal(M * N, actual.Length);
        for (int idx = 0; idx < expected.Length; idx++)
        {
            float e = expected[idx];
            float a = actual[idx];
            float tol = 1e-2f + 1e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol,
                $"int4 mismatch at {idx}: expected {e}, got {a} (tol {tol}, groupSize {groupSize})");
        }
    }

    [Theory]
    [InlineData(0)]    // per-tensor (scaleCount == 1)
    [InlineData(64)]   // per-group over flattened K*N
    [InlineData(128)]  // per-group, different group size
    public void DequantGemmFp8E4M3_MatchesCpuOracle(int groupSize)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;

        const int M = 8, K = 128, N = 64;
        const int kn = K * N;
        var rng = new Random(0xF8F8);

        var act = new float[M * K];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // fp8 e4m3 weights: raw bytes for the GPU, decoded floats (Float8E4M3.ToFloat) for the reference.
        var raws = new byte[kn];
        var decoded = new float[kn];
        for (int i = 0; i < kn; i++)
        {
            var f8 = Float8E4M3.FromFloat((float)(rng.NextDouble() * 8.0 - 4.0)); // within E4M3 finite range
            raws[i] = f8.RawValue;
            decoded[i] = f8.ToFloat();
        }
        // Hardware-validate exact raw E4M3 encodings FromFloat can't produce, incl. exponent-zero raws
        // (0x01/0x07/0x81/0x87) and the ±MaxFinite/±smallest-normal boundaries — GPU decode must match CPU.
        byte[] rawBoundaries = { 0x01, 0x07, 0x81, 0x87, 0x7E, 0xFE, 0x08, 0x88 };
        for (int b = 0; b < rawBoundaries.Length && b < kn; b++)
        {
            var f8 = Float8E4M3.FromRawValue(rawBoundaries[b]);
            raws[b] = f8.RawValue;
            decoded[b] = f8.ToFloat();
        }

        bool perTensor = groupSize <= 0;
        int scaleCount;
        float[] scales;
        int gsArg;
        if (perTensor)
        {
            scaleCount = 1;
            scales = new[] { 0.05f };
            gsArg = kn;
        }
        else
        {
            int totalGroups = (kn + groupSize - 1) / groupSize;
            scaleCount = totalGroups;
            scales = new float[totalGroups];
            for (int g = 0; g < totalGroups; g++) scales[g] = 0.01f + (float)(rng.NextDouble() * 0.05);
            gsArg = groupSize;
        }

        // Reference: C[i,j] = sum_k act[i,k] * decoded[k*N+j] * scale(flat).
        var expected = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                if (perTensor)
                {
                    for (int k = 0; k < K; k++) acc += act[i * K + k] * decoded[k * N + j];
                    acc *= scales[0];
                }
                else
                {
                    for (int k = 0; k < K; k++)
                    {
                        int flat = k * N + j;
                        acc += act[i * K + k] * decoded[flat] * scales[flat / gsArg];
                    }
                }
                expected[i * N + j] = acc;
            }

        float[] actual;
        IGpuBuffer? actBuf = null, scaleBuf = null, wBuf = null, outBuf = null;
        try
        {
            actBuf = backend.AllocateBuffer(act);
            scaleBuf = backend.AllocateBuffer(scales);
            wBuf = backend.AllocateByteBuffer(kn);
            backend.UploadByteBuffer(wBuf, raws);
            outBuf = backend.DequantGemmFp8E4M3(actBuf, wBuf, scaleBuf, M, K, N, gsArg, scaleCount);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            actBuf?.Dispose();
            scaleBuf?.Dispose();
            wBuf?.Dispose();
            outBuf?.Dispose();
        }

        Assert.Equal(M * N, actual.Length);
        for (int idx = 0; idx < expected.Length; idx++)
        {
            float e = expected[idx];
            float a = actual[idx];
            float tol = 1e-2f + 1e-3f * Math.Abs(e);
            Assert.True(Math.Abs(e - a) <= tol,
                $"fp8 mismatch at {idx}: expected {e}, got {a} (tol {tol}, groupSize {groupSize})");
        }
    }
}

#endif
