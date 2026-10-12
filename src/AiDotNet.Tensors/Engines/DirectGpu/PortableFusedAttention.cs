// Copyright (c) AiDotNet. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IFusedAttentionBackend"/> for every backend without the CUDA tiles (#1179), over the shared fa_p_ kernels
/// in <see cref="K.AttentionSource"/>: the same scores, soft-cap, causal offset, broadcast mask and log-sum-exp as the
/// CUDA kernels, one thread per output element. Any head dimension runs.
/// </summary>
internal static class PortableFusedAttention
{
    public static bool Supports(int headDim) => headDim > 0;

    private static int Narrow(long value, string name)
        => value >= int.MinValue && value <= int.MaxValue
            ? (int)value
            : throw new NotSupportedException($"fused attention: stride {name} = {value} does not fit the portable kernels' int32 indexing.");

    // The scalars every fa_p_ kernel reads, in AttentionSource's order (scale and soft-cap as float bits), carried in a float buffer.
    private static IGpuBuffer Meta(IDirectGpuBackend backend, in FusedAttentionParams p, int headDim)
    {
        int[] values =
        {
            p.B, p.H, p.Sq, p.Sk, headDim,
            Narrow(p.Qb, "q batch"), Narrow(p.Qh, "q head"), Narrow(p.Qr, "q row"),
            Narrow(p.Kb, "k batch"), Narrow(p.Kh, "k head"), Narrow(p.Kr, "k row"),
            Narrow(p.Vb, "v batch"), Narrow(p.Vh, "v head"), Narrow(p.Vr, "v row"),
            Narrow(p.Ob, "o batch"), Narrow(p.Oh, "o head"), Narrow(p.Or, "o row"),
            Narrow(p.Mb, "mask batch"), Narrow(p.Mh, "mask head"), Narrow(p.Mi, "mask row"), Narrow(p.Mj, "mask column"),
            p.Causal, p.CausalOffset, p.HasMask,
            BitConverter.ToInt32(BitConverter.GetBytes(p.Scale), 0), BitConverter.ToInt32(BitConverter.GetBytes(p.Softcap), 0),
        };
        var words = new float[K.FaMetaCount];
        for (int i = 0; i < values.Length; i++) words[i] = BitConverter.ToSingle(BitConverter.GetBytes(values[i]), 0);
        return backend.AllocateBuffer(words);
    }

    private static int Elements(int rows, int headDim)
    {
        long total = (long)rows * headDim;
        return total <= int.MaxValue
            ? (int)total
            : throw new NotSupportedException($"fused attention: {total} output elements exceed the portable kernels' int32 grid.");
    }

    public static void Forward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer q, IGpuBuffer k,
        IGpuBuffer v, IGpuBuffer o, IGpuBuffer lse, IGpuBuffer? mask, in FusedAttentionParams p, int headDim)
    {
        if (p.Sq <= 0 || p.B * p.H <= 0) return;
        int rows = checked(p.B * p.H * p.Sq);
        int elements = Elements(rows, headDim);
        using var meta = Meta(backend, p, headDim);
        // Without a mask the kernels never read it (has-mask is 0); the metadata buffer stands in for the binding.
        var maskBuffer = mask ?? meta;
        launch(K.FaLse, rows, q, k, meta, maskBuffer, lse, rows);
        launch(K.FaOut, elements, q, k, v, meta, maskBuffer, lse, o, elements);
    }

    public static void Backward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer q, IGpuBuffer k,
        IGpuBuffer v, IGpuBuffer o, IGpuBuffer dO, IGpuBuffer lse, IGpuBuffer? mask, IGpuBuffer dQ, IGpuBuffer dK,
        IGpuBuffer dV, in FusedAttentionParams p, int headDim)
    {
        if (p.B * p.H <= 0) return;
        int rows = checked(p.B * p.H * p.Sq);
        int keys = checked(p.B * p.H * p.Sk);
        using var meta = Meta(backend, p, headDim);
        // lse in [0, rows), delta in [rows, 2 rows): one binding for both keeps every kernel within 8 storage buffers.
        using var stats = backend.AllocateBuffer(Math.Max(1, 2 * rows));
        var maskBuffer = mask ?? meta;
        if (rows > 0)
        {
            backend.Copy(lse, 0, stats, 0, rows);
            launch(K.FaDelta, rows, o, dO, meta, stats, rows);
            int elements = Elements(rows, headDim);
            launch(K.FaDq, elements, q, k, v, dO, stats, meta, maskBuffer, dQ, elements);
        }
        if (keys > 0)
        {
            int elements = Elements(keys, headDim);
            launch(K.FaDk, elements, q, k, v, dO, stats, meta, maskBuffer, dK, elements);
            launch(K.FaDv, elements, q, k, dO, stats, meta, maskBuffer, dV, elements);
        }
    }
}
