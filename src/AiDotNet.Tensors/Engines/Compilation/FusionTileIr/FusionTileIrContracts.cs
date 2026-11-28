// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.FusionTileIr;

/// <summary>
/// Semantic tile schedule category in the hybrid <c>FusionTileIR</c> system.
/// </summary>
internal enum FusionTileKind : byte
{
    /// <summary>1D/ND grid-stride vectorized pointwise or broadcast tile.</summary>
    Elementwise = 0,

    /// <summary>Row-parallel warp-shuffle shared-memory reduction + affine normalization tile (RMSNorm/LayerNorm).</summary>
    RowNorm = 1,

    /// <summary>Rotary positional embedding pair-transform tile.</summary>
    RotaryPositionEmbedding = 2,

    /// <summary>GEMM prologue/epilogue tile (Bias + Activation + optional ResidualAdd + backward dBias).</summary>
    GemmEpilogue = 3,

    /// <summary>Numerically stable online row-max + log-sum-exp + cross-entropy + fused dLogits tile.</summary>
    LogSoftmaxCrossEntropy = 4,

    /// <summary>SRAM-tiled online-softmax FlashAttention tile (<c>B_r x B_c</c> shared memory blocks).</summary>
    TiledFlashAttention = 5,

    /// <summary>Chunked linear-attention / state-space recurrence scan tile.</summary>
    ChunkedLinearScan = 6
}

/// <summary>
/// Typed instruction opcodes for pre-lowered <see cref="FusionTileDescriptor"/> graphs.
/// </summary>
internal enum FusionTileOpCode : byte
{
    LoadInput = 0,
    LoadScalar = 1,
    Const = 2,
    Add = 3,
    Sub = 4,
    Mul = 5,
    Div = 6,
    Negate = 7,
    Fma = 8,
    Relu = 9,
    Sigmoid = 10,
    Swish = 11,
    Tanh = 12,
    Gelu = 13,
    Exp = 14,
    Log = 15,
    Sqrt = 16,
    Rsqrt = 17,
    Abs = 18,
    Max = 19,
    Min = 20,
    RowSum = 21,
    RowMax = 22,
    StoreOutput = 23,
    ReluStep = 24
}

/// <summary>
/// Compact value-type SSA instruction inside a <see cref="FusionTileDescriptor"/>.
/// Built at compile time by <c>AiDotNet.Tensors.Generators</c> or stitched in microseconds at runtime by <see cref="FusionTileLinker"/>.
/// </summary>
internal readonly struct FusionTileInstruction : IEquatable<FusionTileInstruction>
{
    public FusionTileOpCode Op { get; }
    public short Src0 { get; }
    public short Src1 { get; }
    public short Src2 { get; }
    public float Immediate { get; }

    public FusionTileInstruction(
        FusionTileOpCode op,
        short src0 = -1,
        short src1 = -1,
        short src2 = -1,
        float immediate = 0.0f)
    {
        Op = op;
        Src0 = src0;
        Src1 = src1;
        Src2 = src2;
        Immediate = immediate;
    }

    public bool Equals(FusionTileInstruction other) =>
        Op == other.Op &&
        Src0 == other.Src0 &&
        Src1 == other.Src1 &&
        Src2 == other.Src2 &&
        Immediate.Equals(other.Immediate);

    public override bool Equals(object? obj) => obj is FusionTileInstruction other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int h = (int)Op;
            h = (h * 397) ^ Src0;
            h = (h * 397) ^ Src1;
            h = (h * 397) ^ Src2;
            h = (h * 397) ^ Immediate.GetHashCode();
            return h;
        }
    }
}

/// <summary>
/// Pre-lowered intra-operator or cross-layer linked tile IR descriptor carrying typed SSA instructions,
/// prebaked CUDA device function fragment, optional CPU fast-path executor, and deterministic SHA-256 key.
/// </summary>
internal sealed class FusionTileDescriptor
{
    public string TileName { get; }
    public FusionTileKind Kind { get; }
    public int InputCount { get; }
    public int ScalarCount { get; }
    public int OutputCount { get; }
    public IReadOnlyList<FusionTileInstruction> Instructions { get; }
    public string CudaDeviceFragment { get; }
    public string Sha256Digest { get; }
    public Action<float[][], float[], float[][]>? CpuExecutor { get; }

    public FusionTileDescriptor(
        string tileName,
        FusionTileKind kind,
        int inputCount,
        int scalarCount,
        int outputCount,
        IReadOnlyList<FusionTileInstruction> instructions,
        string cudaDeviceFragment,
        Action<float[][], float[], float[][]>? cpuExecutor = null)
    {
        TileName = tileName ?? throw new ArgumentNullException(nameof(tileName));
        Kind = kind;
        InputCount = inputCount;
        ScalarCount = scalarCount;
        OutputCount = outputCount;
        if (instructions is null) throw new ArgumentNullException(nameof(instructions));
        // Take one immutable snapshot so later mutation of the caller's list cannot desynchronize
        // the instructions, the digest, and any executor built from the same snapshot.
        var snapshot = new FusionTileInstruction[instructions.Count];
        for (int i = 0; i < snapshot.Length; i++) snapshot[i] = instructions[i];
        Instructions = Array.AsReadOnly(snapshot);
        CudaDeviceFragment = cudaDeviceFragment ?? string.Empty;
        CpuExecutor = cpuExecutor;
        Sha256Digest = ComputeDigest(tileName, kind, inputCount, scalarCount, outputCount, Instructions, CudaDeviceFragment);
    }

    private static string ComputeDigest(
        string name,
        FusionTileKind kind,
        int inputs,
        int scalars,
        int outputs,
        IReadOnlyList<FusionTileInstruction> instructions,
        string fragment)
    {
        var sb = new StringBuilder(256 + fragment.Length);
        sb.Append(name).Append('|').Append((int)kind).Append('|')
          .Append(inputs).Append(',').Append(scalars).Append(',').Append(outputs).Append('|');
        for (int i = 0; i < instructions.Count; i++)
        {
            var inst = instructions[i];
            sb.Append((int)inst.Op).Append(':')
              .Append(inst.Src0).Append(',')
              .Append(inst.Src1).Append(',')
              .Append(inst.Src2).Append('@')
              .Append(inst.Immediate.ToString("R", CultureInfo.InvariantCulture)).Append(';');
        }
        sb.Append(fragment);

        using var sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
        var hex = new StringBuilder(64);
        for (int i = 0; i < hash.Length; i++)
        {
            hex.Append(hash[i].ToString("x2", CultureInfo.InvariantCulture));
        }
        return hex.ToString();
    }
}
