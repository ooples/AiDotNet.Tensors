// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// Counter-based RNG family. Mirrors what cuRAND exposes plus the CPU
/// equivalents in <c>torch.Generator</c>. The same enum is used by both
/// the CPU and GPU backends so user code can switch backends without
/// changing the algorithm choice.
/// </summary>
public enum DeviceRngAlgorithm
{
    /// <summary>Philox 4x32-10 — counter-based, parallel-friendly,
    /// same algorithm PyTorch uses on CUDA. Default.</summary>
    Philox,
    /// <summary>MRG32k3a — period 2^191, slower than Philox but useful
    /// for stratified sampling. cuRAND parity.</summary>
    Mrg32K3A,
    /// <summary>Sobol 32-bit quasi-random sequence. cuRAND parity.</summary>
    Sobol,
    /// <summary>Scrambled Sobol 32-bit. cuRAND parity.</summary>
    ScrambledSobol,
}

/// <summary>
/// Counter-based device RNG abstraction. Backends implement this for
/// CPU (managed Philox / Sobol) and each GPU runtime (cuRAND on CUDA,
/// rocRAND on HIP, MPS RNG on Metal, OpenCL kernel RNG on CL).
///
/// <para><b>Determinism contract:</b> seed + offset + algorithm uniquely
/// determines the output bits. The CPU and CUDA backends both implement
/// the same Philox 4x32-10 algorithm so a (seed, offset) pair that
/// produces a given bit on CPU produces the same bit on GPU. This is
/// PyTorch's most-requested cross-backend determinism property —
/// <c>torch.manual_seed(0)</c> + same code on CPU vs CUDA produces
/// different sequences in PyTorch.</para>
/// </summary>
public interface IDeviceRng
{
    /// <summary>RNG algorithm this generator implements.</summary>
    DeviceRngAlgorithm Algorithm { get; }

    /// <summary>The seed configured at construction.</summary>
    ulong Seed { get; }

    /// <summary>Current 64-bit counter offset (advances by one per
    /// 4-element block). Saved/restored for checkpointing.</summary>
    ulong Offset { get; set; }

    /// <summary>Subsequence index within the (seed, offset) stream.
    /// Used for parallel sub-streams that share a seed but advance
    /// independently — typical pattern is per-(epoch, worker, sample)
    /// from the data-loading issue.</summary>
    ulong Subsequence { get; set; }

    /// <summary>Fills <paramref name="output"/> with U(0,1) doubles.</summary>
    void Uniform(Tensor<double> output);

    /// <summary>Fills <paramref name="output"/> with U(0,1) floats.</summary>
    void Uniform(Tensor<float> output);

    /// <summary>Fills <paramref name="output"/> with N(0,1) floats via
    /// the Box-Muller transform on the underlying uniform stream.</summary>
    void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f);

    /// <summary>Fills <paramref name="output"/> with N(0,1) doubles.</summary>
    void Normal(Tensor<double> output, double mean = 0, double stddev = 1);

    /// <summary>Bernoulli(p) draws — output is 0/1 floats.</summary>
    void Bernoulli(Tensor<float> output, float p);
}
