// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). One op × one shape/config = one OpCase.
#if !NETFRAMEWORK

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// A deterministic tensor input shared between the float run, the GPU float run, and the double
/// ORACLE run. The SAME numeric samples (generated once as double[]) feed all three, so any
/// difference is the op's numerics — not the input. <see cref="F"/> materializes a fresh
/// <c>Tensor&lt;float&gt;</c>, <see cref="D"/> a fresh <c>Tensor&lt;double&gt;</c> (fresh each call
/// so an in-place op on one engine can't corrupt a sibling run).
/// </summary>
public sealed class OpInput
{
    private readonly double[] _data;
    public int[] Shape { get; }

    private OpInput(double[] data, int[] shape) { _data = data; Shape = shape; }

    /// <summary>Uniform samples in [lo, hi] from a fixed seed. Deterministic across runs/engines.</summary>
    public static OpInput Rand(int seed, int[] shape, double lo = -1.0, double hi = 1.0)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(seed);
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = lo + rng.NextDouble() * (hi - lo);
        return new OpInput(data, (int[])shape.Clone());
    }

    /// <summary>Strictly-positive samples in [lo, hi] — for log / sqrt / rsqrt / pow domains.</summary>
    public static OpInput RandPositive(int seed, int[] shape, double lo = 0.1, double hi = 4.0)
        => Rand(seed, shape, lo, hi);

    /// <summary>Explicit data (row-major) — for ops needing structured values (indices, masks).</summary>
    public static OpInput From(double[] data, int[] shape) => new OpInput((double[])data.Clone(), shape);

    public Tensor<float> F()
    {
        var f = new float[_data.Length];
        for (int i = 0; i < _data.Length; i++) f[i] = (float)_data[i];
        return new Tensor<float>(f, (int[])Shape.Clone());
    }

    public Tensor<double> D() => new Tensor<double>((double[])_data.Clone(), (int[])Shape.Clone());

    /// <summary>Complex float tensor whose real parts are this input's samples and imaginary parts
    /// are <paramref name="imag"/>'s samples (both must share this shape). For NativeComplex* ops.</summary>
    public Tensor<Complex<float>> CF(OpInput imag)
    {
        if (imag._data.Length != _data.Length)
            throw new ArgumentException($"imag length {imag._data.Length} must match this input's length {_data.Length}.", nameof(imag));
        var c = new Complex<float>[_data.Length];
        for (int i = 0; i < _data.Length; i++) c[i] = new Complex<float>((float)_data[i], (float)imag._data[i]);
        return new Tensor<Complex<float>>(c, (int[])Shape.Clone());
    }

    /// <summary>Complex double tensor (the oracle counterpart of <see cref="CF"/>).</summary>
    public Tensor<Complex<double>> CD(OpInput imag)
    {
        if (imag._data.Length != _data.Length)
            throw new ArgumentException($"imag length {imag._data.Length} must match this input's length {_data.Length}.", nameof(imag));
        var c = new Complex<double>[_data.Length];
        for (int i = 0; i < _data.Length; i++) c[i] = new Complex<double>(_data[i], imag._data[i]);
        return new Tensor<Complex<double>>(c, (int[])Shape.Clone());
    }
}

/// <summary>
/// One parity case: an op invoked on a given engine, in float and in double, plus tolerances.
/// The delegates take the <see cref="IEngine"/> so the harness runs the identical closure on the
/// CPU engine, the GPU engine, and (in double) the CPU oracle. Backward delegates are optional;
/// when present the harness parity-checks the gradient too.
/// </summary>
public sealed class OpCase
{
    /// <summary>Display id, e.g. "Softmax[2,8]". Doubles as the emitted test name.</summary>
    public string Name { get; }

    /// <summary>The IEngine method this case exercises, e.g. "GELU", "TensorMatMul", "Conv2D".
    /// Used by the inventory/coverage report to mark that op covered across the full surface.</summary>
    public string OpMethod { get; }

    /// <summary>Coarse bucket (arithmetic, matmul, activation, reduction, norm, conv, attention, loss, shape).</summary>
    public string Category { get; }

    public Func<IEngine, Tensor<float>> RunFloat { get; }
    public Func<IEngine, Tensor<double>> RunDouble { get; }
    public ParityTol Fwd { get; }

    public Func<IEngine, Tensor<float>>? RunFloatGrad { get; }
    public Func<IEngine, Tensor<double>>? RunDoubleGrad { get; }
    public ParityTol BwdTol { get; }
    public bool HasBackward => RunFloatGrad is not null && RunDoubleGrad is not null;

    /// <summary>Non-null marks a CONFIRMED, tracked CPU/GPU divergence (a real cross-engine bug the
    /// scaffold found that isn't fixed yet). The harness records it and SKIPS with this reason
    /// instead of failing the build — but if the op ever starts passing, the harness fails to prompt
    /// removing the marker, so the fix is noticed. Keeps CI green without hiding the finding.</summary>
    public string? KnownDivergence { get; init; }

    /// <summary>Set when the op's GPU kernel is not just numerically divergent but actively UNSAFE —
    /// it crashes / hangs the host process or corrupts GPU state for subsequent ops (e.g. an OpenCL
    /// kernel that over-allocates private memory and errors the command queue). Unlike
    /// <see cref="KnownDivergence"/>, which still executes both engines before skipping, a GpuUnsafe
    /// op is SKIPPED before the GPU is ever touched, so it can't poison the run. Records the finding
    /// (visible in the report) without executing the crashing kernel. Requires <see cref="KnownDivergence"/>
    /// to carry the reason.</summary>
    public bool GpuUnsafe { get; init; }

    public OpCase(
        string name, string category,
        Func<IEngine, Tensor<float>> runFloat,
        Func<IEngine, Tensor<double>> runDouble,
        ParityTol fwd,
        Func<IEngine, Tensor<float>>? runFloatGrad = null,
        Func<IEngine, Tensor<double>>? runDoubleGrad = null,
        ParityTol bwdTol = default,
        string? opMethod = null)
    {
        Name = name;
        // Default the covered op method to the leading identifier of the display name
        // (e.g. "Softmax[4,16]" -> "Softmax") when not given explicitly.
        OpMethod = opMethod ?? LeadingIdentifier(name);
        Category = category;
        RunFloat = runFloat;
        RunDouble = runDouble;
        Fwd = fwd;
        RunFloatGrad = runFloatGrad;
        RunDoubleGrad = runDoubleGrad;
        BwdTol = bwdTol;
    }

    private static string LeadingIdentifier(string name)
    {
        int i = 0;
        while (i < name.Length && (char.IsLetterOrDigit(name[i]) || name[i] == '_')) i++;
        return i > 0 ? name.Substring(0, i) : name;
    }
}
#endif
