// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend that rounds 32-bit float buffers to a 16-bit float type and back on the device: every element becomes
/// the nearest value of <see cref="GpuElementType.Float16"/> or <see cref="GpuElementType.BFloat16"/> (ties to even,
/// overflow to infinity), still stored as a 32-bit float. What matmul autocast casts a GEMM's operands with.
/// </summary>
internal interface IGpuNarrowFloatRounding
{
    /// <summary>Whether <see cref="RoundToNarrowFloat"/> runs for <paramref name="type"/> on this device.</summary>
    bool SupportsRoundToNarrowFloat(GpuElementType type);

    /// <summary>
    /// <c>destination[i] = (float)(type)source[i]</c> for the first <paramref name="count"/> elements of two 32-bit
    /// float buffers. The destination may be the source.
    /// </summary>
    void RoundToNarrowFloat(GpuElementType type, IGpuBuffer source, IGpuBuffer destination, int count);
}

/// <summary>
/// Matmul autocast for a backend whose GEMMs are its own FP32 kernels (every backend but CUDA with cuBLAS, which has
/// 16-bit tensor-core GEMMs): PyTorch's <c>torch.autocast</c> semantics, the operands of every GEMM rounded to FP16 or
/// BF16 before it and the products accumulated in FP32. The numbers are those of
/// <see cref="Gpu.MatMulAutocastMode.CastOperands"/> on CUDA up to the order of the FP32 accumulation; the device has
/// no 16-bit multiplier to gain speed from.
/// </summary>
/// <remarks>
/// Instance state, as on CUDA: a compiled step's ops may run on other threads, so every GEMM of the backend takes the
/// newest scope still open; scopes may close in any order. <see cref="Gpu.MatMulAutocastMode.FastComputeType"/> asks
/// cuBLAS for a 16-bit compute type; these backends have no such kernel, and rounding the operands is the one way
/// their products reach 16 bits, so both modes round them: a scope gives 16-bit products on every device.
/// </remarks>
internal sealed class GemmOperandAutocast
{
    private readonly List<Scope> _scopes = new();
    // 0: no scope casts. Otherwise the GpuElementType the operands are rounded to, plus one.
    private int _state;
    // Set while this thread is inside a GEMM whose operands are already rounded, so the GEMMs it is built from
    // (a transposed product's inner GEMM, a fused linear's product) do not round them again.
    [ThreadStatic] private static int t_rounded;

    /// <summary>The type GEMM operands are rounded to now, or null.</summary>
    internal GpuElementType? CastType
        => !GemmAutocastExemption.Active && Volatile.Read(ref _state) is int state and not 0
            ? (GpuElementType)(state - 1) : null;

    internal IDisposable Enter(Gpu.PrecisionMode precision, Gpu.MatMulAutocastMode mode)
    {
        int state = precision switch
        {
            Gpu.PrecisionMode.Float32 => 0,
            Gpu.PrecisionMode.Float16 => (int)GpuElementType.Float16 + 1,
            Gpu.PrecisionMode.BFloat16 => (int)GpuElementType.BFloat16 + 1,
            _ => throw new NotSupportedException($"Matmul autocast supports FP16 and BF16, not {precision}."),
        };
        if (mode is not (Gpu.MatMulAutocastMode.CastOperands or Gpu.MatMulAutocastMode.FastComputeType))
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown matmul autocast mode.");
        var scope = new Scope(this, state);
        lock (_scopes)
        {
            _scopes.Add(scope);
            Volatile.Write(ref _state, state);
        }
        return scope;
    }

    /// <summary>
    /// Under an open scope: rounds the first <paramref name="aCount"/> elements of <paramref name="a"/> and the first
    /// <paramref name="bCount"/> of <paramref name="b"/> into scratch buffers and puts those in their place, until the
    /// result is disposed. With no scope, or inside a GEMM that already rounded, nothing changes.
    /// </summary>
    internal RoundedOperands Round(IDirectGpuBackend backend, ref IGpuBuffer a, long aCount, ref IGpuBuffer b, long bCount)
    {
        if (CastType is not { } type || t_rounded > 0) return default;
        if (backend is not IGpuNarrowFloatRounding rounding || !rounding.SupportsRoundToNarrowFloat(type))
            throw new GpuCapabilityUnavailableException(
                $"Matmul autocast is open for {type}, and this device cannot round GEMM operands to it.");
        if (aCount <= 0 || bCount <= 0 || aCount > int.MaxValue || bCount > int.MaxValue) return default;
        IGpuBuffer? roundedA = null, roundedB = null;
        try
        {
            roundedA = backend.AllocateBuffer((int)aCount);
            rounding.RoundToNarrowFloat(type, a, roundedA, (int)aCount);
            // One operand twice (a Gram matrix) is rounded once.
            if (ReferenceEquals(a, b) && aCount == bCount) roundedB = roundedA;
            else
            {
                roundedB = backend.AllocateBuffer((int)bCount);
                rounding.RoundToNarrowFloat(type, b, roundedB, (int)bCount);
            }
        }
        catch
        {
            if (!ReferenceEquals(roundedA, roundedB)) roundedB?.Dispose();
            roundedA?.Dispose();
            throw;
        }
        a = roundedA;
        b = roundedB;
        t_rounded++;
        return new RoundedOperands(roundedA, roundedB);
    }

    /// <summary>The scratch a GEMM's rounded operands live in; disposing releases it once the GEMM is queued.</summary>
    internal readonly struct RoundedOperands : IDisposable
    {
        private readonly IGpuBuffer? _a;
        private readonly IGpuBuffer? _b;

        internal RoundedOperands(IGpuBuffer a, IGpuBuffer b)
        {
            _a = a;
            _b = b;
        }

        /// <summary>Whether the operands were rounded (a scope is open and this is the outermost GEMM).</summary>
        internal bool Rounded => _a is not null;

        public void Dispose()
        {
            if (_a is null) return;
            t_rounded--;
            if (!ReferenceEquals(_a, _b)) _b?.Dispose();
            _a.Dispose();
        }
    }

    private sealed class Scope : IDisposable
    {
        private GemmOperandAutocast? _owner;
        internal readonly int State;

        internal Scope(GemmOperandAutocast owner, int state)
        {
            _owner = owner;
            State = state;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _owner, null) is not { } owner) return;
            lock (owner._scopes)
            {
                owner._scopes.Remove(this);
                Volatile.Write(ref owner._state, owner._scopes.Count == 0 ? 0 : owner._scopes[owner._scopes.Count - 1].State);
            }
        }
    }
}
