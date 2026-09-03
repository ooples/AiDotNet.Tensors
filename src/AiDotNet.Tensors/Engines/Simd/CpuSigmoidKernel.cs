using AiDotNet.Tensors.Engines.CpuJit;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>The type-safe CPU implementation selected for a float sigmoid invocation.</summary>
internal enum CpuSigmoidKernelKind : byte
{
    /// <summary>No implementation has been resolved.</summary>
    Uninitialized = 0,

    /// <summary>The portable CPU-adaptive SIMD path.</summary>
    AdaptiveSimd = 1,

    /// <summary>The Windows x64 runtime-emitted AVX2 kernel.</summary>
    RuntimeJit = 2,
}

/// <summary>
/// Resolves float sigmoid execution once and carries that same decision into eager, into-buffer,
/// in-place, and compiled replay paths.
/// </summary>
/// <remarks>
/// The runtime JIT kernel is specialized for a multiple-of-eight length. The scalar tail remains
/// part of this value's invocation contract, so callers cannot accidentally select the fast path
/// and leave elements unwritten. On platforms where runtime code generation is unavailable, the
/// adaptive SIMD path retains its Intel table / AMD Padé dispatch.
/// </remarks>
internal readonly struct CpuSigmoidKernel
{
    private readonly CpuJitKernels.UnaryKernel? _runtimeJit;
    private readonly int _length;
    private readonly int _runtimeJitLength;

    private CpuSigmoidKernel(
        CpuSigmoidKernelKind kind,
        CpuJitKernels.UnaryKernel? runtimeJit,
        int length,
        int runtimeJitLength)
    {
        Kind = kind;
        _runtimeJit = runtimeJit;
        _length = length;
        _runtimeJitLength = runtimeJitLength;
    }

    /// <summary>The selected implementation.</summary>
    internal CpuSigmoidKernelKind Kind { get; }

    /// <summary>Resolves the fastest supported implementation for <paramref name="length"/>.</summary>
    internal static CpuSigmoidKernel Resolve(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));

        int jitLength = length & ~7;
        if (length >= 64 && CpuJitSelfTest.IsVerified)
        {
            return new CpuSigmoidKernel(
                CpuSigmoidKernelKind.RuntimeJit,
                CpuJitKernels.GetSigmoidKernel(jitLength),
                length,
                jitLength);
        }

        return new CpuSigmoidKernel(CpuSigmoidKernelKind.AdaptiveSimd, null, length, 0);
    }

    /// <summary>Executes the resolved implementation over the complete buffer.</summary>
    internal unsafe void Invoke(float* input, float* output)
    {
        switch (Kind)
        {
            case CpuSigmoidKernelKind.RuntimeJit:
                _runtimeJit!(input, output, _runtimeJitLength);
                for (int i = _runtimeJitLength; i < _length; i++)
                    output[i] = 1.0f / (1.0f + MathF.Exp(-input[i]));
                return;

            case CpuSigmoidKernelKind.AdaptiveSimd:
                SimdKernels.SigmoidUnsafe(input, output, _length);
                return;

            default:
                throw new InvalidOperationException($"Unknown CPU sigmoid kernel kind: {Kind}.");
        }
    }

    /// <summary>Resolves and executes a one-shot invocation.</summary>
    internal static unsafe void Execute(float* input, float* output, int length)
        => Resolve(length).Invoke(input, output);
}
