using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.Tensors.Engines.CpuJit;

/// <summary>
/// Self-test for JIT-compiled kernels. Runs at first use to verify
/// the x86 emitter produces correct machine code on this CPU.
/// </summary>
internal static class CpuJitSelfTest
{
    private static readonly Lazy<bool> _verified = new Lazy<bool>(RunSelfTest, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Returns true if JIT kernels are verified working on this machine.
    /// Runs the self-test on first call, caches the result. Thread-safe.
    /// </summary>
    public static bool IsVerified => _verified.Value;

    private static unsafe bool RunSelfTest()
    {
        if (!CpuJitKernels.IsSupported)
            return false;

        try
        {
            // Test 1: Binary Add kernel (small, 64 elements = 2 loop iterations)
            const int testSize = 64;
            using var srcA = new AlignedBuffer(testSize);
            using var srcB = new AlignedBuffer(testSize);
            using var dst = new AlignedBuffer(testSize);

            var spanA = srcA.AsSpan();
            var spanB = srcB.AsSpan();
            var spanDst = dst.AsSpan();

            // Fill with known values
            for (int i = 0; i < testSize; i++)
            {
                spanA[i] = i * 1.0f;
                spanB[i] = i * 2.0f;
            }

            // Run JIT Add kernel (aligned=true since we're using AlignedBuffer)
            var addKernel = CpuJitKernels.GetBinaryKernel(JitBinaryOp.Add, testSize, aligned: true);
            addKernel(srcA.FloatPtr, srcB.FloatPtr, dst.FloatPtr, testSize);

            // Verify
            for (int i = 0; i < testSize; i++)
            {
                float expected = i * 3.0f;
                if (MathF.Abs(spanDst[i] - expected) > 0.001f)
                    return false;
            }

            // Test 2: Binary Multiply kernel
            var mulKernel = CpuJitKernels.GetBinaryKernel(JitBinaryOp.Multiply, testSize, aligned: true);
            mulKernel(srcA.FloatPtr, srcB.FloatPtr, dst.FloatPtr, testSize);

            for (int i = 0; i < testSize; i++)
            {
                float expected = i * 1.0f * i * 2.0f;
                if (MathF.Abs(spanDst[i] - expected) > 0.01f)
                    return false;
            }

            // Test 3: ReLU kernel
            for (int i = 0; i < testSize; i++)
            {
                spanA[i] = i - 32; // -32 to +31
            }

            var reluKernel = CpuJitKernels.GetReLUKernel(testSize, aligned: true);
            reluKernel(srcA.FloatPtr, dst.FloatPtr, testSize);

            for (int i = 0; i < testSize; i++)
            {
                float expected = Math.Max(i - 32, 0);
                if (MathF.Abs(spanDst[i] - expected) > 0.001f)
                    return false;
            }

            // Test 4: Sigmoid kernel (uses data section constant embedding)
            if (!TestSigmoidKernel())
                return false;

            return true;
        }
        catch
        {
            // Any exception means JIT is broken — fall back to SimdKernels
            return false;
        }
    }

    private static unsafe bool TestSigmoidKernel()
    {
        const int testSize = 64;
        using var src = new AlignedBuffer(testSize);
        using var jitDst = new AlignedBuffer(testSize);

        var spanSrc = src.AsSpan();
        var spanJitDst = jitDst.AsSpan();

        // Test full range including boundary regions
        for (int i = 0; i < testSize; i++)
        {
            spanSrc[i] = (i - 32) * 0.2f; // -6.4 to +6.2
        }

        // Run JIT kernel
        var sigmoidKernel = CpuJitKernels.GetSigmoidKernel(testSize);
        sigmoidKernel(src.FloatPtr, jitDst.FloatPtr, testSize);

        // Compare against SimdKernels (reference implementation using same polynomial)
        float[] simdOutput = new float[testSize];
        fixed (float* pSimdOutput = simdOutput)
        {
            Simd.SimdKernels.SigmoidUnsafe(src.FloatPtr, pSimdOutput, testSize);
        }

        for (int i = 0; i < testSize; i++)
        {
            // JIT and SIMD should produce bit-identical results (same polynomial, same ops)
            // Allow tiny FP rounding differences from VMULPS+VADDPS vs FMA
            if (MathF.Abs(spanJitDst[i] - simdOutput[i]) > 0.001f)
                return false;
        }

        return true;
    }
}
