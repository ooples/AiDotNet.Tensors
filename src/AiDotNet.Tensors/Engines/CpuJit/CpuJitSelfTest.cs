using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.CpuJit;

/// <summary>
/// Self-test for JIT-compiled kernels. Runs at first use to verify
/// the x86 emitter produces correct machine code on this CPU.
/// </summary>
internal static class CpuJitSelfTest
{
    private static bool _tested;
    private static bool _passed;

    /// <summary>
    /// Returns true if JIT kernels are verified working on this machine.
    /// Runs the self-test on first call, caches the result.
    /// </summary>
    public static bool IsVerified
    {
        get
        {
            if (!_tested)
            {
                _passed = RunSelfTest();
                _tested = true;
            }
            return _passed;
        }
    }

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

            // Run JIT Add kernel
            var addKernel = CpuJitKernels.GetAddKernel(testSize);
            addKernel(srcA.FloatPtr, srcB.FloatPtr, dst.FloatPtr, testSize);

            // Verify
            for (int i = 0; i < testSize; i++)
            {
                float expected = i * 3.0f;
                if (MathF.Abs(spanDst[i] - expected) > 0.001f)
                    return false;
            }

            // Test 2: Binary Multiply kernel
            var mulKernel = CpuJitKernels.GetMultiplyKernel(testSize);
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

            var reluKernel = CpuJitKernels.GetReLUKernel(testSize);
            reluKernel(srcA.FloatPtr, dst.FloatPtr, testSize);

            for (int i = 0; i < testSize; i++)
            {
                float expected = Math.Max(i - 32, 0);
                if (MathF.Abs(spanDst[i] - expected) > 0.001f)
                    return false;
            }

            return true;
        }
        catch
        {
            // Any exception means JIT is broken — fall back to SimdKernels
            return false;
        }
    }
}
