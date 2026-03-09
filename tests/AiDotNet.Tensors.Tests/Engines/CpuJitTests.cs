using AiDotNet.Tensors.Engines.CpuJit;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class CpuJitTests
{
    [Fact]
    public void JitSelfTest_PassesOnSupportedHardware()
    {
        if (!CpuJitKernels.IsSupported)
        {
            // Skip on non-AVX2 hardware
            return;
        }

        Assert.True(CpuJitSelfTest.IsVerified, "JIT self-test should pass on AVX2+FMA hardware");
    }

    [Fact]
    public unsafe void JitAddKernel_ProducesCorrectResults()
    {
        if (!CpuJitKernels.IsSupported)
            return;

        const int size = 256;
        using var a = new AlignedBuffer(size);
        using var b = new AlignedBuffer(size);
        using var c = new AlignedBuffer(size);

        var spanA = a.AsSpan();
        var spanB = b.AsSpan();
        for (int i = 0; i < size; i++)
        {
            spanA[i] = i * 0.5f;
            spanB[i] = i * 1.5f;
        }

        var kernel = CpuJitKernels.GetBinaryKernel(JitBinaryOp.Add, size, aligned: true);
        kernel(a.FloatPtr, b.FloatPtr, c.FloatPtr, size);

        var spanC = c.AsSpan();
        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i * 2.0f, spanC[i], 3);
        }
    }

    [Fact]
    public unsafe void JitMultiplyKernel_ProducesCorrectResults()
    {
        if (!CpuJitKernels.IsSupported)
            return;

        const int size = 256;
        using var a = new AlignedBuffer(size);
        using var b = new AlignedBuffer(size);
        using var c = new AlignedBuffer(size);

        var spanA = a.AsSpan();
        var spanB = b.AsSpan();
        for (int i = 0; i < size; i++)
        {
            spanA[i] = 2.0f;
            spanB[i] = i * 0.5f;
        }

        var kernel = CpuJitKernels.GetBinaryKernel(JitBinaryOp.Multiply, size, aligned: true);
        kernel(a.FloatPtr, b.FloatPtr, c.FloatPtr, size);

        var spanC = c.AsSpan();
        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i * 1.0f, spanC[i], 3);
        }
    }

    [Fact]
    public unsafe void JitReLUKernel_ProducesCorrectResults()
    {
        if (!CpuJitKernels.IsSupported)
            return;

        const int size = 256;
        using var src = new AlignedBuffer(size);
        using var dst = new AlignedBuffer(size);

        var spanSrc = src.AsSpan();
        for (int i = 0; i < size; i++)
        {
            spanSrc[i] = i - 128; // -128 to +127
        }

        var kernel = CpuJitKernels.GetReLUKernel(size, aligned: true);
        kernel(src.FloatPtr, dst.FloatPtr, size);

        var spanDst = dst.AsSpan();
        for (int i = 0; i < size; i++)
        {
            float expected = Math.Max(i - 128, 0);
            Assert.Equal(expected, spanDst[i], 3);
        }
    }

    [Fact]
    public unsafe void JitSigmoidKernel_ProducesCorrectResults()
    {
        if (!CpuJitKernels.IsSupported)
            return;

        const int size = 256;
        using var src = new AlignedBuffer(size);
        using var jitDst = new AlignedBuffer(size);
        using var simdDst = new AlignedBuffer(size);

        var spanSrc = src.AsSpan();
        // Full range including clamp boundaries
        for (int i = 0; i < size; i++)
        {
            spanSrc[i] = (i - 128) * 0.1f; // -12.8 to +12.7
        }

        // Run JIT kernel
        var kernel = CpuJitKernels.GetSigmoidKernel(size);
        kernel(src.FloatPtr, jitDst.FloatPtr, size);

        // Run SimdKernels reference (same polynomial)
        AiDotNet.Tensors.Engines.Simd.SimdKernels.SigmoidUnsafe(src.FloatPtr, simdDst.FloatPtr, size);

        var spanJit = jitDst.AsSpan();
        var spanSimd = simdDst.AsSpan();
        for (int i = 0; i < size; i++)
        {
            float x = (i - 128) * 0.1f;
            // Compare JIT vs SIMD (same polynomial, allow small FP rounding from VMULPS+VADDPS vs FMA)
            Assert.True(MathF.Abs(spanJit[i] - spanSimd[i]) <= 0.002f,
                $"JIT vs SIMD mismatch at i={i}, x={x:F2}: jit={spanJit[i]:F6}, simd={spanSimd[i]:F6}, diff={MathF.Abs(spanJit[i] - spanSimd[i]):F6}");
        }
    }

    [Fact]
    public void AlignedBuffer_Is64ByteAligned()
    {
        using var buf = new AlignedBuffer(1024);
        long addr = buf.Pointer.ToInt64();
        Assert.Equal(0, addr % 64);
    }

    [Fact]
    public void ExecutableBuffer_CanAllocateAndFree()
    {
        using var buf = new ExecutableBuffer(4096);
        Assert.NotEqual(IntPtr.Zero, buf.Pointer);
        Assert.Equal(4096, buf.Size);
    }
}
