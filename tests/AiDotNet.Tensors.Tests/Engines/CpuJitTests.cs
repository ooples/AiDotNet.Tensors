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

        var kernel = CpuJitKernels.GetAddKernel(size);
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

        var kernel = CpuJitKernels.GetMultiplyKernel(size);
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

        var kernel = CpuJitKernels.GetReLUKernel(size);
        kernel(src.FloatPtr, dst.FloatPtr, size);

        var spanDst = dst.AsSpan();
        for (int i = 0; i < size; i++)
        {
            float expected = Math.Max(i - 128, 0);
            Assert.Equal(expected, spanDst[i], 3);
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
