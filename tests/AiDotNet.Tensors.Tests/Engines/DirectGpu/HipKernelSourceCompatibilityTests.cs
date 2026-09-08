using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class HipKernelSourceCompatibilityTests
{
    [Theory]
    [InlineData("#include <math.h>\nextern \"C\" __global__ void kernel() {}")]
    [InlineData("#include <math.h>\r\nextern \"C\" __global__ void kernel() {}")]
    [InlineData("#include <math.h>extern \"C\" __global__ void kernel() {}")]
    [InlineData("#include <float.h>\nextern \"C\" __global__ void kernel() {}")]
    public void Prepare_RemovesUnsupportedStandardHeaders(string source)
    {
        string prepared = HipKernelSourceCompatibility.Prepare(source);

        Assert.DoesNotContain("#include <math.h>", prepared, StringComparison.Ordinal);
        Assert.DoesNotContain("#include <float.h>", prepared, StringComparison.Ordinal);
        Assert.Contains("extern \"C\" __global__ void kernel()", prepared, StringComparison.Ordinal);
    }

    [Fact]
    public void Prepare_ProvidesHeaderIndependentFloatAndNullConstants()
    {
        string prepared = HipKernelSourceCompatibility.Prepare("float best = -FLT_MAX; float* pointer = NULL;");

        Assert.Contains("#define FLT_MAX __FLT_MAX__", prepared, StringComparison.Ordinal);
        Assert.Contains("#define NULL nullptr", prepared, StringComparison.Ordinal);
    }

    [Fact]
    public void Prepare_ProvidesInfinityWithoutStandardLibraryHeader()
    {
        string prepared = HipKernelSourceCompatibility.Prepare("float value = -INFINITY;");

        Assert.Contains("#define INFINITY __builtin_huge_valf()", prepared, StringComparison.Ordinal);
    }

    [Fact]
    public void Prepare_PreservesCudaWarpWidthWhenMappingShuffleIntrinsic()
    {
        string prepared = HipKernelSourceCompatibility.Prepare(
            "value += __shfl_down_sync(0xffffffff, value, offset);");

        Assert.Contains(
            "#define __shfl_down_sync(activeMask, value, delta) __shfl_down(value, delta, 32)",
            prepared,
            StringComparison.Ordinal);
    }
}
