using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class RocmNativeLibraryResolverTests
{
    [Fact]
    public void WindowsCandidates_AreIndependentOfBuildOperatingSystem()
    {
        var expectedCandidates = new[]
        {
            (RocmNativeLibraryKind.Runtime, "amdhip64_6.dll"),
            (RocmNativeLibraryKind.RuntimeCompiler, "hiprtc0604.dll"),
            (RocmNativeLibraryKind.HipBlas, "hipblas.dll")
        };

        foreach ((RocmNativeLibraryKind kind, string expectedName) in expectedCandidates)
        {
            IReadOnlyList<string> candidates = RocmNativeLibraryResolver.GetCandidates(
                kind,
                RocmOperatingSystem.Windows,
                rocmBinPath: null);

            Assert.Contains(expectedName, candidates);
        }
    }

    [Fact]
    public void LinuxCandidates_AreIndependentOfBuildOperatingSystem()
    {
        var expectedCandidates = new[]
        {
            (RocmNativeLibraryKind.Runtime, "libamdhip64.so.6"),
            (RocmNativeLibraryKind.RuntimeCompiler, "libhiprtc.so.6"),
            (RocmNativeLibraryKind.HipBlas, "libhipblas.so")
        };

        foreach ((RocmNativeLibraryKind kind, string expectedName) in expectedCandidates)
        {
            IReadOnlyList<string> candidates = RocmNativeLibraryResolver.GetCandidates(
                kind,
                RocmOperatingSystem.Linux,
                rocmBinPath: null);

            Assert.Contains(expectedName, candidates);
        }
    }

    [Fact]
    public void UnsupportedOperatingSystem_HasNoCandidates()
    {
        IReadOnlyList<string> candidates = RocmNativeLibraryResolver.GetCandidates(
            RocmNativeLibraryKind.Runtime,
            RocmOperatingSystem.Unsupported,
            rocmBinPath: null);

        Assert.Empty(candidates);
    }
}
