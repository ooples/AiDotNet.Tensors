using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class HipBlasArchitectureDataTests
{
    [Fact]
    public void HasCompatibleTensileLibrary_AcceptsExactArchitectureData()
    {
        string directory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(Path.Combine(directory, "TensileLibrary_lazy_gfx1012.dat"), string.Empty);

            Assert.True(HipBlasNative.HasCompatibleTensileLibrary(directory, "gfx1012"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void HasCompatibleTensileLibrary_AcceptsGenericArchitectureData()
    {
        string directory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(Path.Combine(directory, "TensileLibrary.dat"), string.Empty);

            Assert.True(HipBlasNative.HasCompatibleTensileLibrary(directory, "gfx1012"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void HasCompatibleTensileLibrary_RejectsAnotherArchitecture()
    {
        string directory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(Path.Combine(directory, "TensileLibrary_lazy_gfx1100.dat"), string.Empty);

            Assert.False(HipBlasNative.HasCompatibleTensileLibrary(directory, "gfx1012"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void FindCompatibleTensileLibrary_SkipsExistingIncompatibleDirectory()
    {
        string incompatibleDirectory = CreateTemporaryDirectory();
        string compatibleDirectory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(
                Path.Combine(incompatibleDirectory, "TensileLibrary_lazy_gfx1100.dat"),
                string.Empty);
            File.WriteAllText(
                Path.Combine(compatibleDirectory, "TensileLibrary_lazy_gfx1012.dat"),
                string.Empty);

            string? selected = HipBlasNative.FindCompatibleTensileLibrary(
                new[] { incompatibleDirectory, compatibleDirectory },
                "gfx1012");

            Assert.Equal(compatibleDirectory, selected);
        }
        finally
        {
            Directory.Delete(incompatibleDirectory, recursive: true);
            Directory.Delete(compatibleDirectory, recursive: true);
        }
    }

    private static string CreateTemporaryDirectory()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"aidotnet-hipblas-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        return directory;
    }
}
