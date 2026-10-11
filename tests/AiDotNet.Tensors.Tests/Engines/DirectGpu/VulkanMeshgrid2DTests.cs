using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #923: the Vulkan fused two-vector Meshgrid2D route. For both admitted shapes and both indexing modes it must give
/// the CPU's grids exactly (it only copies values), stay on the device, and leave every other shape to the composed
/// reshape-plus-broadcast route, which must keep working.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanMeshgrid2DTests
{
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        return backend!;
    }

    // The backend is a process-wide singleton, so the engine around it is non-owning and never disposed.
    private static DirectGpuTensorEngine Engine(VulkanBackend backend) => new DirectGpuTensorEngine(new DirectGpuEngine(backend));

    private static Tensor<float> Vector(int n, int salt)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) t[i] = (float)((rng.NextDouble() - 0.5) * 200);
        // Values a copy must carry bit for bit.
        t[0] = float.NaN; t[1] = -0f; t[2] = float.PositiveInfinity; t[3] = float.Epsilon;
        return t;
    }

    private static void AssertSameBits(string what, Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[i]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: cpu {expected[i]:R} vulkan {actual[i]:R}");
    }

    public static TheoryData<int, int, string> AdmittedCases() => new()
    {
        { 256, 256, "ij" }, { 256, 256, "xy" }, { 1024, 256, "ij" }, { 1024, 256, "xy" },
    };

    [SkippableTheory]
    [MemberData(nameof(AdmittedCases))]
    public void AdmittedShape_MatchesCpu_ThroughEngine(int n0, int n1, string indexing)
    {
        var backend = BackendOrSkip();
        var gpu = Engine(backend);
        var x = Vector(n0, 1);
        var y = Vector(n1, 2);
        Assert.True(((IDirectPtxVisionBackend)backend).CanDirectPtxMeshgrid2D(n0, n1, indexing == "xy"));

        var expected = new CpuEngine().TensorMeshgrid(new[] { x, y }, indexing);
        long readbacks = GpuLaunchProbe.Readbacks;
        var actual = gpu.TensorMeshgrid(new[] { x, y }, indexing);
        // Both grids are built on the device: nothing is read back until the assertions below look at them.
        Assert.Equal(readbacks, GpuLaunchProbe.Readbacks);

        Assert.Equal(2, actual.Length);
        Assert.Equal(indexing == "xy" ? new[] { n1, n0 } : new[] { n0, n1 }, actual[0]._shape);
        AssertSameBits("grid0", expected[0], actual[0]);
        AssertSameBits("grid1", expected[1], actual[1]);
    }

    [SkippableTheory]
    [MemberData(nameof(AdmittedCases))]
    public void AdmittedShape_FusedRoute_MatchesComposedRoute(int n0, int n1, string indexing)
    {
        var backend = BackendOrSkip();
        var gpu = Engine(backend);
        var x = Vector(n0, 3);
        var y = Vector(n1, 4);
        Tensor<float>[] composed;
        try
        {
            VulkanBackend.Meshgrid2DFusedEnabled = false;
            Assert.False(((IDirectPtxVisionBackend)backend).CanDirectPtxMeshgrid2D(n0, n1, indexing == "xy"));
            composed = gpu.TensorMeshgrid(new[] { x, y }, indexing);
            _ = composed[0][0];
            _ = composed[1][0];
        }
        finally
        {
            VulkanBackend.Meshgrid2DFusedEnabled = true;
        }
        var fused = gpu.TensorMeshgrid(new[] { x, y }, indexing);
        AssertSameBits("grid0", composed[0], fused[0]);
        AssertSameBits("grid1", composed[1], fused[1]);
    }

    [SkippableTheory]
    [InlineData(256, 256, false)]
    [InlineData(256, 256, true)]
    [InlineData(1024, 256, false)]
    [InlineData(1024, 256, true)]
    public void Backend_WritesBothGrids_InOneCall(int n0, int n1, bool xy)
    {
        var backend = BackendOrSkip();
        var vision = (IDirectPtxVisionBackend)backend;
        var v0 = new float[n0];
        var v1 = new float[n1];
        for (int i = 0; i < n0; i++) v0[i] = i + 0.25f;
        for (int i = 0; i < n1; i++) v1[i] = -(i + 0.5f);
        using var s0 = backend.AllocateBuffer(v0);
        using var s1 = backend.AllocateBuffer(v1);
        using var o0 = backend.AllocateBuffer(n0 * n1);
        using var o1 = backend.AllocateBuffer(n0 * n1);
        Assert.True(vision.TryDirectPtxMeshgrid2DPair(s0, s1, o0, o1, n0, n1, xy));
        float[] g0 = backend.DownloadBuffer(o0), g1 = backend.DownloadBuffer(o1);
        int rows = xy ? n1 : n0, columns = xy ? n0 : n1;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                Assert.Equal(xy ? v0[c] : v0[r], g0[r * columns + c]);
                Assert.Equal(xy ? v1[r] : v1[c], g1[r * columns + c]);
            }
    }

    [SkippableFact]
    public void OtherShapes_KeepTheComposedRoute()
    {
        var backend = BackendOrSkip();
        var vision = (IDirectPtxVisionBackend)backend;
        Assert.False(vision.CanDirectPtxMeshgrid2D(256, 1024, false));   // the transposed shape is not admitted
        Assert.False(vision.CanDirectPtxMeshgrid2D(300, 200, true));
        Assert.False(vision.CanDirectPtxMeshgrid2D(1, 1, false));

        var gpu = Engine(backend);
        var x = Vector(300, 5);
        var y = Vector(200, 6);
        foreach (string indexing in new[] { "ij", "xy" })
        {
            var expected = new CpuEngine().TensorMeshgrid(new[] { x, y }, indexing);
            var actual = gpu.TensorMeshgrid(new[] { x, y }, indexing);
            AssertSameBits(indexing + " grid0", expected[0], actual[0]);
            AssertSameBits(indexing + " grid1", expected[1], actual[1]);
        }
    }

    [SkippableFact]
    public void Backend_DeclinesAliasedBuffers_AndRejectsUndersizedOnes()
    {
        var backend = BackendOrSkip();
        var vision = (IDirectPtxVisionBackend)backend;
        using var s0 = backend.AllocateBuffer(256);
        using var s1 = backend.AllocateBuffer(256);
        using var grid = backend.AllocateBuffer(256 * 256);
        using var other = backend.AllocateBuffer(256 * 256);
        using var small = backend.AllocateBuffer(64);
        // One buffer for both grids: declined, so the caller takes the composed route.
        Assert.False(vision.TryDirectPtxMeshgrid2DPair(s0, s1, grid, grid, 256, 256, false));
        Assert.Throws<ArgumentException>(() => vision.TryDirectPtxMeshgrid2DPair(s0, s1, grid, small, 256, 256, false));
        Assert.Throws<ArgumentException>(() => vision.TryDirectPtxMeshgrid2DPair(small, s1, grid, other, 256, 256, true));
        // A shape outside the admitted set is declined before any buffer is looked at.
        Assert.False(vision.TryDirectPtxMeshgrid2DPair(small, small, small, small, 8, 8, false));
    }

    // No device needed: libshaderc compiles on the CPU, so this runs wherever the native library loads (CI included).
    [SkippableTheory]
    [InlineData(256, 256, false)]
    [InlineData(256, 256, true)]
    [InlineData(1024, 256, false)]
    [InlineData(1024, 256, true)]
    public void Shader_CompilesToSpirv(int n0, int n1, bool xy)
    {
        VulkanGlslCompiler compiler;
        try { compiler = new VulkanGlslCompiler(); }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or BadImageFormatException) { compiler = null!; }
        Skip.If(compiler is null || !compiler.IsAvailable, "libshaderc is not available.");
        using (compiler)
        {
            uint[]? spirv = compiler!.CompileToSpirv(VulkanBackend.Meshgrid2DSource(n0, n1, xy));
            Assert.True(spirv is { Length: > 5 }, $"meshgrid {n0}x{n1} xy={xy} did not compile: {compiler.LastError}");
        }
    }

    [Fact]
    public void Shader_CompilesTheExtentsIn_AndDiffersPerIndexingMode()
    {
        string ij = VulkanBackend.Meshgrid2DSource(1024, 256, xy: false);
        string xy = VulkanBackend.Meshgrid2DSource(1024, 256, xy: true);
        Assert.NotEqual(ij, xy);
        // 'ij' grids are [1024, 256] (256 columns); 'xy' grids are [256, 1024] (1024 columns).
        Assert.Contains("i / 256u", ij);
        Assert.Contains("i / 1024u", xy);
        Assert.Contains("262144u", ij);
        Assert.True(VulkanBackend.IsMeshgrid2DAdmitted(256, 256));
        Assert.True(VulkanBackend.IsMeshgrid2DAdmitted(1024, 256));
        Assert.False(VulkanBackend.IsMeshgrid2DAdmitted(256, 1024));
    }
}
