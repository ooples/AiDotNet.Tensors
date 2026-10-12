using System;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The Vulkan pipeline lookup every dispatch pays finds a built pipeline by the source string's identity, then by its
/// text, and only then hashes it. Whichever way it is found, one source with one layout must map to one pipeline, and
/// a different binding count or push-constant size must not share it.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanPipelineLookupTests
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

    private static Func<string, int, uint, VulkanComputePipeline?> Lookup(VulkanBackend backend)
    {
        var method = typeof(VulkanBackend).GetMethod("GetOrCreateGlslPipeline", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("GetOrCreateGlslPipeline not found");
        return (Func<string, int, uint, VulkanComputePipeline?>)Delegate.CreateDelegate(
            typeof(Func<string, int, uint, VulkanComputePipeline?>), backend, method);
    }

    private const string Source = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; uint marker; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] + 1151.0;
}";

    [SkippableFact]
    public void OneSource_FoundByInstanceOrByText_IsOnePipeline()
    {
        var lookup = Lookup(BackendOrSkip());
        var first = lookup(Source, 2, 8);
        Assert.NotNull(first);
        Assert.Same(first, lookup(Source, 2, 8));

        // The same text in a different string object, as a source built at run time is.
        string rebuilt = new string(Source.ToCharArray());
        Assert.False(ReferenceEquals(Source, rebuilt));
        Assert.Same(first, lookup(rebuilt, 2, 8));
        Assert.Same(first, lookup(new string(Source.ToCharArray()), 2, 8));
    }

    [SkippableFact]
    public void ADifferentLayout_IsADifferentPipeline()
    {
        var lookup = Lookup(BackendOrSkip());
        var eightBytes = lookup(Source, 2, 8);
        var sixteenBytes = lookup(Source, 2, 16);
        Assert.NotNull(eightBytes);
        Assert.NotNull(sixteenBytes);
        Assert.NotSame(eightBytes, sixteenBytes);
        Assert.Same(sixteenBytes, lookup(Source, 2, 16));
        Assert.Same(eightBytes, lookup(Source, 2, 8));
    }

    [SkippableFact]
    public void AFoundPipeline_StillComputes()
    {
        var backend = BackendOrSkip();
        const int rows = 3, n = 5;
        var x = new float[rows * n];
        for (int i = 0; i < x.Length; i++) x[i] = i * 0.5f;
        using var input = backend.AllocateBuffer(x);
        using var output = backend.AllocateBuffer(x.Length);
        // Three calls: built, found by text or instance, found again.
        for (int pass = 0; pass < 3; pass++)
        {
            backend.Fill(output, float.NaN, x.Length);
            backend.Softmax(input, output, rows, n);
            float[] y = backend.DownloadBuffer(output);
            for (int r = 0; r < rows; r++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++) sum += y[r * n + i];
                Assert.True(Math.Abs(sum - 1.0) < 1e-5, $"pass {pass} row {r} sums to {sum}");
            }
        }
    }
}
