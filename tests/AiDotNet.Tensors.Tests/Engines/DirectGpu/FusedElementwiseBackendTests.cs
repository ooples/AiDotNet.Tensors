using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The fused element-wise members that used to be compositions of other backend members (TENS0008): AddRelu, AddSigmoid
/// and AddGelu against the reference activation of the sum, and Fma with its output aliasing the addend -- the in-place
/// accumulate the old Multiply-into-D-then-Add composition got wrong (2AB instead of AB + C). Runs on every backend this
/// machine has; each one absent is skipped by name.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedElementwiseBackendTests
{
    private const int Length = 4099;          // a float4 body and a scalar tail
    private const double RelativeTolerance = 2e-6;
    private const double AbsoluteTolerance = 1e-6;

    public static IEnumerable<object[]> Backends()
    {
        yield return new object[] { "cuda" };
        yield return new object[] { "hip" };
        yield return new object[] { "opencl" };
        yield return new object[] { "vulkan" };
        yield return new object[] { "metal" };
        yield return new object[] { "webgpu" };
    }

    private static float[] Values(int salt)
    {
        var rng = RandomHelper.CreateSeededRandom(1638 + salt);
        var data = new float[Length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 8 - 4);
        return data;
    }

    private static void AssertClose(double expected, float actual, int index, string op)
        => Assert.True(Math.Abs(expected - actual) <= AbsoluteTolerance + RelativeTolerance * Math.Abs(expected),
            $"{op}[{index}]: expected {expected}, got {actual}");

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void AddActivations_EqualTheActivatedSum(string name)
    {
        using var lease = Create(name, out var backend);
        Skip.If(backend is null, name + " is not available on this machine.");
        var a = Values(1);
        var b = Values(2);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(Length);

        var cases = new (string Op, Action Run, Func<double, double> Reference)[]
        {
            ("AddRelu", () => backend.AddRelu(bufA, bufB, bufC, Length), x => x > 0 ? x : 0),
            ("AddSigmoid", () => backend.AddSigmoid(bufA, bufB, bufC, Length), x => 1 / (1 + Math.Exp(-x))),
            ("AddGelu", () => backend.AddGelu(bufA, bufB, bufC, Length),
                x => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))),
        };
        foreach (var (op, run, reference) in cases)
        {
            run();
            var c = backend.DownloadBuffer(bufC);
            for (int i = 0; i < Length; i++) AssertClose(reference((double)(a[i] + b[i])), c[i], i, op);
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Fma_IntoItsAddend_IsAProductPlusTheAddend(string name)
    {
        using var lease = Create(name, out var backend);
        Skip.If(backend is null, name + " is not available on this machine.");
        var a = Values(3);
        var b = Values(4);
        var c = Values(5);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(c);

        backend.Fma(bufA, bufB, bufC, bufC, Length);   // D is C

        var d = backend.DownloadBuffer(bufC);
        for (int i = 0; i < Length; i++) AssertClose((double)a[i] * b[i] + c[i], d[i], i, "Fma");
    }

    // The backend, or null when unavailable; the returned disposable disposes a backend the test created.
    private static IDisposable? Create(string name, out IDirectGpuBackend? backend)
    {
        backend = null;
        try
        {
            IDirectGpuBackend candidate = name switch
            {
                "cuda" => new CudaBackend(),
                "hip" => new HipBackend(),
                "opencl" => new OpenClBackend(),
                "vulkan" => VulkanBackend.Instance,   // the process-wide instance; never disposed here
                "metal" => new MetalBackend(),
#if NET7_0_OR_GREATER
                "webgpu" => new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend(),
#endif
                _ => throw new PlatformNotSupportedException(name),
            };
            if (candidate is VulkanBackend vulkan && !vulkan.IsAvailable) vulkan.Initialize();
            if (!candidate.IsAvailable)
            {
                if (name != "vulkan") (candidate as IDisposable)?.Dispose();
                return null;
            }
            backend = candidate;
            return name == "vulkan" ? null : candidate as IDisposable;
        }
        catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
        {
            return null;   // no driver for this backend on this machine
        }
    }
}
