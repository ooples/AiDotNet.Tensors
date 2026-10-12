#if NET6_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IMultiTensorGpuOptimizerBackend"/> on every backend (#1142): one multi-tensor step over tensors of
/// different sizes must leave every parameter and moment exactly where the per-tensor Adam/AdamW step leaves it.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MultiTensorOptimizerBackendParityTests
{
    private static readonly int[] Sizes = { 1, 300, 1025, 7 };

    private static float[] Values(int n, int salt)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((i * 7919 + salt * 31) % 61 - 30) * 0.03125f;
        return v;
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var multi = Assert.IsAssignableFrom<IMultiTensorGpuOptimizerBackend>(backend);
        foreach (bool decoupled in new[] { false, true })
        {
            var param = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t))).ToArray();
            var grad = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t + 10))).ToArray();
            var first = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t + 20))).ToArray();
            var second = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t + 30).Select(Math.Abs).ToArray())).ToArray();
            var refParam = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t))).ToArray();
            var refFirst = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t + 20))).ToArray();
            var refSecond = Sizes.Select((n, t) => backend.AllocateBuffer(Values(n, t + 30).Select(Math.Abs).ToArray())).ToArray();
            try
            {
                if (decoupled)
                    multi.AdamWMultiTensorUpdate(param, grad, first, second, Sizes, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 3);
                else
                    multi.AdamMultiTensorUpdate(param, grad, first, second, Sizes, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 3);
                for (int t = 0; t < Sizes.Length; t++)
                {
                    if (decoupled)
                        backend.AdamWUpdate(refParam[t], grad[t], refFirst[t], refSecond[t], 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 3, Sizes[t]);
                    else
                        backend.AdamUpdate(refParam[t], grad[t], refFirst[t], refSecond[t], 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 3, Sizes[t]);
                    var pairs = new[] { (param[t], refParam[t], "param"), (first[t], refFirst[t], "m"), (second[t], refSecond[t], "v") };
                    foreach (var (actual, expected, what) in pairs)
                    {
                        var a = backend.DownloadBuffer(actual);
                        var e = backend.DownloadBuffer(expected);
                        for (int i = 0; i < Sizes[t]; i++)
                            Assert.True(Math.Abs(a[i] - e[i]) <= 1e-6f * (1 + Math.Abs(e[i])),
                                $"{(decoupled ? "AdamW" : "Adam")} tensor {t} {what}[{i}]: multi {a[i]}, per-tensor {e[i]}");
                    }
                }
            }
            finally
            {
                foreach (var buffer in param.Concat(grad).Concat(first).Concat(second).Concat(refParam).Concat(refFirst).Concat(refSecond))
                    buffer.Dispose();
            }
        }
    }

    [SkippableFact]
    public void OpenCl_MultiTensorAdam_MatchesPerTensor()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_MultiTensorAdam_MatchesPerTensor()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_MultiTensorAdam_MatchesPerTensor()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_MultiTensorAdam_MatchesPerTensor()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
