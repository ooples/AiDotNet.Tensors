// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GpuBackendSelectionGlobalStateCollection
{
    public const string Name = "GpuBackendSelectionGlobalState";
}

/// <summary>
/// A CUDA backend that cannot be created is never a quiet switch to another backend or to the CPU: detection records
/// why, warns on stderr, and throws when <see cref="GpuBackendOptions.RequiredBackend"/> asks for CUDA. CUDA
/// initialization failure is simulated, so these run on any machine.
/// </summary>
[Collection(GpuBackendSelectionGlobalStateCollection.Name)]
public class CudaFallbackReportingTests
{
    private const string Simulated = "simulated CUDA init failure";

    private static IDirectGpuBackend Fail() => throw new InvalidOperationException(Simulated);

    private static string CaptureStderr(Action run)
    {
        var previous = Console.Error;
        var writer = new StringWriter();
        Console.SetError(writer);
        try { run(); }
        finally { Console.SetError(previous); }
        return writer.ToString();
    }

    private static void WithCudaFailing(GpuBackendType required, Action run)
    {
        var previousRequired = GpuBackendOptions.RequiredBackend;
        DirectGpuEngine.CudaBackendFactoryForTests = Fail;
        DirectGpuBackendFactory.CudaBackendFactoryForTests = _ => Fail();
        GpuBackendOptions.RequiredBackend = required;
        try { run(); }
        finally
        {
            GpuBackendOptions.RequiredBackend = previousRequired;
            DirectGpuEngine.CudaBackendFactoryForTests = null;
            DirectGpuBackendFactory.CudaBackendFactoryForTests = null;
        }
    }

    [SkippableFact]
    public void Engine_RecordsTheCudaFailure_AndWarnsWhereverItLands()
    {
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS") is { Length: > 0 },
            "A backend order from the environment may leave CUDA out.");
        WithCudaFailing(GpuBackendType.None, () =>
        {
            DirectGpuEngine? engine = null;
            string stderr = CaptureStderr(() => engine = new DirectGpuEngine());
            using (engine)
            {
                Assert.NotNull(engine!.CudaInitializationFailure);
                Assert.Contains(Simulated, engine.CudaInitializationFailure);
                Assert.NotEqual("CUDA", engine.BackendName, StringComparer.OrdinalIgnoreCase);
                Assert.Contains("CUDA backend unavailable", stderr);
                Assert.Contains(Simulated, stderr);
            }
        });
    }

    /// <summary>
    /// A host without the CUDA driver (AMD, Intel, no GPU) has nothing to warn about: the CUDA warning is for a CUDA
    /// that is installed and fails. Before, every engine on such a host printed it.
    /// </summary>
    [SkippableFact]
    public void Engine_OnAHostWithoutTheCudaDriver_DoesNotWarnAboutCuda()
    {
        Skip.If(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "The CUDA driver is installed here.");
        DirectGpuEngine? engine = null;
        string stderr = CaptureStderr(() => engine = new DirectGpuEngine());
        using (engine)
            Assert.DoesNotContain("CUDA backend unavailable", stderr);
    }

    [Fact]
    public void Engine_RequiredCuda_ThrowsWithTheReason_InsteadOfFallingBack()
    {
        WithCudaFailing(GpuBackendType.Cuda, () =>
        {
            var ex = Assert.Throws<InvalidOperationException>(() => new DirectGpuEngine());
            Assert.Contains(Simulated, ex.Message);
            Assert.Contains(nameof(GpuBackendOptions.RequiredBackend), ex.Message);
        });
    }

    /// <summary>
    /// A required backend that fails the owed startup detection must keep failing: before, the detection's finally
    /// marked it done, so the read after the throw silently returned the CPU engine the requirement ruled out. A GPU
    /// engine adopted earlier on another backend (CUDA, on the CUDA lane) must not be reinstated in its place either.
    /// </summary>
    [SkippableFact]
    public void DeferredDetection_RequiredBackendFailing_ThrowsOnEveryRead_NotJustTheFirst()
    {
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU") is { Length: > 0 }, "GPU detection is disabled.");
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            // HIP: absent on the CPU and CUDA lanes, so the required backend fails without a fault hook.
            var previousRequired = GpuBackendOptions.RequiredBackend;
            GpuBackendOptions.RequiredBackend = GpuBackendType.Hip;
            try
            {
                AiDotNetEngine.DeferGpuAutoDetection(verbose: false);
                Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
                Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
                Assert.True(AiDotNetEngine.IsGpuAutoDetectionDeferred);
            }
            finally { GpuBackendOptions.RequiredBackend = previousRequired; }
        }
        finally { AiDotNetEngine.Current = saved; }
        Assert.False(AiDotNetEngine.IsGpuAutoDetectionDeferred);
    }

    /// <summary>
    /// The factory enforces the requirement on every route, not only the NVIDIA one: a contradicting preference is
    /// refused, and a required backend that cannot be created throws instead of returning another backend or null.
    /// </summary>
    [Fact]
    public void Factory_RequiredCuda_ThrowsOnEveryRoute_InsteadOfAnotherBackendOrNull()
    {
        WithCudaFailing(GpuBackendType.Cuda, () =>
        {
            Assert.Throws<ArgumentException>(() => DirectGpuBackendFactory.CreateNew(GpuBackendPreference.OpenCL));
            var ex = Assert.Throws<InvalidOperationException>(() => DirectGpuBackendFactory.CreateNew());
            Assert.Contains(nameof(GpuBackendOptions.RequiredBackend), ex.Message);
        });
    }

    [Fact]
    public void Factory_NvidiaBackend_WarnsOnTheOpenClFallback_AndThrowsWhenCudaIsRequired()
    {
        WithCudaFailing(GpuBackendType.None, () =>
        {
            IDirectGpuBackend? backend = null;
            string stderr = CaptureStderr(() => backend = DirectGpuBackendFactory.CreateNvidiaBackend(0, logger: null));
            (backend as IDisposable)?.Dispose();
            Assert.Contains("CUDA backend unavailable", stderr);
            Assert.Contains(Simulated, stderr);
        });
        WithCudaFailing(GpuBackendType.Cuda, () =>
        {
            var ex = Assert.Throws<InvalidOperationException>(() => DirectGpuBackendFactory.CreateNvidiaBackend(0, logger: null));
            Assert.Contains(Simulated, ex.Message);
        });
    }

    [Theory]
    [InlineData(GpuBackendType.Metal)]
    [InlineData(GpuBackendType.Unknown)]
    public void RequiredBackend_RejectsBackendsDetectionCannotSelect(GpuBackendType backend)
        => Assert.Throws<ArgumentOutOfRangeException>(() => GpuBackendOptions.RequiredBackend = backend);
}
