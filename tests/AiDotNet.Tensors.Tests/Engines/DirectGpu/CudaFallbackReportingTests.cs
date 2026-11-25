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
