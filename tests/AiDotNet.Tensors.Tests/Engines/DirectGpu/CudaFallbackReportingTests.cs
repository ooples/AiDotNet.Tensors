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
                var first = Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
                var second = Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
                // The cached failure, rethrown: a later read does not construct an engine and probe the device again.
                Assert.Same(first, second);
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

    /// <summary>
    /// The opt-out variable and a tripped circuit breaker used to switch to the CPU silently, even with a backend
    /// required: the substitution the requirement exists to forbid.
    /// </summary>
    [Fact]
    public void RequiredBackend_WithGpuDisabledOrTheCircuitOpen_Throws_InsteadOfSwitchingToTheCpu()
    {
        var saved = AiDotNetEngine.Current;
        var previousRequired = GpuBackendOptions.RequiredBackend;
        string? previousDisable = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU");
        try
        {
            GpuBackendOptions.RequiredBackend = GpuBackendType.Hip;
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
            var disabled = Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.Contains("AIDOTNET_DISABLE_GPU", disabled.Message);

            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", null);
            AiDotNetEngine.OpenGpuCircuitBreakerForTests();
            var broken = Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.Contains("circuit breaker", broken.Message);
        }
        finally
        {
            AiDotNetEngine.ResetGpuCircuitBreakerForTests();
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", previousDisable);
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.Current = saved;
        }
    }

    /// <summary>
    /// A parked engine on a backend the requirement rules out used to be overwritten by the new adoption and never
    /// released, holding its device context for the life of the process.
    /// </summary>
    [SkippableFact]
    public void An_adoption_that_supersedes_a_parked_engine_on_another_backend_releases_it()
    {
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU") is { Length: > 0 }, "GPU detection is disabled.");
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "No CUDA device to adopt.");
        var saved = AiDotNetEngine.Current;
        var previousRequired = GpuBackendOptions.RequiredBackend;
        DirectGpuTensorEngine? parked = null;
        try
        {
            GpuBackendOptions.RequiredBackend = GpuBackendType.None;
            var openCl = DirectGpuBackendFactory.CreateNew(GpuBackendPreference.OpenCL);
            Skip.If(openCl is null || !openCl.BackendName.StartsWith("OpenCL", StringComparison.Ordinal), "No OpenCL device to park.");
            parked = new DirectGpuTensorEngine(new DirectGpuEngine(openCl!));
            AiDotNetEngine.Current = new CpuEngine();
            AiDotNetEngine.ParkGpuForTests(parked);

            GpuBackendOptions.RequiredBackend = GpuBackendType.Cuda;
            Assert.True(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.True(AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var adopted));
            Assert.NotSame(parked, adopted);
            Assert.True(parked.IsDisposed, "the superseded OpenCL engine was left open");
        }
        finally
        {
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.Current = saved;
            if (parked is { IsDisposed: false }) parked.Dispose();
        }
    }

    /// <summary>
    /// A tripped circuit breaker publishes the CPU engine. With a backend required, every later read of Current used to
    /// hand that engine out; it now throws the trip.
    /// </summary>
    [Fact]
    public void A_tripped_circuit_breaker_under_a_required_backend_makes_Current_throw()
    {
        var saved = AiDotNetEngine.Current;
        var previousRequired = GpuBackendOptions.RequiredBackend;
        try
        {
            GpuBackendOptions.RequiredBackend = GpuBackendType.Hip;
            AiDotNetEngine.TripGpuCircuitBreaker("test", 700);
            var ex = Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
            Assert.Contains("circuit", ex.Message);
        }
        finally
        {
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.ResetGpuCircuitBreakerForTests();
            AiDotNetEngine.Current = saved;
        }
    }

    /// <summary>
    /// A superseded engine still running an op is not freed under it, and not dropped either: ReleaseGpu releases it
    /// once the op is done.
    /// </summary>
    [SkippableFact]
    public void A_superseded_engine_with_an_op_running_is_released_later_not_leaked()
    {
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU") is { Length: > 0 }, "GPU detection is disabled.");
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "No CUDA device to adopt.");
        var saved = AiDotNetEngine.Current;
        var previousRequired = GpuBackendOptions.RequiredBackend;
        DirectGpuTensorEngine? parked = null;
        try
        {
            GpuBackendOptions.RequiredBackend = GpuBackendType.None;
            var openCl = DirectGpuBackendFactory.CreateNew(GpuBackendPreference.OpenCL);
            Skip.If(openCl is null || !openCl.BackendName.StartsWith("OpenCL", StringComparison.Ordinal), "No OpenCL device to park.");
            parked = new DirectGpuTensorEngine(new DirectGpuEngine(openCl!));
            AiDotNetEngine.Current = new CpuEngine();
            AiDotNetEngine.ParkGpuForTests(parked);
            int pendingBefore = AiDotNetEngine.SupersededGpusPendingForTests;

            GpuBackendOptions.RequiredBackend = GpuBackendType.Cuda;
            using (parked.EnterGpuOp())
            {
                Assert.True(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
                Assert.False(parked.IsDisposed, "the superseded engine was freed under a running op");
                Assert.Equal(pendingBefore + 1, AiDotNetEngine.SupersededGpusPendingForTests);
            }
            AiDotNetEngine.ReleaseGpu();
            Assert.True(parked.IsDisposed, "the superseded engine was never released");
            Assert.Equal(pendingBefore, AiDotNetEngine.SupersededGpusPendingForTests);
        }
        finally
        {
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.Current = saved;
            if (parked is { IsDisposed: false }) parked.Dispose();
        }
    }
}
