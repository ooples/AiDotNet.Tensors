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
        // IsAvailable is false on a host whose driver loads but hides or fails every device (CUDA_VISIBLE_DEVICES=-1);
        // that host reports the failure, so only a driver library that does not load at all is "without the driver".
        Skip.If(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsDriverLibraryPresent,
            "The CUDA driver is installed here.");
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
        var savedSlot = AiDotNetEngine.SnapshotCurrentForTests();
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
            AiDotNetEngine.RestoreCurrentForTests(savedSlot);
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
        DirectGpuTensorEngine? parked = null, previousParked = null;
        try
        {
            GpuBackendOptions.RequiredBackend = GpuBackendType.None;
            var openCl = DirectGpuBackendFactory.CreateNew(GpuBackendPreference.OpenCL);
            Skip.If(openCl is null || !openCl.BackendName.StartsWith("OpenCL", StringComparison.Ordinal), "No OpenCL device to park.");
            parked = new DirectGpuTensorEngine(new DirectGpuEngine(openCl!));
            AiDotNetEngine.Current = new CpuEngine();
            previousParked = AiDotNetEngine.ParkGpuForTests(parked);

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
            // The engine this adoption parked is the test's: disposed, not left holding a context.
            var adoptedHere = AiDotNetEngine.ParkGpuForTests(previousParked);
            if (adoptedHere is not null && !ReferenceEquals(adoptedHere, previousParked) && !ReferenceEquals(adoptedHere, saved)
                && !ReferenceEquals(adoptedHere, parked))
                adoptedHere.Dispose();
            if (parked is { IsDisposed: false }) parked.Dispose();
        }
    }

    private sealed class FakeBuffer : AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer
    {
        private readonly bool _throws;
        public FakeBuffer(bool throws) => _throws = throws;
        public bool Disposed { get; private set; }
        public int Size => 1;
        public long SizeInBytes => 4;
        public GpuElementType ElementType => GpuElementType.Float32;
        public IntPtr Handle => IntPtr.Zero;
        public void Dispose()
        {
            Disposed = true;
            if (_throws) throw new InvalidOperationException("release failed");
        }
    }

    /// <summary>
    /// Releasing host-mask copies used to stop the engine's disposal at the first failure: the later stages (retired
    /// weight buffers, the caches, the backend itself) were skipped and a retry returned at once, so the context
    /// leaked. Every copy is now tried, every later stage runs, and the failures are reported together.
    /// </summary>
    [SkippableFact]
    public void A_failing_host_mask_release_neither_stops_the_other_copies_nor_the_rest_of_the_disposal()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is not OutOfMemoryException) { Skip.If(true, "The GPU engine cannot be constructed: " + ex.Message); return; }
        var failing1 = new FakeBuffer(throws: true);
        var failing2 = new FakeBuffer(throws: true);
        var fine = new FakeBuffer(throws: false);
        var retired = new FakeBuffer(throws: false);
        var copies = new[] { gpu.AddHostMaskDeviceCopyForTests(failing1), gpu.AddHostMaskDeviceCopyForTests(fine),
            gpu.AddHostMaskDeviceCopyForTests(failing2) };
        gpu.AddRetiredWeightBufferForTests(retired);

        var thrown = Record.Exception(() => gpu.Dispose());
        GC.KeepAlive(copies);

        var aggregate = Assert.IsType<AggregateException>(thrown);
        Assert.Equal(2, aggregate.InnerExceptions.Count);
        Assert.True(failing1.Disposed && fine.Disposed && failing2.Disposed, "a host-mask copy was not tried");
        Assert.Equal(0, gpu.LiveHostMaskDeviceCopiesForTests);
        Assert.True(retired.Disposed, "the disposal stopped at the host-mask stage: a later stage never ran");
    }

    [Theory]
    [InlineData("CUDA", true)]
    [InlineData("cuda", true)]
    [InlineData("CUDA-compatible", false)]
    [InlineData("HIP (gfx1100)", false)]
    [InlineData("OpenCL", false)]
    [InlineData(null, false)]
    public void Only_the_CUDA_backend_name_counts_as_CUDA(string? name, bool cuda) =>
        Assert.Equal(cuda, AiDotNetEngine.IsCudaBackendName(name));

    /// <summary>
    /// After a trip every detection returns at the breaker; it still releases a superseded engine on another backend (a
    /// CUDA fault does not affect it), while a CUDA engine stays listed and open on the corrupted context.
    /// </summary>
    [SkippableFact]
    public void After_a_trip_superseded_engines_of_other_backends_are_released_and_CUDA_ones_kept()
    {
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU") is { Length: > 0 }, "GPU detection is disabled.");
        var savedSlot = AiDotNetEngine.SnapshotCurrentForTests();
        var previousRequired = GpuBackendOptions.RequiredBackend;
        IDirectGpuBackend? openCl = null, cuda = null;
        DirectGpuTensorEngine? other = null, onCuda = null;
        try
        {
            openCl = DirectGpuBackendFactory.CreateNew(GpuBackendPreference.OpenCL);
            Skip.If(openCl is null || !openCl.BackendName.StartsWith("OpenCL", StringComparison.Ordinal), "No OpenCL device.");
            cuda = DirectGpuBackendFactory.CreateNew(GpuBackendPreference.CUDA);
            other = new DirectGpuTensorEngine(new DirectGpuEngine(openCl!));
            if (cuda is not null && AiDotNetEngine.IsCudaBackendName(cuda.BackendName))
                onCuda = new DirectGpuTensorEngine(new DirectGpuEngine(cuda));
            GpuBackendOptions.RequiredBackend = GpuBackendType.None;

            AiDotNetEngine.OpenGpuCircuitBreakerForTests();
            // Parked with the breaker open: a parked CUDA engine is kept, the OpenCL one waits for a release pass.
            if (onCuda is not null) AiDotNetEngine.ReleaseSupersededGpuForTests(onCuda);
            AiDotNetEngine.ParkSupersededForTests(other);
            Assert.False(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.True(other.IsDisposed, "a superseded OpenCL engine was kept open after a CUDA trip");
            Assert.False(AiDotNetEngine.IsSupersededForTests(other));
            if (onCuda is not null)
            {
                Assert.False(onCuda.IsDisposed, "a CUDA engine was torn down on the corrupted context");
                Assert.True(AiDotNetEngine.IsSupersededForTests(onCuda));
            }
        }
        finally
        {
            AiDotNetEngine.ResetGpuCircuitBreakerForTests();
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.RestoreCurrentForTests(savedSlot);
            if (onCuda is not null) { AiDotNetEngine.ForgetSupersededForTests(onCuda); onCuda.Dispose(); }
            if (other is not null) { AiDotNetEngine.ForgetSupersededForTests(other); if (!other.IsDisposed) other.Dispose(); }
            // The engines borrow these backends: they are the test's to free.
            openCl?.Dispose();
            cuda?.Dispose();
        }
    }

    /// <summary>
    /// The trip runs inside CUDA error paths that hold a context lock, while an adoption can dispose an engine (which
    /// takes that context lock) under the adoption lock. The trip used to take the adoption lock, so the two could
    /// deadlock; it now takes no lock and completes while another thread holds the adoption lock.
    /// </summary>
    [Fact]
    public void The_circuit_breaker_trip_does_not_wait_for_the_adoption_lock()
    {
        var saved = AiDotNetEngine.Current;
        var savedSlot = AiDotNetEngine.SnapshotCurrentForTests();
        try
        {
            bool tripped = false;
            AiDotNetEngine.RunUnderAdoptionLockForTests(() =>
            {
                var trip = new System.Threading.Thread(() => AiDotNetEngine.TripGpuCircuitBreaker("test", 700));
                trip.IsBackground = true;
                trip.Start();
                tripped = trip.Join(TimeSpan.FromSeconds(10));
            });
            Assert.True(tripped, "the trip blocked on the adoption lock");
            Assert.True(AiDotNetEngine.GpuCircuitBroken);
        }
        finally
        {
            AiDotNetEngine.ResetGpuCircuitBreakerForTests();
            // Restored with its flag: later tests must not inherit a caller-assigned flag from the restore.
            AiDotNetEngine.RestoreCurrentForTests(savedSlot);
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
        var savedSlot = AiDotNetEngine.SnapshotCurrentForTests();
        var previousRequired = GpuBackendOptions.RequiredBackend;
        try
        {
            // Nothing caller-assigned left over from an earlier test: a caller's own engine is exempt from the refusal.
            AiDotNetEngine.ResetToCpu();
            Assert.False(AiDotNetEngine.IsCallerAssigned);
            GpuBackendOptions.RequiredBackend = GpuBackendType.Hip;
            AiDotNetEngine.TripGpuCircuitBreaker("test", 700);
            var ex = Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
            Assert.Contains("circuit", ex.Message);
            // An explicit detection call refuses as well, and does not clear the refusal for later reads.
            Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
            // Without a requirement the CPU engine is handed out; requiring a backend after the trip refuses again.
            GpuBackendOptions.RequiredBackend = GpuBackendType.None;
            Assert.IsType<CpuEngine>(AiDotNetEngine.Current);
            GpuBackendOptions.RequiredBackend = GpuBackendType.Cuda;
            Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
        }
        finally
        {
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.ResetGpuCircuitBreakerForTests();
            // Restored with its flag: later tests must not inherit a caller-assigned flag from the restore.
            AiDotNetEngine.RestoreCurrentForTests(savedSlot);
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
        DirectGpuTensorEngine? parked = null, previousParked = null;
        try
        {
            GpuBackendOptions.RequiredBackend = GpuBackendType.None;
            var openCl = DirectGpuBackendFactory.CreateNew(GpuBackendPreference.OpenCL);
            Skip.If(openCl is null || !openCl.BackendName.StartsWith("OpenCL", StringComparison.Ordinal), "No OpenCL device to park.");
            parked = new DirectGpuTensorEngine(new DirectGpuEngine(openCl!));
            AiDotNetEngine.Current = new CpuEngine();
            previousParked = AiDotNetEngine.ParkGpuForTests(parked);
            int pendingBefore = AiDotNetEngine.SupersededGpusPendingForTests;

            // The op runs on another thread, as another caller's op would, until the adoption has happened.
            GpuBackendOptions.RequiredBackend = GpuBackendType.Cuda;
            using var entered = new System.Threading.ManualResetEventSlim();
            using var finish = new System.Threading.ManualResetEventSlim();
            var engineForOp = parked;
            var op = new System.Threading.Thread(() =>
            {
                using (engineForOp.EnterGpuOp()) { entered.Set(); finish.Wait(); }
            });
            op.Start();
            entered.Wait();
            try
            {
                Assert.True(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
                Assert.False(parked.IsDisposed, "the superseded engine was freed under a running op");
                Assert.Equal(pendingBefore + 1, AiDotNetEngine.SupersededGpusPendingForTests);
            }
            finally
            {
                finish.Set();
                op.Join();
            }
            AiDotNetEngine.ReleaseGpu();
            Assert.True(parked.IsDisposed, "the superseded engine was never released");
            Assert.Equal(pendingBefore, AiDotNetEngine.SupersededGpusPendingForTests);
        }
        finally
        {
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.Current = saved;
            AiDotNetEngine.ParkGpuForTests(previousParked);
            if (parked is { IsDisposed: false }) parked.Dispose();
        }
    }

#if NET6_0_OR_GREATER
    // The module initializer (GpuAutoDetectModuleInit) exists only on .NET 6 and later, and it is the only code that
    // defers the GPU detection. On net462/net471 nothing defers it, so the behaviour tested here does not exist there.
    /// <summary>
    /// With AIDOTNET_DISABLE_GPU set, the module initializer used to skip the detection altogether, so a backend
    /// required afterwards was never checked and Current handed out the CPU engine. The detection is now owed either way.
    /// </summary>
    [Fact]
    public void Module_init_with_the_gpu_disabled_still_enforces_a_backend_required_later()
    {
        var saved = AiDotNetEngine.Current;
        var previousRequired = GpuBackendOptions.RequiredBackend;
        string? previousDisable = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
            GpuAutoDetectModuleInit.AutoDetectGpuIfNotOptedOut();
            Assert.True(AiDotNetEngine.IsGpuAutoDetectionDeferred);
            GpuBackendOptions.RequiredBackend = GpuBackendType.Hip;
            Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.Current);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", previousDisable);
            GpuBackendOptions.RequiredBackend = previousRequired;
            AiDotNetEngine.Current = saved;
        }
    }
#endif
}
