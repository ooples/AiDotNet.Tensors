// Copyright (c) AiDotNet. All rights reserved.
#if !NETFRAMEWORK
using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu
{
    /// <summary>
    /// The launch-diagnostics harness itself: it has to report the right thing, and it has to be
    /// usable by a caller validating their own kernels.
    /// </summary>
    /// <remarks>
    /// These assertions are deliberately about the DIAGNOSTIC, not about any particular kernel. A
    /// harness that silently stops reporting is worse than none, because every later investigation
    /// trusts it.
    /// </remarks>
    /// <remarks>
    /// SERIALIZED DELIBERATELY. The residency counters are process-static and every DirectOpenClBuffer
    /// construction anywhere in the assembly moves them, so a concurrent GPU test could change them
    /// between this class's snapshot and its assertion. "DirectGpuSerial" is declared with
    /// DisableParallelization = true, which means it does not run alongside other collections — so
    /// joining it removes the interference rather than merely narrowing the window, and the delta
    /// assertions can stay exact instead of being relaxed into something that no longer catches a
    /// real imbalance.
    /// </remarks>
    [Collection("DirectGpuSerial")]
    public class GpuKernelDiagnosticsTests
    {
        [Fact]
        public void ReleasedHandle_IsReportedByName_NotFaulted()
        {
            var thrown = Assert.Throws<ObjectDisposedException>(() =>
                GpuKernelDiagnostics.ValidateLaunch(
                    "my_custom_kernel",
                    handleIsValid: false,
                    stagedArgCount: 3,
                    declaredArgCount: 3,
                    globalSize: 256,
                    localSize: 64));

            Assert.Contains("my_custom_kernel", thrown.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void GlobalSizeNotAMultipleOfTheWorkGroup_IsRejected()
        {
            var thrown = Assert.Throws<ArgumentException>(() =>
                GpuKernelDiagnostics.ValidateLaunch(
                    "ragged", handleIsValid: true, stagedArgCount: 1, declaredArgCount: 1,
                    globalSize: 100, localSize: 64));

            Assert.Contains("not a multiple", thrown.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void WorkGroupLargerThanTheKernelAdmits_IsRejected()
        {
            var thrown = Assert.Throws<ArgumentException>(() =>
                GpuKernelDiagnostics.ValidateLaunch(
                    "narrow", handleIsValid: true, stagedArgCount: 1, declaredArgCount: 1,
                    globalSize: 512, localSize: 512, maxWorkGroupSize: 256));

            Assert.Contains("exceeds", thrown.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// The check that would have caught this session's reduction bug at the launch instead of in
        /// a wrong sum four layers away.
        /// </summary>
        [Theory]
        [InlineData(31)]
        [InlineData(33)]
        [InlineData(6)]
        public void NonPowerOfTwoWorkGroup_IsRejectedForKernelsThatRequireOne(int localSize)
        {
            var thrown = Assert.Throws<ArgumentException>(() =>
                GpuKernelDiagnostics.ValidateLaunch(
                    "reduce_sum", handleIsValid: true, stagedArgCount: 4, declaredArgCount: 4,
                    globalSize: localSize, localSize: localSize,
                    requiresPowerOfTwoWorkGroup: true));

            Assert.Contains("power of two", thrown.Message, StringComparison.Ordinal);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(64)]
        [InlineData(256)]
        public void PowerOfTwoWorkGroup_IsAccepted(int localSize)
        {
            GpuKernelDiagnostics.ValidateLaunch(
                "reduce_sum", handleIsValid: true, stagedArgCount: 4, declaredArgCount: 4,
                globalSize: localSize, localSize: localSize,
                requiresPowerOfTwoWorkGroup: true);
        }

        [Fact]
        public void IsPowerOfTwo_AgreesWithTheDefinition()
        {
            foreach (int n in new[] { 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024 })
                Assert.True(GpuKernelDiagnostics.IsPowerOfTwo(n), $"{n} is a power of two");

            foreach (int n in new[] { 0, 3, 5, 6, 7, 9, 31, 33, 255, 257, 1025 })
                Assert.False(GpuKernelDiagnostics.IsPowerOfTwo(n), $"{n} is not a power of two");

            Assert.False(GpuKernelDiagnostics.IsPowerOfTwo(-8), "negative sizes are not valid");
        }

        [Fact]
        public void TheJournalRecordsLaunchesAndSurvivesToDisk()
        {
            GpuKernelDiagnostics.RecordLaunch("journal_probe_alpha", 1024, 256, 3);
            GpuKernelDiagnostics.RecordLaunch("journal_probe_beta", 2048, 128, 5);

            var recent = GpuKernelDiagnostics.RecentLaunches();
            Assert.Contains(recent, line => line.Contains("journal_probe_beta", StringComparison.Ordinal));

            var described = GpuKernelDiagnostics.DescribeRecentLaunches();
            Assert.Contains("journal_probe_beta", described, StringComparison.Ordinal);

            // The point of the journal is that it outlives the process that produced it.
            string path = Path.Combine(Path.GetTempPath(), "aidotnet-gpu-journal-" + Guid.NewGuid().ToString("N") + ".txt");
            try
            {
                GpuKernelDiagnostics.DumpTo(path);
                Assert.True(File.Exists(path), "the journal must be writable to disk for post-mortem use");
                Assert.Contains("journal_probe_beta", File.ReadAllText(path), StringComparison.Ordinal);
            }
            finally
            {
                try { File.Delete(path); } catch (IOException) { }
            }
        }

        [Fact]
        public void DumpTo_NeverThrows_SoACrashHandlerCanCallIt()
        {
            // An unwritable path must not turn a diagnostic into a second failure on the way down.
            GpuKernelDiagnostics.DumpTo(Path.Combine("Z:", "definitely", "not", "writable.txt"));
            GpuKernelDiagnostics.DumpTo(string.Empty);
        }

        /// <summary>
        /// The dump must land even when its directory does not exist yet. Callers build the path from a
        /// results folder that a fresh CI runner has never created, and the process does not run with
        /// the workspace as its working directory. Before this was fixed the write threw
        /// DirectoryNotFoundException -- an IOException -- straight into the swallow-everything catch:
        /// the env var was set on every shard of a full matrix and not one file was produced.
        /// </summary>
        [Fact]
        public void DumpTo_CreatesTheMissingDirectory_ForAbsoluteAndRelativePaths()
        {
            string absoluteRoot = Path.Combine(Path.GetTempPath(), "aidotnet-dump-abs-" + Guid.NewGuid().ToString("N"));
            string relativeRoot = "aidotnet-dump-rel-" + Guid.NewGuid().ToString("N");
            try
            {
                string nested = Path.Combine(absoluteRoot, "results", "gpu", "gpu-diagnostics.txt");
                Assert.False(Directory.Exists(absoluteRoot), "precondition: the target directory must not exist yet");

                GpuKernelDiagnostics.DumpTo(nested);

                Assert.True(File.Exists(nested), "an absolute path into a directory that does not exist yet must still be written");
                Assert.Contains("# GPU launch journal", File.ReadAllText(nested), StringComparison.Ordinal);

                // A RELATIVE path resolves against the process working directory -- which is where a
                // test host's CI step will NOT find it, but that is the caller's problem (pass an
                // absolute path); this method's promise is that the file exists at the resolved location.
                string relative = Path.Combine(relativeRoot, "sub", "gpu-diagnostics.txt");
                string resolved = Path.GetFullPath(relative);

                GpuKernelDiagnostics.DumpTo(relative);

                Assert.True(File.Exists(resolved), $"a relative path must be written at its resolved location {resolved}");
            }
            finally
            {
                foreach (string root in new[] { absoluteRoot, Path.GetFullPath(relativeRoot) })
                {
                    try { if (Directory.Exists(root)) Directory.Delete(root, recursive: true); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
            }
        }

        [Fact]
        public void CapacityCheckIsOptIn_AndReportsTheOverrun()
        {
            if (!GpuKernelDiagnostics.DeepChecksEnabled)
            {
                // Off by default: it must be silent rather than throwing on a legitimate launch.
                GpuKernelDiagnostics.ValidateCapacity("k", "out", bufferElements: 4, addressedElements: 99);
                return;
            }

            var thrown = Assert.Throws<ArgumentException>(() =>
                GpuKernelDiagnostics.ValidateCapacity("k", "out", bufferElements: 4, addressedElements: 99));
            Assert.Contains("99", thrown.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// Buffer accounting has to balance, because its whole purpose is to make an IMBALANCE
        /// visible.
        /// </summary>
        /// <remarks>
        /// Deltas, not absolutes: the counters are process-static and every other GPU test in this
        /// assembly moves them, so asserting an absolute count would be asserting test execution
        /// order. What must hold is that an allocate/release pair leaves the live figures exactly
        /// where it found them.
        /// </remarks>
        [Fact]
        public void BufferAccounting_BalancesAcrossAllocateAndRelease()
        {
            long countBefore = GpuKernelDiagnostics.LiveBufferCount;
            long bytesBefore = GpuKernelDiagnostics.LiveBufferBytes;
            long totalBefore = GpuKernelDiagnostics.TotalBuffersAllocated;

            GpuKernelDiagnostics.RecordBufferAllocated(4096);
            GpuKernelDiagnostics.RecordBufferAllocated(1024);

            Assert.Equal(countBefore + 2, GpuKernelDiagnostics.LiveBufferCount);
            Assert.Equal(bytesBefore + 5120, GpuKernelDiagnostics.LiveBufferBytes);
            Assert.Equal(totalBefore + 2, GpuKernelDiagnostics.TotalBuffersAllocated);

            GpuKernelDiagnostics.RecordBufferReleased(4096);
            GpuKernelDiagnostics.RecordBufferReleased(1024);

            Assert.Equal(countBefore, GpuKernelDiagnostics.LiveBufferCount);
            Assert.Equal(bytesBefore, GpuKernelDiagnostics.LiveBufferBytes);

            // Released buffers must NOT decrement the cumulative total — the gap between "allocated
            // in total" and "live" is exactly the signal a leak hunt is looking for.
            Assert.Equal(totalBefore + 2, GpuKernelDiagnostics.TotalBuffersAllocated);
        }

        [Fact]
        public void PeakResidency_IsAHighWaterMark_AndDoesNotFallBack()
        {
            GpuKernelDiagnostics.RecordBufferAllocated(64 * 1024);
            long peakAtHeight = GpuKernelDiagnostics.PeakLiveBufferBytes;
            Assert.True(
                peakAtHeight >= GpuKernelDiagnostics.LiveBufferBytes,
                "peak must be at least the current live figure");

            GpuKernelDiagnostics.RecordBufferReleased(64 * 1024);

            Assert.Equal(peakAtHeight, GpuKernelDiagnostics.PeakLiveBufferBytes);
        }

        [Fact]
        public void ResidencyDescription_ReportsTheNumbers_AndReachesTheDump()
        {
            GpuKernelDiagnostics.RecordBufferAllocated(2048);
            try
            {
                Assert.Contains("live=", GpuKernelDiagnostics.DescribeBufferResidency(), StringComparison.Ordinal);

                // The residency has to travel with the journal, or a post-mortem never sees it.
                string path = Path.Combine(Path.GetTempPath(), "aidotnet-residency-" + Guid.NewGuid().ToString("N") + ".txt");
                try
                {
                    GpuKernelDiagnostics.DumpTo(path);
                    Assert.Contains("buffers:", File.ReadAllText(path), StringComparison.Ordinal);
                }
                finally
                {
                    try { File.Delete(path); } catch (IOException) { }
                }
            }
            finally
            {
                GpuKernelDiagnostics.RecordBufferReleased(2048);
            }
        }

        /// <summary>
        /// The diagnostics env vars are read ONCE, at type initialization. Setting them later must
        /// not appear to work.
        /// </summary>
        /// <remarks>
        /// This pins a contract rather than an implementation detail. Registration now happens at
        /// assembly load (module initializer), so by the time any caller could call
        /// SetEnvironmentVariable the values have already been latched. Without this test, a future
        /// reader could reasonably assume a mid-process toggle takes effect and spend a debugging
        /// session wondering why their flag did nothing.
        /// </remarks>
        [Fact]
        public void DiagnosticsEnvironmentVariables_AreLatchedAtInitialization_NotReadPerCall()
        {
            bool before = GpuKernelDiagnostics.DeepChecksEnabled;

            string? original = Environment.GetEnvironmentVariable("AIDOTNET_GPU_KERNEL_DIAGNOSTICS");
            try
            {
                // Flip it to whatever it currently is not.
                Environment.SetEnvironmentVariable(
                    "AIDOTNET_GPU_KERNEL_DIAGNOSTICS", before ? "0" : "1");

                Assert.Equal(before, GpuKernelDiagnostics.DeepChecksEnabled);
            }
            finally
            {
                Environment.SetEnvironmentVariable("AIDOTNET_GPU_KERNEL_DIAGNOSTICS", original);
            }
        }

        /// <summary>EnsureRegistered is called from a module initializer, so it must be safe to repeat.</summary>
        [Fact]
        public void EnsureRegistered_IsIdempotentAndDoesNotThrow()
        {
            // A throw here would propagate out of the module initializer and take down the AppDomain,
            // which is the hazard GpuAutoDetectModuleInit's own comments warn about.
            GpuKernelDiagnostics.EnsureRegistered();
            GpuKernelDiagnostics.EnsureRegistered();

            Assert.Contains("live=", GpuKernelDiagnostics.DescribeBufferResidency(), StringComparison.Ordinal);
        }

        /// <summary>A real launch must appear in the journal, which is what makes it useful after a crash.</summary>
        [SkippableFact]
        public void ARealGpuLaunchIsJournalled()
        {
            // Only a missing runtime counts as "no GPU"; a real initialisation failure on a
            // GPU-capable host must fail rather than skip.
            DirectGpuTensorEngine? engine;
            try
            {
                engine = new DirectGpuTensorEngine();
            }
            catch (DllNotFoundException) { engine = null; }
            catch (EntryPointNotFoundException) { engine = null; }
            catch (PlatformNotSupportedException) { engine = null; }

            Skip.If(engine is null, "No direct GPU backend is available on this host.");

            using (engine)
            {
                Skip.If(!engine!.IsGpuAvailable, "No direct GPU backend is available on this host.");
                var backend = engine.GetBackend();
                Skip.If(backend is null, "No direct GPU backend is available on this host.");

                // OpenCL, CUDA and HIP all journal now (CUDA/HIP via their native-launch choke
                // point). This still targets OpenCL because it is the backend this host actually
                // has, so the assertion below is exercised rather than skipped; Metal, Vulkan and
                // WebGpu remain unjournalled (Issue #996) and would fail here for a reason that has
                // nothing to do with the diagnostic being broken.
                Skip.If(
                    backend is not AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend,
                    "This test targets the OpenCL journalling path (CUDA and HIP journal via their native-launch "
                        + "choke point; Metal, Vulkan and WebGpu are not yet journalled - Issue #996). "
                        + $"This host selected {backend!.GetType().Name}.");

                // The journal is STATIC, so a non-empty journal proves nothing — another test in
                // this class fills it. What must be true is that this launch ADVANCES it.
                var before = GpuKernelDiagnostics.RecentLaunches();
                string? newestBefore = before.Count == 0 ? null : before[before.Count - 1];

                var values = Enumerable.Range(1, 64).Select(i => (float)i).ToArray();
                using var buffer = backend!.AllocateBuffer(values);
                backend.Sum(buffer, values.Length);

                var after = GpuKernelDiagnostics.RecentLaunches();
                Assert.NotEmpty(after);

                string newestAfter = after[after.Count - 1];
                Assert.True(
                    newestBefore is null || !string.Equals(newestAfter, newestBefore, StringComparison.Ordinal),
                    "backend.Sum did not add a journal entry: the newest entry is unchanged at "
                        + $"'{newestAfter}'. A launch that is not journalled is invisible after a crash.");
            }
        }
    }
}
#endif
