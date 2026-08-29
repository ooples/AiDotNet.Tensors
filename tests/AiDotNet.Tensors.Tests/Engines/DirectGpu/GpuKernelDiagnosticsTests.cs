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

        /// <summary>A real launch must appear in the journal, which is what makes it useful after a crash.</summary>
        [SkippableFact]
        public void ARealGpuLaunchIsJournalled()
        {
            DirectGpuTensorEngine? engine = null;
            try { engine = new DirectGpuTensorEngine(); }
            catch (Exception) { Skip.If(true, "No direct GPU backend is available on this host."); }

            using (engine)
            {
                Skip.If(engine is null || !engine.IsGpuAvailable, "No direct GPU backend is available on this host.");
                var backend = engine!.GetBackend();
                Skip.If(backend is null, "No direct GPU backend is available on this host.");

                var values = Enumerable.Range(1, 64).Select(i => (float)i).ToArray();
                using var buffer = backend!.AllocateBuffer(values);
                backend.Sum(buffer, values.Length);

                Assert.NotEmpty(GpuKernelDiagnostics.RecentLaunches());
            }
        }
    }
}
#endif
