// Copyright (c) AiDotNet. All rights reserved.
#if NET5_0_OR_GREATER
// The N-tail direct kernels are AVX2/FMA intrinsics, which SimdGemm declares inside
// #if NET5_0_OR_GREATER; SgemmDirectParallelMInto does not exist on net471 at all.
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// Proves the N-tail kernels do not read past B, by making the byte after B unreadable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="SgemmDirectNTailBoundsTests"/> can only show that masking the B loads leaves the
    /// arithmetic unchanged. It cannot show the over-read is gone: the extra lanes were always
    /// discarded by the masked store, so a kernel that reads 60 bytes past its operand and one that
    /// does not produce identical output. A managed array does not help either -- `new float[k * n]`
    /// has no trailing logical elements, but the bytes after it are ordinary readable heap, so the
    /// old unmasked load completed and every assertion passed.
    /// </para>
    /// <para>
    /// So B is placed in raw pages here, positioned so its last element ends exactly at a page
    /// boundary, with the next page mapped NO-ACCESS. Any read past the operand hits that page and
    /// traps. With the loads masked the kernel never touches it and the run completes.
    /// </para>
    /// <para>
    /// A trap is an <c>AccessViolationException</c>, which .NET Core treats as a corrupted state
    /// exception: it cannot be caught, and it takes the whole test host with it. Running the probe
    /// in-process would therefore turn a regression into an aborted run with no attribution -- which
    /// is precisely how the original bug presented in CI. The probe is a separate test that stays
    /// skipped unless <c>AIDOTNET_SGEMM_GUARD_PROBE</c> is set, and
    /// <see cref="OverReadPastB_Traps_OnGuardPage"/> runs it in a child process and reads its exit
    /// code, so a regression is a failing test rather than a dead run.
    /// </para>
    /// </remarks>
    public class SgemmDirectGuardPageTests
    {
        private const string ProbeEnvVar = "AIDOTNET_SGEMM_GUARD_PROBE";

        /// <summary>Where the probe records that it finished.</summary>
        /// <remarks>
        /// A file, not stdout. <c>dotnet test</c> captures a test's console output and reports it
        /// through the logger rather than forwarding it to the parent's stdout, so a marker written
        /// with <c>Console.WriteLine</c> never arrives and the parent cannot tell "the probe ran and
        /// survived" from "the probe never ran". A file the child creates is visible either way.
        /// </remarks>
        private const string MarkerPathEnvVar = "AIDOTNET_SGEMM_GUARD_MARKER";

        private const string ProbeTestName =
            "AiDotNet.Tensors.Tests.Engines.SgemmDirectGuardPageTests.GuardPageProbe_RunsInChildProcess";
        private const string CompletedMarker = "AIDOTNET-GUARD-PROBE-COMPLETED";

        /// <summary>The kernels under test are AVX2 + FMA; elsewhere there is nothing to prove.</summary>
        private static bool KernelsAreLive =>
#if NET5_0_OR_GREATER
            Avx2.IsSupported && Fma.IsSupported;
#else
            false;
#endif

        [Fact]
        public void OverReadPastB_Traps_OnGuardPage()
        {
            if (!KernelsAreLive)
            {
                return;   // no AVX2/FMA path to exercise
            }

            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                && !RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return;   // guard pages are mapped per-OS below; only these two are wired up
            }

            var project = FindTestProject();
            if (project is null)
            {
                return;   // running from a layout where the project file is not locatable
            }

            var psi = new ProcessStartInfo("dotnet")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            psi.ArgumentList.Add("test");
            psi.ArgumentList.Add(project);
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(Configuration);
            psi.ArgumentList.Add("-f");
            psi.ArgumentList.Add(TargetFrameworkMoniker);
            psi.ArgumentList.Add("--no-build");
            psi.ArgumentList.Add("--nologo");
            psi.ArgumentList.Add("--filter");
            psi.ArgumentList.Add("FullyQualifiedName=" + ProbeTestName);

            string markerPath = Path.Combine(
                Path.GetTempPath(),
                "aidotnet-sgemm-guard-" + Guid.NewGuid().ToString("N") + ".marker");
            psi.Environment[ProbeEnvVar] = "1";
            psi.Environment[MarkerPathEnvVar] = markerPath;

            var stdout = new StringBuilder();
            using var child = new Process { StartInfo = psi };
            child.OutputDataReceived += (_, e) => { if (e.Data is not null) stdout.AppendLine(e.Data); };
            child.ErrorDataReceived += (_, e) => { if (e.Data is not null) stdout.AppendLine(e.Data); };
            child.Start();
            child.BeginOutputReadLine();
            child.BeginErrorReadLine();

            if (!child.WaitForExit(milliseconds: 10 * 60 * 1000))
            {
                try { child.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                Assert.Fail("The guard-page probe did not finish within ten minutes.");
            }

            child.WaitForExit();   // let the async output handlers drain
            var log = stdout.ToString();

            string marker = File.Exists(markerPath) ? File.ReadAllText(markerPath) : string.Empty;
            try { File.Delete(markerPath); } catch (IOException) { }

            // A child that never reached the probe says nothing about the kernels, and must not be
            // read as a pass. The marker is written by the probe itself, after the GEMMs return.
            Assert.True(
                marker.Contains(CompletedMarker, StringComparison.Ordinal),
                "The guard-page probe did not report completion, so the kernels were never "
                    + $"exercised. Child exit code {child.ExitCode}. Output:\n{log}");

            Assert.True(
                child.ExitCode == 0,
                "The guard-page probe process died. A read past the end of B landed on the "
                    + "no-access page, which is the over-read this fix removes -- the N-tail kernels "
                    + "must load B through their column masks, not at full 16-wide width. Child "
                    + $"exit code {child.ExitCode}. Output:\n{log}");
        }

        /// <summary>
        /// The probe itself. Never runs in a normal test pass; see the class remarks.
        /// </summary>
        [Fact]
        public void GuardPageProbe_RunsInChildProcess()
        {
            if (Environment.GetEnvironmentVariable(ProbeEnvVar) is null || !KernelsAreLive)
            {
                return;
            }

            // n % 16 == 1: the last column block is a one-wide tail, so the old code read 15 floats
            // past the final row. Routed through SgemmDirectParallelMInto, which passes
            // clearedOutput: true and so lands in DirectKernelMxNMaskedStore.
            RunOnGuardedB(m: 13, k: 9, n: 17, accumulate: false);

            // n % 16 == 8: the only tail width the accumulate path admits (SgemmAddInternal takes
            // the SgemmDirect branch only when n % 8 == 0), and the worst case -- lane 1's load lay
            // entirely past the row. Lands in DirectKernelMxNMasked.
            RunOnGuardedB(m: 13, k: 9, n: 24, accumulate: true);

            // Written only if both GEMMs returned. A fault takes the process down before this line.
            string? markerPath = Environment.GetEnvironmentVariable(MarkerPathEnvVar);
            if (markerPath is not null && markerPath.Length > 0)
            {
                File.WriteAllText(markerPath, CompletedMarker);
            }
        }

        private static unsafe void RunOnGuardedB(int m, int k, int n, bool accumulate)
        {
            int pageSize = Environment.SystemPageSize;
            long operandBytes = (long)k * n * sizeof(float);

            // Round the operand up to whole pages, then add one page to poison. B is placed at the
            // END of the readable region, so its last element is the last readable byte and the
            // very next byte faults.
            long readableBytes = ((operandBytes + pageSize - 1) / pageSize) * pageSize;
            long totalBytes = readableBytes + pageSize;

            IntPtr region = Reserve(totalBytes);
            try
            {
                Poison(region + (IntPtr)readableBytes, pageSize);

                float* b = (float*)(region + (IntPtr)(readableBytes - operandBytes));
                var rng = new Random(4242);
                for (long i = 0; i < (long)k * n; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

                var a = new float[m * k];
                for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

                var c = new float[m * n];
                var bSpan = new ReadOnlySpan<float>(b, k * n);

                if (accumulate)
                {
                    SimdGemm.SgemmAdd(a, bSpan, c, m, k, n);
                }
                else
                {
                    SimdGemm.SgemmDirectParallelMInto(a, bSpan, c, m, k, n);
                }

                // Consume the result so nothing above can be optimized away.
                double sum = 0;
                for (int i = 0; i < c.Length; i++) sum += c[i];
                if (double.IsNaN(sum))
                {
                    throw new InvalidOperationException("guarded GEMM produced NaN");
                }
            }
            finally
            {
                Release(region, totalBytes);
            }
        }

        private static string Configuration =>
#if DEBUG
            "Debug";
#else
            "Release";
#endif

        private static string TargetFrameworkMoniker =>
#if NET10_0_OR_GREATER
            "net10.0";
#elif NET8_0_OR_GREATER
            "net8.0";
#else
            "net471";
#endif

        private static string? FindTestProject()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null)
            {
                string candidate = Path.Combine(dir.FullName, "AiDotNet.Tensors.Tests.csproj");
                if (File.Exists(candidate))
                {
                    return candidate;
                }

                dir = dir.Parent;
            }

            return null;
        }

        // ---------------------------------------------------------------- raw pages, per OS

        private const uint MEM_COMMIT = 0x1000;
        private const uint MEM_RESERVE = 0x2000;
        private const uint MEM_RELEASE = 0x8000;
        private const uint PAGE_READWRITE = 0x04;
        private const uint PAGE_NOACCESS = 0x01;

        private const int PROT_NONE = 0x0;
        private const int PROT_READ = 0x1;
        private const int PROT_WRITE = 0x2;
        private const int MAP_PRIVATE = 0x02;
        private const int MAP_ANONYMOUS_LINUX = 0x20;

        [DllImport("kernel32", SetLastError = true)]
        private static extern IntPtr VirtualAlloc(
            IntPtr lpAddress, UIntPtr dwSize, uint flAllocationType, uint flProtect);

        [DllImport("kernel32", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool VirtualProtect(
            IntPtr lpAddress, UIntPtr dwSize, uint flNewProtect, out uint lpflOldProtect);

        [DllImport("kernel32", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

        [DllImport("libc", SetLastError = true, EntryPoint = "mmap")]
        private static extern IntPtr Mmap(
            IntPtr addr, UIntPtr length, int prot, int flags, int fd, IntPtr offset);

        [DllImport("libc", SetLastError = true, EntryPoint = "mprotect")]
        private static extern int Mprotect(IntPtr addr, UIntPtr len, int prot);

        [DllImport("libc", SetLastError = true, EntryPoint = "munmap")]
        private static extern int Munmap(IntPtr addr, UIntPtr length);

        private static IntPtr Reserve(long totalBytes)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                IntPtr p = VirtualAlloc(
                    IntPtr.Zero, (UIntPtr)totalBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (p == IntPtr.Zero)
                {
                    throw new InvalidOperationException(
                        $"VirtualAlloc failed: {Marshal.GetLastWin32Error()}");
                }

                return p;
            }

            IntPtr q = Mmap(
                IntPtr.Zero, (UIntPtr)totalBytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS_LINUX, -1, IntPtr.Zero);
            if (q == IntPtr.Zero || q == new IntPtr(-1))
            {
                throw new InvalidOperationException($"mmap failed: {Marshal.GetLastWin32Error()}");
            }

            return q;
        }

        private static void Poison(IntPtr pageStart, int pageSize)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (!VirtualProtect(pageStart, (UIntPtr)pageSize, PAGE_NOACCESS, out _))
                {
                    throw new InvalidOperationException(
                        $"VirtualProtect failed: {Marshal.GetLastWin32Error()}");
                }

                return;
            }

            if (Mprotect(pageStart, (UIntPtr)pageSize, PROT_NONE) != 0)
            {
                throw new InvalidOperationException(
                    $"mprotect failed: {Marshal.GetLastWin32Error()}");
            }
        }

        private static void Release(IntPtr region, long totalBytes)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                VirtualFree(region, UIntPtr.Zero, MEM_RELEASE);
                return;
            }

            Munmap(region, (UIntPtr)totalBytes);
        }
    }
}
#endif
