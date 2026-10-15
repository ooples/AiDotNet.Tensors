using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The exact-shape FP32 direct-PTX CSR SpMM family (issue #852). Emitter and shape-domain checks need no GPU; the
/// driver checks need an SM86 device. The specialization is experimental and fails closed until promoted.
/// </summary>
[Collection("DirectGpuSerial")]
public class DirectPtxCsrSpMMTests
{
    private readonly ITestOutputHelper _out;
    public DirectPtxCsrSpMMTests(ITestOutputHelper output) => _out = output;

    public static IEnumerable<object[]> Cells() => new[]
    {
        new object[] { 2708, 2708, 64, 0.0018 },
        new object[] { 19717, 500, 128, 0.001 },
        new object[] { 10000, 10000, 32, 0.0005 },
        new object[] { 512, 512, 64, 0.05 },
    };

    [Fact]
    public void Emitter_IsPointerOnlyRegisterResidentAndVectorized()
    {
        string ptx = PtxCsrSpMMF32Kernel.EmitPtx(8, 6, 2708, 2708, 64);
        Assert.Contains(".maxntid 256, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2708 inner=2708 columns=64 block=256", ptx);
        Assert.Equal(5, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        // 2708 rows do not tile 16-row blocks, so the constant tail guard is present; 512 rows tile 16 exactly.
        Assert.Contains("setp.ge.u32 %p0, %r4, 2708;", ptx);
        Assert.DoesNotContain("setp.ge.u32 %p0", PtxCsrSpMMF32Kernel.EmitPtx(8, 6, 512, 512, 64), StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromoted()
    {
        Assert.True(PtxCsrSpMMF32Kernel.IsSupportedShape(2708, 2708, 64));
        Assert.True(PtxCsrSpMMF32Kernel.IsSupportedShape(19717, 500, 128));
        Assert.True(PtxCsrSpMMF32Kernel.IsSupportedShape(10000, 10000, 32));
        Assert.True(PtxCsrSpMMF32Kernel.IsSupportedShape(512, 512, 64));
        Assert.False(PtxCsrSpMMF32Kernel.IsSupportedShape(2708, 2708, 48));
        Assert.False(PtxCsrSpMMF32Kernel.IsSupportedShape(2709, 2708, 64));
        Assert.False(PtxCsrSpMMF32Kernel.IsPromotedShape(2708, 2708, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => PtxCsrSpMMF32Kernel.EmitPtx(8, 6, 100, 100, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => PtxCsrSpMMF32Kernel.EmitPtx(8, 6, 512, 512, 64, 96));
    }

    [Fact]
    public void SparseCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        var all = DirectPtxSparseCoverageManifest.All;
        Assert.NotEmpty(all);
        Assert.Equal(all.Count, all.Select(c => c.Api).Distinct(StringComparer.Ordinal).Count());
        Assert.All(all, c =>
        {
            Assert.False(string.IsNullOrWhiteSpace(c.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(c.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(c.DirectPtxAssignment));
            Assert.NotEqual(DirectPtxSparseCoverageStatus.PromotedDirectPtx, c.Status);
        });
        Assert.Equal(new[] { "CudaBackend.CsrSpMMVec4" },
            all.Where(c => c.Status == DirectPtxSparseCoverageStatus.ExperimentalDirectPtx).Select(c => c.Api).ToArray());
        // Every NVRTC sparse kernel has a cell.
        foreach (string kernel in AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaSparseKernels.GetKernelNames())
            Assert.Contains(all, c => c.ExistingImplementation.Contains(kernel, StringComparison.Ordinal));
    }

    [SkippableTheory]
    [MemberData(nameof(Cells))]
    public void DriverOnly_MatchesReferenceAndTheNvrtcKernelBitForBit(int rows, int inner, int columns, double density)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "Requires a CUDA backend.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedCsrSpMM(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in CSR SpMM specialization is validated on SM86.");
        using var kernel = new PtxCsrSpMMF32Kernel(runtime, rows, inner, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.Equal("csr-spmm-f32", kernel.Blueprint.Operation);
        _out.WriteLine($"registers {kernel.Audit.Function.RegistersPerThread}, active blocks/SM {kernel.Audit.ActiveBlocksPerMultiprocessor}");

        var csr = SpMMCsr.Build(rows, inner, columns, density, seed: rows + columns);
        using var ptx = SpMMCsr.Upload(runtime, kernel, csr);
        kernel.Launch(ptx.RowPointers, ptx.ColumnIndices, ptx.Values, ptx.Dense, ptx.Output, csr.Nonzeros);
        runtime.Synchronize();
        var actual = new float[rows * columns];
        ptx.OutputBuffer.Download<float>(actual);

        // The NVRTC kernel the PTX ports, on the same data.
        float[] nvrtc = SpMMCsr.RunNvrtc(backend, csr);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(nvrtc[i]),
                $"[{i}] PTX {actual[i]:R} vs NVRTC {nvrtc[i]:R}");

        // And a float64 oracle.
        double maxError = 0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                double sum = 0;
                for (int p = csr.RowPointers[r]; p < csr.RowPointers[r + 1]; p++)
                    sum += (double)csr.Values[p] * csr.Dense[csr.ColumnIndices[p] * columns + c];
                double error = Math.Abs(sum - actual[r * columns + c]);
                maxError = Math.Max(maxError, error);
                Assert.True(error <= 1e-5 * Math.Max(1.0, Math.Abs(sum)), $"[{r},{c}] {actual[r * columns + c]} vs {sum}");
            }
        _out.WriteLine($"nonzeros {csr.Nonzeros}, max |error| vs fp64 {maxError:E2}");
    }

    [SkippableFact]
    public void Launch_RejectsTensorsThatDoNotHoldTheNonzeros()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedCsrSpMM(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in CSR SpMM specialization is validated on SM86.");
        using var kernel = new PtxCsrSpMMF32Kernel(runtime, 512, 512, 64);
        var csr = SpMMCsr.Build(512, 512, 64, 0.05, seed: 7);
        using var ptx = SpMMCsr.Upload(runtime, kernel, csr);
        Assert.Throws<ArgumentException>(() =>
            kernel.Launch(ptx.RowPointers, ptx.ColumnIndices, ptx.Values, ptx.Dense, ptx.Output, csr.Nonzeros + 1));
        // Swapping the index and value tensors breaks the dtype contract.
        Assert.Throws<ArgumentException>(() =>
            kernel.Launch(ptx.RowPointers, ptx.Values, ptx.ColumnIndices, ptx.Dense, ptx.Output, csr.Nonzeros));
    }

    /// <summary>
    /// Resident timing, PTX vs the NVRTC kernels the backend dispatches (vec4, and warp for N &lt;= 64), on the same
    /// device data: 30 warmups, then 101 samples per arm, arms interleaved sample by sample. Reports median, P95, P99,
    /// mean. Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Cells))]
    public void Bench_PtxVsNvrtc(int rows, int inner, int columns, double density)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1",
            "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = new CudaBackend();
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedCsrSpMM(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "SM86 only.");
        using var kernel = new PtxCsrSpMMF32Kernel(runtime, rows, inner, columns);
        var csr = SpMMCsr.Build(rows, inner, columns, density, seed: rows + columns);
        using var ptx = SpMMCsr.Upload(runtime, kernel, csr);
        using var nv = SpMMCsr.UploadBackend(backend, csr);

        const int Launches = 50, Warmups = 30, Samples = 101;
        var arms = new List<(string Name, Action Launch, Action Sync)>
        {
            ("ptx", () => kernel.Launch(ptx.RowPointers, ptx.ColumnIndices, ptx.Values, ptx.Dense, ptx.Output, csr.Nonzeros), runtime.Synchronize),
            ("nvrtc-vec4", () => backend.CsrSpMMVec4(nv.Values, nv.ColumnIndices, nv.RowPointers, nv.Dense, nv.Output, rows, inner, columns, csr.Nonzeros), backend.Synchronize),
        };
        if (columns <= 64)
            arms.Add(("nvrtc-warp", () => backend.CsrSpMMWarp(nv.Values, nv.ColumnIndices, nv.RowPointers, nv.Dense, nv.Output, rows, inner, columns, csr.Nonzeros), backend.Synchronize));

        var samples = arms.Select(_ => new double[Samples]).ToArray();
        for (int s = -Warmups; s < Samples; s++)
            for (int a = 0; a < arms.Count; a++)
            {
                arms[a].Sync();
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < Launches; i++) arms[a].Launch();
                arms[a].Sync();
                sw.Stop();
                if (s >= 0) samples[a][s] = sw.Elapsed.TotalMilliseconds * 1000.0 / Launches;
            }

        double flop = 2.0 * csr.Nonzeros * columns;
        _out.WriteLine($"M={rows} K={inner} N={columns} nnz={csr.Nonzeros} registers={kernel.Audit.Function.RegistersPerThread} activeBlocks/SM={kernel.Audit.ActiveBlocksPerMultiprocessor}");
        _out.WriteLine($"{"arm",-12}{"median us",11}{"p95",9}{"p99",9}{"mean",9}{"GF/s",9}");
        double ptxMedian = 0, bestOther = double.MaxValue;
        for (int a = 0; a < arms.Count; a++)
        {
            var sorted = samples[a].OrderBy(x => x).ToArray();
            double median = sorted[Samples / 2], p95 = sorted[(int)(0.95 * (Samples - 1))], p99 = sorted[(int)(0.99 * (Samples - 1))];
            _out.WriteLine($"{arms[a].Name,-12}{median,11:F2}{p95,9:F2}{p99,9:F2}{sorted.Average(),9:F2}{flop / (median * 1e3),9:F1}");
            if (a == 0) ptxMedian = median; else bestOther = Math.Min(bestOther, median);
        }
        _out.WriteLine($"=> ptx speedup over the fastest NVRTC arm: {bestOther / ptxMedian:F2}x");
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0) { count++; offset += value.Length; }
        return count;
    }

    private sealed class SpMMCsr
    {
        public int Rows, Inner, Columns, Nonzeros;
        public int[] RowPointers = Array.Empty<int>(), ColumnIndices = Array.Empty<int>();
        public float[] Values = Array.Empty<float>(), Dense = Array.Empty<float>();

        public static SpMMCsr Build(int rows, int inner, int columns, double density, int seed)
        {
            var rng = new Random(seed);
            var rowPointers = new int[rows + 1];
            var indices = new List<int>();
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < inner; c++)
                    if (rng.NextDouble() < density) indices.Add(c);
                rowPointers[r + 1] = indices.Count;
            }
            var values = new float[indices.Count];
            for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
            var dense = new float[inner * columns];
            for (int i = 0; i < dense.Length; i++) dense[i] = (float)(rng.NextDouble() * 2 - 1);
            return new SpMMCsr
            {
                Rows = rows, Inner = inner, Columns = columns, Nonzeros = indices.Count,
                RowPointers = rowPointers, ColumnIndices = indices.ToArray(), Values = values, Dense = dense,
            };
        }

        public static PtxBuffers Upload(DirectPtxRuntime runtime, PtxCsrSpMMF32Kernel kernel, SpMMCsr csr)
        {
            var b = new PtxBuffers
            {
                RowPointerBuffer = runtime.AllocateBytes((nuint)csr.RowPointers.Length * sizeof(int)),
                IndexBuffer = runtime.AllocateBytes((nuint)Math.Max(1, csr.Nonzeros) * sizeof(int)),
                ValueBuffer = runtime.AllocateBytes((nuint)Math.Max(1, csr.Nonzeros) * sizeof(float)),
                DenseBuffer = runtime.AllocateBytes((nuint)csr.Dense.Length * sizeof(float)),
                OutputBuffer = runtime.AllocateBytes((nuint)csr.Rows * (nuint)csr.Columns * sizeof(float)),
            };
            b.RowPointerBuffer.Upload<int>(csr.RowPointers);
            if (csr.Nonzeros > 0)
            {
                b.IndexBuffer.Upload<int>(csr.ColumnIndices);
                b.ValueBuffer.Upload<float>(csr.Values);
            }
            b.DenseBuffer.Upload<float>(csr.Dense);
            var t = kernel.Blueprint.Tensors;
            b.RowPointers = DirectPtxTensorView.CreateOwned(b.RowPointerBuffer, t[0]);
            b.ColumnIndices = DirectPtxTensorView.CreateOwned(b.IndexBuffer, t[1]);
            b.Values = DirectPtxTensorView.CreateOwned(b.ValueBuffer, t[2]);
            b.Dense = DirectPtxTensorView.CreateOwned(b.DenseBuffer, t[3]);
            b.Output = DirectPtxTensorView.CreateOwned(b.OutputBuffer, t[4]);
            return b;
        }

        public static BackendBuffers UploadBackend(CudaBackend backend, SpMMCsr csr)
        {
            var b = new BackendBuffers
            {
                Values = backend.AllocateBuffer(csr.Values),
                Dense = backend.AllocateBuffer(csr.Dense),
                Output = backend.AllocateBuffer(csr.Rows * csr.Columns),
                RowPointers = Ints(backend, csr.RowPointers),
                ColumnIndices = Ints(backend, csr.ColumnIndices),
            };
            return b;
        }

        public static float[] RunNvrtc(CudaBackend backend, SpMMCsr csr)
        {
            using var b = UploadBackend(backend, csr);
            backend.CsrSpMMVec4(b.Values, b.ColumnIndices, b.RowPointers, b.Dense, b.Output, csr.Rows, csr.Inner, csr.Columns, csr.Nonzeros);
            backend.Synchronize();
            return backend.DownloadBuffer(b.Output);
        }

        private static IGpuBuffer Ints(CudaBackend backend, int[] values)
        {
            var bytes = new byte[Math.Max(1, values.Length) * sizeof(int)];
            Buffer.BlockCopy(values, 0, bytes, 0, values.Length * sizeof(int));
            var buffer = backend.AllocateByteBuffer(bytes.Length);
            backend.UploadByteBuffer(buffer, bytes);
            return buffer;
        }
    }

    private sealed class PtxBuffers : IDisposable
    {
        public DirectPtxBuffer RowPointerBuffer = null!, IndexBuffer = null!, ValueBuffer = null!, DenseBuffer = null!, OutputBuffer = null!;
        public DirectPtxTensorView RowPointers, ColumnIndices, Values, Dense, Output;

        public void Dispose()
        {
            RowPointerBuffer.Dispose(); IndexBuffer.Dispose(); ValueBuffer.Dispose(); DenseBuffer.Dispose(); OutputBuffer.Dispose();
        }
    }

    private sealed class BackendBuffers : IDisposable
    {
        public IGpuBuffer Values = null!, Dense = null!, Output = null!, RowPointers = null!, ColumnIndices = null!;

        public void Dispose()
        {
            Values.Dispose(); Dense.Dispose(); Output.Dispose(); RowPointers.Dispose(); ColumnIndices.Dispose();
        }
    }
}
