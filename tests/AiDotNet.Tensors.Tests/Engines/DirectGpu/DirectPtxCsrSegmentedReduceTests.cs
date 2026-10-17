using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The exact-shape FP32 direct-PTX CSR segmented max / min family (issue #852). The emitter, shape-domain and manifest
/// checks need no GPU; the instruction check needs NVRTC; the driver checks need an SM86 device. The specialization is
/// experimental and fails closed until promoted.
/// </summary>
[Collection("DirectGpuSerial")]
public class DirectPtxCsrSegmentedReduceTests
{
    private readonly ITestOutputHelper _out;
    public DirectPtxCsrSegmentedReduceTests(ITestOutputHelper output) => _out = output;

    public static IEnumerable<object[]> Cells()
    {
        foreach (var reduction in new[] { "Max", "Min" })
        {
            yield return new object[] { reduction, 2708, 2708, 64, 0.0018 };
            yield return new object[] { reduction, 19717, 500, 128, 0.001 };
            yield return new object[] { reduction, 10000, 10000, 32, 0.0005 };
            yield return new object[] { reduction, 512, 512, 64, 0.05 };
        }
    }

    [Theory]
    [InlineData("Max", "max.ftz.f32", "0fFF7FFFFD")]
    [InlineData("Min", "min.ftz.f32", "0f7F7FFFFD")]
    public void Emitter_IsPointerOnlyRegisterResidentAndVectorized(string reductionName, string op, string initial)
    {
        var reduction = Parse(reductionName);
        string ptx = PtxCsrSegmentedReduceF32Kernel.EmitPtx(8, 6, reduction, 2708, 2708, 64);
        Assert.Contains(".maxntid 256, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2708 inner=2708 columns=64 block=256", ptx);
        Assert.Contains($".visible .entry {PtxCsrSegmentedReduceF32Kernel.EntryPointFor(reduction)}(", ptx);
        Assert.Equal(4, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, op));
        Assert.Equal(4, Count(ptx, $"mov.f32 %f0, {initial};") + Count(ptx, $"mov.f32 %f1, {initial};")
            + Count(ptx, $"mov.f32 %f2, {initial};") + Count(ptx, $"mov.f32 %f3, {initial};"));
        // Both row pointers and every column index go through the truncating conversion.
        Assert.Equal(3, Count(ptx, "cvt.rzi.ftz.s32.f32"));
        Assert.Contains("setp.ge.s32 %p1, %r5, %r6;", ptx);   // empty row -> zeros
        // 2708 rows do not tile 16-row blocks, so the constant tail guard is present; 512 rows tile 16 exactly.
        Assert.Contains("setp.ge.u32 %p0, %r4, 2708;", ptx);
        Assert.DoesNotContain("setp.ge.u32 %p0",
            PtxCsrSegmentedReduceF32Kernel.EmitPtx(8, 6, reduction, 512, 512, 64), StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromoted()
    {
        Assert.True(PtxCsrSegmentedReduceF32Kernel.IsSupportedShape(2708, 2708, 64));
        Assert.True(PtxCsrSegmentedReduceF32Kernel.IsSupportedShape(19717, 500, 128));
        Assert.False(PtxCsrSegmentedReduceF32Kernel.IsSupportedShape(2708, 2708, 48));
        Assert.False(PtxCsrSegmentedReduceF32Kernel.IsPromotedShape(2708, 2708, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxCsrSegmentedReduceF32Kernel.EmitPtx(8, 6, CsrSegmentedReduction.Max, 100, 100, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxCsrSegmentedReduceF32Kernel.EmitPtx(8, 6, CsrSegmentedReduction.Min, 512, 512, 64, 96));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxCsrSegmentedReduceF32Kernel.EmitPtx(8, 6, (CsrSegmentedReduction)7, 512, 512, 64));
        Assert.False(DirectPtxArchitecture.HasTargetedCsrSegmentedReduce(8, 0));
        Assert.True(DirectPtxArchitecture.HasTargetedCsrSegmentedReduce(8, 6));
    }

    [Fact]
    public void SparseCoverageManifest_MarksBothReductionsExperimental()
    {
        Assert.Equal(DirectPtxSparseCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSparseCoverageManifest.Get("CudaBackend.CsrSegmentedMax").Status);
        Assert.Equal(DirectPtxSparseCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSparseCoverageManifest.Get("CudaBackend.CsrSegmentedMin").Status);
        Assert.Equal(DirectPtxSparseCoverageStatus.PlannedDirectPtx,
            DirectPtxSparseCoverageManifest.Get("CudaBackend.CsrSegmentedStdDev").Status);
    }

    /// <summary>
    /// The PTX uses the reduction and conversion instructions, and the starting constants, that NVRTC emits for the
    /// kernels it ports when compiled the way <c>CudaBackend</c> compiles them (sm_86, <c>--use_fast_math</c>). Without
    /// <c>.ftz</c>, a subnormal input would survive here and flush to zero there.
    /// </summary>
    [SkippableTheory]
    [InlineData("Max", "csr_segmented_max")]
    [InlineData("Min", "csr_segmented_min")]
    public void Instructions_AreTheOnesNvrtcEmitsForTheKernelItPorts(string reductionName, string nvrtcKernel)
    {
        var reduction = Parse(reductionName);
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "Requires the NVRTC library.");
        string nvrtcPtx = CompileWithNvrtc(AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaMissingLookupKernels.GetSource());
        string nvrtcBody = EntryBody(nvrtcPtx, nvrtcKernel);
        string ours = PtxCsrSegmentedReduceF32Kernel.EmitPtx(8, 6, reduction, 512, 512, 64);

        Assert.Equal(Arithmetic(nvrtcBody), Arithmetic(ours));
        Assert.Equal(Constants(nvrtcBody), Constants(ours));
    }

    [SkippableTheory]
    [MemberData(nameof(Cells))]
    public void DriverOnly_MatchesTheNvrtcKernelBitForBit(string reductionName, int rows, int inner, int columns,
        double density)
    {
        var reduction = Parse(reductionName);
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "Requires a CUDA backend.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasTargetedCsrSegmentedReduce(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The CSR segmented max / min specialization targets SM86.");
        using var kernel = new PtxCsrSegmentedReduceF32Kernel(runtime, reduction, rows, inner, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        _out.WriteLine($"registers {kernel.Audit.Function.RegistersPerThread}, active blocks/SM {kernel.Audit.ActiveBlocksPerMultiprocessor}");

        var csr = SegmentedCsr.Build(rows, inner, columns, density, seed: rows + columns + (int)reduction);
        float[] actual = csr.RunPtx(runtime, kernel);
        float[] nvrtc = csr.RunNvrtc(backend, reduction);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Bits(actual[i]) == Bits(nvrtc[i]), $"[{i}] PTX {actual[i]:R} vs NVRTC {nvrtc[i]:R}");

        // And the reference semantics: flush-to-zero max / min over the gathered rows, 0 for an empty row.
        float[] expected = csr.Reference(reduction);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(SameValue(actual[i], expected[i]), $"[{i}] PTX {actual[i]:R} vs reference {expected[i]:R}");
        _out.WriteLine($"nonzeros {csr.Nonzeros}, empty rows {csr.EmptyRows}");
    }

    [SkippableFact]
    public void Launch_RejectsTensorsThatDoNotHoldTheNonzeros()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasTargetedCsrSegmentedReduce(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The CSR segmented max / min specialization targets SM86.");
        using var kernel = new PtxCsrSegmentedReduceF32Kernel(runtime, CsrSegmentedReduction.Max, 512, 512, 64);
        var csr = SegmentedCsr.Build(512, 512, 64, 0.05, seed: 7);
        using var buffers = csr.Upload(runtime, kernel);
        Assert.Throws<ArgumentException>(() =>
            kernel.Launch(buffers.ColumnIndices, buffers.RowPointers, buffers.Input, buffers.Output, csr.Nonzeros + 1));
        // Swapping the input and output tensors breaks their access contracts' extents.
        Assert.Throws<ArgumentException>(() =>
            kernel.Launch(buffers.ColumnIndices, buffers.RowPointers, buffers.RowPointers, buffers.Output, csr.Nonzeros));
    }

    private static CsrSegmentedReduction Parse(string name) =>
        (CsrSegmentedReduction)Enum.Parse(typeof(CsrSegmentedReduction), name);

    private static int Count(string text, string token)
    {
        int count = 0, index = 0;
        while ((index = text.IndexOf(token, index, StringComparison.Ordinal)) >= 0) { count++; index += token.Length; }
        return count;
    }

    private static uint Bits(float value) => BitConverter.ToUInt32(BitConverter.GetBytes(value), 0);

    // Max and min are exact; only the sign of a zero may legitimately be either.
    private static bool SameValue(float a, float b) =>
        Bits(a) == Bits(b) || (a == 0 && b == 0) || (float.IsNaN(a) && float.IsNaN(b));

    private static string CompileWithNvrtc(string source)
    {
        // What CudaBackend.CompileKernelModule does before compiling: strip the C headers NVRTC lacks and prepend its
        // macro preamble.
        source = source.Replace("#include <math.h>", "").Replace("#include <float.h>", "").Replace("#include <stdio.h>", "");
        const string preamble =
            "#ifndef INFINITY\n#define INFINITY __int_as_float(0x7f800000)\n#endif\n" +
            "#ifndef NAN\n#define NAN __int_as_float(0x7fffffff)\n#endif\n" +
            "#ifndef FLT_MAX\n#define FLT_MAX 3.402823466e+38f\n#endif\n" +
            "#ifndef FLT_MIN\n#define FLT_MIN 1.175494351e-38f\n#endif\n";
        source = preamble + source;
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success,
            NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "segmented.cu", 0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            // CudaBackend.CompileKernelModule compiles missing_lookup_kernels with its default useFastMath: true.
            string[] options = { "--gpu-architecture=sm_86", "--use_fast_math" };
            Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options));
            NvrtcNativeBindings.nvrtcGetPTXSize(program, out UIntPtr size);
            IntPtr buffer = Marshal.AllocHGlobal(checked((int)size));
            try
            {
                NvrtcNativeBindings.nvrtcGetPTX(program, buffer);
                return Marshal.PtrToStringAnsi(buffer)!;
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }
        finally { NvrtcNativeBindings.nvrtcDestroyProgram(ref program); }
    }

    private static string EntryBody(string ptx, string entry)
    {
        int start = ptx.IndexOf($".entry {entry}(", StringComparison.Ordinal);
        Assert.True(start >= 0, $"NVRTC produced no entry '{entry}'.");
        int next = ptx.IndexOf(".entry ", start + 7, StringComparison.Ordinal);
        return next < 0 ? ptx.Substring(start) : ptx.Substring(start, next - start);
    }

    // The distinct reduction and conversion opcodes a body uses (NVRTC unrolls its loop, so counts differ).
    private static string[] Arithmetic(string body) =>
        Regex.Matches(body, @"\b(max|min|cvt)\.[a-z0-9.]+").Cast<Match>().Select(m => m.Value).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToArray();

    private static string[] Constants(string body) =>
        Regex.Matches(body, @"0f[0-9A-Fa-f]{8}").Cast<Match>().Select(m => m.Value.ToUpperInvariant())
            .Where(v => v != "0F00000000").Distinct().OrderBy(s => s, StringComparer.Ordinal).ToArray();

    private sealed class SegmentedCsr
    {
        public int Rows, Inner, Columns, Nonzeros, EmptyRows;
        public float[] RowPointers = Array.Empty<float>(), ColumnIndices = Array.Empty<float>(), Input = Array.Empty<float>();

        public static SegmentedCsr Build(int rows, int inner, int columns, double density, int seed)
        {
            var rng = new Random(seed);
            var rowPointers = new float[rows + 1];
            var indices = new List<float>();
            int empty = 0;
            for (int r = 0; r < rows; r++)
            {
                int before = indices.Count;
                // Every eighth row is left empty on purpose; the rest draw their columns at the density.
                if (r % 8 != 7)
                    for (int c = 0; c < inner; c++)
                        if (rng.NextDouble() < density)
                            // Some indices carry a fraction: the kernels truncate them (3.7 selects row 3).
                            indices.Add(rng.Next(4) == 0 ? c + 0.75f : c);
                if (indices.Count == before) empty++;
                rowPointers[r + 1] = indices.Count;
            }
            var input = new float[inner * columns];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = (i % 97) switch
                {
                    0 => float.NaN,
                    1 => 1e-40f,     // subnormal: flushed to zero by the ftz reduction
                    2 => -1e-40f,
                    3 => 0f,
                    4 => -0f,
                    5 => -3.4e38f,   // below the max starting value
                    6 => 3.4e38f,    // above the min starting value
                    _ => (float)(rng.NextDouble() * 2 - 1),
                };
            }
            return new SegmentedCsr
            {
                Rows = rows, Inner = inner, Columns = columns, Nonzeros = indices.Count, EmptyRows = empty,
                RowPointers = rowPointers, ColumnIndices = indices.ToArray(), Input = input,
            };
        }

        public float[] Reference(CsrSegmentedReduction reduction)
        {
            var output = new float[Rows * Columns];
            bool max = reduction == CsrSegmentedReduction.Max;
            for (int r = 0; r < Rows; r++)
            {
                int start = (int)RowPointers[r], end = (int)RowPointers[r + 1];
                for (int j = 0; j < Columns; j++)
                {
                    if (start >= end) { output[r * Columns + j] = 0f; continue; }
                    float acc = max ? -3.402823e+38f : 3.402823e+38f;
                    for (int p = start; p < end; p++)
                    {
                        float x = Flush(Input[(int)ColumnIndices[p] * Columns + j]);
                        // PTX max/min: a NaN operand yields the other operand.
                        if (float.IsNaN(x)) continue;
                        acc = max ? Math.Max(acc, x) : Math.Min(acc, x);
                    }
                    output[r * Columns + j] = Flush(acc);
                }
            }
            return output;
        }

        private static float Flush(float x) =>
            x != 0 && !float.IsNaN(x) && Math.Abs(x) < 1.17549435e-38f ? (x < 0 ? -0f : 0f) : x;

        public Buffers Upload(DirectPtxRuntime runtime, PtxCsrSegmentedReduceF32Kernel kernel)
        {
            var b = new Buffers
            {
                IndexBuffer = runtime.AllocateBytes((nuint)Math.Max(1, Nonzeros) * sizeof(float)),
                RowPointerBuffer = runtime.AllocateBytes((nuint)RowPointers.Length * sizeof(float)),
                InputBuffer = runtime.AllocateBytes((nuint)Input.Length * sizeof(float)),
                OutputBuffer = runtime.AllocateBytes((nuint)Rows * (nuint)Columns * sizeof(float)),
            };
            if (Nonzeros > 0) b.IndexBuffer.Upload<float>(ColumnIndices);
            b.RowPointerBuffer.Upload<float>(RowPointers);
            b.InputBuffer.Upload<float>(Input);
            var t = kernel.Blueprint.Tensors;
            b.ColumnIndices = DirectPtxTensorView.CreateOwned(b.IndexBuffer, t[0]);
            b.RowPointers = DirectPtxTensorView.CreateOwned(b.RowPointerBuffer, t[1]);
            b.Input = DirectPtxTensorView.CreateOwned(b.InputBuffer, t[2]);
            b.Output = DirectPtxTensorView.CreateOwned(b.OutputBuffer, t[3]);
            return b;
        }

        public float[] RunPtx(DirectPtxRuntime runtime, PtxCsrSegmentedReduceF32Kernel kernel)
        {
            using var b = Upload(runtime, kernel);
            kernel.Launch(b.ColumnIndices, b.RowPointers, b.Input, b.Output, Nonzeros);
            runtime.Synchronize();
            var output = new float[Rows * Columns];
            b.OutputBuffer.Download<float>(output);
            return output;
        }

        public float[] RunNvrtc(CudaBackend backend, CsrSegmentedReduction reduction)
        {
            using var indices = backend.AllocateBuffer(Nonzeros > 0 ? ColumnIndices : new float[1]);
            using var rowPointers = backend.AllocateBuffer(RowPointers);
            using var input = backend.AllocateBuffer(Input);
            using var output = backend.AllocateBuffer(Rows * Columns);
            if (reduction == CsrSegmentedReduction.Max)
                backend.CsrSegmentedMax(indices, rowPointers, input, output, Rows, Inner, Columns);
            else
                backend.CsrSegmentedMin(indices, rowPointers, input, output, Rows, Inner, Columns);
            backend.Synchronize();
            return backend.DownloadBuffer(output);
        }
    }

    private sealed class Buffers : IDisposable
    {
        public DirectPtxBuffer IndexBuffer = null!, RowPointerBuffer = null!, InputBuffer = null!, OutputBuffer = null!;
        public DirectPtxTensorView ColumnIndices, RowPointers, Input, Output;

        public void Dispose()
        {
            IndexBuffer.Dispose(); RowPointerBuffer.Dispose(); InputBuffer.Dispose(); OutputBuffer.Dispose();
        }
    }
}
