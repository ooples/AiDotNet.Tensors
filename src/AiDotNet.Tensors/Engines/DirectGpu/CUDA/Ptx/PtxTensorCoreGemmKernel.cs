using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// A loaded <see cref="PtxTensorCoreGemmGenerator"/> kernel (#1264): one configuration and operand layout, any admitted
/// shape, TF32 Tensor-Core math with FP32 accumulation (cuBLAS's default FP32 GEMM math on Ampere and newer).
/// </summary>
internal sealed class PtxTensorCoreGemmKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly uint _sharedBytes;

    internal PtxTensorCoreGemmConfiguration Configuration { get; }
    internal PtxGemmLayout Layout { get; }
    internal PtxTensorCorePrecision Precision { get; }
    internal bool HalfOutput { get; }
    /// <summary>Each K tile accumulates in FP16 and is promoted into the FP32 total; a block whose FP16 partial sums
    /// overflowed recomputes its tile with FP32 accumulation (#1549).</summary>
    internal bool Fp16Accumulate { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxTensorCoreGemmKernel(DirectPtxRuntime runtime, PtxTensorCoreGemmConfiguration configuration, PtxGemmLayout layout,
        PtxTensorCorePrecision precision = PtxTensorCorePrecision.Tf32, bool halfOutput = false, bool fp16Accumulate = false)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        Configuration = configuration;
        Layout = layout;
        Precision = precision;
        HalfOutput = halfOutput;
        Fp16Accumulate = fp16Accumulate;
        Ptx = PtxTensorCoreGemmGenerator.Emit(configuration, layout, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            precision, halfOutput, fp16Accumulate);
        _sharedBytes = (uint)configuration.SharedBytes(layout, precision);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, configuration, layout, precision, halfOutput, fp16Accumulate);
        DirectPtxLoadedKernel loaded = DirectPtxResourceInitialization.LoadKernel(
            runtime, Ptx, PtxTensorCoreGemmGenerator.EntryPoint, configuration.Threads, Blueprint);
        _module = loaded.Module;
        _function = loaded.Function;
        Audit = loaded.Audit;
    }

    /// <summary>
    /// <c>C = alpha * op(A) @ op(B) + beta * C</c> for <paramref name="batch"/> problems; pointers are device addresses
    /// of the first problem, strides in elements. Throws when this configuration does not admit the problem.
    /// <paramref name="stream"/> zero means the runtime's stream.
    /// </summary>
    internal unsafe void Launch(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k, int lda, int ldb, int ldc,
        float alpha, float beta, int batch = 1, long strideA = 0, long strideB = 0, long strideC = 0, IntPtr stream = default)
    {
        if (batch < 1 || batch > 65535) throw new ArgumentOutOfRangeException(nameof(batch));
        if (!PtxTensorCoreGemmGenerator.Admits(Configuration, Layout, m, n, k, lda, ldb, ldc, strideA, strideB, strideC, Precision))
            throw new ArgumentException($"Tensor-Core GEMM {Configuration.Name} ({Layout.Name}) does not admit M={m} N={n} K={k}.");
        if ((((long)a | (long)b | (long)c) & 15) != 0)
            throw new ArgumentException("The Tensor-Core GEMM needs 16-byte-aligned operands.");

        ulong aPointer = (ulong)a, bPointer = (ulong)b, cPointer = (ulong)c;
        uint mu = (uint)m, nu = (uint)n, ku = (uint)k, ldau = (uint)lda, ldbu = (uint)ldb, ldcu = (uint)ldc;
        ulong sa = (ulong)strideA, sb = (ulong)strideB, sc = (ulong)strideC;
        void** arguments = stackalloc void*[14];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &cPointer;
        arguments[3] = &mu;
        arguments[4] = &nu;
        arguments[5] = &ku;
        arguments[6] = &ldau;
        arguments[7] = &ldbu;
        arguments[8] = &ldcu;
        arguments[9] = &alpha;
        arguments[10] = &beta;
        arguments[11] = &sa;
        arguments[12] = &sb;
        arguments[13] = &sc;
        uint gridX = (uint)((n + Configuration.BlockN - 1) / Configuration.BlockN);
        uint gridY = (uint)((m + Configuration.BlockM - 1) / Configuration.BlockM);
        _module.Launch(_function, gridX, gridY, (uint)batch, (uint)Configuration.Threads, 1, 1, _sharedBytes, arguments, stream);
    }

    public void Dispose() => _module.Dispose();

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture,
        PtxTensorCoreGemmConfiguration configuration, PtxGemmLayout layout, PtxTensorCorePrecision precision, bool halfOutput,
        bool fp16Accumulate)
    {
        var operandType = precision == PtxTensorCorePrecision.Fp16 ? DirectPtxPhysicalType.Float16 : DirectPtxPhysicalType.Float32;
        var outputType = halfOutput ? DirectPtxPhysicalType.Float16 : DirectPtxPhysicalType.Float32;
        string math = precision == PtxTensorCorePrecision.Fp16
            ? fp16Accumulate
                ? "FP16 Tensor Core (mma.sync m16n8k16), FP16 accumulate per K tile promoted to an FP32 total; a block whose partial sums overflow redoes its tile with FP32 accumulate"
                : "FP16 Tensor Core (mma.sync m16n8k16), FP32 accumulate"
            : "TF32 Tensor Core (mma.sync m16n8k8), FP32 accumulate; operands rounded to TF32 (cvt.rna)";
        var any = new DirectPtxExtent(1, 1);
        return new DirectPtxKernelBlueprint(
            Operation: "tc-sgemm",
            Version: 1,
            Architecture: architecture,
            Variant: configuration.Name + "-" + layout.Name + "-" + precision.ToString().ToLowerInvariant() + (halfOutput ? "-f16out" : "") + (fp16Accumulate ? "-f16acc" : ""),
            Tensors:
            [
                new("a", operandType, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("b", operandType, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("c", outputType, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Read | DirectPtxTensorAccess.Write, DirectPtxExtentMode.AtLeast),
            ],
            // Shared memory is dynamic (sized per launch), so the static budget is zero.
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 255,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C = alpha * op(A) @ op(B) + beta * C, batched over grid z",
                ["math"] = math,
                ["configuration"] = configuration.Name,
                ["layout"] = layout.Name,
                ["dynamic-shared-bytes"] = configuration.SharedBytes(layout, precision).ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["edges"] = "cp.async zero-fill outside the matrix; out-of-range stores skipped",
                ["beta-zero"] = "C is not read",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
            });
    }
}
