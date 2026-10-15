using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// A loaded <see cref="PtxSgemmGenerator"/> kernel (#1262): one configuration and operand layout, any shape. Shapes,
/// leading dimensions and batch strides are launch arguments, so the module is compiled once per configuration.
/// </summary>
internal sealed class PtxSgemmKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal PtxGemmConfiguration Configuration { get; }
    internal PtxGemmLayout Layout { get; }
    internal PtxGemmPrecision Precision { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSgemmKernel(DirectPtxRuntime runtime, PtxGemmConfiguration configuration, PtxGemmLayout layout,
        PtxGemmPrecision precision = PtxGemmPrecision.Fp32)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        Configuration = configuration;
        Layout = layout;
        Precision = precision;
        Ptx = PtxSgemmGenerator.Emit(configuration, layout, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            precision);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, configuration, layout, precision);
        DirectPtxLoadedKernel loaded = DirectPtxResourceInitialization.LoadKernel(
            runtime, Ptx, PtxSgemmGenerator.EntryPoint, configuration.Threads, Blueprint);
        _module = loaded.Module;
        _function = loaded.Function;
        Audit = loaded.Audit;
    }

    /// <summary>
    /// <c>C = alpha * op(A) @ op(B) + beta * C</c> for <paramref name="batch"/> problems; pointers are device addresses
    /// of the first problem, strides are in elements. Throws when this configuration does not admit the problem.
    /// <paramref name="stream"/> zero launches on the runtime's stream. FP64 kernels launch through
    /// <see cref="LaunchDouble"/>.
    /// </summary>
    internal unsafe void Launch(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k, int lda, int ldb, int ldc,
        float alpha, float beta, int batch = 1, long strideA = 0, long strideB = 0, long strideC = 0,
        IntPtr stream = default)
    {
        if (Precision == PtxGemmPrecision.Fp64) throw new InvalidOperationException("An FP64 GEMM launches with LaunchDouble.");
        Validate(a, b, c, m, n, k, lda, ldb, ldc, batch, strideA, strideB, strideC);
        LaunchCore(a, b, c, m, n, k, lda, ldb, ldc, &alpha, &beta, batch, strideA, strideB, strideC, stream);
    }

    /// <summary>The FP64 kernel's launch: as <see cref="Launch"/>, with FP64 scalars.</summary>
    internal unsafe void LaunchDouble(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k, int lda, int ldb, int ldc,
        double alpha, double beta, int batch = 1, long strideA = 0, long strideB = 0, long strideC = 0,
        IntPtr stream = default)
    {
        if (Precision != PtxGemmPrecision.Fp64) throw new InvalidOperationException("Only an FP64 GEMM launches with LaunchDouble.");
        Validate(a, b, c, m, n, k, lda, ldb, ldc, batch, strideA, strideB, strideC);
        LaunchCore(a, b, c, m, n, k, lda, ldb, ldc, &alpha, &beta, batch, strideA, strideB, strideC, stream);
    }

    private void Validate(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k, int lda, int ldb, int ldc, int batch,
        long strideA, long strideB, long strideC)
    {
        if (batch < 1 || batch > 65535) throw new ArgumentOutOfRangeException(nameof(batch));
        if (!PtxSgemmGenerator.Admits(Configuration, Layout, m, n, k, lda, ldb, ldc, strideA, strideB, strideC, Precision))
            throw new ArgumentException($"GEMM configuration {Configuration.Name} ({Layout.Name}) does not admit M={m} N={n} K={k}.");
        if (Configuration.VectorWidth == 4 && (((long)a | (long)b | (long)c) & 15) != 0)
            throw new ArgumentException("A 16-byte vector configuration needs 16-byte-aligned operands.");
    }

    private unsafe void LaunchCore(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k, int lda, int ldb, int ldc,
        void* alpha, void* beta, int batch, long strideA, long strideB, long strideC, IntPtr stream)
    {
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
        arguments[9] = alpha;
        arguments[10] = beta;
        arguments[11] = &sa;
        arguments[12] = &sb;
        arguments[13] = &sc;
        uint gridX = (uint)((n + Configuration.BlockN - 1) / Configuration.BlockN);
        uint gridY = (uint)((m + Configuration.BlockM - 1) / Configuration.BlockM);
        uint dynamicShared = Configuration.IsAsync ? (uint)Configuration.SharedBytes : 0;
        _module.Launch(_function, gridX, gridY, (uint)batch, (uint)Configuration.Threads, 1, 1, dynamicShared, arguments, stream);
    }

    public void Dispose() => _module.Dispose();

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture,
        PtxGemmConfiguration configuration, PtxGemmLayout layout, PtxGemmPrecision precision)
    {
        DirectPtxPhysicalType operandType = precision switch
        {
            PtxGemmPrecision.Fp16InFp32Out or PtxGemmPrecision.Fp16InFp16Out => DirectPtxPhysicalType.Float16,
            PtxGemmPrecision.Bf16InFp32Out => DirectPtxPhysicalType.BFloat16,
            PtxGemmPrecision.Fp64 => DirectPtxPhysicalType.Float64,
            _ => DirectPtxPhysicalType.Float32,
        };
        DirectPtxPhysicalType outputType = precision switch
        {
            PtxGemmPrecision.Fp16InFp16Out => DirectPtxPhysicalType.Float16,
            PtxGemmPrecision.Fp64 => DirectPtxPhysicalType.Float64,
            _ => DirectPtxPhysicalType.Float32,
        };
        // Any shape: the tensors declare a 1 x 1 lower bound and Launch checks the real problem.
        var any = new DirectPtxExtent(1, 1);
        return new DirectPtxKernelBlueprint(
            Operation: "sgemm",
            Version: 1,
            Architecture: architecture,
            Variant: configuration.Name + "-" + layout.Name + (precision == PtxGemmPrecision.Fp32 ? "" : "-" + precision),
            Tensors:
            [
                new("a", operandType, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, configuration.VectorWidth * precision.OperandBytes(), DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("b", operandType, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, configuration.VectorWidth * precision.OperandBytes(), DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("c", outputType, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, configuration.VectorWidth * precision.OutputBytes(), DirectPtxTensorAccess.Read | DirectPtxTensorAccess.Write,
                    DirectPtxExtentMode.AtLeast),
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 255,
                // A cp.async kernel's stages are dynamic shared memory (sized per launch), so its static budget is zero.
                MaxStaticSharedBytes: configuration.IsAsync ? 0 : precision.SharedBytes(configuration),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C = alpha * op(A) @ op(B) + beta * C, batched over grid z",
                ["configuration"] = configuration.Name,
                ["layout"] = layout.Name,
                ["precision"] = precision.ToString(),
                ["edges"] = configuration.IsAsync
                    ? "cp.async zero-fill outside the matrix; out-of-range stores skipped"
                    : "predicated: out-of-range loads read zero, out-of-range stores skipped",
                ["pipeline"] = configuration.IsAsync
                    ? "cp.async, " + configuration.AsyncStages.ToString(System.Globalization.CultureInfo.InvariantCulture) + " shared stages"
                    : configuration.DoubleBuffer ? "register-staged, double-buffered" : "register-staged",
                ["beta-zero"] = "C is not read",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
            });
    }
}
