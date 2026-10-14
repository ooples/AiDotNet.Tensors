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
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSgemmKernel(DirectPtxRuntime runtime, PtxGemmConfiguration configuration, PtxGemmLayout layout)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        Configuration = configuration;
        Layout = layout;
        Ptx = PtxSgemmGenerator.Emit(configuration, layout, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, configuration, layout);
        DirectPtxLoadedKernel loaded = DirectPtxResourceInitialization.LoadKernel(
            runtime, Ptx, PtxSgemmGenerator.EntryPoint, configuration.Threads, Blueprint);
        _module = loaded.Module;
        _function = loaded.Function;
        Audit = loaded.Audit;
    }

    /// <summary>
    /// <c>C = alpha * op(A) @ op(B) + beta * C</c> for <paramref name="batch"/> problems; pointers are device addresses
    /// of the first problem, strides are in elements. Throws when this configuration does not admit the problem.
    /// </summary>
    internal unsafe void Launch(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k, int lda, int ldb, int ldc,
        float alpha, float beta, int batch = 1, long strideA = 0, long strideB = 0, long strideC = 0)
    {
        if (batch < 1 || batch > 65535) throw new ArgumentOutOfRangeException(nameof(batch));
        if (!PtxSgemmGenerator.Admits(Configuration, Layout, m, n, k, lda, ldb, ldc, strideA, strideB, strideC))
            throw new ArgumentException($"GEMM configuration {Configuration.Name} ({Layout.Name}) does not admit M={m} N={n} K={k}.");
        if (Configuration.VectorWidth == 4 && (((long)a | (long)b | (long)c) & 15) != 0)
            throw new ArgumentException("A 16-byte vector configuration needs 16-byte-aligned operands.");

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
        _module.Launch(_function, gridX, gridY, (uint)batch, (uint)Configuration.Threads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture,
        PtxGemmConfiguration configuration, PtxGemmLayout layout)
    {
        // Any shape: the tensors declare a 1 x 1 lower bound and Launch checks the real problem.
        var any = new DirectPtxExtent(1, 1);
        return new DirectPtxKernelBlueprint(
            Operation: "sgemm",
            Version: 1,
            Architecture: architecture,
            Variant: configuration.Name + "-" + layout.Name,
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, configuration.VectorWidth * sizeof(float), DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, configuration.VectorWidth * sizeof(float), DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("c", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, configuration.VectorWidth * sizeof(float), DirectPtxTensorAccess.Read | DirectPtxTensorAccess.Write,
                    DirectPtxExtentMode.AtLeast),
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 255,
                MaxStaticSharedBytes: configuration.SharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C = alpha * op(A) @ op(B) + beta * C, batched over grid z",
                ["configuration"] = configuration.Name,
                ["layout"] = layout.Name,
                ["edges"] = "predicated: out-of-range loads read zero, out-of-range stores skipped",
                ["beta-zero"] = "C is not read",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
            });
    }
}
