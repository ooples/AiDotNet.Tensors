using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise log-sum-exp backward <c>dX[m,n] = exp(x[m,n] - lse[m]) * dY[m]</c> over the last
/// axis (issue #840), where <c>lse</c> is the caller-provided [M] forward log-partition and
/// <c>dY</c> is its [M] upstream gradient. One block owns one row and reuses those forward
/// statistics directly, matching the incumbent CUDA contract while avoiding a redundant max
/// and exp-sum reduction. The baked row extent removes the scalar loop; the default transaction
/// width is derived from the row extent, while the oracle can measure alternate launch widths.
/// Uses <c>ex2.approx.f32</c>, so a
/// promoted specialization carries ~1e-3 approximation error (disclosed on the release gate).
///
/// One block per row (grid = M), 256 or 512 threads, no shared memory. Supported N are
/// multiples of 256 so each thread owns an exact scalar, float2, or float4 pack.
/// </summary>
internal sealed class PtxLogSumExpBackwardKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const string EntryPoint = "aidotnet_logsumexp_backward_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly int _blockThreads;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLogSumExpBackwardKernel(DirectPtxRuntime runtime, int m, int n)
        : this(runtime, m, n, PtxLogSumExpBackwardVariant.ForShape(n))
    {
    }

    internal PtxLogSumExpBackwardKernel(
        DirectPtxRuntime runtime, int m, int n, PtxLogSumExpBackwardVariant variant)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in log-sum-exp-backward specialization is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Log-sum-exp backward");
        variant.Validate(n);
        M = m;
        N = n;
        _blockThreads = variant.BlockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n, variant);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n, variant);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, _blockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, _blockThreads, activeBlocks);
                DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, _blockThreads, activeBlocks,
                    module.JitInfoLog);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView logSumExp,
        DirectPtxTensorView grad,
        DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxAbiGuard.Require(logSumExp, Blueprint.Tensors[1], nameof(logSumExp));
        PtxAbiGuard.Require(grad, Blueprint.Tensors[2], nameof(grad));
        PtxAbiGuard.Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr logSumExpPointer = logSumExp.Pointer;
        IntPtr gradPointer = grad.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &logSumExpPointer;
        arguments[2] = &gradPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, (uint)M, 1, 1, (uint)_blockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
        => EmitPtx(ccMajor, ccMinor, m, n, PtxLogSumExpBackwardVariant.ForShape(n));

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int m, int n, PtxLogSumExpBackwardVariant variant)
    {
        PtxRowShape.Validate(m, n, "Log-sum-exp backward");
        variant.Validate(n);
        int rowBytes = checked(n * sizeof(float));
        int blockThreads = variant.BlockThreads;
        int elementsPerThread = n / blockThreads;
        const string Log2e = "0f3FB8AA3B";

        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// logsumexp-backward-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 lse_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<2>;");
        ptx.AppendLine("    .reg .b64 %rd<5>;");
        ptx.AppendLine("    .reg .f32 %f<9>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [lse_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd0, %rd0, %rd4;");               // &input[m,0]
        ptx.AppendLine("    add.u64 %rd3, %rd3, %rd4;");               // &output[m,0]
        ptx.AppendLine("    mul.wide.u32 %rd4, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd1, %rd1, %rd4;");               // &lse[m]
        ptx.AppendLine("    add.u64 %rd2, %rd2, %rd4;");               // &grad[m]
        ptx.AppendLine($"    ld.global.{variant.LoadModifier}.f32 %f0, [%rd1];"); // lse[m]
        ptx.AppendLine($"    ld.global.{variant.LoadModifier}.f32 %f1, [%rd2];"); // dY[m]
        int vectorWidth = variant.VectorWidth;
        int vectorBytes = vectorWidth * sizeof(float);
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r0, {vectorBytes};");
        ptx.AppendLine("    add.u64 %rd0, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, %rd4;");
        for (int group = 0; group < elementsPerThread; group += vectorWidth)
        {
            if (vectorWidth == 4)
                ptx.AppendLine($"    ld.global.{variant.LoadModifier}.v4.f32 {{%f2,%f3,%f4,%f5}}, [%rd0];");
            else if (vectorWidth == 2)
                ptx.AppendLine($"    ld.global.{variant.LoadModifier}.v2.f32 {{%f2,%f3}}, [%rd0];");
            else
                ptx.AppendLine($"    ld.global.{variant.LoadModifier}.f32 %f2, [%rd0];");
            for (int lane = 2; lane < 2 + vectorWidth; lane++)
            {
                ptx.AppendLine($"    sub.rn.f32 %f{lane}, %f{lane}, %f0;");
                ptx.AppendLine($"    mul.rn.f32 %f{lane}, %f{lane}, {Log2e};");
                ptx.AppendLine($"    ex2.approx.f32 %f{lane}, %f{lane};");
                ptx.AppendLine($"    mul.rn.f32 %f{lane}, %f{lane}, %f1;");
            }
            if (vectorWidth == 4)
                ptx.AppendLine($"    st.global.{variant.StoreModifier}.v4.f32 [%rd3], {{%f2,%f3,%f4,%f5}};");
            else if (vectorWidth == 2)
                ptx.AppendLine($"    st.global.{variant.StoreModifier}.v2.f32 [%rd3], {{%f2,%f3}};");
            else
                ptx.AppendLine($"    st.global.{variant.StoreModifier}.f32 [%rd3], %f2;");
            if (group + vectorWidth < elementsPerThread)
            {
                ptx.AppendLine($"    add.u64 %rd0, %rd0, {blockThreads * vectorBytes};");
                ptx.AppendLine($"    add.u64 %rd3, %rd3, {blockThreads * vectorBytes};");
            }
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n,
        PtxLogSumExpBackwardVariant variant)
    {
        var matrix = new DirectPtxExtent(m, n);
        var vector = new DirectPtxExtent(m);
        return new DirectPtxKernelBlueprint(
            Operation: "logsumexp-backward-row",
            Version: 3,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}-{variant.Name}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("logSumExp", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // The measured SM86 search family reports at most 30 registers; the default
                // 256-thread geometry remains below that envelope.
                MaxRegistersPerThread: 30,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / variant.BlockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dX[m,n] = exp(x[m,n] - lse[m]) * dY[m]",
                ["axis"] = "last",
                ["broadcast"] = "per-row-scalar-upstream-gradient",
                ["block-threads"] = variant.BlockThreads.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["vector-width"] = variant.VectorWidth.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["cache-policy"] = $"ld.{variant.LoadModifier}/st.{variant.StoreModifier}",
                ["exponential"] = "ex2.approx.f32",
                ["loop-shape"] = "baked-row-extent-fully-unrolled",
                ["register-lifetime"] = "pointer-chained-between-unrolled-groups",
                ["forward-statistics"] = "caller-provided-logsumexp-vector",
                ["reduction"] = "none-forward-logsumexp-reused",
                ["global-intermediates"] = "caller-provided-lse-input",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => PtxRowShape.IsPromoted(m, n);

}

internal readonly record struct PtxLogSumExpBackwardVariant(
    int BlockThreads,
    int VectorWidth,
    string LoadModifier,
    string StoreModifier)
{
    internal static readonly PtxLogSumExpBackwardVariant Default = new(256, 4, "nc", "wb");

    internal static PtxLogSumExpBackwardVariant ForShape(int n) => n == 1024
        ? new PtxLogSumExpBackwardVariant(256, 4, "ca", "cg")
        : Default with { VectorWidth = Math.Min(4, n / PtxRowShape.BlockThreads) };

    internal static IEnumerable<PtxLogSumExpBackwardVariant> SearchSpace(int n)
    {
        if (!PtxRowShape.IsSupported(64, n))
            throw new ArgumentOutOfRangeException(nameof(n),
                "Log-sum-exp-backward variant search requires a supported row extent.");
        foreach (int threads in new[] { 128, 256, 512 })
        {
            if (threads > n || n % threads != 0) continue;
            int elementsPerThread = n / threads;
            foreach (int width in new[] { 1, 2, 4 })
                if (elementsPerThread >= width && elementsPerThread % width == 0)
                    yield return new PtxLogSumExpBackwardVariant(threads, width, "nc", "wb");
        }

        PtxLogSumExpBackwardVariant defaultVariant = ForShape(n);
        foreach (string load in new[] { "nc", "ca" })
        foreach (string store in new[] { "wb", "cg", "wt", "cs" })
            if (load != "nc" || store != "wb")
                yield return defaultVariant with { LoadModifier = load, StoreModifier = store };
    }

    internal string Name => $"t{BlockThreads}-v{VectorWidth}-{LoadModifier}-{StoreModifier}-ex2";

    internal void Validate(int n)
    {
        if (BlockThreads is not (128 or 256 or 512) || BlockThreads > n || n % BlockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(BlockThreads),
                "Log-sum-exp-backward variants require 128, 256, or 512 threads that divide N.");
        int elementsPerThread = n / BlockThreads;
        if (VectorWidth is not (1 or 2 or 4) || VectorWidth > elementsPerThread ||
            elementsPerThread % VectorWidth != 0)
            throw new ArgumentOutOfRangeException(nameof(VectorWidth),
                "Transaction width must be 1, 2, or 4 and divide the per-thread row extent.");
        if (LoadModifier is not ("nc" or "ca"))
            throw new ArgumentOutOfRangeException(nameof(LoadModifier));
        if (StoreModifier is not ("wb" or "cg" or "wt" or "cs"))
            throw new ArgumentOutOfRangeException(nameof(StoreModifier));
    }
}
