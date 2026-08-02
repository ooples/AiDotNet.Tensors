using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise numerically-stable softmax <c>output[m,n] = exp(x[m,n] - rowMax) / rowSumExp</c>
/// over the last axis (issue #840). One block owns one row, computes its row max and exp-sum
/// with hierarchical warp reductions, stages exponentials in the final output, then writes the
/// normalized values in place. Max and sum use independent warp-leader scratch regions, so a
/// completed result never needs an extra protection barrier before the next reduction. There is
/// no separate global max/sum/probability allocation; repeated row reads stay cacheable in L1.
/// Uses <c>ex2.approx.f32</c>, so a promoted specialization carries ~1e-3 approximation error
/// (disclosed on the release gate).
///
/// One block per row (grid = M). The measured 1024-column specialization uses 64 threads and
/// keeps four float4 packs per lane live through both reductions; other shapes retain the
/// 256-thread baseline geometry. Shared memory contains two independent warp-leader reduction
/// regions sized from the launch width. Supported N are multiples of 256.
/// </summary>
internal sealed class PtxSoftmaxKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const string EntryPoint = "aidotnet_softmax_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly int _blockThreads;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSoftmaxKernel(DirectPtxRuntime runtime, int m, int n)
        : this(runtime, m, n, PtxSoftmaxVariant.ForShape(n))
    {
    }

    internal PtxSoftmaxKernel(
        DirectPtxRuntime runtime, int m, int n, PtxSoftmaxVariant variant)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in softmax specialization is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Softmax");
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

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxAbiGuard.Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(_function, (uint)M, 1, 1, (uint)_blockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
        => EmitPtx(ccMajor, ccMinor, m, n, PtxSoftmaxVariant.ForShape(n));

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int m, int n, PtxSoftmaxVariant variant)
    {
        PtxRowShape.Validate(m, n, "Softmax");
        variant.Validate(n);
        int rowBytes = checked(n * sizeof(float));
        int blockThreads = variant.BlockThreads;
        bool vectorized = variant.Vectorized;
        int reductionBytes = PtxRowReduce.SharedBytesFor(blockThreads);
        int scratchBytes = reductionBytes * (variant.DoubleBufferedScratch ? 2 : 1);
        string inputCache = variant.CacheInput ? "ca" : "nc";
        const string Log2e = "0f3FB8AA3B";  // 1.4426950408889634
        const string NegInf = "0fFF800000";

        var ptx = new StringBuilder(10_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// softmax-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{scratchBytes}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");                 // &input[m,0]
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");                 // &output[m,0]
        ptx.AppendLine($"    mul.wide.u32 %rd9, %r0, {(vectorized ? 16 : 4)};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd13;");              // &maxRed[tid]

        // ---- Pass 1: partial max; the row remains hot in L1 for pass 2 ----
        ptx.AppendLine($"    mov.f32 %f0, {NegInf};");
        if (vectorized)
        {
            for (int column = 0; column < n; column += blockThreads * 4)
            {
                int[] registers = VectorRegisters(
                    variant.RegisterStage ? column / (blockThreads * 4) : 0);
                ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
                ptx.AppendLine($"    ld.global.{inputCache}.v4.f32 " +
                    $"{{%f{registers[0]},%f{registers[1]},%f{registers[2]},%f{registers[3]}}}, [%rd12];");
                foreach (int register in registers)
                    ptx.AppendLine($"    max.f32 %f0, %f0, %f{register};");
            }
        }
        else
        {
            for (int column = 0; column < n; column += blockThreads)
            {
                ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
                ptx.AppendLine($"    ld.global.{inputCache}.f32 %f1, [%rd12];");
                ptx.AppendLine("    max.f32 %f0, %f0, %f1;");
            }
        }
        PtxRowReduce.Emit(ptx, "max.f32", "%f0", blockThreads);
        ptx.AppendLine("    ld.shared.f32 %f2, [%rd5];");                // rowMax
        if (!variant.DoubleBufferedScratch)
            ptx.AppendLine("    bar.sync 0;");                          // protect max before scratch reuse

        // ---- Pass 2: partial sum of exp(x - rowMax) ----
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        if (vectorized)
        {
            for (int column = 0; column < n; column += blockThreads * 4)
            {
                int[] registers = VectorRegisters(
                    variant.RegisterStage ? column / (blockThreads * 4) : 0);
                if (!variant.RegisterStage)
                {
                    ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                    ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
                    ptx.AppendLine($"    ld.global.{inputCache}.v4.f32 " +
                        $"{{%f{registers[0]},%f{registers[1]},%f{registers[2]},%f{registers[3]}}}, [%rd12];");
                }
                foreach (int register in registers)
                {
                    ptx.AppendLine($"    sub.rn.f32 %f{register}, %f{register}, %f2;");
                    ptx.AppendLine($"    mul.rn.f32 %f{register}, %f{register}, {Log2e};");
                    ptx.AppendLine($"    ex2.approx.f32 %f{register}, %f{register};");
                    ptx.AppendLine($"    add.rn.f32 %f0, %f0, %f{register};");
                }
                if (!variant.RegisterStage)
                {
                    ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
                    ptx.AppendLine("    st.global.v4.f32 [%rd14], {%f1,%f5,%f6,%f7};");
                }
            }
        }
        else
        {
            for (int column = 0; column < n; column += blockThreads)
            {
                ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
                ptx.AppendLine($"    ld.global.{inputCache}.f32 %f1, [%rd12];");
                ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f2;");
                ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {Log2e};");
                ptx.AppendLine("    ex2.approx.f32 %f1, %f1;");
                ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
                ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
                ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
            }
        }
        if (variant.DoubleBufferedScratch)
        {
            ptx.AppendLine($"    add.u64 %rd5, %rd5, {reductionBytes};");
            ptx.AppendLine("    add.u64 %rd10, %rd5, %rd13;");          // &sumRed[tid]
        }
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0", blockThreads);
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");                // sumExp
        ptx.AppendLine("    rcp.approx.f32 %f4, %f3;");                  // 1/sumExp

        // ---- Pass 3: normalize the staged exponentials ----
        if (variant.RegisterStage)
        {
            for (int column = 0; column < n; column += blockThreads * 4)
            {
                int[] registers = VectorRegisters(column / (blockThreads * 4));
                ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
                foreach (int register in registers)
                    ptx.AppendLine($"    mul.rn.f32 %f{register}, %f{register}, %f4;");
                ptx.AppendLine($"    st.global.v4.f32 [%rd14], " +
                    $"{{%f{registers[0]},%f{registers[1]},%f{registers[2]},%f{registers[3]}}};");
            }
        }
        else if (vectorized)
        {
            for (int column = 0; column < n; column += blockThreads * 4)
            {
                ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
                ptx.AppendLine("    ld.global.ca.v4.f32 {%f1,%f5,%f6,%f7}, [%rd14];");
                foreach (int register in new[] { 1, 5, 6, 7 })
                    ptx.AppendLine($"    mul.rn.f32 %f{register}, %f{register}, %f4;");
                ptx.AppendLine("    st.global.v4.f32 [%rd14], {%f1,%f5,%f6,%f7};");
            }
        }
        else
        {
            for (int column = 0; column < n; column += blockThreads)
            {
                ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
                ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
                ptx.AppendLine("    ld.global.ca.f32 %f1, [%rd14];");
                ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f4;");
                ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
            }
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static int[] VectorRegisters(int pack) => pack switch
    {
        0 => new[] { 1, 5, 6, 7 },
        1 => new[] { 12, 13, 14, 15 },
        2 => new[] { 16, 17, 18, 19 },
        3 => new[] { 20, 21, 22, 23 },
        _ => throw new ArgumentOutOfRangeException(nameof(pack))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n, PtxSoftmaxVariant variant)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "softmax-row",
            Version: 3,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}-{variant.Name}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: variant.RegisterStage ? 64 : 32,
                MaxStaticSharedBytes: PtxRowReduce.SharedBytesFor(variant.BlockThreads) *
                    (variant.DoubleBufferedScratch ? 2 : 1),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: Math.Min(16, 1536 / variant.BlockThreads)),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[m,n] = exp(x[m,n] - rowMax[m]) / sum_n exp(x[m,n] - rowMax[m])",
                ["axis"] = "last",
                ["stability"] = "row-max-subtracted",
                ["reduction"] = PtxRowReduce.Strategy,
                ["reduction-scratch"] = variant.DoubleBufferedScratch
                    ? "double-buffered-max-and-sum" : "single-buffer-protected-reuse",
                ["rows-per-block"] = "1",
                ["block-threads"] = variant.BlockThreads.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["vector-width"] = variant.Vectorized ? "4" : "1",
                ["input-cache"] = variant.CacheInput ? "ca" : "nc",
                ["global-intermediates"] = "none",
                ["output-staging"] = variant.RegisterStage
                    ? "one-input-float4-per-lane-kept-through-both-reductions"
                    : "exponentials-normalized-in-place",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => PtxRowShape.IsPromoted(m, n);
}

internal readonly record struct PtxSoftmaxVariant(
    int BlockThreads,
    bool DoubleBufferedScratch,
    bool Vectorized,
    bool CacheInput,
    bool RegisterStage)
{
    internal static readonly PtxSoftmaxVariant Default = new(256, true, true, false, false);

    internal static PtxSoftmaxVariant ForShape(int n) => n == 1024
        ? new PtxSoftmaxVariant(64, true, true, false, true)
        : Default with { Vectorized = n % (PtxRowShape.BlockThreads * 4) == 0 };

    internal static IEnumerable<PtxSoftmaxVariant> SearchSpace(int n)
    {
        if (!PtxRowShape.IsSupported(64, n))
            throw new ArgumentOutOfRangeException(nameof(n),
                "Softmax variant search requires a supported row extent.");
        foreach (int threads in new[] { 64, 128, 256, 512 })
        {
            if (threads > n || n % threads != 0) continue;
            foreach (bool doubleBuffered in new[] { true, false })
            foreach (bool cacheInput in new[] { false, true })
            {
                yield return new PtxSoftmaxVariant(
                    threads, doubleBuffered, false, cacheInput, false);
                if (n % (threads * 4) == 0)
                {
                    yield return new PtxSoftmaxVariant(
                        threads, doubleBuffered, true, cacheInput, false);
                    int registerPacks = n / (threads * 4);
                    if (registerPacks is >= 1 and <= 4)
                        yield return new PtxSoftmaxVariant(
                            threads, doubleBuffered, true, cacheInput, true);
                }
            }
        }
    }

    internal string Name => $"t{BlockThreads}-{(DoubleBufferedScratch ? "db" : "sb")}-" +
        $"{(Vectorized ? "v4" : "s1")}-{(CacheInput ? "ca" : "nc")}-" +
        $"{(RegisterStage ? "reg" : "stage")}";

    internal void Validate(int n)
    {
        if (BlockThreads is not (64 or 128 or 256 or 512) ||
            BlockThreads > n || n % BlockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(BlockThreads),
                "Softmax variants require 64, 128, 256, or 512 threads that divide N.");
        if (Vectorized && n % (BlockThreads * 4) != 0)
            throw new ArgumentOutOfRangeException(nameof(Vectorized),
                "Vectorized softmax variants require N to be a multiple of four values per thread.");
        int registerPacks = n / (BlockThreads * 4);
        if (RegisterStage && (!Vectorized || registerPacks is < 1 or > 4))
            throw new ArgumentOutOfRangeException(nameof(RegisterStage),
                "Register-staged softmax requires one through four float4 packs per thread.");
    }
}
