using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact D=64 row-normalization specializations. One warp owns one row and
/// keeps both lane-owned values in registers across every reduction. The
/// operation and row count are baked into the module identity; only tensor
/// pointers reach the hot launch ABI.
/// </summary>
internal enum DirectPtxRowNormalizationOperation
{
    LayerNormForward,
    LayerNormBackwardInput,
    LayerNormGradParameters,
    Fp16LayerNormForward,
    Fp16LayerNormBackwardInput,
    Fp16LayerNormGradParameters,
    RmsNormForward,
    RmsNormBackwardInput,
    RmsNormGradGamma,
    NormAxis,
    NormBackward,
    NormalizeL2,
    ReduceNormL2,
    LayerNormGradParametersAtomic,
    RmsNormGradGammaAtomic,
    ReduceNormL2Atomic,
    LayerNormBackwardFusedAtomic,
    RmsNormBackwardFusedAtomic
}

internal sealed class PtxRowNormalizationD64Kernel : IDisposable
{
    internal const int Dimension = 64;
    internal const int WarpsPerBlock = 8;
    internal const int RowBlockThreads = WarpsPerBlock * 32;
    internal const int ParameterBlockThreads = 256;
    internal const int ParameterFeaturesPerBlock = 16;
    internal const int ParameterCohorts = ParameterBlockThreads / ParameterFeaturesPerBlock;
    internal const int ParameterPartialCohorts = ParameterCohorts / 2;
    internal const int ParameterRowsPerTile = 256;
    internal const int ReductionBlockThreads = 512;
    internal const int ReductionWarpsPerBlock = ReductionBlockThreads / 32;
    internal const int ReductionSharedBytes = ReductionWarpsPerBlock * sizeof(float);
    internal const int ReductionMaxBlocks = 128;
    internal const int ReductionAccumulatorBanks = 16;
    internal const int FusedLayerNormBackwardWarpsPerBlock = 32;
    internal const int FusedRmsNormBackwardWarpsPerBlock = 20;
    internal const int FusedLayerNormBackwardBlockThreads =
        FusedLayerNormBackwardWarpsPerBlock * 32;
    internal const int FusedRmsNormBackwardBlockThreads =
        FusedRmsNormBackwardWarpsPerBlock * 32;
    internal const int FusedLayerNormBackwardMaxBlocks = 128;
    internal const int FusedRmsNormBackwardMaxBlocks = 136;
    internal const int FusedLayerNormBackwardPlaneSharedBytes =
        FusedLayerNormBackwardWarpsPerBlock * Dimension * sizeof(float);
    internal const int FusedBackwardRmsSharedBytes =
        FusedRmsNormBackwardWarpsPerBlock * Dimension * sizeof(float);
    internal const int FusedBackwardLayerSharedBytes =
        2 * FusedLayerNormBackwardPlaneSharedBytes;
    internal const int FusedBackwardAccumulatorBanks = 4;
    // One fixed backend-owned workspace serves every row-normalization
    // specialization. The largest live layout is LayerNorm's four banks of
    // two D64 accumulators; the final word is a reusable completion counter.
    // Banking reduces cross-block atomic serialization while remaining fixed
    // size and small enough to prewarm once per backend.
    internal const int NormalizationWorkspacePartialFloats =
        FusedBackwardAccumulatorBanks * 2 * Dimension;
    internal const int NormalizationWorkspaceCounterIndex =
        NormalizationWorkspacePartialFloats;
    internal const int NormalizationWorkspaceElements =
        NormalizationWorkspacePartialFloats + 1;
    internal const int NormalizationWorkspaceBytes =
        NormalizationWorkspaceElements * sizeof(float);
    internal const int NormalizationWorkspaceCounterByteOffset =
        NormalizationWorkspaceCounterIndex * sizeof(float);
    internal const int ParameterSharedBytes =
        2 * ParameterFeaturesPerBlock * ParameterPartialCohorts * sizeof(float);

    private static readonly int[] SupportedRowCounts = [256, 2_048, 8_192];
    private static readonly string[] EntryPoints = CreateEntryPoints();
    private static readonly DirectPtxKernelBlueprint[,] AmpereBlueprints =
        CreateAmpereBlueprints();

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxRowNormalizationOperation Operation { get; }
    internal int Rows { get; }
    internal float Epsilon { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxRowNormalizationD64Kernel(
        DirectPtxRuntime runtime,
        DirectPtxRowNormalizationOperation operation,
        int rows,
        float epsilon = 1e-5f)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in row-normalization family is validated only on GA10x/SM86.");
        Validate(operation, rows, epsilon);

        Operation = operation;
        Rows = rows;
        Epsilon = epsilon;
        EntryPoint = GetEntryPoint(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation, rows);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            operation, rows, epsilon);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.NormalizationExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int blockThreads = IsFusedBackward(operation)
            ? GetFusedBackwardBlockThreads(operation)
            : IsParameterGradient(operation)
                ? ParameterBlockThreads
                : IsReduction(operation) ? ReductionBlockThreads : RowBlockThreads;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(
            EntryPoint, info, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            blockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(ReadOnlySpan<DirectPtxTensorView> tensors)
    {
        ValidateTensors(Blueprint, tensors, EntryPoint);
        LaunchPrevalidated(tensors);
    }

    internal unsafe void LaunchPrevalidated(ReadOnlySpan<DirectPtxTensorView> tensors)
    {
        IntPtr pointer0 = tensors[0].Pointer;
        IntPtr pointer1 = tensors[1].Pointer;
        IntPtr pointer2 = tensors.Length > 2 ? tensors[2].Pointer : IntPtr.Zero;
        IntPtr pointer3 = tensors.Length > 3 ? tensors[3].Pointer : IntPtr.Zero;
        IntPtr pointer4 = tensors.Length > 4 ? tensors[4].Pointer : IntPtr.Zero;
        IntPtr pointer5 = tensors.Length > 5 ? tensors[5].Pointer : IntPtr.Zero;
        IntPtr pointer6 = tensors.Length > 6 ? tensors[6].Pointer : IntPtr.Zero;
        IntPtr pointer7 = tensors.Length > 7 ? tensors[7].Pointer : IntPtr.Zero;
        IntPtr pointer8 = tensors.Length > 8 ? tensors[8].Pointer : IntPtr.Zero;
        void** arguments = stackalloc void*[9];
        arguments[0] = &pointer0;
        arguments[1] = &pointer1;
        arguments[2] = &pointer2;
        arguments[3] = &pointer3;
        arguments[4] = &pointer4;
        arguments[5] = &pointer5;
        arguments[6] = &pointer6;
        arguments[7] = &pointer7;
        arguments[8] = &pointer8;

        uint blockThreads = IsFusedBackward(Operation)
            ? (uint)GetFusedBackwardBlockThreads(Operation)
            : IsParameterGradient(Operation)
                ? (uint)ParameterBlockThreads
                : IsReduction(Operation) ? (uint)ReductionBlockThreads : (uint)RowBlockThreads;
        uint grid = IsFusedBackward(Operation)
            ? checked((uint)Math.Min(
                (Rows + GetFusedBackwardWarpsPerBlock(Operation) - 1) /
                    GetFusedBackwardWarpsPerBlock(Operation),
                GetFusedBackwardMaxBlocks(Operation)))
            : Operation == DirectPtxRowNormalizationOperation.ReduceNormL2Atomic
            ? checked((uint)GetReductionBlockCount(Rows))
            : IsAtomicParameterGradient(Operation)
            ? checked((uint)(Dimension / ParameterFeaturesPerBlock *
                (Rows / Math.Min(ParameterRowsPerTile, Rows))))
            : IsParameterGradient(Operation)
                ? Dimension / ParameterFeaturesPerBlock
            : Operation == DirectPtxRowNormalizationOperation.ReduceNormL2
                ? 1u
                : checked((uint)(Rows / WarpsPerBlock));
        _module.Launch(
            _function, grid, 1, 1, blockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedRows(int rows) =>
        rows is 256 or 2_048 or 8_192;

    internal static bool IsPromoted(
        DirectPtxRowNormalizationOperation operation, int rows) => false;

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxRowNormalizationOperation operation,
        int rows,
        float epsilon = 1e-5f)
    {
        Validate(operation, rows, epsilon);
        return operation switch
        {
            DirectPtxRowNormalizationOperation.LayerNormForward =>
                EmitLayerNormForward(ccMajor, ccMinor, rows, epsilon),
            DirectPtxRowNormalizationOperation.LayerNormBackwardInput =>
                EmitLayerNormBackwardInput(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.LayerNormGradParameters =>
                EmitLayerNormGradParameters(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.Fp16LayerNormForward =>
                EmitFp16LayerNormForward(ccMajor, ccMinor, rows, epsilon),
            DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput =>
                EmitFp16LayerNormBackwardInput(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters =>
                EmitFp16LayerNormGradParameters(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.RmsNormForward =>
                EmitRmsNormForward(ccMajor, ccMinor, rows, epsilon),
            DirectPtxRowNormalizationOperation.RmsNormBackwardInput =>
                EmitRmsNormBackwardInput(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.RmsNormGradGamma =>
                EmitRmsNormGradGamma(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.NormAxis =>
                EmitNormAxis(ccMajor, ccMinor, rows, writeNormalized: false),
            DirectPtxRowNormalizationOperation.NormalizeL2 =>
                EmitNormAxis(ccMajor, ccMinor, rows, writeNormalized: true),
            DirectPtxRowNormalizationOperation.NormBackward =>
                EmitNormBackward(ccMajor, ccMinor, rows),
            DirectPtxRowNormalizationOperation.ReduceNormL2 =>
                EmitReduceNormL2(ccMajor, ccMinor, rows, atomic: false),
            DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic =>
                EmitParameterGradient(ccMajor, ccMinor, rows, rms: false, atomic: true),
            DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic =>
                EmitParameterGradient(ccMajor, ccMinor, rows, rms: true, atomic: true),
            DirectPtxRowNormalizationOperation.ReduceNormL2Atomic =>
                EmitReduceNormL2(ccMajor, ccMinor, rows, atomic: true),
            DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic =>
                EmitFusedBackward(ccMajor, ccMinor, rows, rms: false),
            DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic =>
                EmitFusedBackward(ccMajor, ccMinor, rows, rms: true),
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
    }

    private static string EmitLayerNormForward(
        int ccMajor, int ccMinor, int rows, float epsilon)
    {
        var ptx = Begin(ccMajor, ccMinor, GetEntryPoint(DirectPtxRowNormalizationOperation.LayerNormForward),
            "input_ptr", "gamma_ptr", "beta_ptr", "output_ptr", "mean_ptr", "inv_var_ptr");
        EmitRegisters(ptx, predicates: 3, b32: 16, b64: 20, f32: 32);
        EmitPointerLoads(ptx, 6);
        EmitRowAddressSetup(ptx, rows, "LN_FWD_DONE");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd8];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd8+128];");
        ptx.AppendLine("    add.rn.f32 %f2, %f0, %f1;");
        EmitWarpSum(ptx, "%f2", "%f3");
        ptx.AppendLine("    mul.rn.f32 %f4, %f2, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f5, %f0, %f4;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f1, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f7, %f6, %f6, %f7;");
        EmitWarpSum(ptx, "%f7", "%f8");
        ptx.AppendLine("    mul.rn.f32 %f9, %f7, 0f3C800000;");
        ptx.AppendLine($"    add.rn.f32 %f9, %f9, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f10, %f9;");
        ptx.AppendLine("    rcp.approx.f32 %f11, %f10;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f12, [%rd9];");
        ptx.AppendLine("    ld.global.nc.f32 %f13, [%rd9+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f14, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f15, [%rd10+128];");
        ptx.AppendLine("    mul.rn.f32 %f16, %f5, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f16, %f16, %f12, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f6, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f17, %f17, %f13, %f15;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd6;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f16;");
        ptx.AppendLine("    st.global.f32 [%rd11+128], %f17;");
        EmitStatStores(ptx, meanPointerRegister: 4, secondPointerRegister: 5,
            firstValue: "%f4", secondValue: "%f11");
        return End(ptx, "LN_FWD_DONE");
    }

    private static string EmitFp16LayerNormForward(
        int ccMajor, int ccMinor, int rows, float epsilon)
    {
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(DirectPtxRowNormalizationOperation.Fp16LayerNormForward),
            "input_ptr", "gamma_ptr", "beta_ptr", "output_ptr", "mean_ptr", "variance_ptr");
        EmitRegisters(ptx, predicates: 3, b32: 16, b64: 20, f32: 32, b16: 8);
        EmitPointerLoads(ptx, 6);
        EmitFp16RowAddressSetup(ptx, rows, "FP16_LN_FWD_DONE");
        ptx.AppendLine("    ld.global.nc.u16 %rs0, [%rd8];");
        ptx.AppendLine("    ld.global.nc.u16 %rs1, [%rd8+64];");
        ptx.AppendLine("    cvt.f32.f16 %f0, %rs0;");
        ptx.AppendLine("    cvt.f32.f16 %f1, %rs1;");
        ptx.AppendLine("    add.rn.f32 %f2, %f0, %f1;");
        EmitWarpSum(ptx, "%f2", "%f3");
        ptx.AppendLine("    mul.rn.f32 %f4, %f2, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f5, %f0, %f4;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f1, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f7, %f6, %f6, %f7;");
        EmitWarpSum(ptx, "%f7", "%f8");
        ptx.AppendLine("    mul.rn.f32 %f9, %f7, 0f3C800000;");
        ptx.AppendLine("    mov.f32 %f18, %f9;");
        ptx.AppendLine($"    add.rn.f32 %f9, %f9, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f10, %f9;");
        ptx.AppendLine("    rcp.approx.f32 %f11, %f10;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.u16 %rs2, [%rd9];");
        ptx.AppendLine("    ld.global.nc.u16 %rs3, [%rd9+64];");
        ptx.AppendLine("    ld.global.nc.u16 %rs4, [%rd10];");
        ptx.AppendLine("    ld.global.nc.u16 %rs5, [%rd10+64];");
        ptx.AppendLine("    cvt.f32.f16 %f12, %rs2;");
        ptx.AppendLine("    cvt.f32.f16 %f13, %rs3;");
        ptx.AppendLine("    cvt.f32.f16 %f14, %rs4;");
        ptx.AppendLine("    cvt.f32.f16 %f15, %rs5;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f5, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f16, %f16, %f12, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f6, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f17, %f17, %f13, %f15;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs6, %f16;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs7, %f17;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd6;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd7;");
        ptx.AppendLine("    st.global.u16 [%rd11], %rs6;");
        ptx.AppendLine("    st.global.u16 [%rd11+64], %rs7;");
        EmitStatStores(ptx, meanPointerRegister: 4, secondPointerRegister: 5,
            firstValue: "%f4", secondValue: "%f18");
        return End(ptx, "FP16_LN_FWD_DONE");
    }

    private static string EmitFp16LayerNormBackwardInput(
        int ccMajor, int ccMinor, int rows)
    {
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput),
            "grad_output_ptr", "input_ptr", "gamma_ptr", "mean_ptr", "inv_var_ptr", "grad_input_ptr");
        EmitRegisters(ptx, predicates: 2, b32: 16, b64: 22, f32: 36, b16: 8);
        EmitPointerLoads(ptx, 6);
        EmitFp16RowAddressSetup(ptx, rows, "FP16_LN_BWD_DONE");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd7;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.u16 %rs0, [%rd9];");
        ptx.AppendLine("    ld.global.nc.u16 %rs1, [%rd9+64];");
        ptx.AppendLine("    ld.global.nc.u16 %rs2, [%rd10];");
        ptx.AppendLine("    ld.global.nc.u16 %rs3, [%rd10+64];");
        ptx.AppendLine("    ld.global.nc.u16 %rs4, [%rd11];");
        ptx.AppendLine("    ld.global.nc.u16 %rs5, [%rd11+64];");
        ptx.AppendLine("    cvt.f32.f16 %f0, %rs0;");
        ptx.AppendLine("    cvt.f32.f16 %f1, %rs1;");
        ptx.AppendLine("    cvt.f32.f16 %f2, %rs2;");
        ptx.AppendLine("    cvt.f32.f16 %f3, %rs3;");
        ptx.AppendLine("    cvt.f32.f16 %f4, %rs4;");
        ptx.AppendLine("    cvt.f32.f16 %f5, %rs5;");
        EmitRowStatLoads(ptx, meanPointerRegister: 3, secondPointerRegister: 4,
            firstValue: "%f6", secondValue: "%f7");
        ptx.AppendLine("    sub.rn.f32 %f8, %f2, %f6;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f3, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f0, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f1, %f5;");
        ptx.AppendLine("    add.rn.f32 %f12, %f10, %f11;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f10, %f8;");
        ptx.AppendLine("    fma.rn.f32 %f13, %f11, %f9, %f13;");
        EmitWarpSum(ptx, "%f12", "%f14");
        EmitWarpSum(ptx, "%f13", "%f14");
        ptx.AppendLine("    mul.rn.f32 %f15, %f7, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f8, %f15;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f16, %f13;");
        ptx.AppendLine("    add.rn.f32 %f16, %f16, %f12;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f16, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f17, %f10, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f17, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f9, %f15;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f18, %f13;");
        ptx.AppendLine("    add.rn.f32 %f18, %f18, %f12;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f18, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f19, %f11, %f18;");
        ptx.AppendLine("    mul.rn.f32 %f19, %f19, %f7;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs6, %f17;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs7, %f19;");
        ptx.AppendLine("    add.u64 %rd13, %rd5, %rd6;");
        ptx.AppendLine("    add.u64 %rd13, %rd13, %rd7;");
        ptx.AppendLine("    st.global.u16 [%rd13], %rs6;");
        ptx.AppendLine("    st.global.u16 [%rd13+64], %rs7;");
        return End(ptx, "FP16_LN_BWD_DONE");
    }

    private static string EmitFp16LayerNormGradParameters(
        int ccMajor, int ccMinor, int rows)
    {
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters),
            "grad_output_ptr", "input_ptr", "mean_ptr", "inv_var_ptr",
            "grad_gamma_ptr", "grad_beta_ptr");
        EmitRegisters(ptx, predicates: 4, b32: 12, b64: 16, f32: 10,
            blockThreads: ParameterBlockThreads, b16: 4);
        ptx.AppendLine($"    .shared .align 4 .b8 parameter_scratch[{ParameterSharedBytes}];");
        EmitPointerLoads(ptx, 6);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine($"    and.b32 %r1, %r0, {ParameterFeaturesPerBlock - 1};");
        ptx.AppendLine("    shr.u32 %r2, %r0, 4;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r3, %r3, {ParameterFeaturesPerBlock}, %r1;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 2;");
        ptx.AppendLine("    mov.u32 %r4, %r2;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("FP16_PARAM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {rows};");
        ptx.AppendLine("    @%p0 bra FP16_PARAM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 128;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    ld.global.nc.u16 %rs0, [%rd8];");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.u16 %rs1, [%rd8];");
        ptx.AppendLine("    cvt.f32.f16 %f2, %rs0;");
        ptx.AppendLine("    cvt.f32.f16 %f3, %rs1;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd10];");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd10];");
        ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f6, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f6, %f0;");
        ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        ptx.AppendLine($"    add.u32 %r4, %r4, {ParameterCohorts};");
        ptx.AppendLine("    bra FP16_PARAM_LOOP;");
        ptx.AppendLine("FP16_PARAM_DONE:");
        EmitParameterBlockReduction(ptx, "FP16_PARAM", beta: true);
        ptx.AppendLine("    cvt.rn.f16.f32 %rs2, %f0;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs3, %f1;");
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd5, %rd6;");
        ptx.AppendLine("    st.global.u16 [%rd8], %rs2;");
        ptx.AppendLine("    st.global.u16 [%rd9], %rs3;");
        ptx.AppendLine("FP16_PARAM_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string EmitRmsNormForward(
        int ccMajor, int ccMinor, int rows, float epsilon)
    {
        var ptx = Begin(ccMajor, ccMinor, GetEntryPoint(DirectPtxRowNormalizationOperation.RmsNormForward),
            "input_ptr", "gamma_ptr", "output_ptr", "rms_ptr");
        EmitRegisters(ptx, predicates: 3, b32: 16, b64: 18, f32: 24);
        EmitPointerLoads(ptx, 4);
        EmitRowAddressSetup(ptx, rows, "RMS_FWD_DONE");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd8];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd8+128];");
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f2, %f1, %f1, %f2;");
        EmitWarpSum(ptx, "%f2", "%f3");
        ptx.AppendLine("    mul.rn.f32 %f4, %f2, 0f3C800000;");
        ptx.AppendLine($"    add.rn.f32 %f4, %f4, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f5, %f4;");
        ptx.AppendLine("    rcp.approx.f32 %f6, %f5;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f7, [%rd9];");
        ptx.AppendLine("    ld.global.nc.f32 %f8, [%rd9+128];");
        ptx.AppendLine("    mul.rn.f32 %f9, %f0, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f9, %f9, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f1, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f10, %f8;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd6;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f9;");
        ptx.AppendLine("    st.global.f32 [%rd10+128], %f10;");
        EmitSingleStatStore(ptx, pointerRegister: 3, value: "%f5");
        return End(ptx, "RMS_FWD_DONE");
    }

    private static string EmitLayerNormBackwardInput(
        int ccMajor, int ccMinor, int rows)
    {
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(DirectPtxRowNormalizationOperation.LayerNormBackwardInput),
            "grad_output_ptr", "input_ptr", "gamma_ptr", "mean_ptr", "inv_var_ptr", "grad_input_ptr");
        EmitRegisters(ptx, predicates: 2, b32: 16, b64: 22, f32: 36);
        EmitPointerLoads(ptx, 6);
        EmitRowAddressSetup(ptx, rows, "LN_BWD_DONE");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd7;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd9];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd9+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd10+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd11];");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd11+128];");
        EmitRowStatLoads(ptx, meanPointerRegister: 3, secondPointerRegister: 4,
            firstValue: "%f6", secondValue: "%f7");
        ptx.AppendLine("    sub.rn.f32 %f8, %f2, %f6;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f3, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f0, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f1, %f5;");
        ptx.AppendLine("    add.rn.f32 %f12, %f10, %f11;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f10, %f8;");
        ptx.AppendLine("    fma.rn.f32 %f13, %f11, %f9, %f13;");
        EmitWarpSum(ptx, "%f12", "%f14");
        EmitWarpSum(ptx, "%f13", "%f14");
        ptx.AppendLine("    mul.rn.f32 %f15, %f7, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f8, %f15;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f16, %f13;");
        ptx.AppendLine("    add.rn.f32 %f16, %f16, %f12;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f16, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f17, %f10, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f17, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f9, %f15;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f18, %f13;");
        ptx.AppendLine("    add.rn.f32 %f18, %f18, %f12;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f18, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f19, %f11, %f18;");
        ptx.AppendLine("    mul.rn.f32 %f19, %f19, %f7;");
        ptx.AppendLine("    add.u64 %rd13, %rd5, %rd6;");
        ptx.AppendLine("    add.u64 %rd13, %rd13, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f17;");
        ptx.AppendLine("    st.global.f32 [%rd13+128], %f19;");
        return End(ptx, "LN_BWD_DONE");
    }

    private static string EmitRmsNormBackwardInput(
        int ccMajor, int ccMinor, int rows)
    {
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(DirectPtxRowNormalizationOperation.RmsNormBackwardInput),
            "grad_output_ptr", "input_ptr", "gamma_ptr", "rms_ptr", "grad_input_ptr");
        EmitRegisters(ptx, predicates: 2, b32: 16, b64: 20, f32: 30);
        EmitPointerLoads(ptx, 5);
        EmitRowAddressSetup(ptx, rows, "RMS_BWD_DONE");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd7;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd9];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd9+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd10+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd11];");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd11+128];");
        EmitSingleRowStatLoad(ptx, pointerRegister: 3, value: "%f6");
        ptx.AppendLine("    rcp.approx.f32 %f7, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f8, %f7, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f8, %f8, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f9, %f0, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f1, %f5;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f9, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f11, %f10, %f3, %f11;");
        EmitWarpSum(ptx, "%f11", "%f12");
        ptx.AppendLine("    mul.rn.f32 %f13, %f9, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f2, %f11;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f14, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f14, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f15, %f13, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f16, %f10, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f3, %f11;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f17, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f17, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f18, %f16, %f17;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd13, %rd13, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f15;");
        ptx.AppendLine("    st.global.f32 [%rd13+128], %f18;");
        return End(ptx, "RMS_BWD_DONE");
    }

    private static string EmitFusedBackward(
        int ccMajor, int ccMinor, int rows, bool rms)
    {
        DirectPtxRowNormalizationOperation operation = rms
            ? DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic
            : DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic;
        var ptx = rms
            ? Begin(ccMajor, ccMinor, GetEntryPoint(operation),
                "grad_output_ptr", "input_ptr", "gamma_ptr", "rms_ptr",
                "grad_input_ptr", "grad_gamma_ptr", "workspace_ptr")
            : Begin(ccMajor, ccMinor, GetEntryPoint(operation),
                "grad_output_ptr", "input_ptr", "gamma_ptr", "mean_ptr",
                "inv_var_ptr", "grad_input_ptr", "grad_gamma_ptr", "grad_beta_ptr",
                "workspace_ptr");
        int warpsPerBlock = GetFusedBackwardWarpsPerBlock(operation);
        int blockThreads = GetFusedBackwardBlockThreads(operation);
        int planeSharedBytes = rms
            ? FusedBackwardRmsSharedBytes
            : FusedLayerNormBackwardPlaneSharedBytes;
        int accumulatorBankBytes = (rms ? Dimension : 2 * Dimension) * sizeof(float);
        EmitRegisters(ptx, predicates: 6, b32: 16, b64: 24, f32: 36,
            blockThreads: blockThreads);
        ptx.AppendLine($"    .shared .align 16 .b8 fused_scratch[{(rms ? FusedBackwardRmsSharedBytes : FusedBackwardLayerSharedBytes)}];");
        EmitPointerLoads(ptx, rms ? 7 : 9);
        ptx.AppendLine($"    mov.u64 %rd23, %rd{(rms ? 6 : 8)};");

        // A bounded grid lets every warp process a grid-stride row
        // cohort. Input, gradient, gamma, and row statistics stay in registers
        // until grad-input and the block's parameter partials are complete.
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {warpsPerBlock}, %r2;");
        ptx.AppendLine("    mov.u32 %r5, %nctaid.x;");
        ptx.AppendLine($"    mul.lo.u32 %r5, %r5, {warpsPerBlock};");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r1, 8;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.v2.f32 {%f4, %f5}, [%rd9];");
        ptx.AppendLine("    mov.f32 %f20, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f21, 0f00000000;");
        if (!rms)
        {
            ptx.AppendLine("    mov.f32 %f22, 0f00000000;");
            ptx.AppendLine("    mov.f32 %f23, 0f00000000;");
        }
        ptx.AppendLine("FUSED_BWD_ROW_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {rows};");
        ptx.AppendLine("    @%p0 bra FUSED_BWD_PARTIALS;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r4, 256;");
        ptx.AppendLine("    add.u64 %rd11, %rd0, %rd10;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd8;");
        ptx.AppendLine("    add.u64 %rd12, %rd1, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd8;");
        ptx.AppendLine($"    add.u64 %rd13, %rd{(rms ? 4 : 5)}, %rd10;");
        ptx.AppendLine("    add.u64 %rd13, %rd13, %rd8;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0, %f1}, [%rd11];");
        ptx.AppendLine("    ld.global.v2.f32 {%f2, %f3}, [%rd12];");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd15, %rd3, %rd14;");
        ptx.AppendLine("    ld.global.f32 %f6, [%rd15];");
        if (rms)
        {
            ptx.AppendLine("    rcp.approx.f32 %f7, %f6;");
            ptx.AppendLine("    mul.rn.f32 %f8, %f7, %f7;");
            ptx.AppendLine("    mul.rn.f32 %f8, %f8, %f7;");
            ptx.AppendLine("    mul.rn.f32 %f9, %f0, %f4;");
            ptx.AppendLine("    mul.rn.f32 %f10, %f1, %f5;");
            ptx.AppendLine("    mul.rn.f32 %f11, %f9, %f2;");
            ptx.AppendLine("    fma.rn.f32 %f11, %f10, %f3, %f11;");
            EmitWarpSum(ptx, "%f11", "%f12");
            ptx.AppendLine("    mul.rn.f32 %f14, %f11, %f8;");
            ptx.AppendLine("    mul.rn.f32 %f14, %f14, 0fBC800000;");
            ptx.AppendLine("    mul.rn.f32 %f13, %f9, %f7;");
            ptx.AppendLine("    fma.rn.f32 %f15, %f2, %f14, %f13;");
            ptx.AppendLine("    mul.rn.f32 %f16, %f10, %f7;");
            ptx.AppendLine("    fma.rn.f32 %f18, %f3, %f14, %f16;");
            ptx.AppendLine("    mul.rn.f32 %f19, %f0, %f2;");
            ptx.AppendLine("    fma.rn.f32 %f20, %f19, %f7, %f20;");
            ptx.AppendLine("    mul.rn.f32 %f19, %f1, %f3;");
            ptx.AppendLine("    fma.rn.f32 %f21, %f19, %f7, %f21;");
            ptx.AppendLine("    st.global.v2.f32 [%rd13], {%f15, %f18};");
        }
        else
        {
            ptx.AppendLine("    add.u64 %rd16, %rd4, %rd14;");
            ptx.AppendLine("    ld.global.f32 %f7, [%rd16];");
            ptx.AppendLine("    sub.rn.f32 %f8, %f2, %f6;");
            ptx.AppendLine("    sub.rn.f32 %f9, %f3, %f6;");
            ptx.AppendLine("    mul.rn.f32 %f10, %f0, %f4;");
            ptx.AppendLine("    mul.rn.f32 %f11, %f1, %f5;");
            ptx.AppendLine("    add.rn.f32 %f12, %f10, %f11;");
            ptx.AppendLine("    mul.rn.f32 %f13, %f10, %f8;");
            ptx.AppendLine("    fma.rn.f32 %f13, %f11, %f9, %f13;");
            EmitWarpSum(ptx, "%f12", "%f14");
            EmitWarpSum(ptx, "%f13", "%f14");
            ptx.AppendLine("    mul.rn.f32 %f15, %f7, %f7;");
            ptx.AppendLine("    mul.rn.f32 %f16, %f8, %f15;");
            ptx.AppendLine("    mul.rn.f32 %f16, %f16, %f13;");
            ptx.AppendLine("    add.rn.f32 %f16, %f16, %f12;");
            ptx.AppendLine("    mul.rn.f32 %f16, %f16, 0f3C800000;");
            ptx.AppendLine("    sub.rn.f32 %f17, %f10, %f16;");
            ptx.AppendLine("    mul.rn.f32 %f17, %f17, %f7;");
            ptx.AppendLine("    mul.rn.f32 %f18, %f9, %f15;");
            ptx.AppendLine("    mul.rn.f32 %f18, %f18, %f13;");
            ptx.AppendLine("    add.rn.f32 %f18, %f18, %f12;");
            ptx.AppendLine("    mul.rn.f32 %f18, %f18, 0f3C800000;");
            ptx.AppendLine("    sub.rn.f32 %f19, %f11, %f18;");
            ptx.AppendLine("    mul.rn.f32 %f19, %f19, %f7;");
            ptx.AppendLine("    mul.rn.f32 %f24, %f8, %f7;");
            ptx.AppendLine("    fma.rn.f32 %f20, %f0, %f24, %f20;");
            ptx.AppendLine("    mul.rn.f32 %f24, %f9, %f7;");
            ptx.AppendLine("    fma.rn.f32 %f21, %f1, %f24, %f21;");
            ptx.AppendLine("    add.rn.f32 %f22, %f22, %f0;");
            ptx.AppendLine("    add.rn.f32 %f23, %f23, %f1;");
            ptx.AppendLine("    st.global.v2.f32 [%rd13], {%f17, %f19};");
        }
        ptx.AppendLine("    add.u32 %r4, %r4, %r5;");
        ptx.AppendLine("    bra FUSED_BWD_ROW_LOOP;");

        // Every warp publishes its 64 feature partials once. Sixty-four
        // threads fold the warp cohorts and atomically accumulate one value
        // per block/output into a reusable workspace. The last block copies
        // and clears those accumulators, eliminating per-launch output memsets
        // without a second global-memory read pass over every block partial.
        ptx.AppendLine("FUSED_BWD_PARTIALS:");
        ptx.AppendLine($"    mad.lo.u32 %r6, %r2, {Dimension / 2}, %r1;");
        ptx.AppendLine("    mul.lo.u32 %r6, %r6, 2;");
        ptx.AppendLine("    mul.wide.u32 %rd17, %r6, 4;");
        ptx.AppendLine("    mov.u64 %rd18, fused_scratch;");
        ptx.AppendLine("    add.u64 %rd19, %rd18, %rd17;");
        ptx.AppendLine("    st.shared.f32 [%rd19], %f20;");
        ptx.AppendLine("    st.shared.f32 [%rd19+4], %f21;");
        if (!rms)
        {
            ptx.AppendLine($"    st.shared.f32 [%rd19+{planeSharedBytes}], %f22;");
            ptx.AppendLine($"    st.shared.f32 [%rd19+{planeSharedBytes + 4}], %f23;");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r0, {Dimension};");
        ptx.AppendLine("    @%p2 bra FUSED_BWD_PUBLISH_WAIT;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("    mov.f32 %f24, 0f00000000;");
        if (!rms)
            ptx.AppendLine("    mov.f32 %f25, 0f00000000;");
        ptx.AppendLine("FUSED_BWD_BLOCK_REDUCE:");
        ptx.AppendLine($"    setp.ge.u32 %p3, %r6, {warpsPerBlock};");
        ptx.AppendLine("    @%p3 bra FUSED_BWD_PUBLISH;");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r6, {Dimension}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd17, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd19, %rd18, %rd17;");
        ptx.AppendLine("    ld.shared.f32 %f26, [%rd19];");
        ptx.AppendLine("    add.rn.f32 %f24, %f24, %f26;");
        if (!rms)
        {
            ptx.AppendLine($"    ld.shared.f32 %f27, [%rd19+{planeSharedBytes}];");
            ptx.AppendLine("    add.rn.f32 %f25, %f25, %f27;");
        }
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra FUSED_BWD_BLOCK_REDUCE;");
        ptx.AppendLine("FUSED_BWD_PUBLISH:");
        ptx.AppendLine("    mov.u32 %r10, %ctaid.x;");
        ptx.AppendLine($"    and.b32 %r10, %r10, {FusedBackwardAccumulatorBanks - 1};");
        ptx.AppendLine($"    mul.wide.u32 %rd17, %r10, {accumulatorBankBytes};");
        ptx.AppendLine("    add.u64 %rd17, %rd23, %rd17;");
        ptx.AppendLine("    mul.wide.u32 %rd20, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd21, %rd17, %rd20;");
        ptx.AppendLine("    red.global.add.f32 [%rd21], %f24;");
        if (!rms)
        {
            ptx.AppendLine($"    add.u64 %rd22, %rd17, {Dimension * sizeof(float)};");
            ptx.AppendLine("    add.u64 %rd22, %rd22, %rd20;");
            ptx.AppendLine("    red.global.add.f32 [%rd22], %f25;");
        }

        // CUDA's last-block completion protocol: all producers make their
        // accumulator atomics globally visible before one integer completion
        // atomic per block. The last block stores final outputs directly and
        // resets both accumulators and counter for graph replay.
        ptx.AppendLine("FUSED_BWD_PUBLISH_WAIT:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p4, %r0, 0;");
        ptx.AppendLine("    @%p4 bra FUSED_BWD_TICKET_WAIT;");
        ptx.AppendLine("    membar.gl;");
        ptx.AppendLine($"    add.u64 %rd20, %rd23, {NormalizationWorkspaceCounterByteOffset};");
        ptx.AppendLine("    atom.global.add.u32 %r8, [%rd20], 1;");
        ptx.AppendLine("    mov.u32 %r9, %nctaid.x;");
        ptx.AppendLine("    sub.u32 %r9, %r9, 1;");
        ptx.AppendLine("    setp.eq.u32 %p4, %r8, %r9;");
        ptx.AppendLine("    selp.u32 %r8, 1, 0, %p4;");
        ptx.AppendLine("    mov.u64 %rd18, fused_scratch;");
        ptx.AppendLine("    st.shared.u32 [%rd18], %r8;");
        ptx.AppendLine("FUSED_BWD_TICKET_WAIT:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    mov.u64 %rd18, fused_scratch;");
        ptx.AppendLine("    ld.shared.u32 %r8, [%rd18];");
        ptx.AppendLine("    setp.eq.u32 %p4, %r8, 0;");
        ptx.AppendLine("    @%p4 bra FUSED_BWD_RETURN;");
        ptx.AppendLine($"    setp.ge.u32 %p4, %r0, {Dimension};");
        ptx.AppendLine("    @%p4 bra FUSED_BWD_RETURN;");
        ptx.AppendLine("    mul.wide.u32 %rd20, %r0, 4;");
        ptx.AppendLine("    mov.u32 %r10, 0;");
        ptx.AppendLine("    mov.f32 %f24, 0f00000000;");
        if (!rms)
            ptx.AppendLine("    mov.f32 %f25, 0f00000000;");
        ptx.AppendLine("FUSED_BWD_BANK_REDUCE:");
        ptx.AppendLine($"    setp.ge.u32 %p5, %r10, {FusedBackwardAccumulatorBanks};");
        ptx.AppendLine("    @%p5 bra FUSED_BWD_BANK_WRITE;");
        ptx.AppendLine($"    mul.wide.u32 %rd17, %r10, {accumulatorBankBytes};");
        ptx.AppendLine("    add.u64 %rd17, %rd23, %rd17;");
        ptx.AppendLine("    add.u64 %rd21, %rd17, %rd20;");
        ptx.AppendLine("    ld.global.f32 %f26, [%rd21];");
        ptx.AppendLine("    add.rn.f32 %f24, %f24, %f26;");
        ptx.AppendLine("    mov.f32 %f26, 0f00000000;");
        ptx.AppendLine("    st.global.f32 [%rd21], %f26;");
        if (!rms)
        {
            ptx.AppendLine($"    add.u64 %rd21, %rd17, {Dimension * sizeof(float)};");
            ptx.AppendLine("    add.u64 %rd21, %rd21, %rd20;");
            ptx.AppendLine("    ld.global.f32 %f27, [%rd21];");
            ptx.AppendLine("    add.rn.f32 %f25, %f25, %f27;");
            ptx.AppendLine("    st.global.f32 [%rd21], %f26;");
        }
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine("    bra FUSED_BWD_BANK_REDUCE;");
        ptx.AppendLine("FUSED_BWD_BANK_WRITE:");
        ptx.AppendLine($"    add.u64 %rd22, %rd{(rms ? 5 : 6)}, %rd20;");
        ptx.AppendLine("    st.global.f32 [%rd22], %f24;");
        if (!rms)
        {
            ptx.AppendLine("    add.u64 %rd22, %rd7, %rd20;");
            ptx.AppendLine("    st.global.f32 [%rd22], %f25;");
        }
        ptx.AppendLine("    setp.ne.u32 %p4, %r0, 0;");
        ptx.AppendLine("    @%p4 bra FUSED_BWD_RETURN;");
        ptx.AppendLine($"    add.u64 %rd20, %rd23, {NormalizationWorkspaceCounterByteOffset};");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("    st.global.u32 [%rd20], %r8;");
        return End(ptx, "FUSED_BWD_RETURN");
    }

    private static string EmitLayerNormGradParameters(
        int ccMajor, int ccMinor, int rows) =>
        EmitParameterGradient(ccMajor, ccMinor, rows, rms: false, atomic: false);

    private static string EmitRmsNormGradGamma(
        int ccMajor, int ccMinor, int rows) =>
        EmitParameterGradient(ccMajor, ccMinor, rows, rms: true, atomic: false);

    private static string EmitParameterGradient(
        int ccMajor, int ccMinor, int rows, bool rms, bool atomic)
    {
        int rowsPerTile = Math.Min(ParameterRowsPerTile, rows);
        DirectPtxRowNormalizationOperation operation = atomic
            ? rms
                ? DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic
                : DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic
            : rms
                ? DirectPtxRowNormalizationOperation.RmsNormGradGamma
                : DirectPtxRowNormalizationOperation.LayerNormGradParameters;
        var ptx = rms
            ? Begin(ccMajor, ccMinor, GetEntryPoint(operation),
                "grad_output_ptr", "input_ptr", "rms_ptr", "grad_gamma_ptr")
            : Begin(ccMajor, ccMinor, GetEntryPoint(operation),
                "grad_output_ptr", "input_ptr", "mean_ptr", "inv_var_ptr",
                "grad_gamma_ptr", "grad_beta_ptr");
        EmitRegisters(ptx, predicates: 4, b32: 12, b64: 16, f32: 10,
            blockThreads: ParameterBlockThreads);
        ptx.AppendLine($"    .shared .align 4 .b8 parameter_scratch[{ParameterSharedBytes}];");
        EmitPointerLoads(ptx, rms ? 4 : 6);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine($"    and.b32 %r1, %r0, {ParameterFeaturesPerBlock - 1};");
        ptx.AppendLine("    shr.u32 %r2, %r0, 4;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        if (atomic)
            ptx.AppendLine("    and.b32 %r3, %r3, 3;");
        ptx.AppendLine($"    mad.lo.u32 %r3, %r3, {ParameterFeaturesPerBlock}, %r1;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        if (atomic)
        {
            ptx.AppendLine("    mov.u32 %r7, %ctaid.x;");
            ptx.AppendLine("    shr.u32 %r7, %r7, 2;");
            ptx.AppendLine($"    mad.lo.u32 %r4, %r7, {rowsPerTile}, %r2;");
            ptx.AppendLine("    add.u32 %r7, %r7, 1;");
            ptx.AppendLine($"    mul.lo.u32 %r7, %r7, {rowsPerTile};");
        }
        else
        {
            ptx.AppendLine("    mov.u32 %r4, %r2;");
        }
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        if (!rms)
            ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("PARAM_LOOP:");
        ptx.AppendLine(atomic
            ? "    setp.ge.u32 %p0, %r4, %r7;"
            : $"    setp.ge.u32 %p0, %r4, {rows};");
        ptx.AppendLine("    @%p0 bra PARAM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 256;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd8];");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd8];");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r4, 4;");
        if (rms)
        {
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd10];");
            ptx.AppendLine("    rcp.approx.f32 %f5, %f4;");
            ptx.AppendLine("    mul.rn.f32 %f6, %f2, %f3;");
            ptx.AppendLine("    fma.rn.f32 %f0, %f6, %f5, %f0;");
        }
        else
        {
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd10];");
            ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd10];");
            ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f4;");
            ptx.AppendLine("    mul.rn.f32 %f6, %f6, %f5;");
            ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f6, %f0;");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        }
        ptx.AppendLine($"    add.u32 %r4, %r4, {ParameterCohorts};");
        ptx.AppendLine("    bra PARAM_LOOP;");
        ptx.AppendLine("PARAM_DONE:");
        EmitParameterBlockReduction(ptx, "PARAM", beta: !rms);
        if (rms)
        {
            ptx.AppendLine("    add.u64 %rd8, %rd3, %rd6;");
            ptx.AppendLine(atomic
                ? "    atom.global.add.f32 %f2, [%rd8], %f0;"
                : "    st.global.f32 [%rd8], %f0;");
        }
        else
        {
            ptx.AppendLine("    add.u64 %rd8, %rd4, %rd6;");
            ptx.AppendLine("    add.u64 %rd9, %rd5, %rd6;");
            ptx.AppendLine(atomic
                ? "    atom.global.add.f32 %f2, [%rd8], %f0;"
                : "    st.global.f32 [%rd8], %f0;");
            ptx.AppendLine(atomic
                ? "    atom.global.add.f32 %f3, [%rd9], %f1;"
                : "    st.global.f32 [%rd9], %f1;");
        }
        ptx.AppendLine("PARAM_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitParameterBlockReduction(
        StringBuilder ptx, string label, bool beta)
    {
        // Adjacent half-warps own adjacent rows and the same 16 features. Fold
        // each pair with a shuffle, then retain one coalesced partial per warp.
        ptx.AppendLine("    mov.b32 %r8, %f0;");
        ptx.AppendLine("    shfl.sync.bfly.b32 %r9, %r8, 16, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f2, %r9;");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f2;");
        if (beta)
        {
            ptx.AppendLine("    mov.b32 %r8, %f1;");
            ptx.AppendLine("    shfl.sync.bfly.b32 %r9, %r8, 16, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f2, %r9;");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        }
        ptx.AppendLine("    and.b32 %r5, %r2, 1;");
        ptx.AppendLine("    setp.ne.u32 %p1, %r5, 0;");
        ptx.AppendLine($"    @%p1 bra {label}_BARRIER;");
        ptx.AppendLine("    shr.u32 %r5, %r2, 1;");
        ptx.AppendLine($"    mad.lo.u32 %r5, %r5, {ParameterFeaturesPerBlock}, %r1;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r5, 4;");
        ptx.AppendLine("    mov.u64 %rd12, parameter_scratch;");
        ptx.AppendLine("    add.u64 %rd13, %rd12, %rd11;");
        ptx.AppendLine("    st.shared.f32 [%rd13], %f0;");
        if (beta)
        {
            ptx.AppendLine($"    add.u64 %rd14, %rd13, {ParameterSharedBytes / 2};");
            ptx.AppendLine("    st.shared.f32 [%rd14], %f1;");
        }
        ptx.AppendLine($"{label}_BARRIER:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r2, 0;");
        ptx.AppendLine($"    @%p2 bra {label}_RETURN;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        if (beta)
            ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine($"{label}_FINAL_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p3, %r5, {ParameterPartialCohorts};");
        ptx.AppendLine($"    @%p3 bra {label}_STORE;");
        ptx.AppendLine($"    mad.lo.u32 %r6, %r5, {ParameterFeaturesPerBlock}, %r1;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r6, 4;");
        ptx.AppendLine("    mov.u64 %rd12, parameter_scratch;");
        ptx.AppendLine("    add.u64 %rd13, %rd12, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f2, [%rd13];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f2;");
        if (beta)
        {
            ptx.AppendLine($"    add.u64 %rd14, %rd13, {ParameterSharedBytes / 2};");
            ptx.AppendLine("    ld.shared.f32 %f3, [%rd14];");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f3;");
        }
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine($"    bra {label}_FINAL_LOOP;");
        ptx.AppendLine($"{label}_STORE:");
    }

    private static string EmitNormAxis(
        int ccMajor, int ccMinor, int rows, bool writeNormalized)
    {
        DirectPtxRowNormalizationOperation operation = writeNormalized
            ? DirectPtxRowNormalizationOperation.NormalizeL2
            : DirectPtxRowNormalizationOperation.NormAxis;
        var ptx = Begin(ccMajor, ccMinor, GetEntryPoint(operation),
            "input_ptr", "output_ptr");
        EmitRegisters(ptx, predicates: 3, b32: 16, b64: 18, f32: 20);
        EmitPointerLoads(ptx, 2);
        EmitRowAddressSetup(ptx, rows, "NORM_DONE");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd8];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd8+128];");
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f2, %f1, %f1, %f2;");
        EmitWarpSum(ptx, "%f2", "%f3");
        ptx.AppendLine("    sqrt.rn.f32 %f4, %f2;");
        if (writeNormalized)
        {
            ptx.AppendLine("    add.rn.f32 %f5, %f4, 0f2B8CBCCC;");
            ptx.AppendLine("    rcp.approx.f32 %f6, %f5;");
            ptx.AppendLine("    mul.rn.f32 %f7, %f0, %f6;");
            ptx.AppendLine("    mul.rn.f32 %f8, %f1, %f6;");
            ptx.AppendLine("    add.u64 %rd9, %rd1, %rd6;");
            ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
            ptx.AppendLine("    st.global.f32 [%rd9], %f7;");
            ptx.AppendLine("    st.global.f32 [%rd9+128], %f8;");
        }
        else
        {
            EmitSingleStatStore(ptx, pointerRegister: 1, value: "%f4");
        }
        return End(ptx, "NORM_DONE");
    }

    private static string EmitNormBackward(
        int ccMajor, int ccMinor, int rows)
    {
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(DirectPtxRowNormalizationOperation.NormBackward),
            "grad_output_ptr", "input_ptr", "norm_ptr", "grad_input_ptr");
        EmitRegisters(ptx, predicates: 2, b32: 16, b64: 20, f32: 20);
        EmitPointerLoads(ptx, 4);
        EmitRowAddressSetup(ptx, rows, "NORM_BWD_DONE");
        EmitRowStatLoads(ptx, meanPointerRegister: 0, secondPointerRegister: 2,
            firstValue: "%f0", secondValue: "%f1");
        ptx.AppendLine("    max.f32 %f1, %f1, 0f322BCC77;");
        ptx.AppendLine("    rcp.approx.f32 %f2, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f3, %f0, %f2;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd9];");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd9+128];");
        ptx.AppendLine("    mul.rn.f32 %f6, %f4, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f3;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd6;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f6;");
        ptx.AppendLine("    st.global.f32 [%rd10+128], %f7;");
        return End(ptx, "NORM_BWD_DONE");
    }

    private static string EmitReduceNormL2(
        int ccMajor, int ccMinor, int rows, bool atomic)
    {
        DirectPtxRowNormalizationOperation operation = atomic
            ? DirectPtxRowNormalizationOperation.ReduceNormL2Atomic
            : DirectPtxRowNormalizationOperation.ReduceNormL2;
        var ptx = Begin(ccMajor, ccMinor,
            GetEntryPoint(operation),
            atomic
                ? new[] { "input_ptr", "output_ptr", "workspace_ptr" }
                : new[] { "input_ptr", "output_ptr" });
        EmitRegisters(ptx, predicates: 6, b32: 16, b64: 8, f32: 8,
            blockThreads: ReductionBlockThreads);
        ptx.AppendLine($"    .shared .align 4 .b8 reduce_scratch[{ReductionSharedBytes}];");
        EmitPointerLoads(ptx, atomic ? 3 : 2);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        if (atomic)
        {
            ptx.AppendLine("    mov.u32 %r4, %ctaid.x;");
            ptx.AppendLine($"    mad.lo.u32 %r1, %r4, {ReductionBlockThreads}, %r0;");
        }
        else
        {
            ptx.AppendLine("    mov.u32 %r1, %r0;");
        }
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        if (atomic)
        {
            int blockCount = GetReductionBlockCount(rows);
            int threads = blockCount * ReductionBlockThreads;
            int vectorsPerThread = GetReductionVectorsPerThread(rows);
            int gridStrideBytes = checked(threads * 4 * sizeof(float));
            ptx.AppendLine("    mul.wide.u32 %rd6, %r1, 16;");
            ptx.AppendLine("    add.u64 %rd3, %rd0, %rd6;");
            for (int vector = 0; vector < vectorsPerThread; vector++)
            {
                if (vector != 0)
                    ptx.AppendLine($"    add.u64 %rd3, %rd3, {gridStrideBytes};");
                ptx.AppendLine("    ld.global.nc.v4.f32 {%f1, %f2, %f3, %f4}, [%rd3];");
                ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f1, %f0;");
                ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f2, %f0;");
                ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f3, %f0;");
                ptx.AppendLine("    fma.rn.f32 %f0, %f4, %f4, %f0;");
            }
        }
        else
        {
            ptx.AppendLine("REDUCE_NORM_LOOP:");
            ptx.AppendLine($"    setp.ge.u32 %p0, %r1, {rows * Dimension};");
            ptx.AppendLine("    @%p0 bra REDUCE_NORM_WARP;");
            ptx.AppendLine("    mul.wide.u32 %rd6, %r1, 4;");
            ptx.AppendLine("    add.u64 %rd3, %rd0, %rd6;");
            ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd3];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f1, %f0;");
            ptx.AppendLine($"    add.u32 %r1, %r1, {ReductionBlockThreads};");
            ptx.AppendLine("    bra REDUCE_NORM_LOOP;");
        }
        ptx.AppendLine("REDUCE_NORM_WARP:");
        EmitWarpSum(ptx, "%f0", "%f2");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r3, %r0, 5;");
        ptx.AppendLine("    setp.ne.u32 %p1, %r2, 0;");
        ptx.AppendLine("    @%p1 bra REDUCE_NORM_BARRIER;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    mov.u64 %rd5, reduce_scratch;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, %rd4;");
        ptx.AppendLine("    st.shared.f32 [%rd5], %f0;");
        ptx.AppendLine("REDUCE_NORM_BARRIER:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r3, 0;");
        ptx.AppendLine(atomic
            ? "    @%p2 bra REDUCE_NORM_PUBLISH_WAIT;"
            : "    @%p2 bra REDUCE_NORM_DONE;");
        ptx.AppendLine($"    setp.ge.u32 %p3, %r2, {ReductionWarpsPerBlock};");
        ptx.AppendLine(atomic
            ? "    @%p3 bra REDUCE_NORM_PUBLISH_WAIT;"
            : "    @%p3 bra REDUCE_NORM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    mov.u64 %rd7, reduce_scratch;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd7];");
        EmitHalfWarpSum(ptx, "%f3", "%f4");
        ptx.AppendLine("    setp.ne.u32 %p1, %r0, 0;");
        ptx.AppendLine(atomic
            ? "    @%p1 bra REDUCE_NORM_PUBLISH_WAIT;"
            : "    @%p1 bra REDUCE_NORM_DONE;");
        if (!atomic)
        {
            ptx.AppendLine("    st.global.f32 [%rd1], %f3;");
            return End(ptx, "REDUCE_NORM_DONE");
        }

        // Publish into reusable banks rather than materializing one partial per
        // block. Sixteen addresses limit each bank to at most eight publishers,
        // and RED avoids the unused atomic return value. The completion counter
        // orders the final half-warp fold without a second kernel or host clear.
        ptx.AppendLine("    mov.u32 %r5, %ctaid.x;");
        ptx.AppendLine($"    and.b32 %r6, %r5, {ReductionAccumulatorBanks - 1};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd2, %rd3;");
        ptx.AppendLine("    red.global.add.f32 [%rd4], %f3;");
        ptx.AppendLine("    membar.gl;");
        ptx.AppendLine($"    add.u64 %rd5, %rd2, {NormalizationWorkspaceCounterByteOffset};");
        ptx.AppendLine("    atom.global.add.u32 %r5, [%rd5], 1;");
        ptx.AppendLine("    mov.u32 %r6, %nctaid.x;");
        ptx.AppendLine("    sub.u32 %r6, %r6, 1;");
        ptx.AppendLine("    setp.eq.u32 %p4, %r5, %r6;");
        ptx.AppendLine("    selp.u32 %r5, 1, 0, %p4;");
        ptx.AppendLine("    mov.u64 %rd6, reduce_scratch;");
        ptx.AppendLine("    st.shared.u32 [%rd6], %r5;");
        ptx.AppendLine("REDUCE_NORM_PUBLISH_WAIT:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    mov.u64 %rd6, reduce_scratch;");
        ptx.AppendLine("    ld.shared.u32 %r5, [%rd6];");
        ptx.AppendLine("    setp.eq.u32 %p4, %r5, 0;");
        ptx.AppendLine("    @%p4 bra REDUCE_NORM_DONE;");
        ptx.AppendLine("    membar.gl;");

        // The low half of warp zero loads and clears the sixteen banks. Its
        // shuffle mask names exactly those active lanes, avoiding the divergent
        // full-warp helper path while reducing the fold to four inline SHFLs.
        // Only thread zero commits the result and counter reset.
        ptx.AppendLine("    setp.ne.u32 %p5, %r3, 0;");
        ptx.AppendLine("    @%p5 bra REDUCE_NORM_DONE;");
        ptx.AppendLine($"    setp.ge.u32 %p5, %r2, {ReductionAccumulatorBanks};");
        ptx.AppendLine("    @%p5 bra REDUCE_NORM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd4];");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    st.global.f32 [%rd4], %f1;");
        EmitHalfWarpSum(ptx, "%f0", "%f2");
        ptx.AppendLine("    setp.ne.u32 %p5, %r2, 0;");
        ptx.AppendLine("    @%p5 bra REDUCE_NORM_DONE;");
        ptx.AppendLine("    st.global.f32 [%rd1], %f0;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine($"    add.u64 %rd5, %rd2, {NormalizationWorkspaceCounterByteOffset};");
        ptx.AppendLine("    st.global.u32 [%rd5], %r5;");
        return End(ptx, "REDUCE_NORM_DONE");
    }

    private static StringBuilder Begin(
        int ccMajor, int ccMinor, string entryPoint, params string[] parameters)
    {
        var ptx = new StringBuilder(12_288);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// direct-ptx normalization d64 entry={entryPoint}");
        ptx.AppendLine($".visible .entry {entryPoint}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        return ptx;
    }

    private static void EmitRegisters(
        StringBuilder ptx, int predicates, int b32, int b64, int f32,
        int blockThreads = RowBlockThreads, int b16 = 0)
    {
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine($"    .reg .pred %p<{predicates}>;");
        if (b16 > 0) ptx.AppendLine($"    .reg .b16 %rs<{b16}>;");
        ptx.AppendLine($"    .reg .b32 %r<{b32}>;");
        ptx.AppendLine($"    .reg .b64 %rd<{b64}>;");
        ptx.AppendLine($"    .reg .f32 %f<{f32}>;");
    }

    private static void EmitPointerLoads(StringBuilder ptx, int count)
    {
        string[] names =
        [
            "input_ptr", "gamma_ptr", "beta_ptr", "output_ptr", "mean_ptr", "inv_var_ptr"
        ];
        // Parameter names vary by operation. PTX parameter order is stable, so
        // read the emitted declarations from their operation-specific names.
        // The caller replaces these lines immediately below when names differ.
        // Determine names from the entry declaration already accumulated.
        string source = ptx.ToString();
        int cursor = source.IndexOf(".visible .entry", StringComparison.Ordinal);
        string declaration = source.Substring(cursor);
        int register = 0;
        int scan = 0;
        while (register < count)
        {
            int parameter = declaration.IndexOf(".param .u64 ", scan, StringComparison.Ordinal);
            int start = parameter + ".param .u64 ".Length;
            int end = declaration.IndexOfAny([',', '\r', '\n'], start);
            string name = declaration.Substring(start, end - start).Trim();
            ptx.AppendLine($"    ld.param.u64 %rd{register}, [{name}];");
            register++;
            scan = end + 1;
        }
    }

    private static void EmitRowAddressSetup(StringBuilder ptx, int rows, string doneLabel)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r2;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {rows};");
        ptx.AppendLine($"    @%p0 bra {doneLabel};");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r4, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd7;");
    }

    private static void EmitFp16RowAddressSetup(StringBuilder ptx, int rows, string doneLabel)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r2;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {rows};");
        ptx.AppendLine($"    @%p0 bra {doneLabel};");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r4, 128;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r1, 2;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd7;");
    }

    private static void EmitWarpSum(StringBuilder ptx, string value, string temporary)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r8, {value};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r9, %r8, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {temporary}, %r9;");
            ptx.AppendLine($"    add.rn.f32 {value}, {value}, {temporary};");
        }
    }

    private static void EmitHalfWarpSum(StringBuilder ptx, string value, string temporary)
    {
        foreach (int delta in new[] { 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r8, {value};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r9, %r8, {delta}, 15, 0x0000ffff;");
            ptx.AppendLine($"    mov.b32 {temporary}, %r9;");
            ptx.AppendLine($"    add.rn.f32 {value}, {value}, {temporary};");
        }
    }

    private static void EmitStatStores(
        StringBuilder ptx, int meanPointerRegister, int secondPointerRegister,
        string firstValue, string secondValue)
    {
        ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r4, 4;");
        ptx.AppendLine($"    add.u64 %rd15, %rd{meanPointerRegister}, %rd14;");
        ptx.AppendLine($"    add.u64 %rd16, %rd{secondPointerRegister}, %rd14;");
        ptx.AppendLine($"    @%p1 st.global.f32 [%rd15], {firstValue};");
        ptx.AppendLine($"    @%p1 st.global.f32 [%rd16], {secondValue};");
    }

    private static void EmitSingleStatStore(
        StringBuilder ptx, int pointerRegister, string value)
    {
        ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r4, 4;");
        ptx.AppendLine($"    add.u64 %rd15, %rd{pointerRegister}, %rd14;");
        ptx.AppendLine($"    @%p1 st.global.f32 [%rd15], {value};");
    }

    private static void EmitRowStatLoads(
        StringBuilder ptx, int meanPointerRegister, int secondPointerRegister,
        string firstValue, string secondValue)
    {
        ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        ptx.AppendLine($"    add.u64 %rd14, %rd{meanPointerRegister}, %rd12;");
        ptx.AppendLine($"    add.u64 %rd15, %rd{secondPointerRegister}, %rd12;");
        ptx.AppendLine($"    ld.global.nc.f32 {firstValue}, [%rd14];");
        ptx.AppendLine($"    ld.global.nc.f32 {secondValue}, [%rd15];");
    }

    private static void EmitSingleRowStatLoad(
        StringBuilder ptx, int pointerRegister, string value)
    {
        ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        ptx.AppendLine($"    add.u64 %rd14, %rd{pointerRegister}, %rd12;");
        ptx.AppendLine($"    ld.global.nc.f32 {value}, [%rd14];");
    }

    private static string End(StringBuilder ptx, string doneLabel)
    {
        ptx.AppendLine($"{doneLabel}:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxRowNormalizationOperation operation,
        int rows)
    {
        if (architecture != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "Row-normalization blueprints have only been validated on Ampere.");
        int operationIndex = (int)operation;
        if ((uint)operationIndex >= (uint)EntryPoints.Length)
            throw new ArgumentOutOfRangeException(nameof(operation));
        int rowIndex = rows switch
        {
            256 => 0,
            2_048 => 1,
            8_192 => 2,
            _ => throw new ArgumentOutOfRangeException(nameof(rows))
        };
        return AmpereBlueprints[operationIndex, rowIndex];
    }

    private static DirectPtxKernelBlueprint[,] CreateAmpereBlueprints()
    {
        int operationCount = Enum.GetValues(typeof(DirectPtxRowNormalizationOperation)).Length;
        var blueprints = new DirectPtxKernelBlueprint[operationCount, SupportedRowCounts.Length];
        for (int operation = 0; operation < operationCount; operation++)
        {
            for (int row = 0; row < SupportedRowCounts.Length; row++)
            {
                blueprints[operation, row] = CreateBlueprintCore(
                    DirectPtxArchitectureFamily.Ampere,
                    (DirectPtxRowNormalizationOperation)operation,
                    SupportedRowCounts[row]);
            }
        }
        return blueprints;
    }

    private static DirectPtxKernelBlueprint CreateBlueprintCore(
        DirectPtxArchitectureFamily architecture,
        DirectPtxRowNormalizationOperation operation,
        int rows)
    {
        var matrix = new DirectPtxExtent(rows, Dimension);
        var vector = new DirectPtxExtent(Dimension);
        var stats = new DirectPtxExtent(rows);
        var scalar = new DirectPtxExtent(1);
        var workspace = new DirectPtxExtent(NormalizationWorkspaceElements);
        DirectPtxTensorContract Matrix(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                matrix, matrix, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Vector(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                vector, vector, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Stats(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                stats, stats, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Scalar(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                scalar, scalar, 4, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Workspace() =>
            new("workspace", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                workspace, workspace, 16, DirectPtxTensorAccess.ReadWrite,
                DirectPtxExtentMode.Exact);
        DirectPtxTensorContract HalfMatrix(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                matrix, matrix, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract HalfVector(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Vector,
                vector, vector, 16, access, DirectPtxExtentMode.Exact);

        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxRowNormalizationOperation.LayerNormForward =>
            [
                Matrix("input", DirectPtxTensorAccess.Read),
                Vector("gamma", DirectPtxTensorAccess.Read),
                Vector("beta", DirectPtxTensorAccess.Read),
                Matrix("output", DirectPtxTensorAccess.Write),
                Stats("mean", DirectPtxTensorAccess.Write),
                Stats("inv_var", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.LayerNormBackwardInput =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Vector("gamma", DirectPtxTensorAccess.Read),
                Stats("mean", DirectPtxTensorAccess.Read),
                Stats("inv_var", DirectPtxTensorAccess.Read),
                Matrix("grad_input", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.LayerNormGradParameters or
                DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Stats("mean", DirectPtxTensorAccess.Read),
                Stats("inv_var", DirectPtxTensorAccess.Read),
                Vector("grad_gamma", DirectPtxTensorAccess.Write),
                Vector("grad_beta", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.Fp16LayerNormForward =>
            [
                HalfMatrix("input", DirectPtxTensorAccess.Read),
                HalfVector("gamma", DirectPtxTensorAccess.Read),
                HalfVector("beta", DirectPtxTensorAccess.Read),
                HalfMatrix("output", DirectPtxTensorAccess.Write),
                Stats("mean", DirectPtxTensorAccess.Write),
                Stats("variance", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput =>
            [
                HalfMatrix("grad_output", DirectPtxTensorAccess.Read),
                HalfMatrix("input", DirectPtxTensorAccess.Read),
                HalfVector("gamma", DirectPtxTensorAccess.Read),
                Stats("mean", DirectPtxTensorAccess.Read),
                Stats("inv_var", DirectPtxTensorAccess.Read),
                HalfMatrix("grad_input", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters =>
            [
                HalfMatrix("grad_output", DirectPtxTensorAccess.Read),
                HalfMatrix("input", DirectPtxTensorAccess.Read),
                Stats("mean", DirectPtxTensorAccess.Read),
                Stats("inv_var", DirectPtxTensorAccess.Read),
                HalfVector("grad_gamma", DirectPtxTensorAccess.Write),
                HalfVector("grad_beta", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.RmsNormForward =>
            [
                Matrix("input", DirectPtxTensorAccess.Read),
                Vector("gamma", DirectPtxTensorAccess.Read),
                Matrix("output", DirectPtxTensorAccess.Write),
                Stats("rms", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.RmsNormBackwardInput =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Vector("gamma", DirectPtxTensorAccess.Read),
                Stats("rms", DirectPtxTensorAccess.Read),
                Matrix("grad_input", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Vector("gamma", DirectPtxTensorAccess.Read),
                Stats("mean", DirectPtxTensorAccess.Read),
                Stats("inv_var", DirectPtxTensorAccess.Read),
                Matrix("grad_input", DirectPtxTensorAccess.Write),
                Vector("grad_gamma", DirectPtxTensorAccess.Write),
                Vector("grad_beta", DirectPtxTensorAccess.Write),
                Workspace()
            ],
            DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Vector("gamma", DirectPtxTensorAccess.Read),
                Stats("rms", DirectPtxTensorAccess.Read),
                Matrix("grad_input", DirectPtxTensorAccess.Write),
                Vector("grad_gamma", DirectPtxTensorAccess.Write),
                Workspace()
            ],
            DirectPtxRowNormalizationOperation.RmsNormGradGamma or
                DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Stats("rms", DirectPtxTensorAccess.Read),
                Vector("grad_gamma", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.NormAxis =>
            [Matrix("input", DirectPtxTensorAccess.Read), Stats("output", DirectPtxTensorAccess.Write)],
            DirectPtxRowNormalizationOperation.NormalizeL2 =>
            [Matrix("input", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write)],
            DirectPtxRowNormalizationOperation.NormBackward =>
            [
                Stats("grad_output", DirectPtxTensorAccess.Read),
                Matrix("input", DirectPtxTensorAccess.Read),
                Stats("norm", DirectPtxTensorAccess.Read),
                Matrix("grad_input", DirectPtxTensorAccess.Write)
            ],
            DirectPtxRowNormalizationOperation.ReduceNormL2 =>
            [Matrix("input", DirectPtxTensorAccess.Read), Scalar("output", DirectPtxTensorAccess.Write)],
            DirectPtxRowNormalizationOperation.ReduceNormL2Atomic =>
            [
                Matrix("input", DirectPtxTensorAccess.Read),
                Scalar("output", DirectPtxTensorAccess.Write),
                Workspace()
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

        bool parameterGradient = IsParameterGradient(operation);
        bool atomicParameterGradient = IsAtomicParameterGradient(operation);
        bool atomicReduction = operation == DirectPtxRowNormalizationOperation.ReduceNormL2Atomic;
        bool deterministicReduction = operation == DirectPtxRowNormalizationOperation.ReduceNormL2;
        bool fusedBackward = IsFusedBackward(operation);
        return new DirectPtxKernelBlueprint(
            Operation: OperationName(operation),
            Version: 1,
            Architecture: architecture,
            Variant: fusedBackward
                ? $"single-pass-grid-stride-d64-w{GetFusedBackwardWarpsPerBlock(operation)}" +
                    $"-b{GetFusedBackwardMaxBlocks(operation)}-r{rows}"
                : atomicReduction
                ? $"uniform-v4x{GetReductionVectorsPerThread(rows)}-banked-red-half-warp" +
                    $"-t{ReductionBlockThreads}-b{ReductionMaxBlocks}" +
                    $"-a{ReductionAccumulatorBanks}-r{rows}"
                : deterministicReduction
                ? $"single-block-reduction-t{ReductionBlockThreads}-r{rows}"
                : atomicParameterGradient
                ? $"feature-tile-atomic-d64-f{ParameterFeaturesPerBlock}-c{ParameterCohorts}" +
                    $"-rt{Math.Min(ParameterRowsPerTile, rows)}-r{rows}"
                : parameterGradient
                ? $"feature-tile-d64-f{ParameterFeaturesPerBlock}-c{ParameterCohorts}-r{rows}"
                : $"warp-row-d64-w{WarpsPerBlock}-r{rows}",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                // Reductions use fixed shuffle/shared folds. Parameter lanes
                // use four independent feature tiles so row-major loads stay
                // coalesced without output-sized temporary storage.
                MaxRegistersPerThread: parameterGradient || fusedBackward ? 80 : 48,
                MaxStaticSharedBytes: fusedBackward
                    ? operation == DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic
                        ? FusedBackwardLayerSharedBytes
                        : FusedBackwardRmsSharedBytes
                    : parameterGradient ? ParameterSharedBytes :
                    IsReduction(operation) ? ReductionSharedBytes : 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: fusedBackward
                    ? operation == DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic
                        ? 2
                        : 1
                    : parameterGradient || IsReduction(operation) ? 3 : 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = OperationName(operation),
                ["normalized-size"] = Dimension.ToString(CultureInfo.InvariantCulture),
                ["row-count"] = rows.ToString(CultureInfo.InvariantCulture),
                ["global-intermediates"] = fusedBackward
                    ? "bounded-reusable-accumulators"
                    : atomicReduction ? "bounded-reusable-accumulators" : "none",
                ["accumulator-banks"] = fusedBackward
                    ? FusedBackwardAccumulatorBanks.ToString(CultureInfo.InvariantCulture)
                    : atomicReduction
                    ? ReductionAccumulatorBanks.ToString(CultureInfo.InvariantCulture)
                    : "none",
                ["temporary-device-allocation"] = RequiresPersistentWorkspace(operation)
                    ? "persistent-prewarm-workspace" : "none",
                ["shape-stride-parameters"] = "none",
                ["deterministic"] = atomicParameterGradient || fusedBackward || atomicReduction
                    ? "false" : "true",
                ["dataflow"] = fusedBackward
                    ? "one grid-stride pass retains row values in registers for grad-input and parameter partials; shared memory folds warp cohorts before one atomic per block/output"
                    : atomicReduction
                    ? "exact specializations issue one or two uniform coalesced 16-byte loads per thread with no input bounds loop; register FMAs and warp/shared folds publish through sixteen reusable RED banks; the last block performs a half-warp fold and self-reset"
                    : deterministicReduction
                    ? "one 512-thread block uses coalesced scalar loads, register FMAs, and a fixed warp/shared fold before the single scalar store"
                    : atomicParameterGradient
                    ? "256-row tiles accumulate coalesced 16-feature loads in registers, then shuffle/shared fold and one global atomic per output"
                    : parameterGradient
                    ? "16-feature coalesced tiles; row cohorts accumulate in registers, then deterministic shuffle/shared fold"
                    : "two lane values retained through warp reductions"
            });
    }

    internal static void ValidateTensors(
        DirectPtxKernelBlueprint blueprint,
        ReadOnlySpan<DirectPtxTensorView> tensors,
        string operation)
    {
        if (tensors.Length != blueprint.Tensors.Count)
            ThrowTensorCountMismatch(operation, blueprint.Tensors.Count, tensors.Length);
        for (int i = 0; i < tensors.Length; i++)
            Require(tensors[i], blueprint.Tensors[i], i);
        ValidateAliasing(tensors);
    }

    private static void ValidateAliasing(ReadOnlySpan<DirectPtxTensorView> tensors)
    {
        for (int left = 0; left < tensors.Length; left++)
        {
            for (int right = left + 1; right < tensors.Length; right++)
            {
                bool writes = (tensors[left].Access & DirectPtxTensorAccess.Write) != 0 ||
                    (tensors[right].Access & DirectPtxTensorAccess.Write) != 0;
                if (writes && Overlaps(tensors[left], tensors[right]))
                    throw new ArgumentException(
                        "Direct PTX normalization writable tensors may not overlap another tensor.");
            }
        }
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        int index)
    {
        if (view.Pointer == IntPtr.Zero ||
            view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout ||
            !SameExtent(view.LogicalExtent, contract.LogicalExtent) ||
            !SameExtent(view.PhysicalExtent, contract.PhysicalExtent) ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes ||
            view.Access != contract.Access)
            ThrowTensorContractMismatch(index, contract.Name);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void ThrowTensorCountMismatch(string operation, int expected, int actual) =>
        throw new ArgumentException(
            $"{operation} expects {expected} tensor arguments; received {actual}.",
            "tensors");

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void ThrowTensorContractMismatch(int index, string name) =>
        throw new ArgumentException(
            $"tensor[{index}] does not satisfy physical ABI '{name}'.", "view");

    private static bool SameExtent(DirectPtxExtent left, DirectPtxExtent right) =>
        left.Rank == right.Rank && left.D0 == right.D0 && left.D1 == right.D1 &&
        left.D2 == right.D2 && left.D3 == right.D3;

    private static bool IsParameterGradient(DirectPtxRowNormalizationOperation operation) =>
        operation is DirectPtxRowNormalizationOperation.LayerNormGradParameters or
            DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters or
            DirectPtxRowNormalizationOperation.RmsNormGradGamma or
            DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic or
            DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic;

    internal static bool IsAtomicParameterGradient(
        DirectPtxRowNormalizationOperation operation) =>
        operation is DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic or
            DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic;

    internal static bool IsFusedBackward(
        DirectPtxRowNormalizationOperation operation) =>
        operation is DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic or
            DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic;

    internal static bool RequiresPersistentWorkspace(
        DirectPtxRowNormalizationOperation operation) =>
        IsFusedBackward(operation) ||
        operation == DirectPtxRowNormalizationOperation.ReduceNormL2Atomic;

    private static bool IsReduction(
        DirectPtxRowNormalizationOperation operation) =>
        operation is DirectPtxRowNormalizationOperation.ReduceNormL2 or
            DirectPtxRowNormalizationOperation.ReduceNormL2Atomic;

    private static int GetFusedBackwardMaxBlocks(
        DirectPtxRowNormalizationOperation operation) => operation switch
        {
            DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic =>
                FusedLayerNormBackwardMaxBlocks,
            DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic =>
                FusedRmsNormBackwardMaxBlocks,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

    private static int GetReductionBlockCount(int rows) => Math.Min(
        (rows * (Dimension / 4) + ReductionBlockThreads - 1) /
            ReductionBlockThreads,
        ReductionMaxBlocks);

    private static int GetReductionVectorsPerThread(int rows)
    {
        int vectorCount = rows * (Dimension / 4);
        int threads = GetReductionBlockCount(rows) * ReductionBlockThreads;
        if (vectorCount % threads != 0)
            throw new InvalidOperationException(
                "Exact L2 specialization must assign a uniform vector count per thread.");
        return vectorCount / threads;
    }

    private static int GetFusedBackwardWarpsPerBlock(
        DirectPtxRowNormalizationOperation operation) => operation switch
        {
            DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic =>
                FusedLayerNormBackwardWarpsPerBlock,
            DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic =>
                FusedRmsNormBackwardWarpsPerBlock,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

    private static int GetFusedBackwardBlockThreads(
        DirectPtxRowNormalizationOperation operation) => operation switch
        {
            DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic =>
                FusedLayerNormBackwardBlockThreads,
            DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic =>
                FusedRmsNormBackwardBlockThreads,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

    internal static DirectPtxRowNormalizationOperation SelectFastOperation(
        DirectPtxRowNormalizationOperation operation, bool deterministic, int rows) =>
        deterministic ? operation : operation switch
        {
            // The separate bounded atomic parameter lane won only for the
            // largest RMS shape. Deterministic execution always remains on
            // the fixed-order reduction.
            DirectPtxRowNormalizationOperation.RmsNormGradGamma when rows == 8_192 =>
                DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic,
            _ => operation
        };

    private static void Validate(
        DirectPtxRowNormalizationOperation operation, int rows, float epsilon)
    {
        if (!Enum.IsDefined(typeof(DirectPtxRowNormalizationOperation), operation))
            throw new ArgumentOutOfRangeException(nameof(operation));
        if (!IsSupportedRows(rows) || rows % WarpsPerBlock != 0)
            throw new ArgumentOutOfRangeException(nameof(rows),
                "D64 normalization supports exact row counts 256, 2048, and 8192.");
        if (!PtxCompat.IsFinite(epsilon) || epsilon <= 0f)
            throw new ArgumentOutOfRangeException(nameof(epsilon));
    }

    internal static string GetEntryPoint(DirectPtxRowNormalizationOperation operation)
    {
        int index = (int)operation;
        if ((uint)index >= (uint)EntryPoints.Length)
            throw new ArgumentOutOfRangeException(nameof(operation));
        return EntryPoints[index];
    }

    private static string[] CreateEntryPoints()
    {
        int count = Enum.GetValues(typeof(DirectPtxRowNormalizationOperation)).Length;
        var entryPoints = new string[count];
        for (int i = 0; i < count; i++)
            entryPoints[i] = "aidotnet_" +
                OperationName((DirectPtxRowNormalizationOperation)i).Replace('-', '_');
        return entryPoints;
    }

    private static string OperationName(DirectPtxRowNormalizationOperation operation) => operation switch
    {
        DirectPtxRowNormalizationOperation.LayerNormForward => "layernorm-forward-d64",
        DirectPtxRowNormalizationOperation.LayerNormBackwardInput => "layernorm-backward-input-d64",
        DirectPtxRowNormalizationOperation.LayerNormGradParameters => "layernorm-grad-params-d64",
        DirectPtxRowNormalizationOperation.Fp16LayerNormForward => "fp16-layernorm-forward-d64",
        DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput => "fp16-layernorm-backward-input-d64",
        DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters => "fp16-layernorm-grad-params-d64",
        DirectPtxRowNormalizationOperation.RmsNormForward => "rmsnorm-forward-d64",
        DirectPtxRowNormalizationOperation.RmsNormBackwardInput => "rmsnorm-backward-input-d64",
        DirectPtxRowNormalizationOperation.RmsNormGradGamma => "rmsnorm-grad-gamma-d64",
        DirectPtxRowNormalizationOperation.NormAxis => "norm-axis-d64",
        DirectPtxRowNormalizationOperation.NormBackward => "norm-backward-d64",
        DirectPtxRowNormalizationOperation.NormalizeL2 => "normalize-l2-d64",
        DirectPtxRowNormalizationOperation.ReduceNormL2 => "reduce-norm-l2-d64",
        DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic =>
            "layernorm-grad-params-atomic-d64",
        DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic =>
            "rmsnorm-grad-gamma-atomic-d64",
        DirectPtxRowNormalizationOperation.ReduceNormL2Atomic =>
            "reduce-norm-l2-atomic-d64",
        DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic =>
            "layernorm-backward-fused-atomic-d64",
        DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic =>
            "rmsnorm-backward-fused-atomic-d64",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
