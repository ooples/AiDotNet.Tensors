using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxChannelNormalizationOperation
{
    BatchNormTraining,
    BatchNormInference,
    BatchNormRelu,
    BatchNormGelu,
    BatchNormSigmoid,
    BatchNormTanh,
    ResidualBatchNormRelu,
    BatchNormBackward,
    GroupNormForward,
    GroupNormSwish,
    AddGroupNorm,
    GroupNormBackwardInput,
    GroupNormGradParameters,
    InstanceNormForward,
    InstanceNormBackwardInput,
    InstanceNormGradParameters,
    Fp16GroupNormSwish
}

/// <summary>
/// Exact 64-value normalization-unit family for canonical NCHW tensors. The
/// batch, channel, spatial, group, activation, training, epsilon, and momentum
/// semantics are all baked into separate PTX module identities.
/// </summary>
internal sealed class PtxChannelNormalizationD64Kernel : IDisposable
{
    internal const int UnitElements = 64;
    internal const int WarpsPerBlock = 8;
    internal const int RowBlockThreads = 256;
    internal const int ParameterBlockThreads = 256;
    internal const int ParameterSharedBytes = 2 * WarpsPerBlock * sizeof(float);

    internal const int BatchNormBatch = 8;
    internal const int BatchNormChannels = 64;
    internal const int BatchNormSpatial = 8;

    internal const int GroupNormBatch = 32;
    internal const int GroupNormChannels = 64;
    internal const int GroupNormGroups = 8;
    internal const int GroupNormSpatial = 8;

    internal const int InstanceNormBatch = 32;
    internal const int InstanceNormChannels = 64;
    internal const int InstanceNormSpatial = 64;

    private static readonly string[] EntryPoints = CreateEntryPoints();
    private static readonly DirectPtxKernelBlueprint[] AmpereBlueprints =
        CreateAmpereBlueprints();

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxChannelNormalizationOperation Operation { get; }
    internal float Epsilon { get; }
    internal float Momentum { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxChannelNormalizationD64Kernel(
        DirectPtxRuntime runtime,
        DirectPtxChannelNormalizationOperation operation,
        float epsilon = 1e-5f,
        float momentum = 0.1f)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in channel-normalization family is validated only on GA10x/SM86.");
        Validate(operation, epsilon, momentum);
        Operation = operation;
        Epsilon = epsilon;
        Momentum = momentum;
        EntryPoint = GetEntryPoint(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            operation, epsilon, momentum);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.NormalizationExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int blockThreads = IsParameterGradient(operation)
            ? ParameterBlockThreads
            : RowBlockThreads;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, blockThreads, activeBlocks);
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
        void** arguments = stackalloc void*[8];
        arguments[0] = &pointer0;
        arguments[1] = &pointer1;
        arguments[2] = &pointer2;
        arguments[3] = &pointer3;
        arguments[4] = &pointer4;
        arguments[5] = &pointer5;
        arguments[6] = &pointer6;
        arguments[7] = &pointer7;

        uint block = IsParameterGradient(Operation)
            ? (uint)ParameterBlockThreads
            : (uint)RowBlockThreads;
        uint grid = IsParameterGradient(Operation)
            ? BatchNormChannels
            : checked((uint)(UnitCount(Operation) / WarpsPerBlock));
        _module.Launch(_function, grid, 1, 1, block, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsPromoted(
        DirectPtxChannelNormalizationOperation operation) => false;

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxChannelNormalizationOperation operation,
        float epsilon = 1e-5f,
        float momentum = 0.1f)
    {
        Validate(operation, epsilon, momentum);
        return operation switch
        {
            DirectPtxChannelNormalizationOperation.BatchNormTraining =>
                EmitBatchNormTraining(ccMajor, ccMinor, epsilon, momentum),
            DirectPtxChannelNormalizationOperation.BatchNormInference or
            DirectPtxChannelNormalizationOperation.BatchNormRelu or
            DirectPtxChannelNormalizationOperation.BatchNormGelu or
            DirectPtxChannelNormalizationOperation.BatchNormSigmoid or
            DirectPtxChannelNormalizationOperation.BatchNormTanh =>
                EmitBatchNormInference(ccMajor, ccMinor, operation, epsilon),
            DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu =>
                EmitResidualBatchNormRelu(ccMajor, ccMinor, epsilon),
            DirectPtxChannelNormalizationOperation.BatchNormBackward =>
                EmitBackward(ccMajor, ccMinor, operation, UnitLayout.Batch,
                    statisticsAreVariance: false, epsilon),
            DirectPtxChannelNormalizationOperation.GroupNormForward =>
                EmitAffineForward(ccMajor, ccMinor, operation, UnitLayout.Group,
                    DirectPtxActivation.None, half: false, epsilon),
            DirectPtxChannelNormalizationOperation.GroupNormSwish =>
                EmitAffineForward(ccMajor, ccMinor, operation, UnitLayout.Group,
                    DirectPtxActivation.Swish, half: false, epsilon),
            DirectPtxChannelNormalizationOperation.AddGroupNorm =>
                EmitAddGroupNorm(ccMajor, ccMinor, epsilon),
            DirectPtxChannelNormalizationOperation.GroupNormBackwardInput =>
                EmitBackward(ccMajor, ccMinor, operation, UnitLayout.Group,
                    statisticsAreVariance: true, epsilon),
            DirectPtxChannelNormalizationOperation.GroupNormGradParameters =>
                EmitGroupGradParameters(ccMajor, ccMinor, epsilon),
            DirectPtxChannelNormalizationOperation.InstanceNormForward =>
                EmitAffineForward(ccMajor, ccMinor, operation, UnitLayout.Instance,
                    DirectPtxActivation.None, half: false, epsilon),
            DirectPtxChannelNormalizationOperation.InstanceNormBackwardInput =>
                EmitBackward(ccMajor, ccMinor, operation, UnitLayout.Instance,
                    statisticsAreVariance: false, epsilon),
            DirectPtxChannelNormalizationOperation.InstanceNormGradParameters =>
                EmitInstanceGradParameters(ccMajor, ccMinor),
            DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish =>
                EmitAffineForward(ccMajor, ccMinor, operation, UnitLayout.Group,
                    DirectPtxActivation.Swish, half: true, epsilon),
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
    }

    private static string EmitBatchNormTraining(
        int ccMajor, int ccMinor, float epsilon, float momentum)
    {
        DirectPtxChannelNormalizationOperation operation =
            DirectPtxChannelNormalizationOperation.BatchNormTraining;
        var ptx = Begin(ccMajor, ccMinor, operation,
            "input_ptr", "gamma_ptr", "beta_ptr", "running_mean_ptr", "running_var_ptr",
            "output_ptr", "save_mean_ptr", "save_inv_var_ptr");
        EmitRegisters(ptx, predicates: 3, b16: 0, b32: 24, b64: 44, f32: 40);
        EmitPointerLoads(ptx, 8);
        EmitUnitAddressSetup(ptx, UnitLayout.Batch, "BN_TRAIN_DONE");
        EmitInputLoads(ptx, inputRegister: 0, half: false, "%f0", "%f1");
        EmitMeanAndInvStd(ptx, "%f0", "%f1", "%f2", "%f3", "%f4", "%f5", epsilon);
        EmitAffineLoads(ptx, gammaRegister: 1, betaRegister: 2,
            "%f6", "%f7", "%f8", "%f9");
        EmitAffine(ptx, "%f0", "%f1", "%f2", "%f5",
            "%f6", "%f7", "%f8", "%f9", "%f10", "%f11");
        EmitOutputStores(ptx, outputRegister: 5, half: false, "%f10", "%f11");

        ptx.AppendLine("    add.u64 %rd18, %rd3, %rd32;");
        ptx.AppendLine("    add.u64 %rd19, %rd4, %rd32;");
        ptx.AppendLine("    ld.global.nc.f32 %f12, [%rd18];");
        ptx.AppendLine("    ld.global.nc.f32 %f13, [%rd19];");
        ptx.AppendLine($"    mul.rn.f32 %f14, %f12, {FloatLiteral(1f - momentum)};");
        ptx.AppendLine($"    fma.rn.f32 %f14, %f2, {FloatLiteral(momentum)}, %f14;");
        ptx.AppendLine($"    mul.rn.f32 %f15, %f3, {FloatLiteral(64f / 63f)};");
        ptx.AppendLine($"    mul.rn.f32 %f16, %f13, {FloatLiteral(1f - momentum)};");
        ptx.AppendLine($"    fma.rn.f32 %f16, %f15, {FloatLiteral(momentum)}, %f16;");
        ptx.AppendLine("    add.u64 %rd20, %rd6, %rd32;");
        ptx.AppendLine("    add.u64 %rd21, %rd7, %rd32;");
        ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");
        ptx.AppendLine("    @%p1 st.global.f32 [%rd18], %f14;");
        ptx.AppendLine("    @%p1 st.global.f32 [%rd19], %f16;");
        ptx.AppendLine("    @%p1 st.global.f32 [%rd20], %f2;");
        ptx.AppendLine("    @%p1 st.global.f32 [%rd21], %f5;");
        return End(ptx, "BN_TRAIN_DONE");
    }

    private static string EmitBatchNormInference(
        int ccMajor,
        int ccMinor,
        DirectPtxChannelNormalizationOperation operation,
        float epsilon)
    {
        DirectPtxActivation activation = operation switch
        {
            DirectPtxChannelNormalizationOperation.BatchNormRelu => DirectPtxActivation.Relu,
            DirectPtxChannelNormalizationOperation.BatchNormGelu => DirectPtxActivation.Gelu,
            DirectPtxChannelNormalizationOperation.BatchNormSigmoid => DirectPtxActivation.Sigmoid,
            DirectPtxChannelNormalizationOperation.BatchNormTanh => DirectPtxActivation.Tanh,
            _ => DirectPtxActivation.None
        };
        var ptx = Begin(ccMajor, ccMinor, operation,
            "input_ptr", "gamma_ptr", "beta_ptr", "running_mean_ptr", "running_var_ptr", "output_ptr");
        EmitRegisters(ptx, predicates: 2, b16: 0, b32: 24, b64: 44, f32: 40);
        EmitPointerLoads(ptx, 6);
        EmitUnitAddressSetup(ptx, UnitLayout.Batch, "BN_INFER_DONE");
        EmitInputLoads(ptx, 0, half: false, "%f0", "%f1");
        ptx.AppendLine("    add.u64 %rd18, %rd3, %rd32;");
        ptx.AppendLine("    add.u64 %rd19, %rd4, %rd32;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd18];");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd19];");
        ptx.AppendLine($"    add.rn.f32 %f3, %f3, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f4, %f3;");
        ptx.AppendLine("    rcp.approx.f32 %f5, %f4;");
        EmitAffineLoads(ptx, 1, 2, "%f6", "%f7", "%f8", "%f9");
        EmitAffine(ptx, "%f0", "%f1", "%f2", "%f5",
            "%f6", "%f7", "%f8", "%f9", "%f10", "%f11");
        EmitActivation(ptx, activation, "%f10", "%f20", "%f22");
        EmitActivation(ptx, activation, "%f11", "%f21", "%f22");
        EmitOutputStores(ptx, 5, half: false,
            activation == DirectPtxActivation.None ? "%f10" : "%f20",
            activation == DirectPtxActivation.None ? "%f11" : "%f21");
        return End(ptx, "BN_INFER_DONE");
    }

    private static string EmitResidualBatchNormRelu(
        int ccMajor, int ccMinor, float epsilon)
    {
        DirectPtxChannelNormalizationOperation operation =
            DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu;
        var ptx = Begin(ccMajor, ccMinor, operation,
            "input_ptr", "residual_ptr", "gamma_ptr", "beta_ptr",
            "running_mean_ptr", "running_var_ptr", "output_ptr");
        EmitRegisters(ptx, predicates: 2, b16: 0, b32: 24, b64: 44, f32: 40);
        EmitPointerLoads(ptx, 7);
        EmitUnitAddressSetup(ptx, UnitLayout.Batch, "RESIDUAL_BN_RELU_DONE");
        EmitInputLoads(ptx, 0, half: false, "%f0", "%f1");
        EmitInputLoads(ptx, 1, half: false, "%f2", "%f3");
        ptx.AppendLine("    add.u64 %rd18, %rd4, %rd32;");
        ptx.AppendLine("    add.u64 %rd19, %rd5, %rd32;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd18];");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd19];");
        ptx.AppendLine($"    add.rn.f32 %f5, %f5, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f6, %f5;");
        ptx.AppendLine("    rcp.approx.f32 %f7, %f6;");
        EmitAffineLoads(ptx, 2, 3, "%f8", "%f9", "%f10", "%f11");
        EmitAffine(ptx, "%f0", "%f1", "%f4", "%f7",
            "%f8", "%f9", "%f10", "%f11", "%f12", "%f13");
        ptx.AppendLine("    add.rn.f32 %f12, %f12, %f2;");
        ptx.AppendLine("    add.rn.f32 %f13, %f13, %f3;");
        ptx.AppendLine("    max.f32 %f12, %f12, 0f00000000;");
        ptx.AppendLine("    max.f32 %f13, %f13, 0f00000000;");
        EmitOutputStores(ptx, 6, half: false, "%f12", "%f13");
        return End(ptx, "RESIDUAL_BN_RELU_DONE");
    }

    private static string EmitAffineForward(
        int ccMajor,
        int ccMinor,
        DirectPtxChannelNormalizationOperation operation,
        UnitLayout layout,
        DirectPtxActivation activation,
        bool half,
        float epsilon)
    {
        bool saveStatistics = operation is
            DirectPtxChannelNormalizationOperation.GroupNormForward or
            DirectPtxChannelNormalizationOperation.InstanceNormForward;
        var ptx = saveStatistics
            ? Begin(ccMajor, ccMinor, operation,
                "input_ptr", "gamma_ptr", "beta_ptr", "output_ptr", "mean_ptr", "inv_var_ptr")
            : Begin(ccMajor, ccMinor, operation,
                "input_ptr", "gamma_ptr", "beta_ptr", "output_ptr");
        EmitRegisters(ptx, predicates: 3, b16: half ? 6 : 0, b32: 24, b64: 44, f32: 40);
        EmitPointerLoads(ptx, saveStatistics ? 6 : 4);
        EmitUnitAddressSetup(ptx, layout, "AFFINE_FWD_DONE", half);
        EmitInputLoads(ptx, 0, half, "%f0", "%f1");
        EmitMeanAndInvStd(ptx, "%f0", "%f1", "%f2", "%f3", "%f4", "%f5", epsilon);
        EmitAffineLoads(ptx, 1, 2, "%f6", "%f7", "%f8", "%f9");
        EmitAffine(ptx, "%f0", "%f1", "%f2", "%f5",
            "%f6", "%f7", "%f8", "%f9", "%f10", "%f11");
        EmitActivation(ptx, activation, "%f10", "%f20", "%f22");
        EmitActivation(ptx, activation, "%f11", "%f21", "%f22");
        EmitOutputStores(ptx, 3, half,
            activation == DirectPtxActivation.None ? "%f10" : "%f20",
            activation == DirectPtxActivation.None ? "%f11" : "%f21");
        if (saveStatistics)
        {
            ptx.AppendLine("    add.u64 %rd18, %rd4, %rd32;");
            ptx.AppendLine("    add.u64 %rd19, %rd5, %rd32;");
            ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");
            ptx.AppendLine("    @%p1 st.global.f32 [%rd18], %f2;");
            ptx.AppendLine($"    @%p1 st.global.f32 [%rd19], {(layout == UnitLayout.Group ? "%f3" : "%f5")};");
        }
        return End(ptx, "AFFINE_FWD_DONE");
    }

    private static string EmitAddGroupNorm(
        int ccMajor, int ccMinor, float epsilon)
    {
        DirectPtxChannelNormalizationOperation operation =
            DirectPtxChannelNormalizationOperation.AddGroupNorm;
        var ptx = Begin(ccMajor, ccMinor, operation,
            "left_ptr", "right_ptr", "gamma_ptr", "beta_ptr", "output_ptr");
        EmitRegisters(ptx, predicates: 3, b16: 0, b32: 24, b64: 44, f32: 40);
        EmitPointerLoads(ptx, 5);
        EmitUnitAddressSetup(ptx, UnitLayout.Group, "ADD_GROUPNORM_DONE");
        EmitInputLoads(ptx, 0, half: false, "%f0", "%f1");
        EmitInputLoads(ptx, 1, half: false, "%f2", "%f3");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f2;");
        ptx.AppendLine("    add.rn.f32 %f1, %f1, %f3;");
        EmitMeanAndInvStd(ptx, "%f0", "%f1", "%f4", "%f5", "%f6", "%f7", epsilon);
        EmitAffineLoads(ptx, 2, 3, "%f8", "%f9", "%f10", "%f11");
        EmitAffine(ptx, "%f0", "%f1", "%f4", "%f7",
            "%f8", "%f9", "%f10", "%f11", "%f12", "%f13");
        EmitOutputStores(ptx, 4, half: false, "%f12", "%f13");
        return End(ptx, "ADD_GROUPNORM_DONE");
    }

    private static string EmitBackward(
        int ccMajor,
        int ccMinor,
        DirectPtxChannelNormalizationOperation operation,
        UnitLayout layout,
        bool statisticsAreVariance,
        float epsilon)
    {
        bool batch = layout == UnitLayout.Batch;
        var ptx = batch
            ? Begin(ccMajor, ccMinor, operation,
                "grad_output_ptr", "input_ptr", "gamma_ptr", "mean_ptr", "inv_var_ptr",
                "grad_input_ptr", "grad_gamma_ptr", "grad_beta_ptr")
            : Begin(ccMajor, ccMinor, operation,
                "grad_output_ptr", "input_ptr", "gamma_ptr", "mean_ptr", "inv_var_ptr",
                "grad_input_ptr");
        EmitRegisters(ptx, predicates: 3, b16: 0, b32: 24, b64: 44, f32: 44);
        EmitPointerLoads(ptx, batch ? 8 : 6);
        EmitUnitAddressSetup(ptx, layout, "CHANNEL_BWD_DONE");
        EmitInputLoads(ptx, 0, half: false, "%f0", "%f1");
        EmitInputLoads(ptx, 1, half: false, "%f2", "%f3");
        ptx.AppendLine("    add.u64 %rd18, %rd2, %rd30;");
        ptx.AppendLine("    add.u64 %rd19, %rd2, %rd31;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd18];");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd19];");
        ptx.AppendLine("    add.u64 %rd20, %rd3, %rd32;");
        ptx.AppendLine("    add.u64 %rd21, %rd4, %rd32;");
        ptx.AppendLine("    ld.global.nc.f32 %f6, [%rd20];");
        ptx.AppendLine("    ld.global.nc.f32 %f7, [%rd21];");
        if (statisticsAreVariance)
        {
            ptx.AppendLine($"    add.rn.f32 %f7, %f7, {FloatLiteral(epsilon)};");
            ptx.AppendLine("    sqrt.rn.f32 %f24, %f7;");
            ptx.AppendLine("    rcp.approx.f32 %f7, %f24;");
        }
        ptx.AppendLine("    sub.rn.f32 %f8, %f2, %f6;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f3, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f8, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f9, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f12, %f0, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f1, %f5;");
        ptx.AppendLine("    add.rn.f32 %f14, %f12, %f13;");
        ptx.AppendLine("    mul.rn.f32 %f15, %f12, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f15, %f13, %f11, %f15;");
        EmitWarpSum(ptx, "%f14", "%f16");
        EmitWarpSum(ptx, "%f15", "%f16");
        ptx.AppendLine("    add.rn.f32 %f17, %f0, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f0, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f18, %f1, %f11, %f18;");
        EmitWarpSum(ptx, "%f17", "%f19");
        EmitWarpSum(ptx, "%f18", "%f19");
        ptx.AppendLine("    mul.rn.f32 %f20, %f10, %f15;");
        ptx.AppendLine("    add.rn.f32 %f20, %f20, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f20, %f20, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f21, %f12, %f20;");
        ptx.AppendLine("    mul.rn.f32 %f21, %f21, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f22, %f11, %f15;");
        ptx.AppendLine("    add.rn.f32 %f22, %f22, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f22, %f22, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f23, %f13, %f22;");
        ptx.AppendLine("    mul.rn.f32 %f23, %f23, %f7;");
        EmitOutputStores(ptx, 5, half: false, "%f21", "%f23");
        if (batch)
        {
            ptx.AppendLine("    add.u64 %rd22, %rd6, %rd32;");
            ptx.AppendLine("    add.u64 %rd23, %rd7, %rd32;");
            ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");
            ptx.AppendLine("    @%p1 st.global.f32 [%rd22], %f18;");
            ptx.AppendLine("    @%p1 st.global.f32 [%rd23], %f17;");
        }
        return End(ptx, "CHANNEL_BWD_DONE");
    }

    private static string EmitGroupGradParameters(
        int ccMajor, int ccMinor, float epsilon)
    {
        DirectPtxChannelNormalizationOperation operation =
            DirectPtxChannelNormalizationOperation.GroupNormGradParameters;
        var ptx = Begin(ccMajor, ccMinor, operation,
            "grad_output_ptr", "input_ptr", "mean_ptr", "variance_ptr",
            "grad_gamma_ptr", "grad_beta_ptr");
        EmitRegisters(ptx, predicates: 4, b16: 0, b32: 24, b64: 14, f32: 10,
            blockThreads: ParameterBlockThreads);
        ptx.AppendLine($"    .shared .align 4 .b8 parameter_scratch[{ParameterSharedBytes}];");
        EmitPointerLoads(ptx, 6);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("GROUP_PARAM_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r0, 256;");
        ptx.AppendLine("    @%p0 bra GROUP_PARAM_DONE;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 3;");
        ptx.AppendLine("    and.b32 %r3, %r0, 7;");
        ptx.AppendLine("    mul.lo.u32 %r4, %r2, 512;");
        ptx.AppendLine("    mad.lo.u32 %r5, %r1, 8, %r3;");
        ptx.AppendLine("    add.u32 %r4, %r4, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd7];");
        ptx.AppendLine("    shr.u32 %r5, %r1, 3;");
        ptx.AppendLine("    mad.lo.u32 %r5, %r2, 8, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd9];");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd9];");
        ptx.AppendLine($"    add.rn.f32 %f5, %f5, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f5, %f5;");
        ptx.AppendLine("    rcp.approx.f32 %f5, %f5;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f6, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f6, %f0;");
        ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        ptx.AppendLine($"    add.u32 %r0, %r0, {ParameterBlockThreads};");
        ptx.AppendLine("    bra GROUP_PARAM_LOOP;");
        ptx.AppendLine("GROUP_PARAM_DONE:");
        EmitParameterBlockReduction(ptx, "GROUP_PARAM");
        return ptx.ToString();
    }

    private static string EmitInstanceGradParameters(int ccMajor, int ccMinor)
    {
        DirectPtxChannelNormalizationOperation operation =
            DirectPtxChannelNormalizationOperation.InstanceNormGradParameters;
        var ptx = Begin(ccMajor, ccMinor, operation,
            "grad_output_ptr", "input_ptr", "mean_ptr", "inv_var_ptr",
            "grad_gamma_ptr", "grad_beta_ptr");
        EmitRegisters(ptx, predicates: 4, b16: 0, b32: 24, b64: 14, f32: 10,
            blockThreads: ParameterBlockThreads);
        ptx.AppendLine($"    .shared .align 4 .b8 parameter_scratch[{ParameterSharedBytes}];");
        EmitPointerLoads(ptx, 6);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("INSTANCE_PARAM_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r0, 2048;");
        ptx.AppendLine("    @%p0 bra INSTANCE_PARAM_DONE;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 6;");
        ptx.AppendLine("    and.b32 %r3, %r0, 63;");
        ptx.AppendLine("    mul.lo.u32 %r4, %r2, 4096;");
        ptx.AppendLine("    mad.lo.u32 %r5, %r1, 64, %r3;");
        ptx.AppendLine("    add.u32 %r4, %r4, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd7];");
        ptx.AppendLine("    mad.lo.u32 %r5, %r2, 64, %r1;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd9];");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd9];");
        ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f6, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f6, %f0;");
        ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        ptx.AppendLine($"    add.u32 %r0, %r0, {ParameterBlockThreads};");
        ptx.AppendLine("    bra INSTANCE_PARAM_LOOP;");
        ptx.AppendLine("INSTANCE_PARAM_DONE:");
        EmitParameterBlockReduction(ptx, "INSTANCE_PARAM");
        return ptx.ToString();
    }

    private static void EmitParameterBlockReduction(StringBuilder ptx, string label)
    {
        EmitWarpSum(ptx, "%f0", "%f2");
        EmitWarpSum(ptx, "%f1", "%f2");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r6, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r7, %r0, 5;");
        ptx.AppendLine("    setp.ne.u32 %p1, %r6, 0;");
        ptx.AppendLine($"    @%p1 bra {label}_BARRIER;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r7, 4;");
        ptx.AppendLine("    mov.u64 %rd11, parameter_scratch;");
        ptx.AppendLine("    add.u64 %rd12, %rd11, %rd10;");
        ptx.AppendLine("    st.shared.f32 [%rd12], %f0;");
        ptx.AppendLine("    add.u64 %rd13, %rd12, 32;");
        ptx.AppendLine("    st.shared.f32 [%rd13], %f1;");
        ptx.AppendLine($"{label}_BARRIER:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r7, 0;");
        ptx.AppendLine($"    @%p2 bra {label}_RETURN;");
        ptx.AppendLine("    setp.lt.u32 %p3, %r6, 8;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine($"    @!%p3 bra {label}_FINAL;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r6, 4;");
        ptx.AppendLine("    mov.u64 %rd11, parameter_scratch;");
        ptx.AppendLine("    add.u64 %rd12, %rd11, %rd10;");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd12];");
        ptx.AppendLine("    add.u64 %rd13, %rd12, 32;");
        ptx.AppendLine("    ld.shared.f32 %f4, [%rd13];");
        ptx.AppendLine($"{label}_FINAL:");
        EmitWarpSum(ptx, "%f3", "%f5");
        EmitWarpSum(ptx, "%f4", "%f5");
        ptx.AppendLine("    setp.ne.u32 %p1, %r6, 0;");
        ptx.AppendLine($"    @%p1 bra {label}_RETURN;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd5, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f3;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f4;");
        ptx.AppendLine($"{label}_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static StringBuilder Begin(
        int ccMajor,
        int ccMinor,
        DirectPtxChannelNormalizationOperation operation,
        params string[] parameters)
    {
        var ptx = new StringBuilder(16_384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact 64-value channel normalization op={OperationName(operation)}");
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        return ptx;
    }

    private static void EmitRegisters(
        StringBuilder ptx,
        int predicates,
        int b16,
        int b32,
        int b64,
        int f32,
        int blockThreads = RowBlockThreads)
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

    private static void EmitUnitAddressSetup(
        StringBuilder ptx, UnitLayout layout, string doneLabel, bool half = false)
    {
        int units = layout switch
        {
            UnitLayout.Batch => BatchNormChannels,
            UnitLayout.Group => GroupNormBatch * GroupNormGroups,
            _ => InstanceNormBatch * InstanceNormChannels
        };
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r2;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {units};");
        ptx.AppendLine($"    @%p0 bra {doneLabel};");
        switch (layout)
        {
            case UnitLayout.Batch:
                ptx.AppendLine("    shr.u32 %r5, %r1, 3;");
                ptx.AppendLine("    and.b32 %r6, %r1, 7;");
                ptx.AppendLine("    mul.lo.u32 %r7, %r5, 512;");
                ptx.AppendLine("    mad.lo.u32 %r8, %r4, 8, %r6;");
                ptx.AppendLine("    add.u32 %r8, %r8, %r7;");
                ptx.AppendLine("    add.u32 %r9, %r8, 2048;");
                ptx.AppendLine("    mov.u32 %r10, %r4;");
                ptx.AppendLine("    mov.u32 %r11, %r4;");
                break;
            case UnitLayout.Group:
                ptx.AppendLine("    shr.u32 %r5, %r4, 3;");
                ptx.AppendLine("    and.b32 %r6, %r4, 7;");
                ptx.AppendLine("    mul.lo.u32 %r7, %r5, 512;");
                ptx.AppendLine("    mad.lo.u32 %r8, %r6, 64, %r1;");
                ptx.AppendLine("    add.u32 %r8, %r8, %r7;");
                ptx.AppendLine("    add.u32 %r9, %r8, 32;");
                ptx.AppendLine("    shr.u32 %r12, %r1, 3;");
                ptx.AppendLine("    mad.lo.u32 %r10, %r6, 8, %r12;");
                ptx.AppendLine("    add.u32 %r11, %r10, 4;");
                break;
            default:
                ptx.AppendLine("    mad.lo.u32 %r8, %r4, 64, %r1;");
                ptx.AppendLine("    add.u32 %r9, %r8, 32;");
                ptx.AppendLine("    and.b32 %r10, %r4, 63;");
                ptx.AppendLine("    mov.u32 %r11, %r10;");
                break;
        }
        int matrixElementBytes = half ? 2 : 4;
        ptx.AppendLine($"    mul.wide.u32 %rd28, %r8, {matrixElementBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd29, %r9, {matrixElementBytes};");
        ptx.AppendLine("    mul.wide.u32 %rd30, %r10, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd31, %r11, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd32, %r4, 4;");
    }

    private static void EmitInputLoads(
        StringBuilder ptx,
        int inputRegister,
        bool half,
        string first,
        string second)
    {
        ptx.AppendLine($"    add.u64 %rd33, %rd{inputRegister}, %rd28;");
        ptx.AppendLine($"    add.u64 %rd34, %rd{inputRegister}, %rd29;");
        if (half)
        {
            ptx.AppendLine("    ld.global.nc.u16 %rs0, [%rd33];");
            ptx.AppendLine("    ld.global.nc.u16 %rs1, [%rd34];");
            ptx.AppendLine($"    cvt.f32.f16 {first}, %rs0;");
            ptx.AppendLine($"    cvt.f32.f16 {second}, %rs1;");
        }
        else
        {
            ptx.AppendLine($"    ld.global.nc.f32 {first}, [%rd33];");
            ptx.AppendLine($"    ld.global.nc.f32 {second}, [%rd34];");
        }
    }

    private static void EmitMeanAndInvStd(
        StringBuilder ptx,
        string first,
        string second,
        string mean,
        string variance,
        string centeredFirst,
        string invStd,
        float epsilon)
    {
        ptx.AppendLine($"    add.rn.f32 {mean}, {first}, {second};");
        EmitWarpSum(ptx, mean, "%f30");
        ptx.AppendLine($"    mul.rn.f32 {mean}, {mean}, 0f3C800000;");
        ptx.AppendLine($"    sub.rn.f32 {centeredFirst}, {first}, {mean};");
        ptx.AppendLine($"    sub.rn.f32 %f31, {second}, {mean};");
        ptx.AppendLine($"    mul.rn.f32 {variance}, {centeredFirst}, {centeredFirst};");
        ptx.AppendLine($"    fma.rn.f32 {variance}, %f31, %f31, {variance};");
        EmitWarpSum(ptx, variance, "%f30");
        ptx.AppendLine($"    mul.rn.f32 {variance}, {variance}, 0f3C800000;");
        ptx.AppendLine($"    add.rn.f32 %f32, {variance}, {FloatLiteral(epsilon)};");
        ptx.AppendLine("    sqrt.rn.f32 %f33, %f32;");
        ptx.AppendLine($"    rcp.approx.f32 {invStd}, %f33;");
    }

    private static void EmitAffineLoads(
        StringBuilder ptx,
        int gammaRegister,
        int betaRegister,
        string gamma0,
        string gamma1,
        string beta0,
        string beta1)
    {
        ptx.AppendLine($"    add.u64 %rd35, %rd{gammaRegister}, %rd30;");
        ptx.AppendLine($"    add.u64 %rd36, %rd{gammaRegister}, %rd31;");
        ptx.AppendLine($"    add.u64 %rd37, %rd{betaRegister}, %rd30;");
        ptx.AppendLine($"    add.u64 %rd38, %rd{betaRegister}, %rd31;");
        ptx.AppendLine($"    ld.global.nc.f32 {gamma0}, [%rd35];");
        ptx.AppendLine($"    ld.global.nc.f32 {gamma1}, [%rd36];");
        ptx.AppendLine($"    ld.global.nc.f32 {beta0}, [%rd37];");
        ptx.AppendLine($"    ld.global.nc.f32 {beta1}, [%rd38];");
    }

    private static void EmitAffine(
        StringBuilder ptx,
        string input0,
        string input1,
        string mean,
        string invStd,
        string gamma0,
        string gamma1,
        string beta0,
        string beta1,
        string output0,
        string output1)
    {
        ptx.AppendLine($"    sub.rn.f32 %f34, {input0}, {mean};");
        ptx.AppendLine($"    sub.rn.f32 %f35, {input1}, {mean};");
        ptx.AppendLine($"    mul.rn.f32 %f34, %f34, {invStd};");
        ptx.AppendLine($"    mul.rn.f32 %f35, %f35, {invStd};");
        ptx.AppendLine($"    fma.rn.f32 {output0}, %f34, {gamma0}, {beta0};");
        ptx.AppendLine($"    fma.rn.f32 {output1}, %f35, {gamma1}, {beta1};");
    }

    private static void EmitOutputStores(
        StringBuilder ptx,
        int outputRegister,
        bool half,
        string first,
        string second)
    {
        int scale = half ? 2 : 4;
        ptx.AppendLine($"    mul.wide.u32 %rd39, %r8, {scale};");
        ptx.AppendLine($"    mul.wide.u32 %rd40, %r9, {scale};");
        ptx.AppendLine($"    add.u64 %rd41, %rd{outputRegister}, %rd39;");
        ptx.AppendLine($"    add.u64 %rd42, %rd{outputRegister}, %rd40;");
        if (half)
        {
            ptx.AppendLine($"    cvt.rn.f16.f32 %rs2, {first};");
            ptx.AppendLine($"    cvt.rn.f16.f32 %rs3, {second};");
            ptx.AppendLine("    st.global.u16 [%rd41], %rs2;");
            ptx.AppendLine("    st.global.u16 [%rd42], %rs3;");
        }
        else
        {
            ptx.AppendLine($"    st.global.f32 [%rd41], {first};");
            ptx.AppendLine($"    st.global.f32 [%rd42], {second};");
        }
    }

    private static void EmitActivation(
        StringBuilder ptx,
        DirectPtxActivation activation,
        string input,
        string output,
        string temporary)
    {
        switch (activation)
        {
            case DirectPtxActivation.None:
                return;
            case DirectPtxActivation.Relu:
                ptx.AppendLine($"    max.f32 {output}, {input}, 0f00000000;");
                return;
            case DirectPtxActivation.Tanh:
                ptx.AppendLine($"    tanh.approx.f32 {output}, {input};");
                return;
            case DirectPtxActivation.Sigmoid:
                ptx.AppendLine($"    mul.rn.f32 {temporary}, {input}, 0fBFB8AA3B;");
                ptx.AppendLine($"    ex2.approx.f32 {temporary}, {temporary};");
                ptx.AppendLine($"    add.rn.f32 {temporary}, {temporary}, 0f3F800000;");
                ptx.AppendLine($"    rcp.approx.f32 {output}, {temporary};");
                return;
            case DirectPtxActivation.Swish:
                ptx.AppendLine($"    mul.rn.f32 {temporary}, {input}, 0fBFB8AA3B;");
                ptx.AppendLine($"    ex2.approx.f32 {temporary}, {temporary};");
                ptx.AppendLine($"    add.rn.f32 {temporary}, {temporary}, 0f3F800000;");
                ptx.AppendLine($"    rcp.approx.f32 {output}, {temporary};");
                ptx.AppendLine($"    mul.rn.f32 {output}, {output}, {input};");
                return;
            default:
                ptx.AppendLine($"    mul.rn.f32 {temporary}, {input}, {input};");
                ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, {input};");
                ptx.AppendLine($"    fma.rn.f32 {temporary}, {temporary}, 0f3D372713, {input};");
                ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, 0f3F4C422A;");
                ptx.AppendLine($"    tanh.approx.f32 {temporary}, {temporary};");
                ptx.AppendLine($"    add.rn.f32 {temporary}, {temporary}, 0f3F800000;");
                ptx.AppendLine($"    mul.rn.f32 {output}, {temporary}, {input};");
                ptx.AppendLine($"    mul.rn.f32 {output}, {output}, 0f3F000000;");
                return;
        }
    }

    private static void EmitWarpSum(StringBuilder ptx, string value, string temporary)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r20, {value};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r21, %r20, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {temporary}, %r21;");
            ptx.AppendLine($"    add.rn.f32 {value}, {value}, {temporary};");
        }
    }

    private static string End(StringBuilder ptx, string label)
    {
        ptx.AppendLine($"{label}:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxChannelNormalizationOperation operation)
    {
        if (architecture != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "Channel-normalization blueprints have only been validated on Ampere.");
        int operationIndex = (int)operation;
        if ((uint)operationIndex >= (uint)EntryPoints.Length)
            throw new ArgumentOutOfRangeException(nameof(operation));
        return AmpereBlueprints[operationIndex];
    }

    private static DirectPtxKernelBlueprint[] CreateAmpereBlueprints()
    {
        int count = Enum.GetValues(typeof(DirectPtxChannelNormalizationOperation)).Length;
        var blueprints = new DirectPtxKernelBlueprint[count];
        for (int operation = 0; operation < count; operation++)
        {
            blueprints[operation] = CreateBlueprintCore(
                DirectPtxArchitectureFamily.Ampere,
                (DirectPtxChannelNormalizationOperation)operation);
        }
        return blueprints;
    }

    private static DirectPtxKernelBlueprint CreateBlueprintCore(
        DirectPtxArchitectureFamily architecture,
        DirectPtxChannelNormalizationOperation operation)
    {
        bool batch = IsBatch(operation);
        bool group = operation is DirectPtxChannelNormalizationOperation.GroupNormForward or
            DirectPtxChannelNormalizationOperation.GroupNormSwish or
            DirectPtxChannelNormalizationOperation.AddGroupNorm or
            DirectPtxChannelNormalizationOperation.GroupNormBackwardInput or
            DirectPtxChannelNormalizationOperation.GroupNormGradParameters or
            DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish;
        bool half = operation == DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish;
        var matrix = batch
            ? new DirectPtxExtent(BatchNormBatch, BatchNormChannels, BatchNormSpatial)
            : group
                ? new DirectPtxExtent(GroupNormBatch, GroupNormChannels, GroupNormSpatial)
                : new DirectPtxExtent(InstanceNormBatch, InstanceNormChannels, InstanceNormSpatial);
        var channels = new DirectPtxExtent(64);
        var stats = batch
            ? new DirectPtxExtent(BatchNormChannels)
            : group
                ? new DirectPtxExtent(GroupNormBatch, GroupNormGroups)
                : new DirectPtxExtent(InstanceNormBatch, InstanceNormChannels);
        DirectPtxPhysicalType matrixType = half
            ? DirectPtxPhysicalType.Float16
            : DirectPtxPhysicalType.Float32;

        DirectPtxTensorContract Matrix(string name, DirectPtxTensorAccess access) =>
            new(name, matrixType, DirectPtxPhysicalLayout.Nchw,
                matrix, matrix, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Channel(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                channels, channels, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Stat(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                stats, stats, 16, access, DirectPtxExtentMode.Exact);

        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxChannelNormalizationOperation.BatchNormTraining =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Channel("gamma", DirectPtxTensorAccess.Read),
                Channel("beta", DirectPtxTensorAccess.Read), Channel("running_mean", DirectPtxTensorAccess.ReadWrite),
                Channel("running_var", DirectPtxTensorAccess.ReadWrite), Matrix("output", DirectPtxTensorAccess.Write),
                Stat("save_mean", DirectPtxTensorAccess.Write), Stat("save_inv_var", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.BatchNormInference or
            DirectPtxChannelNormalizationOperation.BatchNormRelu or
            DirectPtxChannelNormalizationOperation.BatchNormGelu or
            DirectPtxChannelNormalizationOperation.BatchNormSigmoid or
            DirectPtxChannelNormalizationOperation.BatchNormTanh =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Channel("gamma", DirectPtxTensorAccess.Read),
                Channel("beta", DirectPtxTensorAccess.Read), Channel("running_mean", DirectPtxTensorAccess.Read),
                Channel("running_var", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Matrix("residual", DirectPtxTensorAccess.Read),
                Channel("gamma", DirectPtxTensorAccess.Read), Channel("beta", DirectPtxTensorAccess.Read),
                Channel("running_mean", DirectPtxTensorAccess.Read),
                Channel("running_var", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.BatchNormBackward =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read), Matrix("input", DirectPtxTensorAccess.Read),
                Channel("gamma", DirectPtxTensorAccess.Read), Stat("mean", DirectPtxTensorAccess.Read),
                Stat("inv_var", DirectPtxTensorAccess.Read), Matrix("grad_input", DirectPtxTensorAccess.Write),
                Channel("grad_gamma", DirectPtxTensorAccess.Write), Channel("grad_beta", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.GroupNormForward =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Channel("gamma", DirectPtxTensorAccess.Read),
                Channel("beta", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write),
                Stat("mean", DirectPtxTensorAccess.Write), Stat("variance", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.InstanceNormForward =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Channel("gamma", DirectPtxTensorAccess.Read),
                Channel("beta", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write),
                Stat("mean", DirectPtxTensorAccess.Write), Stat("inv_var", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.GroupNormSwish =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Channel("gamma", DirectPtxTensorAccess.Read),
                Channel("beta", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.AddGroupNorm =>
            [
                Matrix("left", DirectPtxTensorAccess.Read), Matrix("right", DirectPtxTensorAccess.Read),
                Channel("gamma", DirectPtxTensorAccess.Read), Channel("beta", DirectPtxTensorAccess.Read),
                Matrix("output", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.GroupNormBackwardInput =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read), Matrix("input", DirectPtxTensorAccess.Read),
                Channel("gamma", DirectPtxTensorAccess.Read), Stat("mean", DirectPtxTensorAccess.Read),
                Stat("variance", DirectPtxTensorAccess.Read), Matrix("grad_input", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.GroupNormGradParameters =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read), Matrix("input", DirectPtxTensorAccess.Read),
                Stat("mean", DirectPtxTensorAccess.Read), Stat("variance", DirectPtxTensorAccess.Read),
                Channel("grad_gamma", DirectPtxTensorAccess.Write), Channel("grad_beta", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.InstanceNormBackwardInput =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read), Matrix("input", DirectPtxTensorAccess.Read),
                Channel("gamma", DirectPtxTensorAccess.Read), Stat("mean", DirectPtxTensorAccess.Read),
                Stat("inv_var", DirectPtxTensorAccess.Read), Matrix("grad_input", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.InstanceNormGradParameters =>
            [
                Matrix("grad_output", DirectPtxTensorAccess.Read), Matrix("input", DirectPtxTensorAccess.Read),
                Stat("mean", DirectPtxTensorAccess.Read), Stat("inv_var", DirectPtxTensorAccess.Read),
                Channel("grad_gamma", DirectPtxTensorAccess.Write), Channel("grad_beta", DirectPtxTensorAccess.Write)
            ],
            DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish =>
            [
                Matrix("input", DirectPtxTensorAccess.Read), Channel("gamma", DirectPtxTensorAccess.Read),
                Channel("beta", DirectPtxTensorAccess.Read), Matrix("output", DirectPtxTensorAccess.Write)
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

        return new DirectPtxKernelBlueprint(
            Operation: OperationName(operation),
            Version: 1,
            Architecture: architecture,
            Variant: "warp-unit64-exact-sm86",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: IsParameterGradient(operation) ? 80 : 56,
                MaxStaticSharedBytes: IsParameterGradient(operation) ? ParameterSharedBytes : 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 3),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = OperationName(operation),
                ["normalization-unit-elements"] = "64",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["shape-stride-mode-parameters"] = "none",
                ["layout"] = "exact canonical NCHW"
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
        {
            DirectPtxTensorContract contract = blueprint.Tensors[i];
            DirectPtxTensorView view = tensors[i];
            if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
                view.Layout != contract.Layout ||
                !SameExtent(view.LogicalExtent, contract.LogicalExtent) ||
                !SameExtent(view.PhysicalExtent, contract.PhysicalExtent) ||
                view.ByteLength != contract.RequiredBytes ||
                view.AllocationByteLength != contract.RequiredBytes ||
                view.Access != contract.Access)
                ThrowTensorContractMismatch(i, contract.Name);
        }
        for (int left = 0; left < tensors.Length; left++)
        {
            for (int right = left + 1; right < tensors.Length; right++)
            {
                bool writes = (tensors[left].Access & DirectPtxTensorAccess.Write) != 0 ||
                    (tensors[right].Access & DirectPtxTensorAccess.Write) != 0;
                if (writes && Overlaps(tensors[left], tensors[right]))
                    throw new ArgumentException(
                        "Channel-normalization writable tensors may not overlap another tensor.");
            }
        }
    }

    private static bool SameExtent(DirectPtxExtent left, DirectPtxExtent right) =>
        left.Rank == right.Rank && left.D0 == right.D0 && left.D1 == right.D1 &&
        left.D2 == right.D2 && left.D3 == right.D3;

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void ThrowTensorCountMismatch(string operation, int expected, int actual) =>
        throw new ArgumentException(
            $"{operation} expects {expected} tensors; received {actual}.");

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void ThrowTensorContractMismatch(int index, string name) =>
        throw new ArgumentException($"tensor[{index}] does not satisfy '{name}'.");

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        return leftStart < checked(rightStart + right.ByteLength) &&
            rightStart < checked(leftStart + left.ByteLength);
    }

    internal static int UnitCount(DirectPtxChannelNormalizationOperation operation) =>
        IsBatch(operation) ? BatchNormChannels :
        operation is DirectPtxChannelNormalizationOperation.GroupNormForward or
            DirectPtxChannelNormalizationOperation.GroupNormSwish or
            DirectPtxChannelNormalizationOperation.AddGroupNorm or
            DirectPtxChannelNormalizationOperation.GroupNormBackwardInput or
            DirectPtxChannelNormalizationOperation.GroupNormGradParameters or
            DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish
            ? GroupNormBatch * GroupNormGroups
            : InstanceNormBatch * InstanceNormChannels;

    private static bool IsBatch(DirectPtxChannelNormalizationOperation operation) =>
        operation is DirectPtxChannelNormalizationOperation.BatchNormTraining or
            DirectPtxChannelNormalizationOperation.BatchNormInference or
            DirectPtxChannelNormalizationOperation.BatchNormRelu or
            DirectPtxChannelNormalizationOperation.BatchNormGelu or
            DirectPtxChannelNormalizationOperation.BatchNormSigmoid or
            DirectPtxChannelNormalizationOperation.BatchNormTanh or
            DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu or
            DirectPtxChannelNormalizationOperation.BatchNormBackward;

    private static bool IsParameterGradient(DirectPtxChannelNormalizationOperation operation) =>
        operation is DirectPtxChannelNormalizationOperation.GroupNormGradParameters or
            DirectPtxChannelNormalizationOperation.InstanceNormGradParameters;

    internal static string GetEntryPoint(DirectPtxChannelNormalizationOperation operation)
    {
        int index = (int)operation;
        if ((uint)index >= (uint)EntryPoints.Length)
            throw new ArgumentOutOfRangeException(nameof(operation));
        return EntryPoints[index];
    }

    private static string[] CreateEntryPoints()
    {
        int count = Enum.GetValues(typeof(DirectPtxChannelNormalizationOperation)).Length;
        var entryPoints = new string[count];
        for (int i = 0; i < count; i++)
            entryPoints[i] = "aidotnet_" +
                OperationName((DirectPtxChannelNormalizationOperation)i).Replace('-', '_');
        return entryPoints;
    }

    private static string OperationName(DirectPtxChannelNormalizationOperation operation) => operation switch
    {
        DirectPtxChannelNormalizationOperation.BatchNormTraining => "batchnorm-training-unit64",
        DirectPtxChannelNormalizationOperation.BatchNormInference => "batchnorm-inference-unit64",
        DirectPtxChannelNormalizationOperation.BatchNormRelu => "batchnorm-relu-unit64",
        DirectPtxChannelNormalizationOperation.BatchNormGelu => "batchnorm-gelu-unit64",
        DirectPtxChannelNormalizationOperation.BatchNormSigmoid => "batchnorm-sigmoid-unit64",
        DirectPtxChannelNormalizationOperation.BatchNormTanh => "batchnorm-tanh-unit64",
        DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu => "residual-batchnorm-relu-unit64",
        DirectPtxChannelNormalizationOperation.BatchNormBackward => "batchnorm-backward-unit64",
        DirectPtxChannelNormalizationOperation.GroupNormForward => "groupnorm-forward-unit64",
        DirectPtxChannelNormalizationOperation.GroupNormSwish => "groupnorm-swish-unit64",
        DirectPtxChannelNormalizationOperation.AddGroupNorm => "add-groupnorm-unit64",
        DirectPtxChannelNormalizationOperation.GroupNormBackwardInput => "groupnorm-backward-input-unit64",
        DirectPtxChannelNormalizationOperation.GroupNormGradParameters => "groupnorm-grad-params-unit64",
        DirectPtxChannelNormalizationOperation.InstanceNormForward => "instancenorm-forward-unit64",
        DirectPtxChannelNormalizationOperation.InstanceNormBackwardInput => "instancenorm-backward-input-unit64",
        DirectPtxChannelNormalizationOperation.InstanceNormGradParameters => "instancenorm-grad-params-unit64",
        DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish => "fp16-groupnorm-swish-unit64",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static void Validate(
        DirectPtxChannelNormalizationOperation operation,
        float epsilon,
        float momentum)
    {
        if (!Enum.IsDefined(typeof(DirectPtxChannelNormalizationOperation), operation))
            throw new ArgumentOutOfRangeException(nameof(operation));
        if (!PtxCompat.IsFinite(epsilon) || epsilon <= 0f)
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (!PtxCompat.IsFinite(momentum) || momentum < 0f || momentum > 1f)
            throw new ArgumentOutOfRangeException(nameof(momentum));
    }

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    private enum UnitLayout
    {
        Batch,
        Group,
        Instance
    }

    private enum DirectPtxActivation
    {
        None,
        Relu,
        Gelu,
        Sigmoid,
        Tanh,
        Swish
    }
}
