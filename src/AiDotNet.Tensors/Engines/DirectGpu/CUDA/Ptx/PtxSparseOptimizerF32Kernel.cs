#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSparseOptimizerOperation
{
    Sgd,
    SgdMomentum,
    Adam,
    AdamW,
    Rmsprop,
    Adagrad,
    Nag,
    Adadelta,
    Amsgrad,
    Adamax,
    Lion,
    Nadam,
    Ftrl,
    ProximalL1
}

/// <summary>
/// Bit-exact specialization key. P0..P4 retain operation-specific scalar
/// arguments so the emitted kernel ABI remains pointer-only.
/// </summary>
internal readonly record struct DirectPtxSparseOptimizerKey(
    DirectPtxSparseOptimizerOperation Operation,
    int P0Bits,
    int P1Bits,
    int P2Bits,
    int P3Bits,
    int P4Bits,
    int Step)
{
    internal float P0 => BitConverter.Int32BitsToSingle(P0Bits);
    internal float P1 => BitConverter.Int32BitsToSingle(P1Bits);
    internal float P2 => BitConverter.Int32BitsToSingle(P2Bits);
    internal float P3 => BitConverter.Int32BitsToSingle(P3Bits);
    internal float P4 => BitConverter.Int32BitsToSingle(P4Bits);

    internal static DirectPtxSparseOptimizerKey Create(
        DirectPtxSparseOptimizerOperation operation,
        float p0,
        float p1 = 0,
        float p2 = 0,
        float p3 = 0,
        float p4 = 0,
        int step = 0) =>
        new(operation, BitConverter.SingleToInt32Bits(p0), BitConverter.SingleToInt32Bits(p1),
            BitConverter.SingleToInt32Bits(p2), BitConverter.SingleToInt32Bits(p3),
            BitConverter.SingleToInt32Bits(p4), step);
}

/// <summary>
/// Exact sparse optimizer specializations. One thread owns one published
/// non-zero entry; parameters, state, and all hyperparameters are absent from
/// the machine-code ABI except for their device pointers.
/// </summary>
internal sealed class PtxSparseOptimizerF32Kernel : IDisposable
{
    internal const int ParameterElements = 1_048_576;
    internal const int NonZeros = 16_384;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxSparseOptimizerKey Key { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSparseOptimizerF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxSparseOptimizerKey key)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Sparse optimizer has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        if (!SupportsConfiguration(key))
            throw new ArgumentOutOfRangeException(nameof(key), "Unsupported sparse optimizer scalar specialization.");
        Key = key;
        EntryPoint = GetEntryPoint(key.Operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, key);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, key);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int parameterElements, int nonZeros) =>
        parameterElements == ParameterElements && nonZeros == NonZeros;

    internal static bool SupportsConfiguration(DirectPtxSparseOptimizerKey key)
    {
        bool Finite(float value) => float.IsFinite(value);
        bool Basic2(float a, float b) => Finite(a) && Finite(b);
        bool Basic3(float a, float b, float c) => Basic2(a, b) && Finite(c);
        bool Basic4(float a, float b, float c, float d) => Basic3(a, b, c) && Finite(d);
        bool Basic5(float a, float b, float c, float d, float e) => Basic4(a, b, c, d) && Finite(e);

        return key.Operation switch
        {
            DirectPtxSparseOptimizerOperation.Sgd or
            DirectPtxSparseOptimizerOperation.ProximalL1 => Basic2(key.P0, key.P1),
            DirectPtxSparseOptimizerOperation.SgdMomentum or
            DirectPtxSparseOptimizerOperation.Nag or
            DirectPtxSparseOptimizerOperation.Lion => Basic4(key.P0, key.P1, key.P2, key.P3),
            DirectPtxSparseOptimizerOperation.Rmsprop =>
                Basic4(key.P0, key.P1, key.P2, key.P3) && key.P2 > 0,
            DirectPtxSparseOptimizerOperation.Adagrad =>
                Basic3(key.P0, key.P1, key.P2) && key.P1 > 0,
            DirectPtxSparseOptimizerOperation.Adadelta =>
                Basic3(key.P0, key.P1, key.P2) && key.P1 > 0,
            DirectPtxSparseOptimizerOperation.Adam or
            DirectPtxSparseOptimizerOperation.AdamW or
            DirectPtxSparseOptimizerOperation.Amsgrad or
            DirectPtxSparseOptimizerOperation.Adamax or
            DirectPtxSparseOptimizerOperation.Nadam =>
                Basic5(key.P0, key.P1, key.P2, key.P3, key.P4) && key.P3 > 0 &&
                key.P1 >= 0 && key.P1 < 1 && key.P2 >= 0 && key.P2 < 1 &&
                key.Step >= 1 && (key.Operation != DirectPtxSparseOptimizerOperation.Nadam ||
                    key.Step < int.MaxValue),
            DirectPtxSparseOptimizerOperation.Ftrl =>
                Basic4(key.P0, key.P1, key.P2, key.P3) && key.P0 != 0,
            _ => false
        };
    }

    internal unsafe void Launch(
        DirectPtxTensorView parameter,
        DirectPtxTensorView indices,
        DirectPtxTensorView values,
        DirectPtxTensorView state0 = default,
        DirectPtxTensorView state1 = default,
        DirectPtxTensorView state2 = default)
    {
        Require(parameter, Blueprint.Tensors[0], nameof(parameter));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(values, Blueprint.Tensors[2], nameof(values));
        int stateCount = GetStateCount(Key.Operation);
        if (stateCount >= 1) Require(state0, Blueprint.Tensors[3], nameof(state0));
        if (stateCount >= 2) Require(state1, Blueprint.Tensors[4], nameof(state1));
        if (stateCount >= 3) Require(state2, Blueprint.Tensors[5], nameof(state2));
        RequireDisjoint(parameter, indices, values, state0, state1, state2, stateCount);

        IntPtr p0 = parameter.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = values.Pointer;
        IntPtr p3 = state0.Pointer;
        IntPtr p4 = state1.Pointer;
        IntPtr p5 = state2.Pointer;
        uint grid = NonZeros / BlockThreads;
        if (stateCount == 0)
        {
            void** arguments = stackalloc void*[3];
            arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2;
            _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
        }
        else if (stateCount == 1)
        {
            void** arguments = stackalloc void*[4];
            arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2; arguments[3] = &p3;
            _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
        }
        else if (stateCount == 2)
        {
            void** arguments = stackalloc void*[5];
            arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2;
            arguments[3] = &p3; arguments[4] = &p4;
            _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
        }
        else
        {
            void** arguments = stackalloc void*[6];
            arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2;
            arguments[3] = &p3; arguments[4] = &p4; arguments[5] = &p5;
            _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
        }
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxSparseOptimizerKey key)
    {
        if (!SupportsConfiguration(key))
            throw new ArgumentOutOfRangeException(nameof(key));
        var parameterExtent = new DirectPtxExtent(ParameterElements);
        var sparseExtent = new DirectPtxExtent(NonZeros);
        int stateCount = GetStateCount(key.Operation);
        var tensors = new List<DirectPtxTensorContract>(3 + stateCount)
        {
            Exact("parameter", DirectPtxPhysicalLayout.SparseOptimizerState,
                parameterExtent, DirectPtxTensorAccess.ReadWrite),
            Exact("indices", DirectPtxPhysicalLayout.SparseOptimizerFloatIndices,
                sparseExtent, DirectPtxTensorAccess.Read),
            Exact("values", DirectPtxPhysicalLayout.SparseOptimizerValues,
                sparseExtent, DirectPtxTensorAccess.Read)
        };
        for (int state = 0; state < stateCount; state++)
            tensors.Add(Exact($"state-{state}", DirectPtxPhysicalLayout.SparseOptimizerState,
                parameterExtent, DirectPtxTensorAccess.ReadWrite));
        return new DirectPtxKernelBlueprint(
            Operation: GetOperationName(key.Operation),
            Version: 1,
            Architecture: architecture,
            Variant: $"p{ParameterElements}-nnz{NonZeros}-baked-{GetScalarIdentity(key)}",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(40, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["index-encoding"] = "bitcast-int32-first-otherwise-finite-integral-fp32",
                ["duplicate-index-order"] = "unordered-matches-one-thread-per-nonzero-cuda-surface",
                ["scalar-specialization"] = GetScalarDescription(key),
                ["kernel-parameters"] = "pointers-only",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxSparseOptimizerKey key)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        if (!SupportsConfiguration(key)) throw new ArgumentOutOfRangeException(nameof(key));
        int stateCount = GetStateCount(key.Operation);
        var ptx = new StringBuilder(6144);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(key.Operation)}(");
        ptx.AppendLine("    .param .u64 parameter_ptr,");
        ptx.AppendLine("    .param .u64 indices_ptr,");
        ptx.Append($"    .param .u64 values_ptr{(stateCount == 0 ? string.Empty : ",")}").AppendLine();
        for (int state = 0; state < stateCount; state++)
            ptx.Append($"    .param .u64 state{state}_ptr{(state + 1 == stateCount ? string.Empty : ",")}").AppendLine();
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<10>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [parameter_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [values_ptr];");
        for (int state = 0; state < stateCount; state++)
            ptx.AppendLine($"    ld.param.u64 %rd{3 + state}, [state{state}_ptr];");
        EmitIndexDecode(ptx);
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd8;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd7];");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd9];");
        EmitOperation(ptx, key);
        ptx.AppendLine("OPT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitIndexDecode(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.b32 %r3, [%rd7];");
        ptx.AppendLine("    mov.b32 %f0, %r3;");
        ptx.AppendLine("    setp.ge.s32 %p0, %r3, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r3, {ParameterElements};");
        ptx.AppendLine("    and.pred %p2, %p0, %p1;");
        ptx.AppendLine("    @%p2 mov.u32 %r4, %r3;");
        ptx.AppendLine("    @%p2 bra OPT_INDEX_OK;");
        ptx.AppendLine("    and.b32 %r5, %r3, 2139095040;");
        ptx.AppendLine("    setp.eq.u32 %p3, %r5, 2139095040;");
        ptx.AppendLine("    @%p3 bra OPT_DONE;");
        ptx.AppendLine("    setp.ge.f32 %p4, %f0, 0f00000000;");
        ptx.AppendLine($"    setp.lt.f32 %p5, %f0, {F(ParameterElements)};");
        ptx.AppendLine("    and.pred %p6, %p4, %p5;");
        ptx.AppendLine("    @!%p6 bra OPT_DONE;");
        ptx.AppendLine("    cvt.rzi.s32.f32 %r4, %f0;");
        ptx.AppendLine("    cvt.rn.f32.s32 %f1, %r4;");
        ptx.AppendLine("    setp.eq.f32 %p7, %f0, %f1;");
        ptx.AppendLine("    @!%p7 bra OPT_DONE;");
        ptx.AppendLine("OPT_INDEX_OK:");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
    }

    private static void EmitOperation(StringBuilder ptx, DirectPtxSparseOptimizerKey key)
    {
        switch (key.Operation)
        {
            case DirectPtxSparseOptimizerOperation.Sgd:
                EmitCoupledGradient(ptx, key.P1);
                ptx.AppendLine($"    fma.rn.f32 %f3, {F(-key.P0)}, %f2, %f3;");
                StoreParameter(ptx);
                break;
            case DirectPtxSparseOptimizerOperation.SgdMomentum:
                EmitCoupledGradient(ptx, key.P2);
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine($"    fma.rn.f32 %f5, {F(key.P1)}, %f4, %f2;");
                StoreState(ptx, 0, "%f5");
                ptx.AppendLine($"    fma.rn.f32 %f3, {F(-key.P0)}, %f5, %f3;");
                StoreParameter(ptx);
                break;
            case DirectPtxSparseOptimizerOperation.Adam:
                EmitAdamMoments(ptx, key, coupledDecay: true);
                EmitAdamUpdate(ptx, key, "%f5", "%f7");
                break;
            case DirectPtxSparseOptimizerOperation.AdamW:
                EmitAdamMoments(ptx, key, coupledDecay: false);
                if (key.P4 > 0)
                {
                    ptx.AppendLine($"    mul.rn.f32 %f8, %f3, {F(key.P4)};");
                    ptx.AppendLine($"    mul.rn.f32 %f8, %f8, {F(key.P0)};");
                    ptx.AppendLine("    sub.rn.f32 %f3, %f3, %f8;");
                }
                EmitAdamUpdate(ptx, key, "%f5", "%f7");
                break;
            case DirectPtxSparseOptimizerOperation.Rmsprop:
                EmitCoupledGradient(ptx, key.P3);
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine("    mul.rn.f32 %f5, %f2, %f2;");
                ptx.AppendLine($"    mul.rn.f32 %f4, %f4, {F(key.P1)};");
                ptx.AppendLine($"    fma.rn.f32 %f5, {F(1f - key.P1)}, %f5, %f4;");
                StoreState(ptx, 0, "%f5");
                EmitNormalizedGradient(ptx, key.P0, key.P2, "%f5");
                break;
            case DirectPtxSparseOptimizerOperation.Adagrad:
                EmitCoupledGradient(ptx, key.P2);
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine("    fma.rn.f32 %f5, %f2, %f2, %f4;");
                StoreState(ptx, 0, "%f5");
                EmitNormalizedGradient(ptx, key.P0, key.P1, "%f5");
                break;
            case DirectPtxSparseOptimizerOperation.Nag:
                EmitCoupledGradient(ptx, key.P2);
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine($"    fma.rn.f32 %f5, {F(key.P1)}, %f4, %f2;");
                StoreState(ptx, 0, "%f5");
                ptx.AppendLine($"    mul.rn.f32 %f6, %f5, {F(1f + key.P1)};");
                ptx.AppendLine($"    fma.rn.f32 %f6, {F(-key.P1)}, %f4, %f6;");
                ptx.AppendLine($"    fma.rn.f32 %f3, {F(-key.P0)}, %f6, %f3;");
                StoreParameter(ptx);
                break;
            case DirectPtxSparseOptimizerOperation.Adadelta:
                EmitCoupledGradient(ptx, key.P2);
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine("    mul.rn.f32 %f5, %f2, %f2;");
                ptx.AppendLine($"    mul.rn.f32 %f4, %f4, {F(key.P0)};");
                ptx.AppendLine($"    fma.rn.f32 %f5, {F(1f - key.P0)}, %f5, %f4;");
                StoreState(ptx, 0, "%f5");
                LoadState(ptx, 1, "%f6");
                ptx.AppendLine($"    add.rn.f32 %f7, %f6, {F(key.P1)};");
                ptx.AppendLine("    sqrt.rn.f32 %f7, %f7;");
                ptx.AppendLine($"    add.rn.f32 %f8, %f5, {F(key.P1)};");
                ptx.AppendLine("    sqrt.rn.f32 %f8, %f8;");
                ptx.AppendLine("    div.rn.f32 %f7, %f7, %f8;");
                ptx.AppendLine("    mul.rn.f32 %f7, %f7, %f2;");
                ptx.AppendLine("    mul.rn.f32 %f8, %f7, %f7;");
                ptx.AppendLine($"    mul.rn.f32 %f6, %f6, {F(key.P0)};");
                ptx.AppendLine($"    fma.rn.f32 %f6, {F(1f - key.P0)}, %f8, %f6;");
                StoreState(ptx, 1, "%f6");
                ptx.AppendLine("    sub.rn.f32 %f3, %f3, %f7;");
                StoreParameter(ptx);
                break;
            case DirectPtxSparseOptimizerOperation.Amsgrad:
                EmitAdamMoments(ptx, key, coupledDecay: true);
                LoadState(ptx, 2, "%f8");
                ptx.AppendLine("    max.f32 %f8, %f8, %f7;");
                StoreState(ptx, 2, "%f8");
                ptx.AppendLine("    mov.f32 %f7, %f8;");
                EmitAdamUpdate(ptx, key, "%f5", "%f7");
                break;
            case DirectPtxSparseOptimizerOperation.Adamax:
                EmitCoupledGradient(ptx, key.P4);
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine($"    mul.rn.f32 %f5, %f4, {F(key.P1)};");
                ptx.AppendLine($"    fma.rn.f32 %f5, {F(1f - key.P1)}, %f2, %f5;");
                StoreState(ptx, 0, "%f5");
                LoadState(ptx, 1, "%f6");
                ptx.AppendLine($"    mul.rn.f32 %f6, %f6, {F(key.P2)};");
                ptx.AppendLine("    abs.f32 %f7, %f2;");
                ptx.AppendLine("    max.f32 %f6, %f6, %f7;");
                StoreState(ptx, 1, "%f6");
                ptx.AppendLine($"    mul.rn.f32 %f5, %f5, {F(key.P0 / BiasCorrection(key.P1, key.Step))};");
                ptx.AppendLine($"    add.rn.f32 %f6, %f6, {F(key.P3)};");
                ptx.AppendLine("    div.rn.f32 %f5, %f5, %f6;");
                ptx.AppendLine("    sub.rn.f32 %f3, %f3, %f5;");
                StoreParameter(ptx);
                break;
            case DirectPtxSparseOptimizerOperation.Lion:
                LoadState(ptx, 0, "%f4");
                ptx.AppendLine($"    mul.rn.f32 %f5, %f4, {F(key.P1)};");
                ptx.AppendLine($"    fma.rn.f32 %f5, {F(1f - key.P1)}, %f2, %f5;");
                ptx.AppendLine("    setp.gt.f32 %p8, %f5, 0f00000000;");
                ptx.AppendLine("    setp.lt.f32 %p9, %f5, 0f00000000;");
                ptx.AppendLine("    selp.f32 %f6, 0f3f800000, 0f00000000, %p8;");
                ptx.AppendLine("    @%p9 mov.f32 %f6, 0fbf800000;");
                if (key.P3 > 0)
                    ptx.AppendLine($"    fma.rn.f32 %f6, {F(key.P3)}, %f3, %f6;");
                ptx.AppendLine($"    fma.rn.f32 %f3, {F(-key.P0)}, %f6, %f3;");
                StoreParameter(ptx);
                ptx.AppendLine($"    mul.rn.f32 %f4, %f4, {F(key.P2)};");
                ptx.AppendLine($"    fma.rn.f32 %f4, {F(1f - key.P2)}, %f2, %f4;");
                StoreState(ptx, 0, "%f4");
                break;
            case DirectPtxSparseOptimizerOperation.Nadam:
                EmitAdamMoments(ptx, key, coupledDecay: true);
                ptx.AppendLine($"    mul.rn.f32 %f8, %f5, {F(key.P1 / BiasCorrection(key.P1, key.Step))};");
                ptx.AppendLine($"    fma.rn.f32 %f8, {F((1f - key.P1) / BiasCorrection(key.P1, key.Step + 1))}, %f2, %f8;");
                ptx.AppendLine($"    mul.rn.f32 %f9, %f7, {F(1f / BiasCorrection(key.P2, key.Step))};");
                EmitParameterFromNumerator(ptx, key.P0, key.P3, "%f8", "%f9");
                break;
            case DirectPtxSparseOptimizerOperation.Ftrl:
                LoadState(ptx, 1, "%f4");
                ptx.AppendLine("    fma.rn.f32 %f5, %f2, %f2, %f4;");
                StoreState(ptx, 1, "%f5");
                ptx.AppendLine("    sqrt.rn.f32 %f6, %f5;");
                ptx.AppendLine("    sqrt.rn.f32 %f7, %f4;");
                ptx.AppendLine("    sub.rn.f32 %f8, %f6, %f7;");
                ptx.AppendLine($"    mul.rn.f32 %f8, %f8, {F(1f / key.P0)};");
                LoadState(ptx, 0, "%f7");
                ptx.AppendLine("    mul.rn.f32 %f9, %f8, %f3;");
                ptx.AppendLine("    sub.rn.f32 %f9, %f2, %f9;");
                ptx.AppendLine("    add.rn.f32 %f7, %f7, %f9;");
                StoreState(ptx, 0, "%f7");
                ptx.AppendLine("    abs.f32 %f8, %f7;");
                ptx.AppendLine($"    setp.le.f32 %p8, %f8, {F(key.P1)};");
                ptx.AppendLine("    @%p8 mov.f32 %f3, 0f00000000;");
                ptx.AppendLine("    @%p8 bra OPT_STORE_PARAMETER;");
                ptx.AppendLine("    setp.gt.f32 %p9, %f7, 0f00000000;");
                ptx.AppendLine("    selp.f32 %f8, 0f3f800000, 0fbf800000, %p9;");
                ptx.AppendLine($"    mul.rn.f32 %f8, %f8, {F(key.P1)};");
                ptx.AppendLine("    sub.rn.f32 %f8, %f8, %f7;");
                ptx.AppendLine($"    add.rn.f32 %f6, %f6, {F(key.P3)};");
                ptx.AppendLine($"    mul.rn.f32 %f6, %f6, {F(1f / key.P0)};");
                ptx.AppendLine($"    add.rn.f32 %f6, %f6, {F(key.P2)};");
                ptx.AppendLine("    div.rn.f32 %f3, %f8, %f6;");
                ptx.AppendLine("OPT_STORE_PARAMETER:");
                StoreParameter(ptx);
                break;
            case DirectPtxSparseOptimizerOperation.ProximalL1:
                ptx.AppendLine($"    fma.rn.f32 %f3, {F(-key.P0)}, %f2, %f3;");
                ptx.AppendLine($"    setp.gt.f32 %p8, %f3, {F(key.P0 * key.P1)};");
                ptx.AppendLine($"    setp.lt.f32 %p9, %f3, {F(-(key.P0 * key.P1))};");
                ptx.AppendLine($"    @%p8 add.rn.f32 %f3, %f3, {F(-(key.P0 * key.P1))};");
                ptx.AppendLine($"    @%p9 add.rn.f32 %f3, %f3, {F(key.P0 * key.P1)};");
                ptx.AppendLine("    or.pred %p8, %p8, %p9;");
                ptx.AppendLine("    @!%p8 mov.f32 %f3, 0f00000000;");
                StoreParameter(ptx);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(key));
        }
    }

    private static void EmitAdamMoments(
        StringBuilder ptx,
        DirectPtxSparseOptimizerKey key,
        bool coupledDecay)
    {
        if (coupledDecay) EmitCoupledGradient(ptx, key.P4);
        LoadState(ptx, 0, "%f4");
        ptx.AppendLine($"    mul.rn.f32 %f5, %f4, {F(key.P1)};");
        ptx.AppendLine($"    fma.rn.f32 %f5, {F(1f - key.P1)}, %f2, %f5;");
        StoreState(ptx, 0, "%f5");
        LoadState(ptx, 1, "%f6");
        ptx.AppendLine("    mul.rn.f32 %f7, %f2, %f2;");
        ptx.AppendLine($"    mul.rn.f32 %f6, %f6, {F(key.P2)};");
        ptx.AppendLine($"    fma.rn.f32 %f7, {F(1f - key.P2)}, %f7, %f6;");
        StoreState(ptx, 1, "%f7");
    }

    private static void EmitAdamUpdate(
        StringBuilder ptx,
        DirectPtxSparseOptimizerKey key,
        string moment,
        string variance)
    {
        ptx.AppendLine($"    mul.rn.f32 %f8, {moment}, {F(1f / BiasCorrection(key.P1, key.Step))};");
        ptx.AppendLine($"    mul.rn.f32 %f9, {variance}, {F(1f / BiasCorrection(key.P2, key.Step))};");
        EmitParameterFromNumerator(ptx, key.P0, key.P3, "%f8", "%f9");
    }

    private static void EmitParameterFromNumerator(
        StringBuilder ptx,
        float learningRate,
        float epsilon,
        string numerator,
        string variance)
    {
        ptx.AppendLine($"    sqrt.rn.f32 %f9, {variance};");
        ptx.AppendLine($"    add.rn.f32 %f9, %f9, {F(epsilon)};");
        ptx.AppendLine($"    div.rn.f32 %f8, {numerator}, %f9;");
        ptx.AppendLine($"    fma.rn.f32 %f3, {F(-learningRate)}, %f8, %f3;");
        StoreParameter(ptx);
    }

    private static void EmitNormalizedGradient(
        StringBuilder ptx,
        float learningRate,
        float epsilon,
        string accumulator)
    {
        ptx.AppendLine($"    sqrt.rn.f32 %f6, {accumulator};");
        ptx.AppendLine($"    add.rn.f32 %f6, %f6, {F(epsilon)};");
        ptx.AppendLine("    div.rn.f32 %f6, %f2, %f6;");
        ptx.AppendLine($"    fma.rn.f32 %f3, {F(-learningRate)}, %f6, %f3;");
        StoreParameter(ptx);
    }

    private static void EmitCoupledGradient(StringBuilder ptx, float weightDecay)
    {
        if (weightDecay > 0)
            ptx.AppendLine($"    fma.rn.f32 %f2, {F(weightDecay)}, %f3, %f2;");
    }

    private static void LoadState(StringBuilder ptx, int state, string destination)
    {
        ptx.AppendLine($"    add.u64 %rd10, %rd{3 + state}, %rd8;");
        ptx.AppendLine($"    ld.global.f32 {destination}, [%rd10];");
    }

    private static void StoreState(StringBuilder ptx, int state, string source)
    {
        ptx.AppendLine($"    add.u64 %rd10, %rd{3 + state}, %rd8;");
        ptx.AppendLine($"    st.global.f32 [%rd10], {source};");
    }

    private static void StoreParameter(StringBuilder ptx) =>
        ptx.AppendLine("    st.global.f32 [%rd9], %f3;");

    private static float BiasCorrection(float beta, int step) =>
        1f - MathF.Pow(beta, step);

    private static string F(float value) =>
        $"0f{BitConverter.SingleToInt32Bits(value):x8}";

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, DirectPtxPhysicalType.Float32, layout, extent, extent, 16, access,
            DirectPtxExtentMode.Exact);

    internal static int GetStateCount(DirectPtxSparseOptimizerOperation operation) => operation switch
    {
        DirectPtxSparseOptimizerOperation.Sgd or DirectPtxSparseOptimizerOperation.ProximalL1 => 0,
        DirectPtxSparseOptimizerOperation.SgdMomentum or DirectPtxSparseOptimizerOperation.Rmsprop or
        DirectPtxSparseOptimizerOperation.Adagrad or DirectPtxSparseOptimizerOperation.Nag or
        DirectPtxSparseOptimizerOperation.Lion => 1,
        DirectPtxSparseOptimizerOperation.Adam or DirectPtxSparseOptimizerOperation.AdamW or
        DirectPtxSparseOptimizerOperation.Adadelta or DirectPtxSparseOptimizerOperation.Adamax or
        DirectPtxSparseOptimizerOperation.Nadam or DirectPtxSparseOptimizerOperation.Ftrl => 2,
        DirectPtxSparseOptimizerOperation.Amsgrad => 3,
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static string GetEntryPoint(DirectPtxSparseOptimizerOperation operation) =>
        $"aidotnet_{GetOperationName(operation).Replace('-', '_')}_p1048576_nnz16384";

    private static string GetOperationName(DirectPtxSparseOptimizerOperation operation) => operation switch
    {
        DirectPtxSparseOptimizerOperation.Sgd => "sparse-sgd-update-f32",
        DirectPtxSparseOptimizerOperation.SgdMomentum => "sparse-sgd-momentum-update-f32",
        DirectPtxSparseOptimizerOperation.Adam => "sparse-adam-update-f32",
        DirectPtxSparseOptimizerOperation.AdamW => "sparse-adamw-update-f32",
        DirectPtxSparseOptimizerOperation.Rmsprop => "sparse-rmsprop-update-f32",
        DirectPtxSparseOptimizerOperation.Adagrad => "sparse-adagrad-update-f32",
        DirectPtxSparseOptimizerOperation.Nag => "sparse-nag-update-f32",
        DirectPtxSparseOptimizerOperation.Adadelta => "sparse-adadelta-update-f32",
        DirectPtxSparseOptimizerOperation.Amsgrad => "sparse-amsgrad-update-f32",
        DirectPtxSparseOptimizerOperation.Adamax => "sparse-adamax-update-f32",
        DirectPtxSparseOptimizerOperation.Lion => "sparse-lion-update-f32",
        DirectPtxSparseOptimizerOperation.Nadam => "sparse-nadam-update-f32",
        DirectPtxSparseOptimizerOperation.Ftrl => "sparse-ftrl-update-f32",
        DirectPtxSparseOptimizerOperation.ProximalL1 => "sparse-proximal-l1-update-f32",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static string GetScalarIdentity(DirectPtxSparseOptimizerKey key) =>
        $"{key.P0Bits:x8}-{key.P1Bits:x8}-{key.P2Bits:x8}-{key.P3Bits:x8}-{key.P4Bits:x8}-s{key.Step}";

    private static string GetScalarDescription(DirectPtxSparseOptimizerKey key) =>
        string.Create(CultureInfo.InvariantCulture,
            $"p0={key.P0:R};p1={key.P1:R};p2={key.P2:R};p3={key.P3:R};p4={key.P4:R};step={key.Step}");

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static void RequireDisjoint(
        DirectPtxTensorView parameter,
        DirectPtxTensorView indices,
        DirectPtxTensorView values,
        DirectPtxTensorView state0,
        DirectPtxTensorView state1,
        DirectPtxTensorView state2,
        int stateCount)
    {
        if (Overlaps(parameter, indices) || Overlaps(parameter, values) || Overlaps(indices, values) ||
            (stateCount >= 1 && (Overlaps(state0, parameter) || Overlaps(state0, indices) || Overlaps(state0, values))) ||
            (stateCount >= 2 && (Overlaps(state1, parameter) || Overlaps(state1, indices) ||
                Overlaps(state1, values) || Overlaps(state1, state0))) ||
            (stateCount >= 3 && (Overlaps(state2, parameter) || Overlaps(state2, indices) ||
                Overlaps(state2, values) || Overlaps(state2, state0) || Overlaps(state2, state1))))
            throw new ArgumentException("Sparse optimizer parameter, index, value, and state buffers must be disjoint.");
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
