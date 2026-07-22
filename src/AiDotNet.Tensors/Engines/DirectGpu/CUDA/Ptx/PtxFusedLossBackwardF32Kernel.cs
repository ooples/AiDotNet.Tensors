using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Which loss gradient a <see cref="PtxFusedLossBackwardF32Kernel"/> emits.
/// The two operators have different launch ABIs - MSE takes a broadcast
/// gradOutput scalar and a baked 1/N, MAE takes neither - so each is a separate
/// module and a separate issue-#847 coverage cell.
/// </summary>
internal enum DirectPtxLossBackwardOp
{
    /// <summary>gradInput[i] = gradOutput[0] * 2 * (pred[i] - target[i]) * invN.</summary>
    MeanSquaredError,

    /// <summary>gradInput[i] = sign(pred[i] - target[i]), with zero at zero and at NaN.</summary>
    MeanAbsoluteError
}

/// <summary>
/// Exact contiguous FP32 loss-gradient kernels for issue #847. Both operators
/// are pure elementwise maps, so each thread handles one FP32x4 vector: the
/// inputs are read once, everything stays in registers, and one vector is
/// written. There is no shared memory, no local memory, no global intermediate,
/// no temporary allocation, no stride parameter, and no tail branch.
///
/// The arithmetic mirrors the established NVRTC kernels term for term, in the
/// same association order, because a loss gradient that rounds differently from
/// the kernel it replaces would silently change training results:
///
/// - mse_loss_backward computes <c>gradOutput[0] * 2.0f * (p - t) * inv_n</c>,
///   which C evaluates as <c>((g * 2) * d) * invN</c>. gradOutput[0] is a
///   broadcast scalar, so <c>g * 2</c> is loop-invariant and is hoisted once;
///   the per-element work is then exactly two multiplies in that same order.
///   inv_n is baked into the module as an IEEE-754 literal, so the kernel keys
///   on its exact bit pattern rather than dividing.
/// - mae_gradient computes <c>(d &gt; 0) ? 1 : ((d &lt; 0) ? -1 : 0)</c>. Two
///   predicates and two selects reproduce it exactly, including the NaN case:
///   both comparisons are false, so the result is +0.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedLossBackwardF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const int ElementsPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxLossBackwardOp Op { get; }
    internal int Size { get; }
    internal int BlockThreads { get; }
    internal float InvN { get; }
    internal int ElementsPerBlock => BlockThreads * ElementsPerThread;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(DirectPtxLossBackwardOp op) => op switch
    {
        DirectPtxLossBackwardOp.MeanSquaredError => "aidotnet_fused_mse_loss_backward_f32",
        DirectPtxLossBackwardOp.MeanAbsoluteError => "aidotnet_fused_mae_gradient_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    internal PtxFusedLossBackwardF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxLossBackwardOp op,
        int size,
        float invN = 1f,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedLossBackward(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in loss-gradient specializations are admitted only on SM86.");
        ValidateOp(op);
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        ValidateInvN(op, invN);
        Op = op;
        Size = size;
        InvN = invN;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, size, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            op, size, invN, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(op), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(op), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    /// <summary>
    /// Launches the gradient. <paramref name="gradOutput"/> is the broadcast
    /// scalar required by the MSE operator and must be null for MAE, whose
    /// established kernel takes no upstream gradient.
    /// </summary>
    internal unsafe void Launch(
        DirectPtxTensorView predictions,
        DirectPtxTensorView targets,
        DirectPtxTensorView gradInput,
        DirectPtxTensorView? gradOutput = null)
    {
        bool needsGradOutput = Op == DirectPtxLossBackwardOp.MeanSquaredError;
        if (needsGradOutput && !gradOutput.HasValue)
            throw new ArgumentException(
                "The MSE loss gradient requires the broadcast gradOutput scalar.", nameof(gradOutput));
        if (!needsGradOutput && gradOutput.HasValue)
            throw new ArgumentException(
                "The MAE gradient does not take an upstream gradOutput.", nameof(gradOutput));

        IntPtr gradOutputPointer = IntPtr.Zero;
        int cursor = 0;
        // Pattern-match rather than read .Value: this narrows the nullable for
        // the compiler without a null-forgiving operator. The guards above have
        // already proved presence matches the operator.
        if (gradOutput is { } gradOutputView)
        {
            Require(gradOutputView, Blueprint.Tensors[cursor++], nameof(gradOutput));
            gradOutputPointer = gradOutputView.Pointer;
        }
        Require(predictions, Blueprint.Tensors[cursor++], nameof(predictions));
        Require(targets, Blueprint.Tensors[cursor++], nameof(targets));
        Require(gradInput, Blueprint.Tensors[cursor], nameof(gradInput));

        IntPtr predictionsPointer = predictions.Pointer;
        IntPtr targetsPointer = targets.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;

        void** arguments = stackalloc void*[4];
        int argumentCount = 0;
        if (needsGradOutput)
            arguments[argumentCount++] = &gradOutputPointer;
        arguments[argumentCount++] = &predictionsPointer;
        arguments[argumentCount++] = &targetsPointer;
        arguments[argumentCount] = &gradInputPointer;

        _module.Launch(
            _function,
            checked((uint)(Size / ElementsPerBlock)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxLossBackwardOp op,
        int size,
        float invN = 1f,
        int blockThreads = DefaultBlockThreads)
    {
        ValidateOp(op);
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        ValidateInvN(op, invN);
        bool isMse = op == DirectPtxLossBackwardOp.MeanSquaredError;
        string entryPoint = EntryPointFor(op);

        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape size={size} block={blockThreads} elems-per-thread={ElementsPerThread} strategy=linear-vec4 op={OpTag(op)}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        if (isMse)
            ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 predictions_ptr,");
        ptx.AppendLine("    .param .u64 targets_ptr,");
        ptx.AppendLine("    .param .u64 grad_input_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        if (!isMse)
            ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        // p0..p3, t0..t3, d0..d3, plus the hoisted scalar for MSE.
        ptx.AppendLine("    .reg .f32 %f<14>;");

        int paramCursor = 0;
        if (isMse)
            ptx.AppendLine($"    ld.param.u64 %rd{paramCursor++}, [grad_output_ptr];");
        int predParam = paramCursor++;
        int targetParam = paramCursor++;
        int gradInParam = paramCursor;
        ptx.AppendLine($"    ld.param.u64 %rd{predParam}, [predictions_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd{targetParam}, [targets_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd{gradInParam}, [grad_input_ptr];");

        if (isMse)
        {
            // gradOutput[0] is a broadcast scalar, so g * 2 is loop-invariant.
            // Hoisting it keeps the per-element association identical to
            // ((g * 2) * d) * invN in the established kernel.
            ptx.AppendLine("    ld.global.ca.f32 %f12, [%rd0];");
            ptx.AppendLine("    mul.rn.f32 %f12, %f12, 0f40000000;");   // * 2.0
        }

        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r2, {ElementsPerThread * sizeof(float)};");
        ptx.AppendLine($"    add.u64 %rd7, %rd{predParam}, %rd6;");
        ptx.AppendLine($"    add.u64 %rd8, %rd{targetParam}, %rd6;");
        ptx.AppendLine($"    add.u64 %rd9, %rd{gradInParam}, %rd6;");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f0, %f1, %f2, %f3}, [%rd7];");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f4, %f5, %f6, %f7}, [%rd8];");

        for (int i = 0; i < ElementsPerThread; i++)
        {
            int diff = 8 + i;
            ptx.AppendLine($"    sub.rn.f32 %f{diff}, %f{i}, %f{4 + i};");
            if (isMse)
            {
                string invLiteral = "0f" + PtxCompat.SingleToUInt32Bits(invN)
                    .ToString("X8", System.Globalization.CultureInfo.InvariantCulture);
                ptx.AppendLine($"    mul.rn.f32 %f{diff}, %f12, %f{diff};");
                ptx.AppendLine($"    mul.rn.f32 %f{diff}, %f{diff}, {invLiteral};");
            }
            else
            {
                // sign(d): both predicates false for NaN and for exact zero, so
                // the result is +0 - matching the established ternary chain.
                ptx.AppendLine($"    setp.gt.f32 %p1, %f{diff}, 0f00000000;");
                ptx.AppendLine($"    setp.lt.f32 %p2, %f{diff}, 0f00000000;");
                ptx.AppendLine($"    selp.f32 %f{diff}, 0f3F800000, 0f00000000, %p1;");
                ptx.AppendLine($"    selp.f32 %f{diff}, 0fBF800000, %f{diff}, %p2;");
            }
        }

        ptx.AppendLine("    st.global.v4.f32 [%rd9], {%f8, %f9, %f10, %f11};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string OpTag(DirectPtxLossBackwardOp op) => op switch
    {
        DirectPtxLossBackwardOp.MeanSquaredError => "mse-backward",
        DirectPtxLossBackwardOp.MeanAbsoluteError => "mae-gradient",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxLossBackwardOp op,
        int size,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(size);
        var scalarExtent = new DirectPtxExtent(1);
        var tensors = new List<DirectPtxTensorContract>();
        if (op == DirectPtxLossBackwardOp.MeanSquaredError)
            tensors.Add(new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                scalarExtent, scalarExtent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast));
        tensors.Add(new("predictions", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
            extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        tensors.Add(new("targets", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
            extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        tensors.Add(new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
            extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact));

        return new DirectPtxKernelBlueprint(
            Operation: $"loss-backward-{OpTag(op)}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"linear-vec4-b{blockThreads}-n{size}",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = op == DirectPtxLossBackwardOp.MeanSquaredError
                    ? "gradInput[i] = ((gradOutput[0] * 2) * (predictions[i] - targets[i])) * invN"
                    : "gradInput[i] = sign(predictions[i] - targets[i]); zero at zero and at NaN",
                ["mode"] = $"training-backward-{OpTag(op)}",
                ["elements-per-thread"] = ElementsPerThread.ToString(),
                ["global-input-reads"] = op == DirectPtxLossBackwardOp.MeanSquaredError
                    ? "one-vector-per-input-plus-one-broadcast-scalar"
                    : "one-vector-per-input",
                ["global-output-writes"] = "one-vector-per-thread",
                ["association-order"] = "matches the established NVRTC kernel term for term",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["division"] = op == DirectPtxLossBackwardOp.MeanSquaredError
                    ? "none-invN-baked-as-literal"
                    : "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int size) =>
        size is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int size) => false;

    internal static bool IsSupportedOp(DirectPtxLossBackwardOp op) =>
        op is DirectPtxLossBackwardOp.MeanSquaredError or DirectPtxLossBackwardOp.MeanAbsoluteError;

    private static void ValidateOp(DirectPtxLossBackwardOp op)
    {
        if (!IsSupportedOp(op))
            throw new ArgumentOutOfRangeException(nameof(op),
                "The loss-gradient family covers MeanSquaredError and MeanAbsoluteError.");
    }

    private static void Validate(int size)
    {
        if (!IsSupportedShape(size))
            throw new ArgumentOutOfRangeException(nameof(size),
                "The first loss-gradient family supports exact sizes " +
                "65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int size, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            size % (blockThreads * ElementsPerThread) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Loss-gradient block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    /// <summary>
    /// invN is baked into the module, so a non-finite value would silently
    /// poison every gradient the cache serves for that key.
    /// </summary>
    private static void ValidateInvN(DirectPtxLossBackwardOp op, float invN)
    {
        if (op != DirectPtxLossBackwardOp.MeanSquaredError)
            return;
        if (!PtxCompat.IsFinite(invN))
            throw new ArgumentOutOfRangeException(nameof(invN),
                "The MSE gradient scale must be finite.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
