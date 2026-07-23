using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 fused SGD-with-momentum update step (issue #848):
/// <code>
///   g' = grad + weightDecay * param
///   v  = momentum * v + g'
///   param = param - learningRate * v
/// </code>
/// applied elementwise. Each thread loads one FP32x4 vector of param, grad, and
/// velocity, applies the three fused-multiply-adds entirely in registers, and
/// commits the updated velocity and param — so the intermediate decayed
/// gradient and the velocity update are <b>never materialized</b> to global
/// memory (the naive path is two or three separate elementwise kernels, each a
/// full global round-trip). The learning rate, momentum, and weight decay are
/// module identity (baked immediates), exactly like the residual-RMSNorm
/// kernel's epsilon; the pointer-only launch ABI carries no scalar parameters.
/// There are no shared-memory, local-memory, global-intermediate, temporary-
/// allocation, division, remainder, or stride parameters. Disabled by default;
/// fails closed until three clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedSgdMomentumF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_sgd_momentum_f32";
    internal const int DefaultBlockThreads = 256;
    /// <summary>Whether this module emits the weight-decay term.</summary>
    internal bool HasWeightDecay { get; }

    internal const int ElementsPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Size { get; }
    internal float LearningRate { get; }
    internal float Momentum { get; }
    internal float WeightDecay { get; }
    internal int BlockThreads { get; }
    internal int ElementsPerBlock => BlockThreads * ElementsPerThread;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedSgdMomentumF32Kernel(
        DirectPtxRuntime runtime,
        int size,
        float learningRate,
        float momentum,
        float weightDecay,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSgdMomentum(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 SGD-momentum specialization is admitted only on SM86.");
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        ValidateHyperparameters(learningRate, momentum, weightDecay);
        Size = size;
        LearningRate = learningRate;
        Momentum = momentum;
        WeightDecay = weightDecay;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, size, learningRate, momentum, weightDecay, blockThreads);
        HasWeightDecay = weightDecay != 0f;
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            size, HasWeightDecay, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView param,
        DirectPtxTensorView gradient,
        DirectPtxTensorView velocity)
    {
        Require(param, Blueprint.Tensors[0], nameof(param));
        Require(gradient, Blueprint.Tensors[1], nameof(gradient));
        Require(velocity, Blueprint.Tensors[2], nameof(velocity));
        if (Overlaps(param, gradient) || Overlaps(param, velocity) || Overlaps(gradient, velocity))
            throw new ArgumentException("The SGD-momentum specialization does not admit aliasing.");

        IntPtr paramPointer = param.Pointer;
        IntPtr gradientPointer = gradient.Pointer;
        IntPtr velocityPointer = velocity.Pointer;
        // Negated once here so the parameter update stays a single fma per
        // element rather than a multiply followed by a subtract.
        float negLearningRate = -LearningRate;
        float momentum = Momentum;
        float weightDecay = WeightDecay;
        void** arguments = stackalloc void*[6];
        arguments[0] = &paramPointer;
        arguments[1] = &gradientPointer;
        arguments[2] = &velocityPointer;
        arguments[3] = &negLearningRate;
        arguments[4] = &momentum;
        arguments[5] = &weightDecay;
        _module.Launch(
            _function,
            checked((uint)(Size / ElementsPerBlock)),
            1, 1,
            checked((uint)BlockThreads),
            1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string HexFloat(float value) =>
        "0f" + PtxCompat.SingleToUInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    /// <summary>
    /// Emits the module. Only the SHAPE and the weight-decay presence are baked;
    /// the learning rate, momentum, and decay arrive as launch parameters, so
    /// one module serves an entire run and the key space stays finite.
    /// </summary>
    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int size,
        bool hasWeightDecay,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        var ptx = new StringBuilder(3_072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape size={size} block={blockThreads} elems-per-thread={ElementsPerThread} strategy=linear-vec4 op=sgd-momentum wd={(hasWeightDecay ? 1 : 0)}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 param_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 vel_ptr,");
        // The host negates the learning rate once so the update stays a single
        // fma per element rather than a multiply and a subtract.
        ptx.AppendLine("    .param .f32 neg_learning_rate,");
        ptx.AppendLine("    .param .f32 momentum,");
        ptx.AppendLine("    .param .f32 weight_decay");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<9>;");
        ptx.AppendLine("    .reg .f32 %f<15>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [param_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [vel_ptr];");
        ptx.AppendLine("    ld.param.f32 %f12, [neg_learning_rate];");
        ptx.AppendLine("    ld.param.f32 %f13, [momentum];");
        if (hasWeightDecay)
            ptx.AppendLine("    ld.param.f32 %f14, [weight_decay];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {ElementsPerThread * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        // f0-3 = param, f4-7 = grad, f8-11 = velocity.
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f0, %f1, %f2, %f3}, [%rd4];");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f4, %f5, %f6, %f7}, [%rd5];");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f8, %f9, %f10, %f11}, [%rd6];");
        for (int i = 0; i < ElementsPerThread; i++)
        {
            int p = i, g = 4 + i, v = 8 + i;
            if (hasWeightDecay)
                ptx.AppendLine($"    fma.rn.f32 %f{g}, %f{p}, %f14, %f{g};");   // g = wd*param + grad
            ptx.AppendLine($"    fma.rn.f32 %f{v}, %f{v}, %f13, %f{g};");       // v = momentum*v + g
            ptx.AppendLine($"    fma.rn.f32 %f{p}, %f{v}, %f12, %f{p};");       // param = -lr*v + param
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd6], {%f8, %f9, %f10, %f11};");  // velocity out
        ptx.AppendLine("    st.global.v4.f32 [%rd4], {%f0, %f1, %f2, %f3};");    // param out
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int size,
        float learningRate,
        float momentum,
        float weightDecay,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(size);
        return new DirectPtxKernelBlueprint(
            Operation: "sgd-momentum-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"linear-vec4-b{blockThreads}-n{size}-lr{PtxCompat.SingleToUInt32Bits(learningRate):X8}-m{PtxCompat.SingleToUInt32Bits(momentum):X8}-wd{PtxCompat.SingleToUInt32Bits(weightDecay):X8}",
            Tensors:
            [
                new("param", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("gradient", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("velocity", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // Measured by the offline gate at sm86: 22 registers without
                // weight decay and 24 with it, so the previous budget of 24
                // sat exactly on the limit and any codegen drift would have
                // tripped ResourceBudget.Validate on a real device.
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "g'=grad+wd*param; v=momentum*v+g'; param-=lr*v",
                ["mode"] = "inference-forward-fused-optimizer-step",
                ["learning-rate"] = learningRate.ToString("R", CultureInfo.InvariantCulture),
                ["momentum"] = momentum.ToString("R", CultureInfo.InvariantCulture),
                ["weight-decay"] = weightDecay.ToString("R", CultureInfo.InvariantCulture),
                ["input"] = "fp32",
                ["output"] = "fp32-param-and-velocity-in-place",
                ["elements-per-thread"] = ElementsPerThread.ToString(),
                ["global-input-reads"] = "one-param-one-grad-one-velocity-vector-per-thread",
                ["global-output-writes"] = "one-param-one-velocity-vector-per-thread",
                ["lane-vector-transaction"] = "aligned-fp32x4",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none-no-materialized-decayed-grad-or-velocity-temp",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["alias-policy"] = "param-grad-velocity-disjoint",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int size) =>
        size is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int size) => false;

    private static void Validate(int size)
    {
        if (!IsSupportedShape(size))
            throw new ArgumentOutOfRangeException(nameof(size),
                "The first SGD-momentum family supports exact sizes 65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int size, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            size % (blockThreads * ElementsPerThread) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "SGD-momentum block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    private static void ValidateHyperparameters(float learningRate, float momentum, float weightDecay)
    {
        if (!PtxCompat.IsFinite(learningRate) || !PtxCompat.IsFinite(momentum) ||
            !PtxCompat.IsFinite(weightDecay))
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "SGD-momentum hyperparameters must be finite.");
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

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
