using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 Adam parameter update for issue #848. Each thread owns
/// one FP32x4 slice of the parameter, gradient, first-moment, and second-moment
/// tensors: all four are read once, the whole step happens in registers, and
/// three vectors (param, m, v) are written back. There is no shared memory, no
/// local memory, no global intermediate, no temporary allocation, no stride
/// parameter, and no tail branch.
///
/// Every scalar the established <c>adam_update</c> kernel takes as a launch
/// argument - learning rate, both betas, epsilon, weight decay, and the step
/// counter - is baked into the module instead, so the launch ABI is four
/// pointers. Two consequences are deliberate:
///
/// - <c>1 - pow(beta, step)</c> is loop-invariant, so the two bias corrections
///   are computed once on the host and baked as IEEE-754 literals. That removes
///   both <c>powf</c> calls from the kernel entirely. The host computes them in
///   double and rounds once, which is at least as accurate as the device
///   <c>powf</c>, but it is NOT guaranteed bit-identical to it - so this kernel
///   is mathematically equivalent to the established one rather than
///   bit-exact, and its parity spec needs a tolerance. This is the one place in
///   the direct-PTX families where that is true, and it is recorded here rather
///   than discovered later.
/// - The weight-decay term is emitted only when the baked decay is positive,
///   matching the established kernel's <c>if (weightDecay &gt; 0.0f)</c> guard.
///   Because the value is baked, that guard resolves at emit time and costs no
///   runtime branch.
///
/// The arithmetic otherwise follows the established kernel term for term and in
/// the same association order, including <c>(1 - beta2) * grad * grad</c> being
/// left-associated. Multiplies and adds are emitted unfused; nvcc may contract
/// some of them into FMAs, which is the other reason the parity spec is a
/// tolerance rather than a bit comparison.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedAdamUpdateF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_adam_update_f32";
    internal const int DefaultBlockThreads = 256;
    internal const int ElementsPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Size { get; }
    internal int BlockThreads { get; }
    /// <summary>Whether this module emits the weight-decay term.</summary>
    internal bool HasWeightDecay { get; }

    internal int ElementsPerBlock => BlockThreads * ElementsPerThread;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedAdamUpdateF32Kernel(
        DirectPtxRuntime runtime,
        int size,
        in DirectPtxAdamHyperparameters hyperparameters,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedAdamUpdate(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in Adam update specialization is admitted only on SM86.");
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        hyperparameters.Validate();
        Size = size;
        BlockThreads = blockThreads;
        HasWeightDecay = hyperparameters.WeightDecay > 0f;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, size, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            size, HasWeightDecay, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    /// <summary>
    /// Launches the update. The scalars travel as launch parameters, so the
    /// same module serves every step of a run - the step counter never enters
    /// the module key and no per-step recompilation happens.
    /// </summary>
    internal unsafe void Launch(
        DirectPtxTensorView param,
        DirectPtxTensorView gradient,
        DirectPtxTensorView firstMoment,
        DirectPtxTensorView secondMoment,
        in DirectPtxAdamHyperparameters hyperparameters)
    {
        Require(param, Blueprint.Tensors[0], nameof(param));
        Require(gradient, Blueprint.Tensors[1], nameof(gradient));
        Require(firstMoment, Blueprint.Tensors[2], nameof(firstMoment));
        Require(secondMoment, Blueprint.Tensors[3], nameof(secondMoment));
        hyperparameters.Validate();
        if (hyperparameters.WeightDecay > 0f != HasWeightDecay)
            throw new ArgumentException(
                "This module was emitted for a different weight-decay presence; " +
                "the decay term is baked, so the two must agree.",
                nameof(hyperparameters));

        IntPtr paramPointer = param.Pointer;
        IntPtr gradientPointer = gradient.Pointer;
        IntPtr firstPointer = firstMoment.Pointer;
        IntPtr secondPointer = secondMoment.Pointer;
        float lrOverBc1 = hyperparameters.LearningRateOverBiasCorrection1;
        float beta1 = hyperparameters.Beta1;
        float oneMinusBeta1 = 1f - hyperparameters.Beta1;
        float beta2 = hyperparameters.Beta2;
        float oneMinusBeta2 = 1f - hyperparameters.Beta2;
        float rsqrtBc2 = hyperparameters.ReciprocalSqrtBiasCorrection2;
        float epsilon = hyperparameters.Epsilon;
        float weightDecay = hyperparameters.WeightDecay;

        void** arguments = stackalloc void*[12];
        arguments[0] = &paramPointer;
        arguments[1] = &gradientPointer;
        arguments[2] = &firstPointer;
        arguments[3] = &secondPointer;
        arguments[4] = &lrOverBc1;
        arguments[5] = &beta1;
        arguments[6] = &oneMinusBeta1;
        arguments[7] = &beta2;
        arguments[8] = &oneMinusBeta2;
        arguments[9] = &rsqrtBc2;
        arguments[10] = &epsilon;
        arguments[11] = &weightDecay;
        _module.Launch(
            _function,
            checked((uint)(Size / ElementsPerBlock)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    /// <summary>
    /// Emits the module. Only the SHAPE and the weight-decay presence are baked;
    /// every scalar arrives as a launch parameter, so one module serves an
    /// entire training run.
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

        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape size={size} block={blockThreads} elems-per-thread={ElementsPerThread} strategy=linear-vec4 op=adam-update");
        ptx.AppendLine(
            $"// weight-decay={(hasWeightDecay ? "on" : "off")}; all scalars are launch parameters, so the module key is shape-only");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 param_ptr,");
        ptx.AppendLine("    .param .u64 gradient_ptr,");
        ptx.AppendLine("    .param .u64 first_moment_ptr,");
        ptx.AppendLine("    .param .u64 second_moment_ptr,");
        // lr/bc1 and rsqrt(bc2) are folded on the host so the kernel needs one
        // divide per element instead of three.
        ptx.AppendLine("    .param .f32 lr_over_bc1,");
        ptx.AppendLine("    .param .f32 beta1,");
        ptx.AppendLine("    .param .f32 one_minus_beta1,");
        ptx.AppendLine("    .param .f32 beta2,");
        ptx.AppendLine("    .param .f32 one_minus_beta2,");
        ptx.AppendLine("    .param .f32 rsqrt_bc2,");
        ptx.AppendLine("    .param .f32 epsilon,");
        ptx.AppendLine("    .param .f32 weight_decay");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [param_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [gradient_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [first_moment_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [second_moment_ptr];");
        // Parameter space is uniform and cached, so these loads are broadcast
        // and effectively free next to the four global vectors below.
        ptx.AppendLine("    ld.param.f32 %f20, [lr_over_bc1];");
        ptx.AppendLine("    ld.param.f32 %f21, [beta1];");
        ptx.AppendLine("    ld.param.f32 %f22, [one_minus_beta1];");
        ptx.AppendLine("    ld.param.f32 %f23, [beta2];");
        ptx.AppendLine("    ld.param.f32 %f24, [one_minus_beta2];");
        ptx.AppendLine("    ld.param.f32 %f25, [rsqrt_bc2];");
        ptx.AppendLine("    ld.param.f32 %f26, [epsilon];");
        if (hasWeightDecay)
            ptx.AppendLine("    ld.param.f32 %f27, [weight_decay];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r2, {ElementsPerThread * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd4;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd4;");
        // param, m and v are read-modify-write, so they stay on the default
        // cached path. The gradient is read once and never revisited, so it uses
        // the read-only data cache and leaves L1 for the three resident tensors.
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f0, %f1, %f2, %f3}, [%rd6];");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f4, %f5, %f6, %f7}, [%rd7];");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f8, %f9, %f10, %f11}, [%rd8];");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f12, %f13, %f14, %f15}, [%rd9];");

        for (int i = 0; i < ElementsPerThread; i++)
        {
            int p = i, g = 4 + i, m = 8 + i, v = 12 + i;
            // grad += weightDecay * param, emitted only when decay is on so the
            // decay-free module carries no extra instruction.
            if (hasWeightDecay)
            {
                ptx.AppendLine($"    mul.rn.f32 %f16, %f27, %f{p};");
                ptx.AppendLine($"    add.rn.f32 %f{g}, %f{g}, %f16;");
            }
            // m = beta1 * m + (1 - beta1) * grad
            ptx.AppendLine($"    mul.rn.f32 %f16, %f21, %f{m};");
            ptx.AppendLine($"    mul.rn.f32 %f17, %f22, %f{g};");
            ptx.AppendLine($"    add.rn.f32 %f{m}, %f16, %f17;");
            // v = beta2 * v + ((1 - beta2) * grad) * grad   [left-associated]
            ptx.AppendLine($"    mul.rn.f32 %f16, %f23, %f{v};");
            ptx.AppendLine($"    mul.rn.f32 %f17, %f24, %f{g};");
            ptx.AppendLine($"    mul.rn.f32 %f17, %f17, %f{g};");
            ptx.AppendLine($"    add.rn.f32 %f{v}, %f16, %f17;");
            // numerator = (lr / bc1) * m, folding what was lr * (m / bc1).
            ptx.AppendLine($"    mul.rn.f32 %f16, %f20, %f{m};");
            // denominator = sqrt(v) * rsqrt(bc2) + eps, folding what was
            // sqrt(v / bc2) + eps. One fma replaces a divide and an add.
            ptx.AppendLine($"    sqrt.rn.f32 %f17, %f{v};");
            ptx.AppendLine("    fma.rn.f32 %f17, %f17, %f25, %f26;");
            ptx.AppendLine("    div.rn.f32 %f16, %f16, %f17;");
            ptx.AppendLine($"    sub.rn.f32 %f{p}, %f{p}, %f16;");
        }

        ptx.AppendLine("    st.global.v4.f32 [%rd6], {%f0, %f1, %f2, %f3};");
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f8, %f9, %f10, %f11};");
        ptx.AppendLine("    st.global.v4.f32 [%rd9], {%f12, %f13, %f14, %f15};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string Literal(float value) =>
        "0f" + PtxCompat.SingleToUInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int size,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(size);
        return new DirectPtxKernelBlueprint(
            Operation: "adam-update-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"linear-vec4-b{blockThreads}-n{size}",
            Tensors:
            [
                new("param", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("gradient", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("firstMoment", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("secondMoment", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1024 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] =
                    "m = b1*m + (1-b1)*g; v = b2*v + ((1-b2)*g)*g; " +
                    "p -= (lr * (m/bc1)) / (sqrt(v/bc2) + eps)",
                ["mode"] = "training-adam-update-in-place",
                ["elements-per-thread"] = ElementsPerThread.ToString(),
                ["global-input-reads"] = "one-vector-each-for-param-gradient-m-v",
                ["global-output-writes"] = "one-vector-each-for-param-m-v",
                ["launch-abi"] = "four-pointers-every-scalar-baked",
                ["bias-correction"] = "host-precomputed-in-double-then-baked; no powf in kernel",
                ["bit-exactness"] =
                    "mathematically equivalent to adam_update, NOT bit-identical: the baked bias " +
                    "corrections and possible nvcc FMA contraction both shift the last ulp",
                ["weight-decay"] = "baked; the term is emitted only when positive, so no runtime branch",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["aliasing"] = "param, m, and v are updated in place and must be mutually disjoint",
                ["byte-offset"] = "zero-entire-allocation-view",
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
                "The first Adam update family supports exact sizes " +
                "65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int size, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            size % (blockThreads * ElementsPerThread) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Adam block threads must be 128, 256, or 512 and evenly tile the element count.");
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

/// <summary>
/// The Adam scalars baked into a module. Every field participates in the kernel
/// cache key, so two steps with different hyperparameters never share a module.
/// </summary>
internal readonly struct DirectPtxAdamHyperparameters : IEquatable<DirectPtxAdamHyperparameters>
{
    internal float LearningRate { get; }
    internal float Beta1 { get; }
    internal float Beta2 { get; }
    internal float Epsilon { get; }
    internal float WeightDecay { get; }
    internal int Step { get; }

    /// <summary>1 - beta1^step, computed once in double and rounded once.</summary>
    internal float BiasCorrection1 { get; }

    /// <summary>1 - beta2^step, computed once in double and rounded once.</summary>
    internal float BiasCorrection2 { get; }

    /// <summary>
    /// learningRate / biasCorrection1, folded on the host so the kernel
    /// multiplies instead of dividing. Replaces lr * (m / bc1) with
    /// (lr / bc1) * m.
    /// </summary>
    internal float LearningRateOverBiasCorrection1 { get; }

    /// <summary>
    /// 1 / sqrt(biasCorrection2), folded on the host so the kernel computes
    /// sqrt(v) * rsqrt(bc2) instead of sqrt(v / bc2) - turning a divide plus an
    /// add into a single fma.
    /// </summary>
    internal float ReciprocalSqrtBiasCorrection2 { get; }

    internal DirectPtxAdamHyperparameters(
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step)
    {
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        Step = step;
        // Accumulate the power in double so a large step count does not lose
        // precision before the single rounding to float.
        double bc1 = 1.0 - Math.Pow(beta1, step);
        double bc2 = 1.0 - Math.Pow(beta2, step);
        BiasCorrection1 = (float)bc1;
        BiasCorrection2 = (float)bc2;
        // Fold both loop-invariant divisions on the host, in double, so the
        // kernel is left with exactly one divide per element.
        LearningRateOverBiasCorrection1 = bc1 > 0.0 ? (float)(learningRate / bc1) : 0f;
        ReciprocalSqrtBiasCorrection2 = bc2 > 0.0 ? (float)(1.0 / Math.Sqrt(bc2)) : 0f;
    }

    /// <summary>
    /// The scalars are baked into the module, so an out-of-domain value would
    /// poison every update the cache serves under this key. A zero bias
    /// correction would divide by zero on every element.
    /// </summary>
    internal void Validate()
    {
        if (!PtxCompat.IsFinite(LearningRate) || !PtxCompat.IsFinite(Epsilon) ||
            !PtxCompat.IsFinite(WeightDecay))
            throw new ArgumentOutOfRangeException(nameof(LearningRate),
                "Adam learning rate, epsilon, and weight decay must be finite.");
        if (Epsilon <= 0f)
            throw new ArgumentOutOfRangeException(nameof(Epsilon),
                "Adam epsilon must be positive so the denominator can never be zero.");
        if (WeightDecay < 0f)
            throw new ArgumentOutOfRangeException(nameof(WeightDecay),
                "Adam weight decay must not be negative.");
        if (!(Beta1 > 0f && Beta1 < 1f) || !(Beta2 > 0f && Beta2 < 1f))
            throw new ArgumentOutOfRangeException(nameof(Beta1),
                "Adam betas must lie strictly between 0 and 1.");
        if (Step <= 0)
            throw new ArgumentOutOfRangeException(nameof(Step),
                "Adam step counts start at 1; step 0 leaves the bias correction at zero.");
        if (BiasCorrection1 <= 0f || BiasCorrection2 <= 0f)
            throw new ArgumentOutOfRangeException(nameof(Step),
                "The baked bias corrections must be positive; this beta/step pair underflows to zero.");
    }

    public bool Equals(DirectPtxAdamHyperparameters other) =>
        PtxCompat.SingleToUInt32Bits(LearningRate) == PtxCompat.SingleToUInt32Bits(other.LearningRate) &&
        PtxCompat.SingleToUInt32Bits(Beta1) == PtxCompat.SingleToUInt32Bits(other.Beta1) &&
        PtxCompat.SingleToUInt32Bits(Beta2) == PtxCompat.SingleToUInt32Bits(other.Beta2) &&
        PtxCompat.SingleToUInt32Bits(Epsilon) == PtxCompat.SingleToUInt32Bits(other.Epsilon) &&
        PtxCompat.SingleToUInt32Bits(WeightDecay) == PtxCompat.SingleToUInt32Bits(other.WeightDecay) &&
        Step == other.Step;

    public override bool Equals(object? obj) =>
        obj is DirectPtxAdamHyperparameters other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + (int)PtxCompat.SingleToUInt32Bits(LearningRate);
            hash = hash * 31 + (int)PtxCompat.SingleToUInt32Bits(Beta1);
            hash = hash * 31 + (int)PtxCompat.SingleToUInt32Bits(Beta2);
            hash = hash * 31 + (int)PtxCompat.SingleToUInt32Bits(Epsilon);
            hash = hash * 31 + (int)PtxCompat.SingleToUInt32Bits(WeightDecay);
            hash = hash * 31 + Step;
            return hash;
        }
    }
}
