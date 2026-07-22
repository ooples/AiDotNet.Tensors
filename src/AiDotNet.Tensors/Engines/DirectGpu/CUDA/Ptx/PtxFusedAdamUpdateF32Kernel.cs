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
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, size, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            size, hyperparameters, blockThreads);
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
        DirectPtxTensorView firstMoment,
        DirectPtxTensorView secondMoment)
    {
        Require(param, Blueprint.Tensors[0], nameof(param));
        Require(gradient, Blueprint.Tensors[1], nameof(gradient));
        Require(firstMoment, Blueprint.Tensors[2], nameof(firstMoment));
        Require(secondMoment, Blueprint.Tensors[3], nameof(secondMoment));

        IntPtr paramPointer = param.Pointer;
        IntPtr gradientPointer = gradient.Pointer;
        IntPtr firstPointer = firstMoment.Pointer;
        IntPtr secondPointer = secondMoment.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &paramPointer;
        arguments[1] = &gradientPointer;
        arguments[2] = &firstPointer;
        arguments[3] = &secondPointer;
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
        int size,
        in DirectPtxAdamHyperparameters hyperparameters,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        hyperparameters.Validate();

        string lr = Literal(hyperparameters.LearningRate);
        string beta1 = Literal(hyperparameters.Beta1);
        string beta2 = Literal(hyperparameters.Beta2);
        string oneMinusBeta1 = Literal(1f - hyperparameters.Beta1);
        string oneMinusBeta2 = Literal(1f - hyperparameters.Beta2);
        string epsilon = Literal(hyperparameters.Epsilon);
        string weightDecay = Literal(hyperparameters.WeightDecay);
        string biasCorrection1 = Literal(hyperparameters.BiasCorrection1);
        string biasCorrection2 = Literal(hyperparameters.BiasCorrection2);
        bool hasWeightDecay = hyperparameters.WeightDecay > 0f;

        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape size={size} block={blockThreads} elems-per-thread={ElementsPerThread} strategy=linear-vec4 op=adam-update");
        ptx.AppendLine(
            $"// baked step={hyperparameters.Step} bias-corrections precomputed on host; no powf in kernel");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 param_ptr,");
        ptx.AppendLine("    .param .u64 gradient_ptr,");
        ptx.AppendLine("    .param .u64 first_moment_ptr,");
        ptx.AppendLine("    .param .u64 second_moment_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [param_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [gradient_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [first_moment_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [second_moment_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r2, {ElementsPerThread * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd4;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd4;");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f0, %f1, %f2, %f3}, [%rd6];");    // param
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f4, %f5, %f6, %f7}, [%rd7];");    // gradient
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f8, %f9, %f10, %f11}, [%rd8];");  // m
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f12, %f13, %f14, %f15}, [%rd9];");// v

        for (int i = 0; i < ElementsPerThread; i++)
        {
            int p = i, g = 4 + i, m = 8 + i, v = 12 + i;
            // grad += weightDecay * param  (emitted only when the baked decay is
            // positive, matching the established kernel's guard).
            if (hasWeightDecay)
            {
                ptx.AppendLine($"    mul.rn.f32 %f16, {weightDecay}, %f{p};");
                ptx.AppendLine($"    add.rn.f32 %f{g}, %f{g}, %f16;");
            }
            // m = beta1 * m + (1 - beta1) * grad
            ptx.AppendLine($"    mul.rn.f32 %f16, {beta1}, %f{m};");
            ptx.AppendLine($"    mul.rn.f32 %f17, {oneMinusBeta1}, %f{g};");
            ptx.AppendLine($"    add.rn.f32 %f{m}, %f16, %f17;");
            // v = beta2 * v + ((1 - beta2) * grad) * grad   [left-associated]
            ptx.AppendLine($"    mul.rn.f32 %f16, {beta2}, %f{v};");
            ptx.AppendLine($"    mul.rn.f32 %f17, {oneMinusBeta2}, %f{g};");
            ptx.AppendLine($"    mul.rn.f32 %f17, %f17, %f{g};");
            ptx.AppendLine($"    add.rn.f32 %f{v}, %f16, %f17;");
            // mHat = m / bc1 ; vHat = v / bc2
            ptx.AppendLine($"    div.rn.f32 %f16, %f{m}, {biasCorrection1};");
            ptx.AppendLine($"    div.rn.f32 %f17, %f{v}, {biasCorrection2};");
            // param -= (lr * mHat) / (sqrt(vHat) + eps)
            ptx.AppendLine("    sqrt.rn.f32 %f17, %f17;");
            ptx.AppendLine($"    add.rn.f32 %f17, %f17, {epsilon};");
            ptx.AppendLine($"    mul.rn.f32 %f16, {lr}, %f16;");
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
        BiasCorrection1 = (float)(1.0 - Math.Pow(beta1, step));
        BiasCorrection2 = (float)(1.0 - Math.Pow(beta2, step));
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
