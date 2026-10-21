// Copyright (c) AiDotNet. All rights reserved.
// Declared programs as resident-program stages (#890 phase 4 on phase 3's resident program, #1430): a chain of
// declared calls -- LayerNorm, then Softmax, then ... -- runs as ONE persistent cooperative launch composed from the
// stages' own kernels (PtxResidentComposer), a grid barrier between them. Every stage is the same kernel the
// declared route launches on its own, so the chain's results are bit-identical to running the stages separately;
// what residency removes is the launches.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>One declared call as a resident stage: the program for <see cref="Key"/> and the caller's buffers.</summary>
internal readonly struct DeclaredResidentStage
{
    internal DeclaredResidentStage(DeclaredKey key, Func<DeclaredKey, CodegenProgram> build, string[] names, DeclaredBindings bound,
        Func<CodegenKernelSpec, PtxPassLowering?>? lowerPass = null)
    {
        LowerPass = lowerPass;
        Key = key;
        Build = build ?? throw new ArgumentNullException(nameof(build));
        Names = names ?? throw new ArgumentNullException(nameof(names));
        Bound = bound;
    }

    internal DeclaredKey Key { get; }
    internal Func<DeclaredKey, CodegenProgram> Build { get; }
    internal string[] Names { get; }
    internal DeclaredBindings Bound { get; }

    /// <summary>The family's own pass lowering, as its routed call uses it: one key is one program however it is first built.</summary>
    internal Func<CodegenKernelSpec, PtxPassLowering?>? LowerPass { get; }
}

public sealed partial class CudaBackend
{
    private sealed class DeclaredResident : IDisposable
    {
        internal DeclaredResident(DirectPtxModule module, IntPtr function, PtxResidentComposition composition,
            (int Stage, int Slot)[] argumentSources, int grid)
        {
            Module = module;
            Function = function;
            Composition = composition;
            ArgumentSources = argumentSources;
            Grid = grid;
        }

        internal DirectPtxModule Module { get; }
        internal IntPtr Function { get; }
        internal PtxResidentComposition Composition { get; }

        /// <summary>Per argument: the stage and its caller slot (>= 0) or scratch (-1 - index); stage -1 is the barrier.</summary>
        internal (int Stage, int Slot)[] ArgumentSources { get; }

        internal int Grid { get; }

        public void Dispose() => Module.Dispose();
    }

    // Composed chains by their stages' keys, in order; a null entry records a chain that cannot be composed.
    private readonly Dictionary<string, DeclaredResident?> _declaredResidents = new(StringComparer.Ordinal);

    /// <summary>LayerNorm as a resident stage, in CudaBackend.LayerNorm's contract.</summary>
    internal static DeclaredResidentStage LayerNormStage(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon) =>
        new(LayerNormKey(batchSize, normalizedSize, epsilon), s_buildLayerNorm, s_layerNormNames,
            new DeclaredBindings(input, gamma, beta, output, saveMean, saveInvVar));

    /// <summary>RmsNorm as a resident stage, in CudaBackend.RmsNorm's contract.</summary>
    internal static DeclaredResidentStage RmsNormStage(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon) =>
        new(RmsNormKey(batchSize, normalizedSize, epsilon), s_buildRmsNorm, s_rmsNormNames,
            new DeclaredBindings(input, gamma, output, saveRms));

    /// <summary>GroupNorm as a resident stage, in CudaBackend.GroupNorm's contract.</summary>
    internal static DeclaredResidentStage GroupNormStage(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon) =>
        new(GroupNormKey(batch, channels, numGroups, spatialSize, epsilon), s_buildGroupNorm, s_groupNormNames,
            new DeclaredBindings(input, gamma, beta, output, saveMean, saveInvVar));

    /// <summary>Training BatchNorm as a resident stage, in CudaBackend.BatchNorm's contract.</summary>
    internal static DeclaredResidentStage BatchNormTrainingStage(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum) =>
        new(BatchNormKey(batch, channels, spatialSize, epsilon, momentum), s_buildBatchNorm, s_batchNormNames,
            new DeclaredBindings(input, gamma, beta, runningMean, runningVar, output, saveMean, saveInvVar, runningMean, runningVar));

    /// <summary>LayerNorm backward as a resident stage, in CudaBackend.LayerNormBackward's contract.</summary>
    internal static DeclaredResidentStage LayerNormBackwardStage(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize) =>
        new(LayerNormBackwardKey(batchSize, normalizedSize), s_buildLayerNormBackward, s_normBackwardNames,
            new DeclaredBindings(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta));

    /// <summary>RmsNorm backward as a resident stage, in CudaBackend.RmsNormBackward's contract.</summary>
    internal static DeclaredResidentStage RmsNormBackwardStage(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveRms, IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize) =>
        new(RmsNormBackwardKey(batchSize, normalizedSize), s_buildRmsNormBackward, s_rmsNormBackwardNames,
            new DeclaredBindings(gradOutput, input, gamma, saveRms, gradInput, gradGamma));

    /// <summary>BatchNorm backward as a resident stage, in CudaBackend.BatchNormBackward's contract.</summary>
    internal static DeclaredResidentStage BatchNormBackwardStage(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize) =>
        new(BatchNormBackwardKey(batch, channels, spatialSize), s_buildBatchNormBackward, s_normBackwardNames,
            new DeclaredBindings(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta));

    /// <summary>
    /// A derived elementwise kernel (relu, gelu, sinf, leaky_relu, ...) as a resident stage over <paramref name="length"/>
    /// elements: the same library function the routed call applies, with any scalars fixed for the stage.
    /// </summary>
    internal DeclaredResidentStage ElementwiseStage(string kernel, IGpuBuffer output, int length, IGpuBuffer input0,
        IGpuBuffer? input1 = null, IGpuBuffer? input2 = null, params float[] scalars)
    {
        var function = ResolveElementwise(kernel) ?? throw new NotSupportedException(ElementwiseRefusal(kernel));
        int inputs = input2 is not null ? 3 : input1 is not null ? 2 : 1;
        if (function.Inputs != inputs || function.Scalars.Length != scalars.Length)
            throw new ArgumentException(kernel + " takes " + function.Inputs + " inputs and " + function.Scalars.Length + " scalars.");
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        var name = new System.Text.StringBuilder(kernel);
        foreach (float s in scalars) name.Append(':').Append(s.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        var names = new string[inputs + 1];
        for (int i = 0; i < inputs; i++) names[i] = "in" + i.ToString(System.Globalization.CultureInfo.InvariantCulture);
        names[inputs] = "out";
        var fixedScalars = (float[])scalars.Clone();
        return new DeclaredResidentStage(new DeclaredKey(DeclaredFamily.Elementwise, length, 0, 0, 0, 0, 0, name.ToString()),
            key =>
            {
                var arguments = new CodegenExpr[function.Inputs + fixedScalars.Length];
                var bindings = new CodegenTensorBinding[function.Inputs];
                for (int i = 0; i < function.Inputs; i++)
                {
                    arguments[i] = CodegenExpr.Operand(i);
                    bindings[i] = new CodegenTensorBinding(i, names[i], new[] { key.D0 }, new[] { CodegenAffineExpr.Axis(0) });
                }
                for (int s = 0; s < fixedScalars.Length; s++) arguments[function.Inputs + s] = CodegenExpr.Constant(fixedScalars[s]);
                var body = CodegenExpr.Call(function.Handle, arguments);
                var products = body.Operands();
                var spec = new CodegenKernelSpec("elementwise_" + kernel, new CodegenIterationSpace(CodegenAxis.Parallel("i", key.D0)),
                    bindings, new CodegenTensorBinding(function.Inputs, "out", new[] { key.D0 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true),
                    products, CodegenReduceKind.None, body: body);
                return new CodegenProgram(new[] { spec }, new long[] { key.D0 }, "elementwise " + kernel + " " + key.D0);
            },
            names, new DeclaredBindings(input0, input1 ?? output, input2 ?? output, output, output));
    }

    /// <summary>Row softmax as a resident stage, in CudaBackend.Softmax's contract.</summary>
    internal static DeclaredResidentStage SoftmaxStage(IGpuBuffer input, IGpuBuffer output, int batchSize, int features) =>
        new(SoftmaxKey(batchSize, features), s_buildSoftmax, s_softmaxNames, new DeclaredBindings(input, output));

    /// <summary>Row softmax backward as a resident stage, in CudaBackend.SoftmaxBackward's contract.</summary>
    internal static DeclaredResidentStage SoftmaxBackwardStage(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput,
        int batchSize, int features) =>
        new(SoftmaxBackwardKey(batchSize, features), s_buildSoftmaxBackward, s_softmaxBackwardNames,
            new DeclaredBindings(gradOutput, output, gradInput));

    /// <summary>
    /// Runs <paramref name="stages"/> in order as ONE cooperative launch of their composed kernels, a grid barrier
    /// between kernels. Returns the grid it used. Throws NotSupportedException when the chain cannot be composed (a
    /// stage that declines, mixed block sizes, a device without cooperative launch); nothing is launched then.
    /// </summary>
    internal unsafe int RunDeclaredResidentProgram(ReadOnlySpan<DeclaredResidentStage> stages, IGpuBuffer barrier)
    {
        if (stages.Length == 0) throw new ArgumentException("A resident program needs at least one stage.", nameof(stages));
        if (barrier is null || barrier.Size < 2)
            throw new ArgumentException("The barrier must be the two-word buffer from AllocateResidentBarrier.", nameof(barrier));
        if (!SupportsResidentProgram || !DirectPtxArchitecture.HasCodegenConveyor(_ccMajor, _ccMinor))
            throw new NotSupportedException("A declared resident program needs a cooperative launch and the sm_75+ codegen conveyor.");

        var programs = new DeclaredProgram[stages.Length];
        var keys = new string[stages.Length];
        DeclaredResident? resident;
        lock (_directPtxLock)
        {
            for (int i = 0; i < stages.Length; i++)
            {
                var stage = stages[i];
                if (!_declaredPrograms.TryGetValue(stage.Key, out var program))
                {
                    if (IsStreamCapturing()) throw new NotSupportedException("A declared resident program is built outside capture.");
                    program = BuildDeclaredProgram(stage.Key, stage.Build, stage.Names, stage.LowerPass);
                    _declaredPrograms[stage.Key] = program;
                }
                programs[i] = program ?? throw new NotSupportedException(
                    "Stage " + i + " (" + stage.Key + ") has no declared lowering: " + DeclaredDeclineReasonUnlocked(stage.Key));
                keys[i] = stage.Key.ToString();
            }
            string chain = string.Join("|", keys);
            if (!_declaredResidents.TryGetValue(chain, out resident))
            {
                if (IsStreamCapturing()) throw new NotSupportedException("A declared resident program is built outside capture.");
                resident = BuildDeclaredResident(chain, programs);
                _declaredResidents[chain] = resident;
            }
        }
        if (resident is null) throw new NotSupportedException("The chain cannot be composed into one resident kernel: " + LastResidentDecline);

        for (int i = 0; i < stages.Length; i++)
        {
            var names = stages[i].Names;
            for (int k = 0; k < names.Length; k++)
            {
                long needed = programs[i].BoundElements[k];
                if (needed != 0 && stages[i].Bound[k].Size < needed)
                    throw new ArgumentException("Stage " + i + ": '" + names[k] + "' holds " + stages[i].Bound[k].Size +
                        " elements; the program needs " + needed + ".");
            }
        }

        using var _ = PushContext();
        var sources = resident.ArgumentSources;
        IntPtr* handles = stackalloc IntPtr[sources.Length];
        void** argv = stackalloc void*[sources.Length];
        for (int a = 0; a < sources.Length; a++)
        {
            var (stage, slot) = sources[a];
            handles[a] = stage < 0 ? barrier.Handle
                : slot >= 0 ? stages[stage].Bound[slot].Handle
                : programs[stage].Scratch[-1 - slot].Handle;
            argv[a] = &handles[a];
        }
        lock (resident) LaunchCooperativeKernel(resident.Function, (uint)resident.Grid, resident.Composition.BlockX, 0, argv);
        System.Threading.Interlocked.Increment(ref _declaredProgramLaunches);
        return resident.Grid;
    }

    /// <summary>The same stages, each through its declared program on its own (what residency is measured against).</summary>
    internal void RunDeclaredStagesSeparately(ReadOnlySpan<DeclaredResidentStage> stages)
    {
        foreach (var stage in stages)
            if (!TryRunDeclaredProgram(stage.Key, stage.Build, stage.Names, stage.Bound, stage.LowerPass))
                throw new NotSupportedException("Stage " + stage.Key + " did not run through its declared program.");
    }

    private DeclaredResident? BuildDeclaredResident(string chain, DeclaredProgram[] programs)
    {
        var steps = new List<PtxProgramStep>();
        var owners = new List<(int Stage, int Step)>();
        for (int i = 0; i < programs.Length; i++)
            for (int s = 0; s < programs[i].Lowering.Steps.Count; s++)
            {
                steps.Add(programs[i].Lowering.Steps[s]);
                owners.Add((i, s));
            }
        PtxResidentComposition composition;
        try
        {
            composition = PtxResidentComposer.Compose(steps, "resident_declared", _ccMajor, _ccMinor);
        }
        catch (NotSupportedException ex)
        {
            LastResidentDecline = chain + ": " + ex.Message;
            return null;
        }

        // Each composed argument comes from the step that declared it: that stage's caller slot or scratch.
        var sources = new (int Stage, int Slot)[composition.Arguments.Count];
        int at = 0;
        for (int t = 0; t < steps.Count; t++)
        {
            var (stage, step) = owners[t];
            int[] slots = programs[stage].ArgumentSlots[step];
            for (int a = 0; a < slots.Length; a++) sources[at++] = (stage, slots[a]);
        }
        sources[at] = (-1, 0);   // the barrier

        _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
        DirectPtxModule module;
        try
        {
            module = composition.Calls.Count == 0
                ? _directPtxRuntime.LoadModule(composition.Ptx, allowExperimentalJitFallback: true)
                : _directPtxRuntime.LoadLinkedModule(composition.Ptx, LibrariesFor(composition.Calls));
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            LastResidentDecline = chain + ": " + ex.Message;
            return null;
        }
        try
        {
            IntPtr function = module.GetFunction(composition.EntryPoint, out _);
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out int perSm, function,
                    (int)composition.BlockX, 0, 0),
                "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(resident_declared)");
            int grid = Math.Max(1, (int)Math.Min((long)perSm * _multiProcessorCount, composition.MaxStageBlocks));
            return new DeclaredResident(module, function, composition, sources, grid);
        }
        catch (InvalidOperationException ex)
        {
            module.Dispose();
            LastResidentDecline = chain + ": " + ex.Message;
            return null;
        }
    }

    /// <summary>Why the last chain that could not be composed was refused, for diagnostics.</summary>
    internal string? LastResidentDecline { get; private set; }

    private string? DeclaredDeclineReasonUnlocked(DeclaredKey key) =>
        _declaredDeclines.TryGetValue(key, out string? reason) ? reason : null;

    private void DisposeDeclaredResidents()
    {
        foreach (var resident in _declaredResidents.Values) resident?.Dispose();
        _declaredResidents.Clear();
        foreach (var resident in _keyResidents.Values) resident?.Dispose();
        _keyResidents.Clear();
        _keyResidentBarrier?.Dispose();
        _keyResidentBarrier = null;
    }

    // ---- one program as one launch ---------------------------------------------------------------------------------

    // A program's composition as one cooperative kernel, by key; null records a key that does not compose.
    private readonly Dictionary<DeclaredKey, DeclaredResident?> _keyResidents = new();

    // The grid barrier the per-key residents share: launches on the backend's stream run one after another, and the
    // barrier is reusable launch after launch once zeroed.
    private IGpuBuffer? _keyResidentBarrier;

    /// <summary>
    /// Runs a built multi-step program as ONE cooperative launch -- its steps composed, a grid barrier between them --
    /// when every step is at most a wave of blocks: such a program is launch-bound, and residency replaces its launches
    /// with barriers (the results are the same kernels', so bit-identical). False, nothing launched, when it does not
    /// apply or does not compose. A call allocates nothing once the key's composition is built.
    /// </summary>
    private unsafe bool TryLaunchDeclaredResident(DeclaredKey key, DeclaredProgram program, in DeclaredBindings bound)
    {
        var steps = program.Lowering.Steps;
        if (steps.Count < 2 || !SupportsResidentProgram) return false;
        int wave = Math.Max(1, _multiProcessorCount);
        for (int s = 0; s < steps.Count; s++)
            if (steps[s].Blocks > (uint)wave) return false;

        DeclaredResident? resident;
        IGpuBuffer? barrier;
        lock (_directPtxLock)
        {
            if (!_keyResidents.TryGetValue(key, out resident))
            {
                if (IsStreamCapturing()) return false;
                resident = BuildDeclaredResident(key.ToString(), new[] { program });
                _keyResidents[key] = resident;
                if (resident is not null) _keyResidentBarrier ??= AllocateResidentBarrier();
            }
            barrier = _keyResidentBarrier;
        }
        if (resident is null || barrier is null) return false;

        using var _ = PushContext();
        var sources = resident.ArgumentSources;
        IntPtr* handles = stackalloc IntPtr[sources.Length];
        void** argv = stackalloc void*[sources.Length];
        for (int a = 0; a < sources.Length; a++)
        {
            var (stage, slot) = sources[a];
            handles[a] = stage < 0 ? barrier.Handle : slot >= 0 ? bound[slot].Handle : program.Scratch[-1 - slot].Handle;
            argv[a] = &handles[a];
        }
        lock (resident) LaunchCooperativeKernel(resident.Function, (uint)resident.Grid, resident.Composition.BlockX, 0, argv);
        System.Threading.Interlocked.Increment(ref _declaredProgramLaunches);
        return true;
    }
}
