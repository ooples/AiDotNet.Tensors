// Copyright (c) AiDotNet. All rights reserved.
// Routes backend calls to DECLARED programs (#890 phase 4): a normalisation call is served by the program lowered from
// its spec (PtxProgramLowering) instead of a hand-written kernel, in the hand-written kernel's own contract
// (CodegenBackendPrograms). A program is lowered once per shape and its modules and scratch are cached; a shape the
// declared path cannot lower declines, and the caller falls through to the hand-written kernel.
//
// A call allocates nothing: the shape is a value key, the caller's buffers travel in a fixed-size struct in the order
// of the family's tensor names, and each launch argument was resolved to a caller slot or a scratch buffer when the
// program was built. (A string key, a per-call name dictionary and a closure cost ~800 bytes a call, and a small
// LayerNorm is launch-bound, so host time is its time.)
//
// AIDOTNET_DECLARED_KERNELS=0 turns the route off for the process.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>The backend call a declared program serves.</summary>
internal enum DeclaredFamily
{
    LayerNorm,
    RmsNorm,
    GroupNorm,
    BatchNormTraining,
    LayerNormBackward,
    RmsNormBackward,
    BatchNormBackward,
    GroupNormBackward,
    Softmax,
    SoftmaxBackward,

    /// <summary>A derived elementwise kernel at a fixed length (a resident stage); the key's Name says which.</summary>
    Elementwise,
    FusedLinear,
    FusedLinearBackward,
    Gemm,
    GemmTransposedA,
    GemmTransposedB,
    ScaledDotProductAttention,
    ScaledDotProductAttentionBackward,
    Embedding,
    FlashAttentionBackward,

    /// <summary>A convolution pass (dense, grouped or a transposed one's adjoint); the key's Name is its DeclaredConvKey.</summary>
    Convolution,
    IndexSelect,
    TakeAlongDim,

    /// <summary>Paged attention's decode step: the KV cache reached through a block table (#890 phase 4d).</summary>
    PagedAttentionDecode,

    /// <summary>Paged attention's causal prefill step over the same cache (#890 phase 4d).</summary>
    PagedAttentionPrefill,

    /// <summary>The deterministic scatter-add, and the segment sum that is the same operator (#890 phase 4e).</summary>
    ScatterAddDeterministic,

    /// <summary>The deterministic scatter-add onto the destination's existing value (#890 phase 4e).</summary>
    ScatterAddAccumulateDeterministic,

    /// <summary>The deterministic segment maximum, which a sparse matrix's row indices make a CSR one (#890 phase 4e).</summary>
    SegmentMax,

    /// <summary>The deterministic segment minimum (#890 phase 4e).</summary>
    SegmentMin,

    /// <summary><c>dst[i] = src[offset + i * stride]</c> (#1441).</summary>
    StridedGather,

    /// <summary><c>dst[offset + i * stride] = src[i]</c>, in place (#1441).</summary>
    StridedScatter,

    /// <summary>The row-wise scatter mean with its counts (#1441).</summary>
    ScatterMeanRows,

    /// <summary>The row-wise scatter softmax (#1441).</summary>
    ScatterSoftmaxRows,
}

/// <summary>One declared program: its family and shape (unused dimensions are 0).</summary>
internal readonly struct DeclaredKey : IEquatable<DeclaredKey>
{
    internal DeclaredKey(DeclaredFamily family, int d0, int d1, int d2, int d3, float epsilon, float momentum, string? name = null,
        int d4 = 0, int d5 = 0)
    {
        Name = name;
        D4 = d4;
        D5 = d5;
        Family = family;
        D0 = d0;
        D1 = d1;
        D2 = d2;
        D3 = d3;
        Epsilon = epsilon;
        Momentum = momentum;
    }

    internal DeclaredFamily Family { get; }
    internal int D0 { get; }
    internal int D1 { get; }
    internal int D2 { get; }
    internal int D3 { get; }
    internal int D4 { get; }
    internal int D5 { get; }
    internal float Epsilon { get; }
    internal float Momentum { get; }

    /// <summary>What the dimensions do not say -- an elementwise stage's kernel and scalars; null otherwise.</summary>
    internal string? Name { get; }

    public bool Equals(DeclaredKey other) =>
        Family == other.Family && D0 == other.D0 && D1 == other.D1 && D2 == other.D2 && D3 == other.D3 &&
        D4 == other.D4 && D5 == other.D5 && Epsilon.Equals(other.Epsilon) && Momentum.Equals(other.Momentum) && string.Equals(Name, other.Name, StringComparison.Ordinal);

    public override bool Equals(object? obj) => obj is DeclaredKey other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int h = (int)Family;
            h = h * 31 + D0;
            h = h * 31 + D1;
            h = h * 31 + D2;
            h = h * 31 + D3;
            h = h * 31 + D4;
            h = h * 31 + D5;
            h = h * 31 + Epsilon.GetHashCode();
            h = h * 31 + (Name is null ? 0 : StringComparer.Ordinal.GetHashCode(Name));
            return h * 31 + Momentum.GetHashCode();
        }
    }

    /// <summary>A readable name, used for entry points and diagnostics (built once per program, never per call).</summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder(Family.ToString().ToLowerInvariant());
        foreach (int d in new[] { D0, D1, D2, D3, D4, D5 })
            if (d != 0) sb.Append(sb.Length > Family.ToString().Length ? "x" : ":").Append(d.ToString(CultureInfo.InvariantCulture));
        if (Epsilon != 0) sb.Append(":e").Append(Epsilon.ToString("R", CultureInfo.InvariantCulture));
        if (Momentum != 0) sb.Append(":m").Append(Momentum.ToString("R", CultureInfo.InvariantCulture));
        if (Name is not null) sb.Append(':').Append(Name);
        return sb.ToString();
    }
}

/// <summary>
/// The caller's buffers for one call, in the order of the family's tensor names. Up to ten (training BatchNorm binds
/// its running statistics under two names each); a struct, so a call does not allocate.
/// </summary>
internal readonly struct DeclaredBindings
{
    internal const int Capacity = 10;
    private readonly IGpuBuffer? _b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7, _b8, _b9;

    internal DeclaredBindings(IGpuBuffer b0, IGpuBuffer b1, IGpuBuffer? b2 = null, IGpuBuffer? b3 = null,
        IGpuBuffer? b4 = null, IGpuBuffer? b5 = null, IGpuBuffer? b6 = null, IGpuBuffer? b7 = null,
        IGpuBuffer? b8 = null, IGpuBuffer? b9 = null)
    {
        _b0 = b0; _b1 = b1; _b2 = b2; _b3 = b3; _b4 = b4; _b5 = b5; _b6 = b6; _b7 = b7; _b8 = b8; _b9 = b9;
    }

    internal IGpuBuffer this[int slot] => (slot switch
    {
        0 => _b0,
        1 => _b1,
        2 => _b2,
        3 => _b3,
        4 => _b4,
        5 => _b5,
        6 => _b6,
        7 => _b7,
        8 => _b8,
        9 => _b9,
        _ => null,
    }) ?? throw new ArgumentException("No buffer is bound to slot " + slot + ".");
}

public sealed partial class CudaBackend
{
    private static readonly bool s_declaredKernels =
        Environment.GetEnvironmentVariable("AIDOTNET_DECLARED_KERNELS") != "0";

    // Lowered programs by key; a null entry records a shape that declined, so it is not lowered again.
    private readonly Dictionary<DeclaredKey, DeclaredProgram?> _declaredPrograms = new();

    // Why each declined key declined, for diagnostics.
    private readonly Dictionary<DeclaredKey, string> _declaredDeclines = new();

    private long _declaredProgramLaunches;

    // Device memory the cached programs' scratch holds (attention's [batch, heads, seqQ, seqK] logits, the weights
    // when the caller does not keep them, the backward's dP and dS, a split's partials, and the causal biases),
    // against an admission budget.
    //
    // The second-sighting rule stops autoregressive decoding from compiling a program per token; it does NOT bound
    // the number of RECURRING shapes. Training with dynamic padding or bucketed sequence lengths, or a varying last
    // batch, makes many keys recur, and each one pinned its scratch for as long as the backend lived -- up to
    // AttentionMaxScoreElements (256 MiB) per buffer -- so device memory grew until an allocation failed.
    //
    // Admission rather than eviction: a program is reached through _directPtxLock but LAUNCHED after the lock is
    // released, so evicting one thread's program is a use-after-free for another, and a captured graph has the
    // scratch's addresses baked into it. Declining instead is the route's existing failure mode - the same one a
    // soft-cap, an oversized score tensor or a first sighting under capture already takes - so the shape keeps the
    // hand-written kernel, which needs no scratch at all. The decline is memoised like any other, which is accurate
    // because a cached program's scratch is only released when the backend is disposed.
    private long _declaredScratchBytes;
    private long _declaredScratchBudget;

    /// <summary>Device bytes the cached declared programs' scratch holds, and the budget it is admitted against.</summary>
    internal long DeclaredScratchBytes { get { lock (_directPtxLock) return _declaredScratchBytes; } }

    internal long DeclaredScratchBudgetBytes { get { lock (_directPtxLock) return DeclaredScratchBudgetUnlocked(); } }

    // An eighth of the device, at least 256 MiB, or AIDOTNET_DECLARED_SCRATCH_BUDGET_MB when it parses. Resolved once
    // per backend, since the device's capacity does not change. The caller holds _directPtxLock.
    private long DeclaredScratchBudgetUnlocked()
    {
        if (_declaredScratchBudget != 0) return _declaredScratchBudget;
        const long floor = 256L * 1024 * 1024;
        string? configured = Environment.GetEnvironmentVariable("AIDOTNET_DECLARED_SCRATCH_BUDGET_MB");
        if (configured is not null &&
            long.TryParse(configured, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out long megabytes) && megabytes > 0)
            return _declaredScratchBudget = megabytes * 1024 * 1024;
        _declaredScratchBudget = TryGetDeviceMemory(out _, out ulong total) && total != 0
            ? Math.Max(floor, (long)(total / 8))
            : floor;
        return _declaredScratchBudget;
    }

    // Whether this many scratch bytes fit, and the decline when they do not. The caller holds _directPtxLock.
    private bool DeclaredScratchFits(long bytes, out string? decline)
    {
        long budget = DeclaredScratchBudgetUnlocked();
        if (bytes <= 0 || _declaredScratchBytes + bytes <= budget)
        {
            decline = null;
            return true;
        }
        decline = "the program needs " + (bytes >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture) +
            " MiB of scratch and the declared route already holds " +
            (_declaredScratchBytes >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture) + " MiB of its " +
            (budget >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture) +
            " MiB budget (AIDOTNET_DECLARED_SCRATCH_BUDGET_MB); the hand-written kernel, which needs none, serves this shape.";
        return false;
    }

    /// <summary>
    /// Overrides the process switch for this backend: the route-equivalence tooling runs the same call with the
    /// declared route on and off. Null follows AIDOTNET_DECLARED_KERNELS.
    /// </summary>
    internal bool? DeclaredKernelsSwitch { get; set; }

    /// <summary>Whether the declared-program route is on for this backend (switch and architecture).</summary>
    internal bool DeclaredKernelsEnabled =>
        (DeclaredKernelsSwitch ?? s_declaredKernels) && IsAvailable &&
        DirectPtxArchitecture.HasReleasedCodegenKernels(_ccMajor, _ccMinor);

    /// <summary>Number of calls served by declared programs, for routing tests and diagnostics.</summary>
    internal long DeclaredProgramLaunches => System.Threading.Interlocked.Read(ref _declaredProgramLaunches);

    private sealed class DeclaredProgram : IDisposable
    {
        internal DeclaredProgram(PtxProgramLowering lowering, List<(DirectPtxModule Module, IntPtr Function)> kernels,
            IGpuBuffer[] scratch, int[][] argumentSlots, long[] boundElements)
        {
            foreach (var step in lowering.Steps)
                VectorAccess |= s_vectorGlobalAccess.IsMatch(step.Ptx);
            Lowering = lowering;
            Kernels = kernels;
            Scratch = scratch;
            ArgumentSlots = argumentSlots;
            BoundElements = boundElements;
        }

        internal PtxProgramLowering Lowering { get; }
        internal List<(DirectPtxModule Module, IntPtr Function)> Kernels { get; }
        internal IGpuBuffer[] Scratch { get; }

        /// <summary>Per step, per argument: a caller slot (k >= 0) or a scratch buffer (-1 - index).</summary>
        internal int[][] ArgumentSlots { get; }

        /// <summary>Elements the program needs in each caller slot (0 for a slot it does not name).</summary>
        internal long[] BoundElements { get; }

        /// <summary>Whether a kernel reads or writes global memory with vector instructions, which need every bound
        /// buffer 16-byte aligned (<see cref="AlignedFor"/>).</summary>
        internal bool VectorAccess { get; }

        public void Dispose()
        {
            foreach (var (module, _) in Kernels) module.Dispose();
            foreach (var buffer in Scratch) buffer.Dispose();
        }
    }

    private static readonly System.Text.RegularExpressions.Regex s_vectorGlobalAccess = new(
        @"\b(ld|st)\.global(\.nc)?\.v[24]\.", System.Text.RegularExpressions.RegexOptions.Compiled);

    // Whether the caller's buffers suit the program: a program with vector global access needs each buffer it names
    // 16-byte aligned (a view into a larger allocation need not be), or a vector instruction faults the context.
    private static bool AlignedFor(DeclaredProgram program, string[] names, in DeclaredBindings bound)
    {
        if (!program.VectorAccess) return true;
        for (int k = 0; k < names.Length; k++)
            if (program.BoundElements[k] != 0 && ((long)bound[k].Handle & 15) != 0) return false;
        return true;
    }

    /// <summary>
    /// Runs the program <paramref name="build"/> makes for <paramref name="key"/>, with <paramref name="bound"/>
    /// supplying the tensors named by <paramref name="names"/> (slot k is names[k]); every other tensor of the program
    /// is scratch this backend allocates once per key. Returns false -- having launched nothing -- when the route is
    /// off or the program cannot be lowered for this shape.
    /// </summary>
    internal bool TryRunDeclaredProgram(DeclaredKey key, Func<DeclaredKey, CodegenProgram> build,
        string[] names, in DeclaredBindings bound, Func<CodegenKernelSpec, PtxPassLowering?>? lowerPass = null,
        bool residentWhenSmall = false)
    {
        if (!DeclaredKernelsEnabled) return false;
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(key, out program))
            {
                // Building loads modules and allocates scratch, which CUDA graph capture forbids: a program first
                // met under capture is served by the hand-written kernel, and is built on the first call outside.
                if (IsStreamCapturing()) return false;
                program = BuildDeclaredProgram(key, build, names, lowerPass);
                _declaredPrograms[key] = program;
            }
        }
        if (program is null || !AlignedFor(program, names, bound)) return false;
        if (residentWhenSmall && TryLaunchDeclaredResidentChecked(key, program, names, bound)) return true;
        RunDeclaredProgram(program, names, bound);
        return true;
    }

    // The bound-size check of RunDeclaredProgram, then the one-launch path.
    private bool TryLaunchDeclaredResidentChecked(DeclaredKey key, DeclaredProgram program, string[] names,
        in DeclaredBindings bound)
    {
        for (int k = 0; k < names.Length; k++)
        {
            long needed = program.BoundElements[k];
            if (needed != 0 && bound[k].Size < needed)
                throw new ArgumentException(
                    "'" + names[k] + "' holds " + bound[k].Size + " elements; the program needs " + needed + ".");
        }
        return TryLaunchDeclaredResident(key, program, bound);
    }

    // Checks the bound buffers against the program's sizes, then launches its steps on the main stream.
    private void RunDeclaredProgram(DeclaredProgram program, string[] names, in DeclaredBindings bound)
    {
        for (int k = 0; k < names.Length; k++)
        {
            long needed = program.BoundElements[k];
            if (needed != 0 && bound[k].Size < needed)
                throw new ArgumentException(
                    "'" + names[k] + "' holds " + bound[k].Size + " elements; the program needs " + needed + ".");
        }

        using var _ = PushContext();
        // One caller at a time per program: its scratch (a split's partials, the statistics) is shared, and two
        // callers' launches interleaved on the stream would read each other's partials.
        lock (program) LaunchSteps(program, bound);
        System.Threading.Interlocked.Increment(ref _declaredProgramLaunches);
    }

    private static unsafe void LaunchSteps(DeclaredProgram program, in DeclaredBindings bound)
    {
        var steps = program.Lowering.Steps;
        // One argument block sized to the widest step, reused by every launch.
        int widest = 0;
        for (int s = 0; s < steps.Count; s++) widest = Math.Max(widest, steps[s].Arguments.Count);
        IntPtr* handles = stackalloc IntPtr[widest];
        void** argv = stackalloc void*[widest];
        for (int s = 0; s < steps.Count; s++)
        {
            var step = steps[s];
            int[] slots = program.ArgumentSlots[s];
            for (int a = 0; a < slots.Length; a++)
            {
                int slot = slots[a];
                handles[a] = (slot >= 0 ? bound[slot] : program.Scratch[-1 - slot]).Handle;
                argv[a] = &handles[a];
            }
            var (module, function) = program.Kernels[s];
            // The caller pushed the context; Launch would query it again on every step.
            module.LaunchCurrentContext(function, step.Blocks, 1, 1, step.BlockX, step.BlockY, 1,
                step.DynamicSharedMemoryBytes, argv);
        }
    }

    private DeclaredProgram? BuildDeclaredProgram(DeclaredKey key, Func<DeclaredKey, CodegenProgram> build, string[] names,
        Func<CodegenKernelSpec, PtxPassLowering?>? lowerPass = null)
    {
        CodegenProgram source;
        try
        {
            source = build(key);
        }
        catch (NotSupportedException ex)
        {
            _declaredDeclines[key] = ex.Message;   // the family cannot express this key (a builder's own refusal)
            return null;
        }
        var program = BuildDeclaredProgram(source, key.ToString(), names, lowerPass, out string? decline);
        if (decline is not null) _declaredDeclines[key] = decline;
        return program;
    }

    // Lowers and loads a program, or returns null with the reason it has no declared lowering (the hand-written
    // kernel then serves the shape). lowerPass replaces the default lowering of a pass, as in PtxProgramLowering.
    private DeclaredProgram? BuildDeclaredProgram(CodegenProgram source, string name, string[] names,
        Func<CodegenKernelSpec, PtxPassLowering?>? lowerPass, out string? decline)
    {
        decline = null;
        if (names.Length > DeclaredBindings.Capacity)
            throw new ArgumentException("A declared call binds at most " + DeclaredBindings.Capacity + " buffers.", nameof(names));
        PtxProgramLowering lowering;
        try
        {
            lowering = PtxProgramLowering.Create(source, name, _ccMajor, _ccMinor, lowerPass);
        }
        catch (NotSupportedException ex)
        {
            decline = ex.Message;
            return null;
        }

        var slotOf = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int k = 0; k < names.Length; k++) slotOf[names[k]] = k;
        var boundElements = new long[names.Length];
        for (int k = 0; k < names.Length; k++)
            if (lowering.Buffers.TryGetValue(names[k], out long elements)) boundElements[k] = elements;

        // The scratch this program would pin for the backend's lifetime, decided BEFORE any of it is allocated.
        long scratchBytes = 0;
        foreach (var pair in lowering.Buffers)
            if (!slotOf.ContainsKey(pair.Key)) scratchBytes = checked(scratchBytes + pair.Value * sizeof(float));
        if (!DeclaredScratchFits(scratchBytes, out string? overBudget))
        {
            decline = overBudget;
            return null;
        }

        _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
        var kernels = new List<(DirectPtxModule, IntPtr)>();
        var scratch = new List<IGpuBuffer>();
        try
        {
            var scratchOf = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var pair in lowering.Buffers)
            {
                if (slotOf.ContainsKey(pair.Key)) continue;
                scratchOf[pair.Key] = scratch.Count;
                scratch.Add(AllocateBuffer(checked((int)pair.Value)));
            }
            var argumentSlots = new int[lowering.Steps.Count][];
            for (int s = 0; s < lowering.Steps.Count; s++)
            {
                var step = lowering.Steps[s];
                // A step that calls math-library functions is linked with their libraries.
                var module = step.Calls.Count == 0
                    ? _directPtxRuntime.LoadModule(step.Ptx, allowExperimentalJitFallback: true)
                    : _directPtxRuntime.LoadLinkedModule(step.Ptx, LibrariesFor(step.Calls));
                IntPtr function = module.GetFunction(step.EntryPoint, out _);
                if (step.DynamicSharedMemoryBytes != 0)
                    module.SetMaxDynamicSharedMemory(function, checked((int)step.DynamicSharedMemoryBytes));
                kernels.Add((module, function));
                var slots = new int[step.Arguments.Count];
                for (int a = 0; a < slots.Length; a++)
                    slots[a] = slotOf.TryGetValue(step.Arguments[a], out int k) ? k : -1 - scratchOf[step.Arguments[a]];
                argumentSlots[s] = slots;
            }
            // Charged only once the program is certain to be returned: both catch arms below dispose the scratch.
            _declaredScratchBytes = checked(_declaredScratchBytes + scratchBytes);
            return new DeclaredProgram(lowering, kernels, scratch.ToArray(), argumentSlots, boundElements);
        }
        catch (InvalidOperationException ex)
        {
            // The driver rejected a kernel (a JIT failure, or local memory the direct-kernel contract
            // forbids): decline, so the hand-written kernel serves the shape.
            foreach (var (module, _) in kernels) module.Dispose();
            foreach (var buffer in scratch) buffer.Dispose();
            decline = ex.Message;
            return null;
        }
        catch
        {
            foreach (var (module, _) in kernels) module.Dispose();
            foreach (var buffer in scratch) buffer.Dispose();
            throw;
        }
    }

    /// <summary>Why the program for <paramref name="key"/> declined, or null when it did not.</summary>
    internal string? DeclaredDeclineReason(DeclaredKey key)
    {
        lock (_directPtxLock) return _declaredDeclines.TryGetValue(key, out string? reason) ? reason : null;
    }

    /// <summary>The launches a cached program makes ("entry blocks x threads" per step), or null when the key has
    /// not been lowered or declined. Diagnostics only.</summary>
    internal IReadOnlyList<string>? DescribeDeclaredProgram(DeclaredKey key)
    {
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(key, out var program) || program is null) return null;
            var steps = new List<string>();
            foreach (var step in program.Lowering.Steps)
                steps.Add(step.EntryPoint + " " + step.Blocks + "x" + step.BlockX + (step.BlockY > 1 ? "x" + step.BlockY : ""));
            return steps;
        }
    }

    private void DisposeDeclaredPrograms()
    {
        DisposeDeclaredResidents();
        DisposeDeclaredElementwise();
        DisposeDeclaredConvolutions();
        DisposeCausalBiases();

        foreach (var program in _declaredPrograms.Values) program?.Dispose();
        _declaredPrograms.Clear();
        _declaredStridedPrograms = 0;
        _declaredScatterRowsPrograms = 0;
    }

    // ---- LayerNorm ------------------------------------------------------------------------------------------------

    private static readonly string[] s_layerNormNames = { "x", "gamma", "beta", "y", "mean", "rstd" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildLayerNorm =
        key => CodegenBackendPrograms.LayerNorm(key.D0, key.D1, key.Epsilon);

    internal static DeclaredKey LayerNormKey(int batchSize, int normalizedSize, float epsilon) =>
        new(DeclaredFamily.LayerNorm, batchSize, normalizedSize, 0, 0, epsilon, 0);

    /// <summary>LayerNorm through its declared program, in CudaBackend.LayerNorm's contract.</summary>
    private bool TryDeclaredLayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        if (batchSize <= 0 || normalizedSize <= 0) return false;
        return TryRunDeclaredProgram(LayerNormKey(batchSize, normalizedSize, epsilon), s_buildLayerNorm,
            s_layerNormNames, new DeclaredBindings(input, gamma, beta, output, saveMean, saveInvVar));
    }

    // ---- RmsNorm --------------------------------------------------------------------------------------------------

    private static readonly string[] s_rmsNormNames = { "x", "gamma", "y", "rms" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildRmsNorm =
        key => CodegenBackendPrograms.RmsNorm(key.D0, key.D1, key.Epsilon);

    internal static DeclaredKey RmsNormKey(int batchSize, int normalizedSize, float epsilon) =>
        new(DeclaredFamily.RmsNorm, batchSize, normalizedSize, 0, 0, epsilon, 0);

    /// <summary>RmsNorm through its declared program, in CudaBackend.RmsNorm's contract (saveRms = rms).</summary>
    private bool TryDeclaredRmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (batchSize <= 0 || normalizedSize <= 0) return false;
        return TryRunDeclaredProgram(RmsNormKey(batchSize, normalizedSize, epsilon), s_buildRmsNorm,
            s_rmsNormNames, new DeclaredBindings(input, gamma, output, saveRms));
    }

    // ---- GroupNorm ------------------------------------------------------------------------------------------------

    private static readonly string[] s_groupNormNames = { "x", "gamma", "beta", "y", "mean", "rstd" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildGroupNorm =
        key => CodegenBackendPrograms.GroupNorm(key.D0, key.D1, key.D2, key.D3, key.Epsilon);

    internal static DeclaredKey GroupNormKey(int batch, int channels, int numGroups, int spatialSize, float epsilon) =>
        new(DeclaredFamily.GroupNorm, batch, channels, numGroups, spatialSize, epsilon, 0);

    /// <summary>GroupNorm through its declared program, in CudaBackend.GroupNorm's contract.</summary>
    private bool TryDeclaredGroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        if (batch <= 0 || numGroups <= 0 || channels <= 0 || spatialSize <= 0 || channels % numGroups != 0) return false;
        return TryRunDeclaredProgram(GroupNormKey(batch, channels, numGroups, spatialSize, epsilon), s_buildGroupNorm,
            s_groupNormNames, new DeclaredBindings(input, gamma, beta, output, saveMean, saveInvVar));
    }

    // ---- BatchNorm (training) -------------------------------------------------------------------------------------

    // The running statistics are bound twice, as the value read and the value written: one buffer, updated in place.
    // Each element is read and then written by its channel's leader, and no later pass reads the written value.
    private static readonly string[] s_batchNormNames =
    {
        "x", "gamma", "beta", "running_mean", "running_var", "y", "mean", "rstd", "running_mean_out", "running_var_out",
    };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildBatchNorm =
        key => CodegenBackendPrograms.BatchNorm2DTraining(key.D0, key.D1, key.D2, 1, key.Epsilon, key.Momentum);

    internal static DeclaredKey BatchNormKey(int batch, int channels, int spatialSize, float epsilon, float momentum) =>
        new(DeclaredFamily.BatchNormTraining, batch, channels, spatialSize, 0, epsilon, momentum);

    /// <summary>Training BatchNorm through its declared program, in CudaBackend.BatchNorm's contract.</summary>
    private bool TryDeclaredBatchNormTraining(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum)
    {
        if (batch <= 0 || channels <= 0 || spatialSize <= 0 || !(momentum >= 0 && momentum <= 1)) return false;
        return TryRunDeclaredProgram(BatchNormKey(batch, channels, spatialSize, epsilon, momentum), s_buildBatchNorm,
            s_batchNormNames, new DeclaredBindings(input, gamma, beta, runningMean, runningVar, output, saveMean,
                saveInvVar, runningMean, runningVar));
    }

    // ---- Backward -------------------------------------------------------------------------------------------------

    private static readonly string[] s_normBackwardNames = { "dy", "x", "gamma", "mean", "rstd", "dx", "dgamma", "dbeta" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildLayerNormBackward =
        key => CodegenBackendPrograms.LayerNormBackward(key.D0, key.D1);

    internal static DeclaredKey LayerNormBackwardKey(int batchSize, int normalizedSize) =>
        new(DeclaredFamily.LayerNormBackward, batchSize, normalizedSize, 0, 0, 0, 0);

    /// <summary>LayerNorm backward through its declared program, in CudaBackend.LayerNormBackward's contract.</summary>
    private bool TryDeclaredLayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize)
    {
        if (batchSize <= 0 || normalizedSize <= 0) return false;
        return TryRunDeclaredProgram(LayerNormBackwardKey(batchSize, normalizedSize), s_buildLayerNormBackward,
            s_normBackwardNames, new DeclaredBindings(gradOutput, input, gamma, saveMean, saveInvVar, gradInput,
                gradGamma, gradBeta));
    }

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildBatchNormBackward =
        key => CodegenBackendPrograms.BatchNorm2DBackward(key.D0, key.D1, key.D2, 1);

    internal static DeclaredKey BatchNormBackwardKey(int batch, int channels, int spatialSize) =>
        new(DeclaredFamily.BatchNormBackward, batch, channels, spatialSize, 0, 0, 0);

    /// <summary>BatchNorm backward through its declared program, in CudaBackend.BatchNormBackward's contract.</summary>
    private bool TryDeclaredBatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize)
    {
        if (batch <= 0 || channels <= 0 || spatialSize <= 0) return false;
        return TryRunDeclaredProgram(BatchNormBackwardKey(batch, channels, spatialSize), s_buildBatchNormBackward,
            s_normBackwardNames, new DeclaredBindings(gradOutput, input, gamma, saveMean, saveInvVar, gradInput,
                gradGamma, gradBeta));
    }

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildGroupNormBackward =
        key => CodegenBackendPrograms.GroupNormBackward(key.D0, key.D1, key.D2, key.D3);

    internal static DeclaredKey GroupNormBackwardKey(int batch, int channels, int numGroups, int spatialSize) =>
        new(DeclaredFamily.GroupNormBackward, batch, channels, numGroups, spatialSize, 0, 0);

    /// <summary>
    /// GroupNorm backward through its declared program. <paramref name="saveInvVar"/> is the forward's slot,
    /// 1/sqrt(var + eps), not the variance: the program folds epsilon in nowhere else. False declines the shape, and
    /// the caller composes the gradient from elementwise kernels as it did before this route existed.
    /// </summary>
    internal bool TryDeclaredGroupNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int numGroups, int channels, int spatialSize)
    {
        if (batch <= 0 || numGroups <= 0 || channels <= 0 || spatialSize <= 0 || channels % numGroups != 0) return false;
        return TryRunDeclaredProgram(GroupNormBackwardKey(batch, channels, numGroups, spatialSize),
            s_buildGroupNormBackward, s_normBackwardNames, new DeclaredBindings(gradOutput, input, gamma, saveMean,
                saveInvVar, gradInput, gradGamma, gradBeta));
    }

    private static readonly string[] s_rmsNormBackwardNames = { "dy", "x", "gamma", "rms", "dx", "dgamma" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildRmsNormBackward =
        key => CodegenBackendPrograms.RmsNormBackward(key.D0, key.D1);

    internal static DeclaredKey RmsNormBackwardKey(int batchSize, int normalizedSize) =>
        new(DeclaredFamily.RmsNormBackward, batchSize, normalizedSize, 0, 0, 0, 0);

    /// <summary>RmsNorm backward through its declared program, in CudaBackend.RmsNormBackward's contract.</summary>
    private bool TryDeclaredRmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveRms, IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize)
    {
        if (batchSize <= 0 || normalizedSize <= 0) return false;
        return TryRunDeclaredProgram(RmsNormBackwardKey(batchSize, normalizedSize), s_buildRmsNormBackward,
            s_rmsNormBackwardNames, new DeclaredBindings(gradOutput, input, gamma, saveRms, gradInput, gradGamma));
    }

    // ---- Softmax --------------------------------------------------------------------------------------------------

    private static readonly string[] s_softmaxNames = { "x", "y" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildSoftmax =
        key => CodegenFusedStatistics.Softmax(key.D0, key.D1);

    internal static DeclaredKey SoftmaxKey(int batchSize, int features) =>
        new(DeclaredFamily.Softmax, batchSize, features, 0, 0, 0, 0);

    /// <summary>Row softmax through its declared program, in CudaBackend.Softmax's contract.</summary>
    private bool TryDeclaredSoftmax(IGpuBuffer input, IGpuBuffer output, int batchSize, int features)
    {
        if (batchSize <= 0 || features <= 0) return false;
        return TryRunDeclaredProgram(SoftmaxKey(batchSize, features), s_buildSoftmax,
            s_softmaxNames, new DeclaredBindings(input, output));
    }

    private static readonly string[] s_softmaxBackwardNames = { "dy", "y", "dx" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildSoftmaxBackward =
        key => CodegenFusedStatistics.SoftmaxBackward(key.D0, key.D1);

    internal static DeclaredKey SoftmaxBackwardKey(int batchSize, int features) =>
        new(DeclaredFamily.SoftmaxBackward, batchSize, features, 0, 0, 0, 0);

    /// <summary>Row softmax backward through its declared program, in CudaBackend.SoftmaxBackward's contract.</summary>
    private bool TryDeclaredSoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput,
        int batchSize, int features)
    {
        if (batchSize <= 0 || features <= 0) return false;
        return TryRunDeclaredProgram(SoftmaxBackwardKey(batchSize, features), s_buildSoftmaxBackward,
            s_softmaxBackwardNames, new DeclaredBindings(gradOutput, output, gradInput));
    }
}
