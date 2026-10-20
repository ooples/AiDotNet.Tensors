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
}

/// <summary>One declared program: its family and shape (unused dimensions are 0).</summary>
internal readonly struct DeclaredKey : IEquatable<DeclaredKey>
{
    internal DeclaredKey(DeclaredFamily family, int d0, int d1, int d2, int d3, float epsilon, float momentum)
    {
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
    internal float Epsilon { get; }
    internal float Momentum { get; }

    public bool Equals(DeclaredKey other) =>
        Family == other.Family && D0 == other.D0 && D1 == other.D1 && D2 == other.D2 && D3 == other.D3 &&
        Epsilon.Equals(other.Epsilon) && Momentum.Equals(other.Momentum);

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
            h = h * 31 + Epsilon.GetHashCode();
            return h * 31 + Momentum.GetHashCode();
        }
    }

    /// <summary>A readable name, used for entry points and diagnostics (built once per program, never per call).</summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder(Family.ToString().ToLowerInvariant());
        foreach (int d in new[] { D0, D1, D2, D3 })
            if (d != 0) sb.Append(sb.Length > Family.ToString().Length ? "x" : ":").Append(d.ToString(CultureInfo.InvariantCulture));
        if (Epsilon != 0) sb.Append(":e").Append(Epsilon.ToString("R", CultureInfo.InvariantCulture));
        if (Momentum != 0) sb.Append(":m").Append(Momentum.ToString("R", CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}

/// <summary>
/// The caller's buffers for one call, in the order of the family's tensor names. Up to eight; a struct, so a call
/// does not allocate.
/// </summary>
internal readonly struct DeclaredBindings
{
    internal const int Capacity = 8;
    private readonly IGpuBuffer? _b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7;

    internal DeclaredBindings(IGpuBuffer b0, IGpuBuffer b1, IGpuBuffer b2, IGpuBuffer b3,
        IGpuBuffer? b4 = null, IGpuBuffer? b5 = null, IGpuBuffer? b6 = null, IGpuBuffer? b7 = null)
    {
        _b0 = b0; _b1 = b1; _b2 = b2; _b3 = b3; _b4 = b4; _b5 = b5; _b6 = b6; _b7 = b7;
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

    /// <summary>
    /// Overrides the process switch for this backend: the route-equivalence tooling runs the same call with the
    /// declared route on and off. Null follows AIDOTNET_DECLARED_KERNELS.
    /// </summary>
    internal bool? DeclaredKernelsSwitch { get; set; }

    /// <summary>Whether the declared-program route is on for this backend (switch and architecture).</summary>
    internal bool DeclaredKernelsEnabled =>
        (DeclaredKernelsSwitch ?? s_declaredKernels) && IsAvailable &&
        DirectPtxArchitecture.HasCodegenConveyor(_ccMajor, _ccMinor);

    /// <summary>Number of calls served by declared programs, for routing tests and diagnostics.</summary>
    internal long DeclaredProgramLaunches => System.Threading.Interlocked.Read(ref _declaredProgramLaunches);

    private sealed class DeclaredProgram : IDisposable
    {
        internal DeclaredProgram(PtxProgramLowering lowering, List<(DirectPtxModule Module, IntPtr Function)> kernels,
            IGpuBuffer[] scratch, int[][] argumentSlots, long[] boundElements)
        {
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

        public void Dispose()
        {
            foreach (var (module, _) in Kernels) module.Dispose();
            foreach (var buffer in Scratch) buffer.Dispose();
        }
    }

    /// <summary>
    /// Runs the program <paramref name="build"/> makes for <paramref name="key"/>, with <paramref name="bound"/>
    /// supplying the tensors named by <paramref name="names"/> (slot k is names[k]); every other tensor of the program
    /// is scratch this backend allocates once per key. Returns false -- having launched nothing -- when the route is
    /// off or the program cannot be lowered for this shape.
    /// </summary>
    internal bool TryRunDeclaredProgram(DeclaredKey key, Func<DeclaredKey, CodegenProgram> build,
        string[] names, in DeclaredBindings bound)
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
                program = BuildDeclaredProgram(key, build, names);
                _declaredPrograms[key] = program;
            }
        }
        if (program is null) return false;

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
        return true;
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
            module.Launch(function, step.Blocks, 1, 1, step.BlockX, step.BlockY, 1, 0, argv);
        }
    }

    private DeclaredProgram? BuildDeclaredProgram(DeclaredKey key, Func<DeclaredKey, CodegenProgram> build, string[] names)
    {
        if (names.Length > DeclaredBindings.Capacity)
            throw new ArgumentException("A declared call binds at most " + DeclaredBindings.Capacity + " buffers.", nameof(names));
        PtxProgramLowering lowering;
        try
        {
            lowering = PtxProgramLowering.Create(build(key), key.ToString(), _ccMajor, _ccMinor);
        }
        catch (NotSupportedException ex)
        {
            _declaredDeclines[key] = ex.Message;
            return null;   // this shape has no declared lowering; the hand-written kernel serves it
        }

        var slotOf = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int k = 0; k < names.Length; k++) slotOf[names[k]] = k;
        var boundElements = new long[names.Length];
        for (int k = 0; k < names.Length; k++)
            if (lowering.Buffers.TryGetValue(names[k], out long elements)) boundElements[k] = elements;

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
                var module = _directPtxRuntime.LoadModule(step.Ptx, allowExperimentalJitFallback: true);
                kernels.Add((module, module.GetFunction(step.EntryPoint, out _)));
                var slots = new int[step.Arguments.Count];
                for (int a = 0; a < slots.Length; a++)
                    slots[a] = slotOf.TryGetValue(step.Arguments[a], out int k) ? k : -1 - scratchOf[step.Arguments[a]];
                argumentSlots[s] = slots;
            }
            return new DeclaredProgram(lowering, kernels, scratch.ToArray(), argumentSlots, boundElements);
        }
        catch (InvalidOperationException ex)
        {
            // The driver rejected a kernel (a JIT failure, or local memory the direct-kernel contract
            // forbids): decline, so the hand-written kernel serves the shape.
            foreach (var (module, _) in kernels) module.Dispose();
            foreach (var buffer in scratch) buffer.Dispose();
            _declaredDeclines[key] = ex.Message;
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
        foreach (var program in _declaredPrograms.Values) program?.Dispose();
        _declaredPrograms.Clear();
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
}
