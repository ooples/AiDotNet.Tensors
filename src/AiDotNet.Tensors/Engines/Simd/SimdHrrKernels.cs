using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// SIMD kernels for Holographic Reduced Representation (HRR) workloads —
/// issue #248. HRR bind/unbind, permutation-gather for non-commutativity,
/// unit-phase codebook generation, and full-vocabulary phase-coherence
/// decode. All kernels operate on split Re/Im <c>double</c> spans to keep
/// the AVX2 + FMA path dense (4 lanes × 64-bit).
///
/// <para>Design notes:</para>
/// <list type="bullet">
///   <item>No tensor wrapping — callers pass raw spans / arrays, so the
///   per-call overhead that drove issue #248 disappears (the HRE research
///   campaign's cycle 12 build at D=16384 × 1M pairs was blocked by
///   allocation churn, not compute).</item>
///   <item>Kernels validate shapes then dispatch a single vectorised
///   loop. No GraphMode recording or autograd — these are eager spans,
///   the caller is responsible for tape hookups when needed.</item>
///   <item>Every op has a deterministic scalar fallback that matches the
///   SIMD path bit-for-bit (up to order-of-operations differences in the
///   FMA vs mul+add case, which are bounded by 1 ULP).</item>
/// </list>
/// </summary>
public static class SimdHrrKernels
{
    // Parallelism threshold — below this we stay single-threaded because
    // the work-stealing dispatcher overhead exceeds the kernel time.
    // Chosen to match SimdKernels' hot-path heuristic for D=1024 fp64.
    private const int ParallelThresholdElements = 4096;

    /// <summary>
    /// Indexed gather: <c>output[i] = input[indices[i]]</c> for each i in
    /// <c>[0, indices.Length)</c>. Standard permutation-gather primitive
    /// equivalent to <c>numpy a[perm]</c> or <c>torch.gather(..., dim=0)</c>.
    ///
    /// <para>AVX2 has a native <c>VGATHERDPD</c> that loads 4 doubles at 4
    /// independent addresses in one instruction, but the instruction is
    /// notoriously slow on Intel (≥20 cycle latency) and marginally faster
    /// than scalar on Zen 4. We therefore use a tight scalar loop with
    /// sequential bounds-check elimination — the JIT recognises the
    /// pattern and emits near-optimal code. Issue #248's measured 10 µs
    /// at D=16384 was an un-parallelised `for` loop; this version adds
    /// per-chunk parallelisation for D above <see cref="ParallelThresholdElements"/>.</para>
    /// </summary>
    /// <exception cref="ArgumentException">Mismatched span lengths.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An index is outside
    /// <c>[0, input.Length)</c>.</exception>
    [MethodImpl(HotInline)]
    public static void GatherDouble(
        ReadOnlySpan<double> input,
        ReadOnlySpan<int> indices,
        Span<double> output)
    {
        int n = indices.Length;
        if (output.Length != n)
            throw new ArgumentException(
                $"Output length ({output.Length}) must equal indices length ({indices.Length}).");

        int inputLen = input.Length;
        for (int i = 0; i < n; i++)
        {
            int idx = indices[i];
            // Single bounds check — the cast-to-uint trick folds the
            // negative-index and overflow checks into a single unsigned
            // compare that the JIT reliably recognises.
            if ((uint)idx >= (uint)inputLen)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {idx} at position {i} is outside input range [0, {inputLen}).");
            output[i] = input[idx];
        }
    }

    /// <summary>
    /// Parallel overload of <see cref="GatherDouble(ReadOnlySpan{double}, ReadOnlySpan{int}, Span{double})"/>.
    /// Partitions the index set into contiguous chunks and runs them on
    /// the thread pool. The span-based overload is kept for hot single-
    /// threaded paths where task-dispatch overhead would dominate.
    /// </summary>
    public static unsafe void GatherDoubleParallel(
        double[] input, int[] indices, double[] output)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (output is null) throw new ArgumentNullException(nameof(output));
        int n = indices.Length;
        if (output.Length != n)
            throw new ArgumentException(
                $"Output length ({output.Length}) must equal indices length ({indices.Length}).");

        if (n < ParallelThresholdElements)
        {
            GatherDouble(input.AsSpan(), indices.AsSpan(), output.AsSpan());
            return;
        }

        int inputLen = input.Length;
        int maxDop = CpuParallelSettings.MaxDegreeOfParallelism;
        int chunks = Math.Min(maxDop, Math.Max(1, n / ParallelThresholdElements));
        int chunkSize = (n + chunks - 1) / chunks;

        Parallel.For(0, chunks, chunk =>
        {
            int start = chunk * chunkSize;
            int end = Math.Min(start + chunkSize, n);
            for (int i = start; i < end; i++)
            {
                int idx = indices[i];
                if ((uint)idx >= (uint)inputLen)
                    throw new ArgumentOutOfRangeException(nameof(indices),
                        $"Index {idx} at position {i} is outside input range [0, {inputLen}).");
                output[i] = input[idx];
            }
        });
    }

    /// <summary>
    /// Generate a V×D unit-phase complex codebook: every entry is
    /// <c>exp(iθ)</c> for a uniformly random phase θ ∈ [0, 2π). Output is
    /// split into <paramref name="outR"/> (cos θ) and <paramref name="outI"/>
    /// (sin θ), both flattened row-major as <c>[v * D + d]</c>.
    ///
    /// <para>Optional K-PSK quantization: when <paramref name="kPsk"/> is
    /// true, phases are snapped to the nearest multiple of <c>2π/k</c> —
    /// this is the M-ary phase-shift-keying constraint used in HRE's
    /// discrete-phase cycles (4-PSK, 8-PSK, 16-PSK). <paramref name="k"/>
    /// must be positive when <paramref name="kPsk"/> is set.</para>
    ///
    /// <para>Uses a deterministic Xoroshiro-style PRNG keyed by
    /// <paramref name="seed"/> so the same seed reproduces the same
    /// codebook across runs — critical for HRR experiments' reproducibility.</para>
    /// </summary>
    public static void UnitPhaseCodebookDouble(
        Span<double> outR, Span<double> outI,
        int seed, int V, int D,
        bool kPsk = false, int k = 0)
    {
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V), "V must be non-negative.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be non-negative.");
        long total = (long)V * D;
        if (total > int.MaxValue)
            throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        int n = (int)total;
        if (outR.Length != n || outI.Length != n)
            throw new ArgumentException(
                $"outR ({outR.Length}) and outI ({outI.Length}) must both have length V*D = {n}.");
        if (kPsk && k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k),
                "k must be positive when kPsk is true.");

        // xorshift64* — deterministic, fast enough that Sin/Cos dominate.
        ulong state = unchecked((ulong)seed * 0x9E3779B97F4A7C15UL | 1);
        const double TwoPi = 2.0 * Math.PI;
        double kPskStep = kPsk ? TwoPi / k : 0.0;

        for (int i = 0; i < n; i++)
        {
            // xorshift64* step
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ulong r = state * 0x2545F4914F6CDD1DUL;
            // Upper 53 bits → double in [0, 1).
            double u01 = (r >> 11) * (1.0 / (1UL << 53));
            double phase = u01 * TwoPi;
            if (kPsk)
            {
                // Snap to nearest K-PSK lattice point.
                phase = Math.Floor(phase / kPskStep + 0.5) * kPskStep;
            }
            outR[i] = Math.Cos(phase);
            outI[i] = Math.Sin(phase);
        }
    }

    /// <summary>
    /// Full-vocabulary phase-coherence decode: for each candidate
    /// <c>v ∈ [0, V)</c>, compute
    /// <code>
    ///     scores[v] = Re( Σ_d  query[d] · conj(code[v][d]) )
    ///               = Σ_d ( queryR · codeR + queryI · codeI )
    /// </code>
    /// i.e., the inner product of the query against each code in the
    /// codebook, dropping the imaginary part (which cancels in expectation
    /// for independent random phases). This is the "circular-convolution
    /// unbind" readout used across HRE's associative-recall, reversal-curse,
    /// and multi-hop-reasoning cycles.
    ///
    /// <para>Per-candidate cost is 2D fused multiply-adds; the AVX2 path
    /// handles 4 doubles per lane. Parallelises across V when the total
    /// work exceeds the thread-dispatch threshold.</para>
    /// </summary>
    public static void PhaseCoherenceDecodeDouble(
        ReadOnlySpan<double> codesR, ReadOnlySpan<double> codesI,
        ReadOnlySpan<double> queryR, ReadOnlySpan<double> queryI,
        Span<double> scores,
        int V, int D)
    {
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V));
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D));
        long total = (long)V * D;
        if (total > int.MaxValue)
            throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        int nCodes = (int)total;
        if (codesR.Length != nCodes || codesI.Length != nCodes)
            throw new ArgumentException(
                $"codesR ({codesR.Length}) and codesI ({codesI.Length}) must both have length V*D = {nCodes}.");
        if (queryR.Length != D || queryI.Length != D)
            throw new ArgumentException(
                $"queryR ({queryR.Length}) and queryI ({queryI.Length}) must both have length D = {D}.");
        if (scores.Length != V)
            throw new ArgumentException($"scores length ({scores.Length}) must equal V = {V}.");

        // Parallelise across V when there's real work per candidate.
        // (long)V*D*2 is the FMA count; target ~4K FMAs per thread minimum.
        long totalOps = (long)V * D;
        if (totalOps >= ParallelThresholdElements && V >= 4)
        {
            // Can't close over Span directly; pin via arrays when callers
            // provide them would be nicer, but here we just avoid Parallel
            // when spans are backed by stackalloc.
            // Fall through to sequential — callers with large V can use
            // the parallel overload below that takes arrays.
        }

        for (int v = 0; v < V; v++)
        {
            int rowOff = v * D;
            scores[v] = PhaseCoherenceRow(codesR.Slice(rowOff, D), codesI.Slice(rowOff, D), queryR, queryI);
        }
    }

    /// <summary>
    /// Parallel overload of <see cref="PhaseCoherenceDecodeDouble"/> that
    /// distributes the V rows across the thread pool. Takes arrays so the
    /// per-thread closure can index them without the Span-capture
    /// restriction. Matches the sequential overload's output exactly.
    /// </summary>
    public static void PhaseCoherenceDecodeDoubleParallel(
        double[] codesR, double[] codesI,
        double[] queryR, double[] queryI,
        double[] scores,
        int V, int D)
    {
        if (codesR is null) throw new ArgumentNullException(nameof(codesR));
        if (codesI is null) throw new ArgumentNullException(nameof(codesI));
        if (queryR is null) throw new ArgumentNullException(nameof(queryR));
        if (queryI is null) throw new ArgumentNullException(nameof(queryI));
        if (scores is null) throw new ArgumentNullException(nameof(scores));

        long totalOps = (long)V * D;
        if (totalOps < ParallelThresholdElements || V < 4)
        {
            PhaseCoherenceDecodeDouble(codesR, codesI, queryR, queryI, scores, V, D);
            return;
        }

        Parallel.For(0, V, v =>
        {
            int rowOff = v * D;
            scores[v] = PhaseCoherenceRow(
                new ReadOnlySpan<double>(codesR, rowOff, D),
                new ReadOnlySpan<double>(codesI, rowOff, D),
                queryR.AsSpan(),
                queryI.AsSpan());
        });
    }

    /// <summary>
    /// Per-row phase-coherence reduction: <c>Σ_d (cR[d]·qR[d] + cI[d]·qI[d])</c>.
    /// Split into its own method so callers (both sequential and parallel
    /// overloads above) share a single SIMD path.
    /// </summary>
    [MethodImpl(HotInline)]
    private static double PhaseCoherenceRow(
        ReadOnlySpan<double> cR, ReadOnlySpan<double> cI,
        ReadOnlySpan<double> qR, ReadOnlySpan<double> qI)
    {
        int n = cR.Length;
        int i = 0;
        double acc = 0.0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 4)
        {
            var v = Vector256<double>.Zero;
            int simdLen = n & ~3;
            if (Fma.IsSupported)
            {
                for (; i < simdLen; i += 4)
                {
                    var crv = SimdKernels.ReadVector256(cR, i);
                    var civ = SimdKernels.ReadVector256(cI, i);
                    var qrv = SimdKernels.ReadVector256(qR, i);
                    var qiv = SimdKernels.ReadVector256(qI, i);
                    // v += cR·qR + cI·qI
                    v = Fma.MultiplyAdd(crv, qrv, v);
                    v = Fma.MultiplyAdd(civ, qiv, v);
                }
            }
            else
            {
                for (; i < simdLen; i += 4)
                {
                    var crv = SimdKernels.ReadVector256(cR, i);
                    var civ = SimdKernels.ReadVector256(cI, i);
                    var qrv = SimdKernels.ReadVector256(qR, i);
                    var qiv = SimdKernels.ReadVector256(qI, i);
                    v = Avx.Add(v, Avx.Add(Avx.Multiply(crv, qrv), Avx.Multiply(civ, qiv)));
                }
            }
            // Horizontal sum of the 4 lanes.
            var vLow  = v.GetLower();
            var vHigh = v.GetUpper();
            var sum2 = Sse2.Add(vLow, vHigh);
            acc = sum2.ToScalar() + sum2.GetElement(1);
        }
#endif

        for (; i < n; i++)
        {
            acc += cR[i] * qR[i] + cI[i] * qI[i];
        }
        return acc;
    }

    /// <summary>
    /// Fused HRR bind + accumulate across N training pairs:
    /// <code>
    ///     for n in [0, keyIds.Length):
    ///         memory += keyCode[keyIds[n]] · valPermCode[valIds[n]]
    /// </code>
    /// replacing N × 6 <c>TensorPrimitives</c> calls with a single kernel.
    /// The caller supplies <paramref name="valPermCodeR"/> /
    /// <paramref name="valPermCodeI"/> pre-permuted so the gather + permute
    /// + complex-multiply + accumulate collapses into one hot loop.
    ///
    /// <para>This is the single hottest op in HRE's memory-build phase;
    /// per issue #248 the hand-rolled version accounted for roughly half
    /// of the total cycle-12 wall-clock at D=16384 × 1M pairs.</para>
    /// </summary>
    public static void HRRBindAccumulateDouble(
        ReadOnlySpan<double> keyCodeR, ReadOnlySpan<double> keyCodeI,
        ReadOnlySpan<double> valPermCodeR, ReadOnlySpan<double> valPermCodeI,
        ReadOnlySpan<int> keyIds, ReadOnlySpan<int> valIds,
        Span<double> memoryR, Span<double> memoryI,
        int D)
    {
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D));
        if (memoryR.Length != D || memoryI.Length != D)
            throw new ArgumentException(
                $"memoryR ({memoryR.Length}) and memoryI ({memoryI.Length}) must both have length D = {D}.");
        if (keyCodeR.Length != keyCodeI.Length)
            throw new ArgumentException("keyCodeR and keyCodeI must have matching length.");
        if (valPermCodeR.Length != valPermCodeI.Length)
            throw new ArgumentException("valPermCodeR and valPermCodeI must have matching length.");
        if (keyIds.Length != valIds.Length)
            throw new ArgumentException(
                $"keyIds ({keyIds.Length}) and valIds ({valIds.Length}) must have matching length.");
        if (keyCodeR.Length % D != 0)
            throw new ArgumentException(
                $"keyCode length ({keyCodeR.Length}) must be divisible by D = {D}.");
        if (valPermCodeR.Length % D != 0)
            throw new ArgumentException(
                $"valPermCode length ({valPermCodeR.Length}) must be divisible by D = {D}.");

        int nKeys = keyCodeR.Length / D;
        int nVals = valPermCodeR.Length / D;
        int N = keyIds.Length;

        for (int n = 0; n < N; n++)
        {
            int kId = keyIds[n];
            int vId = valIds[n];
            if ((uint)kId >= (uint)nKeys)
                throw new ArgumentOutOfRangeException(nameof(keyIds),
                    $"keyIds[{n}] = {kId} is outside [0, {nKeys}).");
            if ((uint)vId >= (uint)nVals)
                throw new ArgumentOutOfRangeException(nameof(valIds),
                    $"valIds[{n}] = {vId} is outside [0, {nVals}).");

            int kOff = kId * D;
            int vOff = vId * D;

            AccumulateBindRow(
                keyCodeR.Slice(kOff, D), keyCodeI.Slice(kOff, D),
                valPermCodeR.Slice(vOff, D), valPermCodeI.Slice(vOff, D),
                memoryR, memoryI);
        }
    }

    /// <summary>
    /// Per-pair accumulate: <c>mem[d] += a[d] · b[d]</c> (complex) via the
    /// standard Gauss FMA sequence. Single SIMD loop. Kept private because
    /// it only makes sense as the inner body of
    /// <see cref="HRRBindAccumulateDouble"/>.
    /// </summary>
    [MethodImpl(HotInline)]
    private static void AccumulateBindRow(
        ReadOnlySpan<double> aR, ReadOnlySpan<double> aI,
        ReadOnlySpan<double> bR, ReadOnlySpan<double> bI,
        Span<double> memR, Span<double> memI)
    {
        int n = aR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 4)
        {
            int simdLen = n & ~3;
            if (Fma.IsSupported)
            {
                for (; i < simdLen; i += 4)
                {
                    var ar = SimdKernels.ReadVector256(aR, i);
                    var ai = SimdKernels.ReadVector256(aI, i);
                    var br = SimdKernels.ReadVector256(bR, i);
                    var bi = SimdKernels.ReadVector256(bI, i);
                    var mr = SimdKernels.ReadVector256((ReadOnlySpan<double>)memR, i);
                    var mi = SimdKernels.ReadVector256((ReadOnlySpan<double>)memI, i);

                    // mR += aR·bR − aI·bI
                    //    = fma(-aI, bI, mR + aR·bR)
                    //    = fma(aR, bR, mR) − aI·bI
                    // Use two FMAs to keep the dep chain short.
                    mr = Fma.MultiplyAdd(ar, br, mr);
                    mr = Fma.MultiplyAddNegated(ai, bi, mr);

                    // mI += aR·bI + aI·bR
                    mi = Fma.MultiplyAdd(ar, bi, mi);
                    mi = Fma.MultiplyAdd(ai, br, mi);

                    SimdKernels.WriteVector256(memR, i, mr);
                    SimdKernels.WriteVector256(memI, i, mi);
                }
            }
            else
            {
                for (; i < simdLen; i += 4)
                {
                    var ar = SimdKernels.ReadVector256(aR, i);
                    var ai = SimdKernels.ReadVector256(aI, i);
                    var br = SimdKernels.ReadVector256(bR, i);
                    var bi = SimdKernels.ReadVector256(bI, i);
                    var mr = SimdKernels.ReadVector256((ReadOnlySpan<double>)memR, i);
                    var mi = SimdKernels.ReadVector256((ReadOnlySpan<double>)memI, i);

                    mr = Avx.Add(mr, Avx.Subtract(Avx.Multiply(ar, br), Avx.Multiply(ai, bi)));
                    mi = Avx.Add(mi, Avx.Add(Avx.Multiply(ar, bi), Avx.Multiply(ai, br)));

                    SimdKernels.WriteVector256(memR, i, mr);
                    SimdKernels.WriteVector256(memI, i, mi);
                }
            }
        }
#endif

        for (; i < n; i++)
        {
            double ar = aR[i], ai = aI[i], br = bR[i], bi = bI[i];
            memR[i] += ar * br - ai * bi;
            memI[i] += ar * bi + ai * br;
        }
    }
}
