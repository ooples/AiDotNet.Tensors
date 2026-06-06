using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Mixed-dtype reverse-mode autograd over a paired FP32 / FP16 tape — the industry-standard AMP
/// graph (docs/fp16-activation-storage-design.md, Tensors #555 Phase 1). Activations on autocast-
/// eligible ops live as <see cref="Tensor{Half}"/> nodes recorded on an inner FP16
/// <see cref="GradientTape{T}"/>; everything else (params, norms, softmax, loss) stays FP32 on a
/// second inner tape. The two tapes are joined at differentiable cast boundaries by
/// <see cref="MixedPrecisionCast"/>: a cast's local Jacobian is the identity, so its backward simply
/// moves the gradient across the dtype boundary.
///
/// <para><b>Why a fixpoint and not one pass.</b> A single-type <see cref="GradientTape{T}"/> cannot
/// record a cast edge — the FP32 tape sees an up-cast output as a parentless leaf and the FP16 tape
/// sees a down-cast output as a leaf. So the backward is driven as a Gauss-Seidel sweep over the cast
/// boundaries: run the FP32 backward (ones at loss + any grads bridged in from FP16 at down-cast
/// inputs), bridge the FP32 grads at up-cast outputs down into FP16 seeds, run the FP16 backward,
/// bridge the FP16 grads at down-cast outputs up into FP32 seeds, repeat. Each boundary tensor's grad
/// depends only on strictly-downstream boundaries, so the sweep reaches the exact reverse-mode result
/// in at most (#cast boundaries + 1) rounds. Each round recomputes from a fresh grad dict given the
/// current boundary seeds, so there is no double-counting. A down-cast input's only consumer is the
/// cast into FP16, so seeding it with the bridged grad is its complete gradient (plus any FP32-side
/// consumers, which the walk accumulates).</para>
///
/// <para>The default all-FP32 training path never constructs this type, so it is byte-identical; this
/// is opt-in behind autocast. Gated by MixedPrecisionTapeGradientCheckTests (finite-difference vs an
/// all-FP32 reference, within FP16 tolerance) on a multi-layer interleaved chain.</para>
/// </summary>
internal sealed class MixedPrecisionTape : IDisposable
{
    private readonly GradientTape<float> _fp32;
    private readonly GradientTape<Half> _fp16;

    // Cast boundary edges, captured at forward time.
    // up-cast:   FP16 activation -> FP32 (e.g. matmul output consumed by an FP32 norm).
    private readonly List<(Tensor<Half> HalfInput, Tensor<float> FloatOutput)> _upCasts = new();
    // down-cast: FP32 activation -> FP16 (e.g. an FP32 norm output feeding an FP16 matmul).
    private readonly List<(Tensor<float> FloatInput, Tensor<Half> HalfOutput)> _downCasts = new();

    private readonly IEngine _engine;
    private bool _disposed;

    public MixedPrecisionTape(GradientTapeOptions? options = null)
    {
        _engine = AiDotNetEngine.Current;
        // Order matters only for disposal symmetry; both set their own ThreadStatic Current<T>.
        _fp32 = new GradientTape<float>(options);
        _fp16 = new GradientTape<Half>(options);
    }

    public GradientTape<float> Fp32Tape => _fp32;
    public GradientTape<Half> Fp16Tape => _fp16;

    /// <summary>Down-cast an FP32 activation into FP16 storage and record the bridge edge.</summary>
    public Tensor<Half> CastToFp16(Tensor<float> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        var h = MixedPrecisionCast.CastToFp16(x);
        _downCasts.Add((x, h));
        return h;
    }

    /// <summary>Up-cast an FP16 activation to FP32 (e.g. at a norm/softmax boundary) and record the edge.</summary>
    public Tensor<float> CastToFp32(Tensor<Half> h)
    {
        if (h is null) throw new ArgumentNullException(nameof(h));
        var f = MixedPrecisionCast.CastToFp32(h);
        _upCasts.Add((h, f));
        return f;
    }

    /// <summary>Result of a mixed-precision backward: FP32 grads (inputs, FP32 params) and FP16 grads (FP16 params).</summary>
    public sealed class MixedGrads
    {
        public Dictionary<Tensor<float>, Tensor<float>> Fp32 { get; }
        public Dictionary<Tensor<Half>, Tensor<Half>> Fp16 { get; }
        public MixedGrads(Dictionary<Tensor<float>, Tensor<float>> fp32, Dictionary<Tensor<Half>, Tensor<Half>> fp16)
        {
            Fp32 = fp32;
            Fp16 = fp16;
        }
    }

    /// <summary>
    /// Reverse-mode gradients of <paramref name="loss"/> (FP32 scalar) across the mixed-dtype graph.
    /// Returns the last sweep's FP32 and FP16 grad dictionaries (exact reverse-mode values).
    /// </summary>
    public MixedGrads ComputeGradients(Tensor<float> loss)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MixedPrecisionTape));
        if (loss is null) throw new ArgumentNullException(nameof(loss));

        // Ones seed at the loss (FP32), constructed once.
        var onesData = new float[loss.Length];
        for (int i = 0; i < onesData.Length; i++) onesData[i] = 1f;
        var lossSeed = new Tensor<float>(onesData, loss._shape);

        // Boundary seeds carried across sweeps.
        // bridgedToFp32: grad bridged FROM FP16 INTO the FP32 tape at each down-cast's FP32 input.
        var bridgedToFp32 = new Dictionary<Tensor<float>, Tensor<float>>(
            ReferenceEqualityComparer<Tensor<float>>.Instance);
        // bridgedToFp16: grad bridged FROM FP32 INTO the FP16 tape at each up-cast's FP16 input.
        var bridgedToFp16 = new Dictionary<Tensor<Half>, Tensor<Half>>(
            ReferenceEqualityComparer<Tensor<Half>>.Instance);

        Dictionary<Tensor<float>, Tensor<float>> fp32Grads = new(ReferenceEqualityComparer<Tensor<float>>.Instance);
        Dictionary<Tensor<Half>, Tensor<Half>> fp16Grads = new(ReferenceEqualityComparer<Tensor<Half>>.Instance);

        int maxRounds = _upCasts.Count + _downCasts.Count + 2;
        int prevToFp32Keys = -1, prevToFp16Keys = -1;

        for (int round = 0; round < maxRounds; round++)
        {
            // ── FP32 backward: ones at loss + everything bridged up from FP16 so far. ──
            var seeds32 = new List<KeyValuePair<Tensor<float>, Tensor<float>>>(1 + bridgedToFp32.Count)
            {
                new(loss, lossSeed)
            };
            foreach (var kv in bridgedToFp32) seeds32.Add(kv);

            if (_fp32.EntryCount > 0)
                fp32Grads = _fp32.ComputeGradients(loss, sources: null, createGraph: false, seedOverride: seeds32);

            // Bridge FP32 -> FP16 at up-cast outputs: grad at the FP32 cast output, cast DOWN to FP16,
            // accumulated at the FP16 cast input (several up-casts may share one FP16 input).
            var nextToFp16 = new Dictionary<Tensor<Half>, Tensor<Half>>(
                ReferenceEqualityComparer<Tensor<Half>>.Instance);
            foreach (var (halfIn, floatOut) in _upCasts)
            {
                if (!fp32Grads.TryGetValue(floatOut, out var gOut)) continue;
                var bridged = MixedPrecisionCast.CastToFp32Backward(gOut); // FP32 grad -> FP16 grad
                nextToFp16[halfIn] = nextToFp16.TryGetValue(halfIn, out var ex)
                    ? _engine.TensorAdd(ex, bridged)
                    : bridged;
            }
            bridgedToFp16 = nextToFp16;

            // ── FP16 backward: seeded at the up-cast FP16 inputs. ──
            if (_fp16.EntryCount > 0 && bridgedToFp16.Count > 0)
            {
                var seeds16 = new List<KeyValuePair<Tensor<Half>, Tensor<Half>>>(bridgedToFp16.Count);
                foreach (var kv in bridgedToFp16) seeds16.Add(kv);
                // `loss` arg is unused on the seeded slow path (guards bypass it); pass a recorded
                // FP16 tensor as a harmless placeholder.
                var placeholder = seeds16[0].Key;
                fp16Grads = _fp16.ComputeGradients(placeholder, sources: null, createGraph: false, seedOverride: seeds16);
            }

            // Bridge FP16 -> FP32 at down-cast outputs: grad at the FP16 cast output, cast UP to FP32,
            // accumulated at the FP32 cast input.
            var nextToFp32 = new Dictionary<Tensor<float>, Tensor<float>>(
                ReferenceEqualityComparer<Tensor<float>>.Instance);
            foreach (var (floatIn, halfOut) in _downCasts)
            {
                if (!fp16Grads.TryGetValue(halfOut, out var gHalf)) continue;
                var bridged = MixedPrecisionCast.CastToFp16Backward(gHalf); // FP16 grad -> FP32 grad
                nextToFp32[floatIn] = nextToFp32.TryGetValue(floatIn, out var ex)
                    ? _engine.TensorAdd(ex, bridged)
                    : bridged;
            }
            bridgedToFp32 = nextToFp32;

            // Convergence: once the set of seeded boundaries stops growing, one more round has already
            // refreshed values from fully-seeded downstream, so the result is exact.
            if (bridgedToFp32.Count == prevToFp32Keys && bridgedToFp16.Count == prevToFp16Keys && round > 0)
                break;
            prevToFp32Keys = bridgedToFp32.Count;
            prevToFp16Keys = bridgedToFp16.Count;
        }

        return new MixedGrads(fp32Grads, fp16Grads);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Dispose in reverse construction order so each restores the correct ThreadStatic Current<T>.
        _fp16.Dispose();
        _fp32.Dispose();
    }
}
