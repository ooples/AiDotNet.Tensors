using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// #1650 — captures a model's EAGER GPU forward into a CUDA graph and replays it per call, the
/// inference analog of the #638 training-step capture. Drives the resident capture path on
/// <see cref="DirectGpuTensorEngine"/> so the forward runs host-round-trip-free (capturable), then
/// collapses its hundreds of per-kernel launches into one <c>cuGraphLaunch</c> — the TensorRT-LLM /
/// vLLM decode-graph lever, which dominates latency at the low batch a diffusion denoising step runs.
/// <para>
/// The caller supplies, per step, the FRESH input tensors and a forward closure. This class keeps
/// STABLE copies of the inputs (fixed device addresses), copies the fresh data in, runs a short eager
/// warmup, captures the forward once, and thereafter replays — refreshing the stable input buffers in
/// place and forcing a fresh download of the (stable) output. If the forward isn't yet capture-pure
/// (some op still does a host round-trip), capture fails and it transparently falls back to eager, so
/// correctness is never affected. The forward MUST be run on the tensors this class passes it (the
/// stable copies), never the originals.
/// </para>
/// </summary>
public sealed class ResidentInferenceGraph : System.IDisposable
{
    private readonly DirectGpuTensorEngine _engine;
    private readonly int _warmupRuns;
    // Serializes Forward on a single graph instance: _stableInputs/_exec/_output and the stable device
    // buffers are shared mutable state, so two concurrent calls could interleave input copy/refresh/launch
    // and replay one request with another's inputs.
    private readonly object _gate = new object();
    private System.IntPtr _exec;
    private object? _stableInputs;   // Tensor<T>[]
    private object? _output;         // Tensor<T> captured output (its resident buffer is what the graph writes)
    private int[]? _outShape;
    private int _warmup;
    private bool _disabled;
    private System.IDisposable? _scope;

    /// <param name="engine">The CUDA-backed engine to capture on.</param>
    /// <param name="warmupRuns">Eager forwards before capture (lazy allocs / autotune settle). Default 2.</param>
    public ResidentInferenceGraph(DirectGpuTensorEngine engine, int warmupRuns = 2)
    {
        _engine = engine ?? throw new System.ArgumentNullException(nameof(engine));
        // Benchmarking knob: AIDOTNET_INFGRAPH_WARMUP overrides the eager-warmup count so a timing run can
        // collect many eager-forward samples before capture (see AIDOTNET_INFGRAPH_TIMING).
        var wenv = System.Environment.GetEnvironmentVariable("AIDOTNET_INFGRAPH_WARMUP");
        if (wenv is not null && int.TryParse(wenv, out var w) && w >= 1) warmupRuns = w;
        _warmupRuns = warmupRuns < 1 ? 1 : warmupRuns;
    }

    /// <summary>True once the forward has been captured into a replayable graph.</summary>
    public bool HasGraph => _exec != System.IntPtr.Zero;

    /// <summary>True once capture was abandoned (a non-capturable op) — the forward runs eager from here.</summary>
    public bool Disabled => _disabled;

    /// <summary>
    /// Runs the forward for this step. Returns the forward's output. On the capture path, <paramref name="forward"/>
    /// is invoked with this class's STABLE input copies (run the model on those, not the originals); on a replay
    /// it is not invoked at all (the graph runs) and the captured output is downloaded fresh.
    /// </summary>
    public Tensor<T> Forward<T>(Tensor<T>[] inputs, System.Func<Tensor<T>[], Tensor<T>> forward)
    {
        if (inputs is null) throw new System.ArgumentNullException(nameof(inputs));
        if (forward is null) throw new System.ArgumentNullException(nameof(forward));
        // Validate every element up front — the diagnostic loop and shape handling below dereference
        // inputs[i], so a null slot must surface as a clear contract error, not an NRE deep inside.
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is null) throw new System.ArgumentException($"inputs[{i}] is null.", nameof(inputs));

        // Serialize the whole forward/capture/replay on this instance (shared mutable graph state — see _gate).
        lock (_gate)
        {
        if (System.Environment.GetEnvironmentVariable("AIDOTNET_INFGRAPH_DIAG") == "1")
        {
            var sh = new System.Text.StringBuilder();
            for (int i = 0; i < inputs.Length; i++) sh.Append($"[{string.Join("x", inputs[i].Shape._dims)}]");
            var stb = _stableInputs as Tensor<T>[];
            var ss = new System.Text.StringBuilder();
            if (stb != null) for (int i = 0; i < stb.Length; i++) ss.Append($"[{string.Join("x", stb[i].Shape._dims)}]");
            try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_infgraph_diag.txt"),
                $"RIG.Forward<{typeof(T).Name}>: disabled={_disabled} warmup={_warmup} hasGraph={_exec != System.IntPtr.Zero} inShapes={sh} stableShapes={ss}\n"); } catch { }
        }

        if (_disabled || typeof(T) != typeof(float) || !_engine.SupportsInferenceGraphCapture)
            return forward(inputs); // eager passthrough (no capture possible / already abandoned)

        // Stable input copies — fixed device addresses the captured graph reads from.
        var stable = (Tensor<T>[]?)_stableInputs;
        // (Re)build the stable input copies on the first call OR when the shape changes — e.g. the first
        // PredictNoise is an internal lazy-shape-resolution probe at a smaller size ([1,3,16,16]) before the
        // real denoising steps ([1,3,32,32]); lock onto the steady-state shape rather than the probe. A shape
        // change discards any graph captured for the old shape and restarts the warmup.
        bool needRebuild = stable is null || stable.Length != inputs.Length;
        if (!needRebuild)
        {
            for (int i = 0; i < inputs.Length; i++)
                if (!ShapeMatches(inputs[i], stable![i])) { needRebuild = true; break; }
        }
        if (needRebuild)
        {
            if (_exec != System.IntPtr.Zero) { _engine.DestroyGpuGraph(_exec); _exec = System.IntPtr.Zero; }
            _scope?.Dispose();
            _scope = null;
            stable = new Tensor<T>[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                stable[i] = new Tensor<T>(inputs[i].Shape._dims);
            _stableInputs = stable;
            _warmup = 0;
        }
        var s = stable!; // non-null from here: either pre-existing or rebuilt above
        for (int i = 0; i < inputs.Length; i++)
            inputs[i].AsSpan().CopyTo(s[i].AsWritableSpan());

        // REPLAY: refresh stable input buffers in place, one cuGraphLaunch, fresh download of the output.
        if (_exec != System.IntPtr.Zero)
        {
          try
          {
            bool timing = System.Environment.GetEnvironmentVariable("AIDOTNET_INFGRAPH_TIMING") == "1";
            bool breakdown = timing && System.Environment.GetEnvironmentVariable("AIDOTNET_INFGRAPH_TIMING_BREAKDOWN") == "1";
            double freq = System.Diagnostics.Stopwatch.Frequency / 1_000_000.0;
            long t0 = timing ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            for (int i = 0; i < s.Length; i++) _engine.RefreshResidentInputInPlace(s[i]);
            long tRefresh = breakdown ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            _engine.LaunchGpuGraph(_exec);
            if (breakdown) _engine.SynchronizeStream(); // block so the next stamp isolates GPU-compute time
            long tCompute = breakdown ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            var outR = DownloadOutput<T>();
            if (timing)
            {
                double us = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / freq;
                string line = breakdown
                    ? $"REPLAY {us:F1} us  [refresh(HtoD)={(tRefresh - t0) / freq:F1} compute={(tCompute - tRefresh) / freq:F1} download(DtoH)={(System.Diagnostics.Stopwatch.GetTimestamp() - tCompute) / freq:F1}]\n"
                    : $"REPLAY {us:F1} us\n";
                try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_infgraph_timing.txt"), line); } catch { }
            }
            if (System.Environment.GetEnvironmentVariable("AIDOTNET_REPLAY_VERIFY") == "1")
            {
                double inSum = 0; var isp = s[0].AsSpan();
                for (int i = 0; i < isp.Length; i++) inSum += System.Convert.ToDouble(isp[i]);
                double tSum = 0; if (s.Length > 1) { var tsp = s[1].AsSpan(); for (int i = 0; i < tsp.Length; i++) tSum += System.Convert.ToDouble(tsp[i]); }
                double outSum = 0; var osp = outR.AsSpan();
                for (int i = 0; i < osp.Length; i++) outSum += System.Convert.ToDouble(osp[i]);
                try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_replay_verify.txt"),
                    $"REPLAY inSum(s0)={inSum:F4} timeSum(s1)={tSum:F4} -> outSum={outSum:F4}\n"); } catch { }
            }
            return outR;
          }
          catch (System.InvalidOperationException)
          {
            // Resident replay/download failed — fail closed: discard the graph and run eager so the
            // caller never receives a stale (frozen-buffer) result.
            if (_exec != System.IntPtr.Zero) { _engine.DestroyGpuGraph(_exec); _exec = System.IntPtr.Zero; }
            _output = null;
            _disabled = true;
            return forward(s);
          }
        }

        // WARMUP: a few eager forwards on the stable inputs so lazy allocs / autotune settle.
        // <= so warmupRuns eager forwards run before capture (the count is incremented first, so a
        // strict < would run only warmupRuns-1 — and zero at the clamped minimum of 1).
        _warmup++;
        if (_warmup <= _warmupRuns)
        {
            if (System.Environment.GetEnvironmentVariable("AIDOTNET_INFGRAPH_TIMING") == "1")
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                var w = forward(s);
                _ = w.AsSpan(); // force materialization so timing includes the eager download
                double us = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1_000_000.0 / System.Diagnostics.Stopwatch.Frequency;
                try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_infgraph_timing.txt"), $"EAGER {us:F1} us\n"); } catch { }
                return w;
            }
            return forward(s);
        }

        // CAPTURE: enter the resident capture path (kept alive for the graph's lifetime), do a
        // pre-residency pass so every op binds its stable resident buffer, then record the forward.
        try
        {
            // RESIDENT-VS-CPU VERIFY (AIDOTNET_RESIDENT_VERIFY=1): run the SAME input eagerly first (outside
            // the scope → ResidentStepActive off → CPU/host reference), then compare the resident pre-residency
            // forward to it. Localizes a wrong resident op immediately, per build, with NO capture needed.
            float[]? verifyRef = null;
            bool verify = System.Environment.GetEnvironmentVariable("AIDOTNET_RESIDENT_VERIFY") == "1";
            if (verify)
            {
                var refOut = forward(s);
                verifyRef = refOut.AsSpan().ToArray() as float[];
            }

            _scope = _engine.EnterResidentCaptureScope();
            if (_scope is null) { _disabled = true; return forward(s); }

            // Bind each external stable input to a fixed resident buffer so the captured forward reads a
            // stable device address (no re-upload mid-capture). The forward's own intermediates become
            // resident under the capture path.
            for (int i = 0; i < s.Length; i++) _engine.EnsureResidentInput(s[i]);
            var preResOut = forward(s); // pre-residency: bind resident buffers (sync uploads here are fine — not capturing)
            // Allocate the STABLE output buffer (sync, pre-capture) the captured graph's final op will copy into,
            // so the externally-read output address survives AUTO_FREE relaunch (else replays read stale output).
            _engine.PrepareStableCaptureOutput(preResOut);
            if (verify && verifyRef is not null)
            {
                var got = preResOut.AsSpan();
                double maxd = 0; int n = System.Math.Min(got.Length, verifyRef.Length);
                for (int i = 0; i < n; i++) maxd = System.Math.Max(maxd, System.Math.Abs(System.Convert.ToDouble(got[i]) - verifyRef[i]));
                try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_resident_verify.txt"),
                    $"resident-vs-cpu maxAbsDiff={maxd:E4} n={got.Length}\n"); } catch { }
            }
            for (int i = 0; i < s.Length; i++) _engine.RefreshResidentInputInPlace(s[i]);

            // RESIDENCY AUDIT (AIDOTNET_RESIDENT_AUDIT=1): run one more resident pass with the audit log
            // cleared first, so only GENUINELY non-resident ops (sync HtoD/DtoH every pass — not one-time
            // cache fills) are recorded. Lists the whole residency work-list in ONE run; then bail (no capture).
            if (System.Environment.GetEnvironmentVariable("AIDOTNET_RESIDENT_AUDIT") == "1")
            {
                try { System.IO.File.Delete(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_resident_audit.txt")); } catch { }
                var auditOut = forward(s);
                _scope.Dispose(); _scope = null; _disabled = true;
                return auditOut;
            }

            Tensor<T>? captured = null;
            // Capture the forward AND a final copy of its output into the stable buffer — both become graph nodes,
            // so replay recomputes into the stable address and DownloadOutput reads fresh data every launch.
            var exec = _engine.CaptureGpuGraph(() => { var o = forward(s); captured = _engine.WriteOutputToStableCapture(o); });
            if (exec != System.IntPtr.Zero && captured is not null)
            {
                _exec = exec;
                _output = captured;
                _outShape = captured.Shape._dims;
                _engine.LaunchGpuGraph(exec); // capture records but does not execute — run once for this call
                var out1 = DownloadOutput<T>();

                // REPLAY DETERMINISM (AIDOTNET_REPLAY_VERIFY=1): relaunch the SAME graph on the SAME (unrefreshed)
                // input buffers and compare. Identical ⇒ replay is bit-stable (any later run-vs-run divergence is
                // from input refresh, not the graph). Nonzero ⇒ the graph replay itself corrupts (AUTO_FREE
                // realloc moving a buffer's VA, or a captured pointer into a non-graph-managed buffer).
                if (System.Environment.GetEnvironmentVariable("AIDOTNET_REPLAY_VERIFY") == "1")
                {
                    _engine.LaunchGpuGraph(exec);
                    var out2 = DownloadOutput<T>();
                    var a = out1.AsSpan(); var b = out2.AsSpan();
                    double maxd = 0, a0 = 0, b0 = 0; int n = System.Math.Min(a.Length, b.Length);
                    for (int i = 0; i < n; i++)
                    {
                        double ai = System.Convert.ToDouble(a[i]), bi = System.Convert.ToDouble(b[i]);
                        if (i == 0) { a0 = ai; b0 = bi; }
                        maxd = System.Math.Max(maxd, System.Math.Abs(ai - bi));
                    }
                    // Does the graph actually CONSUME a refreshed input? Mutate s[0], refresh, relaunch.
                    // If the output is unchanged, RefreshResidentInputInPlace isn't reaching the buffer the
                    // graph's first op reads (the run-vs-run divergence then comes entirely from this).
                    var orig = s[0].AsSpan().ToArray();
                    var w = s[0].AsWritableSpan();
                    var zeroT = (T)(object)0f; // typeof(T)==float guaranteed above
                    for (int i = 0; i < w.Length; i++) w[i] = zeroT;
                    _engine.RefreshResidentInputInPlace(s[0]);
                    _engine.LaunchGpuGraph(exec);
                    var out3 = DownloadOutput<T>();
                    var c = out3.AsSpan();
                    double mutd = 0, c0 = 0; int n3 = System.Math.Min(a.Length, c.Length);
                    for (int i = 0; i < n3; i++)
                    {
                        double ci = System.Convert.ToDouble(c[i]);
                        if (i == 0) c0 = ci;
                        mutd = System.Math.Max(mutd, System.Math.Abs(System.Convert.ToDouble(a[i]) - ci));
                    }
                    orig.AsSpan().CopyTo(s[0].AsWritableSpan()); // restore
                    _engine.RefreshResidentInputInPlace(s[0]);
                    _engine.LaunchGpuGraph(exec);
                    out1 = DownloadOutput<T>();
                    try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_replay_verify.txt"),
                        $"relaunch-same-input maxAbsDiff={maxd:E4} n={a.Length} launch1[0]={a0:F4} launch2[0]={b0:F4} | zeroed-input maxAbsDiffVsOrig={mutd:E4} zeroOut[0]={c0:F4} (≈0 ⇒ refresh INEFFECTIVE)\n"); } catch { }
                }
                return out1;
            }

            // A non-capturable op remains → abandon capture, resume eviction, run eager from here.
            _scope.Dispose();
            _scope = null;
            _disabled = true;
            return forward(s);
        }
        catch (System.Exception ex)
        {
            if (System.Environment.GetEnvironmentVariable("AIDOTNET_GRAPH_CAPTURE_DEBUG") == "1")
            {
                try { System.IO.File.AppendAllText(
                    System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_graphcapture_diag.txt"),
                    $"[RESIDENT-INF-GRAPH] capture threw: {ex.GetType().Name}: {ex.Message} | {ex.StackTrace?.Replace("\r", "").Replace("\n", " >> ")}\n"); } catch { }
            }
            // A launch/download can throw after _exec was assigned (Lines above). Destroy the partially
            // captured graph so HasGraph reports false and the handle is not leaked until Dispose.
            if (_exec != System.IntPtr.Zero) { _engine.DestroyGpuGraph(_exec); _exec = System.IntPtr.Zero; }
            _output = null;
            _scope?.Dispose();
            _scope = null;
            _disabled = true;
            return forward(inputs);
        }
        } // lock (_gate)
    }


    private Tensor<T> DownloadOutput<T>()
    {
        if (_output is not Tensor<T> outT)
            throw new System.InvalidOperationException("Resident graph output is not available for download.");
        var data = _engine.DownloadResidentBuffer(outT);
        if (data is null || _outShape is null)
            // Fail closed: the graph wrote the resident buffer, not necessarily outT's host data, so
            // returning outT would surface a stale (frozen-buffer) result. Throwing lets the caller
            // disable capture and fall back to eager — never a stale inference result.
            throw new System.InvalidOperationException("Resident output download failed; refusing to return stale inference result.");
        return new Tensor<T>(_outShape, new Vector<T>((T[])(object)data));
    }

    private static bool ShapeMatches<T>(Tensor<T> a, Tensor<T> b)
    {
        var ad = a.Shape._dims; var bd = b.Shape._dims;
        if (ad.Length != bd.Length) return false;
        for (int i = 0; i < ad.Length; i++) if (ad[i] != bd[i]) return false;
        return true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_exec != System.IntPtr.Zero) { _engine.DestroyGpuGraph(_exec); _exec = System.IntPtr.Zero; }
        _scope?.Dispose();
        _scope = null;
    }
}
