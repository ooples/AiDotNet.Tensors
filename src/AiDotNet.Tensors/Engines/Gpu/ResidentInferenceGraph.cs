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
            for (int i = 0; i < s.Length; i++) _engine.RefreshResidentInputInPlace(s[i]);
            _engine.LaunchGpuGraph(_exec);
            return DownloadOutput<T>();
        }

        // WARMUP: a few eager forwards on the stable inputs so lazy allocs / autotune settle.
        _warmup++;
        if (_warmup < _warmupRuns)
            return forward(s);

        // CAPTURE: enter the resident capture path (kept alive for the graph's lifetime), do a
        // pre-residency pass so every op binds its stable resident buffer, then record the forward.
        try
        {
            _scope = _engine.EnterResidentCaptureScope();
            if (_scope is null) { _disabled = true; return forward(s); }

            // Bind each external stable input to a fixed resident buffer so the captured forward reads a
            // stable device address (no re-upload mid-capture). The forward's own intermediates become
            // resident under the capture path.
            for (int i = 0; i < s.Length; i++) _engine.EnsureResidentInput(s[i]);
            _ = forward(s); // pre-residency: bind resident buffers (sync uploads here are fine — not capturing)
            for (int i = 0; i < s.Length; i++) _engine.RefreshResidentInputInPlace(s[i]);

            Tensor<T>? captured = null;
            var exec = _engine.CaptureGpuGraph(() => { captured = forward(s); });
            if (exec != System.IntPtr.Zero && captured is not null)
            {
                _exec = exec;
                _output = captured;
                _outShape = captured.Shape._dims;
                _engine.LaunchGpuGraph(exec); // capture records but does not execute — run once for this call
                return DownloadOutput<T>();
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
            _scope?.Dispose();
            _scope = null;
            _disabled = true;
            return forward(inputs);
        }
    }

    private Tensor<T> DownloadOutput<T>()
    {
        var outT = (Tensor<T>)_output!;
        var data = _engine.DownloadResidentBuffer(outT);
        if (data is not null && _outShape is not null)
            return new Tensor<T>(_outShape, new Vector<T>((T[])(object)data));
        return outT; // fallback: the tensor's own (possibly stale) data
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
