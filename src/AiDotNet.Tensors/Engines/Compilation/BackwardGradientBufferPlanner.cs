namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Liveness-based buffer planner for the compiled training plan's backward
/// gradient buffers (#1624 prototype).
///
/// <para><b>The problem.</b> <see cref="CompiledTrainingPlan{T}"/> currently
/// pre-allocates ONE gradient buffer per traced tensor (every forward step's
/// output and input, plus every parameter) and holds them ALL resident for the
/// plan's lifetime — see the <c>allGrads</c> loop in the builder. For a deep
/// model that is the entire backward's intermediate-gradient working set held at
/// once: a 14-layer transformer measured ~1,450 weight-shaped buffers (~3.4 GB)
/// rooted by the plan, which is what OOMs the #1624 SimCSE repro under a 16 GB
/// heap.</para>
///
/// <para><b>The bound.</b> In the reverse (backward) walk, an intermediate
/// tensor's gradient buffer is only LIVE from the first backward step that
/// writes it (its earliest consumer) until the backward step that reads it (its
/// own producer). Two intermediate gradients with DISJOINT live intervals and
/// the SAME shape can share one physical buffer. This is classic linear-scan
/// register allocation: the resident set shrinks from "every traced tensor" to
/// "the peak simultaneously-live frontier", which for a feed-forward stack is a
/// few layers' worth rather than all of them.</para>
///
/// <para><b>Scope.</b> This planner is a pure, side-effect-free analysis. It
/// computes the assignment + the achievable byte bound so the win can be
/// measured before the (correctness-sensitive) buffer sharing is wired into the
/// hot <c>Step()</c> path. Parameters — and anything else read AFTER backward by
/// the optimizer — are marked persistent and always get a dedicated buffer.</para>
/// </summary>
internal static class BackwardGradientBufferPlanner
{
    /// <summary>One backward step's data dependency, expressed over tensor ids.
    /// Steps are given in FORWARD order; the backward walk is their reverse.</summary>
    internal readonly struct GradStep
    {
        /// <summary>Tensor id of this step's output (whose gradient the backward READS).</summary>
        public readonly int OutputId;
        /// <summary>Tensor ids of this step's inputs (whose gradients the backward WRITES/accumulates).</summary>
        public readonly int[] InputIds;

        public GradStep(int outputId, int[] inputIds)
        {
            OutputId = outputId;
            InputIds = inputIds;
        }
    }

    internal sealed class PlanResult
    {
        /// <summary>tensorId -&gt; physical buffer index. Persistent tensors (parameters,
        /// loss seed, etc.) and tensors with no backward liveness get a unique index;
        /// poolable intermediates may share an index with disjoint-lifetime peers.</summary>
        public int[] BufferOfTensor { get; }
        /// <summary>Per physical buffer, the element count it must hold (max over its tenants —
        /// always exact since only same-shape tensors share).</summary>
        public int[] BufferElemCount { get; }
        /// <summary>Number of distinct physical buffers after pooling.</summary>
        public int PhysicalBufferCount => BufferElemCount.Length;
        /// <summary>Naive bytes: one buffer per traced tensor (the current behaviour).</summary>
        public long NaiveBytes { get; }
        /// <summary>Pooled bytes: distinct physical buffers only.</summary>
        public long PooledBytes { get; }
        /// <summary>
        /// Re-zero schedule, indexed by backward POSITION (0 = the backward step of
        /// the LAST forward step). <c>ReZeroAtPosition[k]</c> lists the physical
        /// buffer indices that must be cleared to zero BEFORE the backward action at
        /// position k runs, because a buffer is being handed from a previous tenant
        /// to a NEW (non-first) tenant whose gradient accumulation begins at k. The
        /// first tenant of every buffer is covered by the step-start clear, so it is
        /// NOT listed here. Null entries mean "nothing to re-zero at this position".
        /// </summary>
        public int[][] ReZeroAtPosition { get; }

        public PlanResult(int[] bufferOfTensor, int[] bufferElemCount, long naiveBytes,
            long pooledBytes, int[][] reZeroAtPosition)
        {
            BufferOfTensor = bufferOfTensor;
            BufferElemCount = bufferElemCount;
            NaiveBytes = naiveBytes;
            PooledBytes = pooledBytes;
            ReZeroAtPosition = reZeroAtPosition;
        }
    }

    /// <summary>
    /// One backward ACTION's gradient data-dependency, expressed over tensor ids,
    /// in the order the actions actually execute. Unlike <see cref="GradStep"/>
    /// (one entry per forward step, reverse-walked positionally), this models the
    /// FINAL backward action stream after fusion/analytic/skip transforms — where a
    /// single fused action covers a contiguous range of forward steps and runs as
    /// one atomic unit. Liveness is computed in ACTION-index units so a buffer is
    /// only shared across an action BOUNDARY (never within a fused action, whose
    /// internal grad hand-offs cannot be re-zeroed) and the re-zero schedule maps
    /// directly onto backward action indices.
    /// </summary>
    internal readonly struct GradAction
    {
        /// <summary>Tensor ids whose gradient this action READS (the output grads of the steps it covers).</summary>
        public readonly int[] ReadIds;
        /// <summary>Tensor ids whose gradient this action WRITES/accumulates (the input grads of the steps it covers).</summary>
        public readonly int[] WriteIds;

        public GradAction(int[] readIds, int[] writeIds)
        {
            ReadIds = readIds;
            WriteIds = writeIds;
        }
    }

    /// <summary>
    /// Liveness-based physical-buffer assignment computed over the FINAL backward
    /// ACTION stream (fusion-compatible). Each action is treated atomically: a
    /// tensor's gradient buffer is live from the first action that writes it to the
    /// last action that touches (reads or writes) it, in action-index units. Two
    /// same-shape tensors with disjoint action intervals share one physical buffer;
    /// the returned re-zero schedule is indexed by backward ACTION index.
    /// <para>Because intervals are in action units, two tensors that share a buffer
    /// are guaranteed to live in DIFFERENT actions, so the hand-off (re-zero) always
    /// falls on an action boundary that <c>Step()</c> can inject a clear before —
    /// this is what makes pooling correct under fusion (where a single action runs
    /// several forward steps' backward internally).</para>
    /// </summary>
    /// <param name="tensorCount">Number of distinct traced tensors (ids 0..tensorCount-1).</param>
    /// <param name="tensorElemCount">Per-tensor gradient element count.</param>
    /// <param name="tensorShapeKey">Per-tensor shape key; only equal keys may share a buffer.</param>
    /// <param name="actions">The backward actions in execution order, with per-action read/write tensor ids.</param>
    /// <param name="isPersistent">Per-tensor: true =&gt; never pooled (parameters / read after backward).</param>
    /// <param name="elementBytes">Bytes per gradient element (4 for float, 8 for double).</param>
    internal static PlanResult PlanOverActions(
        int tensorCount,
        int[] tensorElemCount,
        long[] tensorShapeKey,
        System.Collections.Generic.IReadOnlyList<GradAction> actions,
        bool[] isPersistent,
        int elementBytes)
    {
        int m = actions.Count;
        const int NONE = int.MaxValue;
        var firstWrite = new int[tensorCount];
        var lastUse = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++) { firstWrite[i] = NONE; lastUse[i] = -1; }

        for (int a = 0; a < m; a++)
        {
            var act = actions[a];
            var reads = act.ReadIds;
            if (reads != null)
                for (int j = 0; j < reads.Length; j++)
                {
                    int t = reads[j];
                    if (t < 0) continue;
                    if (a > lastUse[t]) lastUse[t] = a;
                }
            var writes = act.WriteIds;
            if (writes != null)
                for (int j = 0; j < writes.Length; j++)
                {
                    int t = writes[j];
                    if (t < 0) continue;
                    if (a < firstWrite[t]) firstWrite[t] = a;
                    if (a > lastUse[t]) lastUse[t] = a;
                }
        }

        return AssignBuffers(tensorCount, tensorElemCount, tensorShapeKey, isPersistent,
            elementBytes, firstWrite, lastUse, NONE, m);
    }

    /// <summary>
    /// Computes a liveness-based physical-buffer assignment for the backward
    /// gradient buffers.
    /// </summary>
    /// <param name="tensorCount">Number of distinct traced tensors (ids 0..tensorCount-1).</param>
    /// <param name="tensorElemCount">Per-tensor gradient element count.</param>
    /// <param name="tensorShapeKey">Per-tensor shape key; only equal keys may share a buffer.</param>
    /// <param name="forwardOrderSteps">Backward dependency per forward step, in forward order.</param>
    /// <param name="isPersistent">Per-tensor: true =&gt; never pooled (parameters / read after backward).</param>
    /// <param name="elementBytes">Bytes per gradient element (4 for float, 8 for double).</param>
    internal static PlanResult Plan(
        int tensorCount,
        int[] tensorElemCount,
        long[] tensorShapeKey,
        System.Collections.Generic.IReadOnlyList<GradStep> forwardOrderSteps,
        bool[] isPersistent,
        int elementBytes)
    {
        int n = forwardOrderSteps.Count;

        // Backward execution position of forward step s is (n-1-s): the LAST
        // forward step runs its backward FIRST. We compute, per tensor, the
        // [firstWrite, lastRead] interval in backward-position units.
        //
        //   firstWrite(X) = earliest backward step that ACCUMULATES into grad(X)
        //                 = the backward of X's latest-in-forward consumer
        //   lastRead(X)   = backward step that READS grad(X) to propagate it
        //                 = the backward of X's producer step
        //
        // A tensor that is never an input (no consumer) is never written by the
        // backward; a tensor that is never an output (a leaf/parameter) is read
        // only by the optimizer. Both are handled by the persistence / no-interval
        // guards below.
        const int NONE = int.MaxValue;
        var firstWrite = new int[tensorCount];
        var lastRead = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++) { firstWrite[i] = NONE; lastRead[i] = -1; }

        for (int s = 0; s < n; s++)
        {
            int bpos = n - 1 - s;
            var step = forwardOrderSteps[s];
            // This backward step reads grad(output): extend output's lastRead.
            if (step.OutputId >= 0 && bpos > lastRead[step.OutputId])
                lastRead[step.OutputId] = bpos;
            // ...and writes grad(input) for each input: extend input's firstWrite
            // (earliest) and lastRead (an accumulate is also the latest touch until
            // the input's own producer backward reads it later).
            var inputs = step.InputIds;
            if (inputs != null)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    int inp = inputs[j];
                    if (inp < 0) continue;
                    if (bpos < firstWrite[inp]) firstWrite[inp] = bpos;
                    if (bpos > lastRead[inp]) lastRead[inp] = bpos;
                }
            }
        }

        return AssignBuffers(tensorCount, tensorElemCount, tensorShapeKey, isPersistent,
            elementBytes, firstWrite, lastRead, NONE, n);
    }

    /// <summary>
    /// Shared linear-scan register allocation over precomputed liveness intervals
    /// (<paramref name="firstWrite"/> / <paramref name="lastUse"/>, in whatever
    /// unit the caller computed them — backward POSITION for <see cref="Plan"/>,
    /// backward ACTION index for <see cref="PlanOverActions"/>). Same-shape tensors
    /// with disjoint intervals share a physical buffer; a reused buffer's non-first
    /// tenant is scheduled for re-zero before its interval start.
    /// </summary>
    /// <param name="scheduleLen">Length of the re-zero schedule (position/action count).</param>
    private static PlanResult AssignBuffers(
        int tensorCount,
        int[] tensorElemCount,
        long[] tensorShapeKey,
        bool[] isPersistent,
        int elementBytes,
        int[] firstWrite,
        int[] lastUse,
        int none,
        int scheduleLen)
    {
        var bufferOfTensor = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++) bufferOfTensor[i] = -1;

        long naiveBytes = 0;
        for (int i = 0; i < tensorCount; i++)
            naiveBytes += (long)tensorElemCount[i] * elementBytes;

        // Poolable tensors: not persistent AND have a real backward interval.
        var poolable = new System.Collections.Generic.List<int>(tensorCount);
        for (int i = 0; i < tensorCount; i++)
        {
            bool hasInterval = firstWrite[i] != none && lastUse[i] >= firstWrite[i];
            if (!isPersistent[i] && hasInterval) poolable.Add(i);
        }
        poolable.Sort((a, b) =>
        {
            int c = firstWrite[a].CompareTo(firstWrite[b]);
            return c != 0 ? c : lastUse[a].CompareTo(lastUse[b]);
        });

        var bufferElem = new System.Collections.Generic.List<int>();
        // Free physical buffers keyed by shape: shapeKey -> stack of buffer ids.
        var freeByShape = new System.Collections.Generic.Dictionary<long, System.Collections.Generic.Stack<int>>();
        // Active intervals.
        var active = new System.Collections.Generic.List<(int end, int tensorId, int bufferId)>();
        // Re-zero schedule, indexed by position/action.
        var reZero = new System.Collections.Generic.List<int>[scheduleLen];

        foreach (int t in poolable)
        {
            int start = firstWrite[t];
            // Expire every active interval that ended strictly before this start —
            // its buffer is free to reuse for a same-shape tensor.
            for (int k = active.Count - 1; k >= 0; k--)
            {
                if (active[k].end < start)
                {
                    int freedBuf = active[k].bufferId;
                    long key = tensorShapeKey[active[k].tensorId];
                    if (!freeByShape.TryGetValue(key, out var stack))
                        freeByShape[key] = stack = new System.Collections.Generic.Stack<int>();
                    stack.Push(freedBuf);
                    active.RemoveAt(k);
                }
            }

            long shapeKey = tensorShapeKey[t];
            int buf;
            if (freeByShape.TryGetValue(shapeKey, out var avail) && avail.Count > 0)
            {
                // Reusing a buffer => t is a NON-first tenant. Its accumulation
                // starts at `start`, and the buffer still holds the previous
                // tenant's gradient, so it MUST be cleared before position `start`.
                buf = avail.Pop();
                if (start >= 0 && start < scheduleLen)
                    (reZero[start] ??= new System.Collections.Generic.List<int>()).Add(buf);
            }
            else
            {
                buf = bufferElem.Count;
                bufferElem.Add(tensorElemCount[t]);
            }
            bufferOfTensor[t] = buf;
            active.Add((lastUse[t], t, buf));
        }

        // Persistent tensors (and any non-poolable, e.g. unreferenced) each get
        // their own dedicated physical buffer so correctness never depends on the
        // pool: parameters survive to the optimizer, loss seed is external, etc.
        for (int i = 0; i < tensorCount; i++)
        {
            if (bufferOfTensor[i] < 0)
            {
                bufferOfTensor[i] = bufferElem.Count;
                bufferElem.Add(tensorElemCount[i]);
            }
        }

        long pooledBytes = 0;
        for (int i = 0; i < bufferElem.Count; i++)
            pooledBytes += (long)bufferElem[i] * elementBytes;

        var reZeroArr = new int[scheduleLen][];
        for (int k = 0; k < scheduleLen; k++)
            reZeroArr[k] = reZero[k] != null ? reZero[k].ToArray() : System.Array.Empty<int>();

        return new PlanResult(bufferOfTensor, bufferElem.ToArray(), naiveBytes, pooledBytes, reZeroArr);
    }
}
