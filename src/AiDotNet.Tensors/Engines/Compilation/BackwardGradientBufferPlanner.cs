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

        public PlanResult(int[] bufferOfTensor, int[] bufferElemCount, long naiveBytes, long pooledBytes)
        {
            BufferOfTensor = bufferOfTensor;
            BufferElemCount = bufferElemCount;
            NaiveBytes = naiveBytes;
            PooledBytes = pooledBytes;
        }
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

        var bufferOfTensor = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++) bufferOfTensor[i] = -1;

        long naiveBytes = 0;
        for (int i = 0; i < tensorCount; i++)
            naiveBytes += (long)tensorElemCount[i] * elementBytes;

        // Poolable tensors: not persistent AND have a real backward interval.
        // Build (tensorId, start, end) and linear-scan by start position.
        var poolable = new System.Collections.Generic.List<int>(tensorCount);
        for (int i = 0; i < tensorCount; i++)
        {
            bool hasInterval = firstWrite[i] != NONE && lastRead[i] >= firstWrite[i];
            if (!isPersistent[i] && hasInterval) poolable.Add(i);
        }
        poolable.Sort((a, b) =>
        {
            int c = firstWrite[a].CompareTo(firstWrite[b]);
            return c != 0 ? c : lastRead[a].CompareTo(lastRead[b]);
        });

        var bufferElem = new System.Collections.Generic.List<int>();
        // Free physical buffers keyed by shape: shapeKey -> stack of buffer ids.
        var freeByShape = new System.Collections.Generic.Dictionary<long, System.Collections.Generic.Stack<int>>();
        // Active intervals, ordered by end position, so we can expire cheaply.
        var active = new System.Collections.Generic.List<(int end, int tensorId, int bufferId)>();

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
                buf = avail.Pop();
            }
            else
            {
                buf = bufferElem.Count;
                bufferElem.Add(tensorElemCount[t]);
            }
            bufferOfTensor[t] = buf;
            active.Add((lastRead[t], t, buf));
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

        return new PlanResult(bufferOfTensor, bufferElem.ToArray(), naiveBytes, pooledBytes);
    }
}
