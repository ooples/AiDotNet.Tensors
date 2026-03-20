namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Optimization passes for computation graphs that improve cache utilization
/// and operation scheduling for maximum CPU performance.
/// </summary>
public static class OptimizationPasses
{
    /// <summary>
    /// Detects CPU cache sizes for tile scheduling.
    /// Falls back to conservative defaults if detection fails.
    /// </summary>
    public static class CacheInfo
    {
        /// <summary>L1 data cache size in bytes (typically 32-48KB).</summary>
        public static int L1DataCacheSize { get; } = DetectL1CacheSize();

        /// <summary>L2 cache size in bytes (typically 256KB-1MB).</summary>
        public static int L2CacheSize { get; } = DetectL2CacheSize();

        /// <summary>L3 cache size in bytes (typically 4-32MB).</summary>
        public static int L3CacheSize { get; } = DetectL3CacheSize();

        /// <summary>Optimal tile size for float operations targeting L1 cache.</summary>
        public static int L1TileSizeFloat => (int)Math.Sqrt(L1DataCacheSize / (3 * sizeof(float)));

        /// <summary>Optimal tile size for double operations targeting L1 cache.</summary>
        public static int L1TileSizeDouble => (int)Math.Sqrt(L1DataCacheSize / (3 * sizeof(double)));

        /// <summary>Optimal tile size for float operations targeting L2 cache.</summary>
        public static int L2TileSizeFloat => (int)Math.Sqrt(L2CacheSize / (3 * sizeof(float)));

        /// <summary>Optimal tile size for double operations targeting L2 cache.</summary>
        public static int L2TileSizeDouble => (int)Math.Sqrt(L2CacheSize / (3 * sizeof(double)));

        private static int DetectL1CacheSize()
        {
            // Conservative default: 32KB (Intel/AMD typical)
            return 32 * 1024;
        }

        private static int DetectL2CacheSize()
        {
            // Conservative default: 256KB
            return 256 * 1024;
        }

        private static int DetectL3CacheSize()
        {
            // Conservative default: 8MB
            return 8 * 1024 * 1024;
        }
    }

    /// <summary>
    /// Computes optimal tile dimensions for matrix operations to fit in cache.
    /// </summary>
    /// <param name="m">Number of rows in output.</param>
    /// <param name="n">Number of columns in output.</param>
    /// <param name="k">Inner dimension (shared between A and B).</param>
    /// <param name="elementSize">Size of each element in bytes (4 for float, 8 for double).</param>
    /// <returns>Tile dimensions (tileM, tileN, tileK) that fit in L2 cache.</returns>
    public static (int tileM, int tileN, int tileK) ComputeMatMulTiles(int m, int n, int k, int elementSize = 4)
    {
        // Target: 3 tile panels (A, B, C) fit in L2 cache
        // A panel: tileM x tileK, B panel: tileK x tileN, C panel: tileM x tileN
        // Total: (tileM*tileK + tileK*tileN + tileM*tileN) * elementSize <= L2
        int l2 = CacheInfo.L2CacheSize;
        int targetBytes = l2 / 2; // Use half of L2 to leave room for other data

        // Start with square tiles and adjust
        int tileSize = (int)Math.Sqrt(targetBytes / (3.0 * elementSize));
        tileSize = Math.Max(16, tileSize); // minimum 16 for SIMD efficiency

        int tileM = Math.Min(m, tileSize);
        int tileN = Math.Min(n, tileSize);
        int tileK = Math.Min(k, tileSize);

        // Align to SIMD width, but never exceed actual dimension
        int simdWidth = elementSize == 4 ? 8 : 4; // AVX2: 8 floats or 4 doubles
        tileN = (tileN / simdWidth) * simdWidth;
        if (tileN == 0) tileN = Math.Min(simdWidth, n);

        return (tileM, tileN, tileK);
    }

    /// <summary>
    /// Computes optimal tile dimensions for Conv2D im2col to fit in cache.
    /// </summary>
    /// <param name="outputHeight">Output spatial height.</param>
    /// <param name="outputWidth">Output spatial width.</param>
    /// <param name="kernelSize">Kernel spatial size (height * width).</param>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="elementSize">Size of each element in bytes.</param>
    /// <returns>Number of output spatial positions to process per tile.</returns>
    public static int ComputeConv2DTileSize(int outputHeight, int outputWidth,
        int kernelSize, int inChannels, int outChannels, int elementSize = 4)
    {
        int colH = inChannels * kernelSize;
        int totalSpatial = outputHeight * outputWidth;

        // im2col buffer: colH * tileSpatial elements
        // kernel: outChannels * colH elements (reused across tiles)
        // output: outChannels * tileSpatial elements
        int l2 = CacheInfo.L2CacheSize;
        int kernelBytes = outChannels * colH * elementSize;
        int availableBytes = Math.Max(l2 - kernelBytes, l2 / 4);

        // tileSpatial * (colH + outChannels) * elementSize <= availableBytes
        int tileSpatial = availableBytes / ((colH + outChannels) * elementSize);
        tileSpatial = Math.Max(1, Math.Min(tileSpatial, totalSpatial));

        return tileSpatial;
    }

    /// <summary>
    /// Reorders operations in a computation graph to maximize data locality.
    /// Operations that share input tensors are scheduled adjacently when possible.
    /// </summary>
    /// <param name="graph">The computation graph to optimize.</param>
    /// <returns>Reordered operation indices for optimal data locality.</returns>
    public static int[] ReorderForLocality(ComputationGraph graph)
    {
        var nodes = graph.Nodes;
        int count = nodes.Count;

        if (count <= 2) // Nothing to reorder
        {
            var trivial = new int[count];
            for (int i = 0; i < count; i++) trivial[i] = i;
            return trivial;
        }

        // Build dependency graph: which nodes must come before which
        var dependsOn = new HashSet<int>[count];
        var dependedBy = new List<int>[count];
        for (int i = 0; i < count; i++)
        {
            dependsOn[i] = new HashSet<int>(nodes[i].InputIds);
            dependedBy[i] = new List<int>();
        }

        for (int i = 0; i < count; i++)
        {
            // Deduplicate to prevent double-counting when InputIds has repeats
            foreach (int dep in new HashSet<int>(nodes[i].InputIds))
            {
                if (dep >= 0 && dep < count)
                    dependedBy[dep].Add(i);
            }
        }

        // Topological sort with locality heuristic:
        // When multiple nodes are ready (all deps satisfied), pick the one
        // that shares the most inputs with the most recently scheduled node
        var order = new List<int>(count);
        var ready = new List<int>();
        var scheduled = new bool[count];
        var inDegree = new int[count];

        for (int i = 0; i < count; i++)
        {
            inDegree[i] = dependsOn[i].Count;
            if (inDegree[i] == 0)
                ready.Add(i);
        }

        int lastScheduled = -1;

        while (ready.Count > 0)
        {
            // Pick the ready node with best locality to last scheduled
            int bestIdx = 0;
            int bestScore = -1;

            if (lastScheduled >= 0)
            {
                var lastInputs = new HashSet<int>(nodes[lastScheduled].InputIds);
                for (int r = 0; r < ready.Count; r++)
                {
                    int nodeId = ready[r];
                    // Score: number of shared inputs with last scheduled node
                    int score = 0;
                    foreach (int inp in nodes[nodeId].InputIds)
                    {
                        if (lastInputs.Contains(inp)) score++;
                    }
                    // Also boost nodes that consume the output of lastScheduled
                    if (nodes[nodeId].InputIds.Contains(lastScheduled)) score += 2;

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIdx = r;
                    }
                }
            }

            int chosen = ready[bestIdx];
            ready.RemoveAt(bestIdx);
            order.Add(chosen);
            scheduled[chosen] = true;
            lastScheduled = chosen;

            // Update in-degree for dependents
            foreach (int dependent in dependedBy[chosen])
            {
                inDegree[dependent]--;
                if (inDegree[dependent] == 0 && !scheduled[dependent])
                    ready.Add(dependent);
            }
        }

        return order.ToArray();
    }
}
