// Copyright (c) AiDotNet. All rights reserved.
// Secondary interface for backends that ship native approximate-nearest-neighbour (ANN) kernels.
// Follows the same optional-capability pattern as IFftBackend / IAudioBackend: if a backend implements
// this interface, the engine dispatches ANN ops to the custom GPU kernel; otherwise the call
// transparently falls through to the managed CPU reference. Supply-chain-clean — the kernels are custom
// (no FAISS / cuVS / MKL dependency) and live alongside each backend.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Distance/similarity metric for ANN kernels. Cosine is caller-normalized (treated as inner product
    /// over L2-normalized vectors), so the kernels only implement L2 and inner product.
    /// </summary>
    public enum AnnMetric
    {
        /// <summary>Squared Euclidean (L2²) distance — smaller is nearer.</summary>
        L2 = 0,
        /// <summary>Inner product — larger is nearer (use with L2-normalized vectors for cosine).</summary>
        InnerProduct = 1,
    }

    /// <summary>
    /// Optional capability interface for GPU backends that ship native ANN kernels backing the IVF, PQ,
    /// IVFPQ and HNSW index families. Three fused primitives cover all four:
    /// <list type="bullet">
    ///   <item><see cref="ComputeDistances"/> — dense query×database distance matrix (flat/exact scan,
    ///     IVF inverted-list scan, and HNSW candidate scoring).</item>
    ///   <item><see cref="IvfAssign"/> — nearest-centroid assignment (IVF coarse quantizer + the inner
    ///     loop of Lloyd k-means used to train IVF/PQ codebooks).</item>
    ///   <item><see cref="PqComputeDistanceTables"/> + <see cref="PqAdcScan"/> — product-quantization
    ///     asymmetric distance computation (PQ and IVFPQ search).</item>
    /// </list>
    /// Top-k selection over the produced distances is done by the caller (managed), keeping the kernels
    /// simple and portable. All buffers are float32 <see cref="IGpuBuffer"/>s; f64 callers downcast first.
    /// </summary>
    public interface IAnnBackend
    {
        /// <summary>
        /// Computes the full distance matrix between <paramref name="numQueries"/> query vectors and
        /// <paramref name="numDatabase"/> database vectors, both row-major <c>[n, dim]</c>.
        /// </summary>
        /// <param name="queries">Query vectors, <c>numQueries·dim</c> float32.</param>
        /// <param name="database">Database vectors, <c>numDatabase·dim</c> float32.</param>
        /// <param name="distances">Output, row-major <c>[numQueries, numDatabase]</c>.</param>
        /// <param name="numQueries">Number of query rows.</param>
        /// <param name="numDatabase">Number of database rows.</param>
        /// <param name="dim">Vector dimensionality.</param>
        /// <param name="metric">Distance metric.</param>
        void ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
            int numQueries, int numDatabase, int dim, AnnMetric metric);

        /// <summary>
        /// Assigns each of <paramref name="numVectors"/> vectors to its nearest centroid (argmin/argmax by
        /// metric). Used as the IVF coarse quantizer and as the assignment step of Lloyd k-means.
        /// </summary>
        /// <param name="vectors">Input vectors, <c>numVectors·dim</c> float32, row-major.</param>
        /// <param name="centroids">Centroids, <c>numCentroids·dim</c> float32, row-major.</param>
        /// <param name="assignments">Output centroid index per vector, <c>numVectors</c> int32 (packed as float bits or an int buffer per backend contract).</param>
        /// <param name="numVectors">Number of input vectors.</param>
        /// <param name="numCentroids">Number of centroids.</param>
        /// <param name="dim">Vector dimensionality.</param>
        /// <param name="metric">Distance metric.</param>
        void IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
            int numVectors, int numCentroids, int dim, AnnMetric metric);

        /// <summary>
        /// Precomputes the PQ asymmetric distance tables: for each query and each of <paramref name="m"/>
        /// subspaces, the distance from the query subvector to all <paramref name="ksub"/> sub-centroids.
        /// </summary>
        /// <param name="queries">Query vectors, <c>numQueries·(m·dsub)</c> float32, row-major.</param>
        /// <param name="codebooks">PQ codebooks, <c>m·ksub·dsub</c> float32 (subspace-major).</param>
        /// <param name="tables">Output distance tables, <c>numQueries·m·ksub</c> float32.</param>
        /// <param name="numQueries">Number of queries.</param>
        /// <param name="m">Number of subspaces.</param>
        /// <param name="ksub">Sub-centroids per subspace (typically 256).</param>
        /// <param name="dsub">Dimensionality of each subspace.</param>
        /// <param name="metric">Distance metric.</param>
        void PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
            int numQueries, int m, int ksub, int dsub, AnnMetric metric);

        /// <summary>
        /// Scans PQ codes with the precomputed tables (ADC): for each query and each coded database vector,
        /// sums the per-subspace table lookups into an approximate distance.
        /// </summary>
        /// <param name="codes">Database PQ codes, <c>numCodes·m</c> bytes (one code per subspace), uploaded as a byte buffer.</param>
        /// <param name="tables">Distance tables from <see cref="PqComputeDistanceTables"/>, <c>numQueries·m·ksub</c> float32.</param>
        /// <param name="distances">Output approximate distances, row-major <c>[numQueries, numCodes]</c>.</param>
        /// <param name="numQueries">Number of queries.</param>
        /// <param name="numCodes">Number of coded database vectors.</param>
        /// <param name="m">Number of subspaces.</param>
        /// <param name="ksub">Sub-centroids per subspace (table stride).</param>
        void PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
            int numQueries, int numCodes, int m, int ksub);
    }
}
