// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Ann
{
    /// <summary>
    /// Managed, dependency-free CPU reference for the fused ANN primitives declared by
    /// <see cref="AiDotNet.Tensors.Engines.DirectGpu.IAnnBackend"/>. This is both the correctness oracle
    /// the GPU kernels are validated against and the fallback compute path when no GPU ANN backend is
    /// present, so the IVF / PQ / IVFPQ / HNSW index families run fully on the AiDotNet stack with no
    /// external FAISS / MKL dependency.
    /// </summary>
    public static class AnnPrimitives
    {
        /// <summary>Squared Euclidean (L2²) distance — smaller is nearer.</summary>
        public const int MetricL2 = 0;
        /// <summary>Inner product — larger is nearer.</summary>
        public const int MetricInnerProduct = 1;

        /// <summary>
        /// Distance between two length-<paramref name="dim"/> subvectors at the given offsets.
        /// L2 returns squared distance; InnerProduct returns the dot product.
        /// </summary>
        public static float Distance(float[] a, int aOffset, float[] b, int bOffset, int dim, int metric)
        {
            if (metric == MetricInnerProduct)
            {
                float ip = 0f;
                for (int k = 0; k < dim; k++) ip += a[aOffset + k] * b[bOffset + k];
                return ip;
            }

            float sum = 0f;
            for (int k = 0; k < dim; k++)
            {
                float d = a[aOffset + k] - b[bOffset + k];
                sum += d * d;
            }
            return sum;
        }

        /// <summary>True when <paramref name="candidate"/> is a nearer/better score than <paramref name="best"/> for the metric.</summary>
        public static bool IsBetter(float candidate, float best, int metric)
            => metric == MetricInnerProduct ? candidate > best : candidate < best;

        /// <summary>The worst possible score for the metric (used to seed a running best).</summary>
        public static float WorstScore(int metric)
            => metric == MetricInnerProduct ? float.NegativeInfinity : float.PositiveInfinity;

        /// <summary>
        /// Dense query×database distance matrix, row-major <c>[numQueries, numDatabase]</c>.
        /// Backs flat/exact scan, IVF list scan, and HNSW candidate scoring.
        /// </summary>
        public static void ComputeDistances(float[] queries, float[] database, float[] distances,
            int numQueries, int numDatabase, int dim, int metric)
        {
            if (queries == null) throw new ArgumentNullException(nameof(queries));
            if (database == null) throw new ArgumentNullException(nameof(database));
            if (distances == null) throw new ArgumentNullException(nameof(distances));
            if (distances.Length < (long)numQueries * numDatabase)
                throw new ArgumentException("distances buffer too small", nameof(distances));

            for (int q = 0; q < numQueries; q++)
            {
                int qOff = q * dim;
                int rowOff = q * numDatabase;
                for (int j = 0; j < numDatabase; j++)
                    distances[rowOff + j] = Distance(queries, qOff, database, j * dim, dim, metric);
            }
        }

        /// <summary>
        /// Assigns each vector to its nearest centroid (IVF coarse quantizer / Lloyd k-means assignment).
        /// </summary>
        public static void IvfAssign(float[] vectors, float[] centroids, int[] assignments,
            int numVectors, int numCentroids, int dim, int metric)
        {
            if (numCentroids <= 0) throw new ArgumentException("numCentroids must be positive", nameof(numCentroids));
            for (int i = 0; i < numVectors; i++)
            {
                int vOff = i * dim;
                int best = 0;
                float bestScore = WorstScore(metric);
                for (int c = 0; c < numCentroids; c++)
                {
                    float score = Distance(vectors, vOff, centroids, c * dim, dim, metric);
                    if (IsBetter(score, bestScore, metric))
                    {
                        bestScore = score;
                        best = c;
                    }
                }
                assignments[i] = best;
            }
        }

        /// <summary>
        /// PQ asymmetric distance tables: per query, per subspace, distance to each of <paramref name="ksub"/>
        /// sub-centroids. Layout of <paramref name="tables"/> is <c>[query][subspace][ksub]</c> row-major.
        /// Codebooks layout is <c>[subspace][ksub][dsub]</c>.
        /// </summary>
        public static void PqComputeDistanceTables(float[] queries, float[] codebooks, float[] tables,
            int numQueries, int m, int ksub, int dsub, int metric)
        {
            int subDim = dsub;
            int qStride = m * dsub;
            int cbSubStride = ksub * dsub;
            int tblQStride = m * ksub;
            for (int q = 0; q < numQueries; q++)
            {
                for (int s = 0; s < m; s++)
                {
                    int qSubOff = q * qStride + s * subDim;
                    int cbOff = s * cbSubStride;
                    int tblOff = q * tblQStride + s * ksub;
                    for (int c = 0; c < ksub; c++)
                        tables[tblOff + c] = Distance(queries, qSubOff, codebooks, cbOff + c * subDim, subDim, metric);
                }
            }
        }

        /// <summary>
        /// PQ ADC scan: per query, per coded vector, sum the per-subspace table lookups into an approximate
        /// distance. <paramref name="codes"/> layout is <c>[code][m]</c>; output is <c>[numQueries, numCodes]</c>.
        /// </summary>
        public static void PqAdcScan(byte[] codes, float[] tables, float[] distances,
            int numQueries, int numCodes, int m, int ksub)
        {
            int tblQStride = m * ksub;
            for (int q = 0; q < numQueries; q++)
            {
                int tblQOff = q * tblQStride;
                int outOff = q * numCodes;
                for (int i = 0; i < numCodes; i++)
                {
                    int codeOff = i * m;
                    float sum = 0f;
                    for (int s = 0; s < m; s++)
                        sum += tables[tblQOff + s * ksub + codes[codeOff + s]];
                    distances[outOff + i] = sum;
                }
            }
        }
    }
}
