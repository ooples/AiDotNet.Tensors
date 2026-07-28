// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Linq;
using AiDotNet.Tensors.Ann;
using Xunit;

namespace AiDotNet.Tensors.Tests.Ann
{
    /// <summary>
    /// Correctness tests for the dependency-free ANN stack (Flat/IVF/PQ/IVFPQ) and the underlying
    /// <see cref="AnnPrimitives"/> managed reference — validated on separable synthetic clusters so the
    /// expected nearest neighbour is unambiguous. These are the oracle tests the GPU kernels are checked against.
    /// </summary>
    public class AnnIndexTests
    {
        private const int Dim = 16;
        private const int Clusters = 8;
        private const int PerCluster = 40;

        // Deterministic separable clusters: cluster c centered at a far-apart lattice point.
        private static (float[] flat, long[] ids, float[] centers) MakeData(int seed)
        {
            var rng = new Random(seed);
            int n = Clusters * PerCluster;
            var flat = new float[n * Dim];
            var ids = new long[n];
            var centers = new float[Clusters * Dim];
            for (int c = 0; c < Clusters; c++)
                for (int d = 0; d < Dim; d++)
                    centers[c * Dim + d] = c * 50f + d; // widely separated
            int i = 0;
            for (int c = 0; c < Clusters; c++)
                for (int p = 0; p < PerCluster; p++, i++)
                {
                    ids[i] = i;
                    for (int d = 0; d < Dim; d++)
                        flat[i * Dim + d] = centers[c * Dim + d] + (float)(rng.NextDouble() - 0.5) * 2f;
                }
            return (flat, ids, centers);
        }

        private static float[] Row(float[] flat, int i) => flat.Skip(i * Dim).Take(Dim).ToArray();
        private static int ClusterOf(long id) => (int)(id / PerCluster);

        [Theory]
        [InlineData(AnnIndexType.Flat)]
        [InlineData(AnnIndexType.Ivf)]
        [InlineData(AnnIndexType.Pq)]
        [InlineData(AnnIndexType.IvfPq)]
        public void Search_ReturnsNeighboursFromTheQueryCluster(AnnIndexType type)
        {
            var (flat, ids, centers) = MakeData(seed: 7);
            var index = new AnnIndex(type, Dim, AnnPrimitives.MetricL2, nlist: Clusters, nprobe: 3, m: 4, ksub: 32);
            if (index.IndexType != AnnIndexType.Flat)
                index.Train(flat, ids.Length);
            for (int i = 0; i < ids.Length; i++) index.Add(ids[i], Row(flat, i));

            Assert.Equal(ids.Length, index.Count);

            // Query at cluster 5's center → the nearest neighbours must belong to cluster 5.
            int targetCluster = 5;
            var query = new float[Dim];
            Array.Copy(centers, targetCluster * Dim, query, 0, Dim);

            var (resultIds, _) = index.Search(query, k: 5);
            Assert.NotEmpty(resultIds);
            // Top-1 must be in the target cluster for every index type on well-separated data.
            Assert.Equal(targetCluster, ClusterOf(resultIds[0]));
            // Majority of top-5 should also be in-cluster.
            Assert.True(resultIds.Count(r => ClusterOf(r) == targetCluster) >= 3);
        }

        [Fact]
        public void Flat_MatchesBruteForceExactly()
        {
            var (flat, ids, _) = MakeData(seed: 11);
            var index = new AnnIndex(AnnIndexType.Flat, Dim, AnnPrimitives.MetricL2);
            for (int i = 0; i < ids.Length; i++) index.Add(ids[i], Row(flat, i));

            var query = Row(flat, 123);
            var (resultIds, _) = index.Search(query, 1);
            Assert.Equal(123L, resultIds[0]); // a stored vector is its own nearest neighbour
        }

        [Fact]
        public void Primitives_ComputeDistances_L2_IsSquaredEuclidean()
        {
            var q = new float[] { 0, 0 };
            var db = new float[] { 3, 4, 1, 0 }; // dist² = 25, 1
            var dist = new float[2];
            AnnPrimitives.ComputeDistances(q, db, dist, 1, 2, 2, AnnPrimitives.MetricL2);
            Assert.Equal(25f, dist[0], 3);
            Assert.Equal(1f, dist[1], 3);
        }

        [Fact]
        public void Primitives_IvfAssign_PicksNearestCentroid()
        {
            var vectors = new float[] { 0.1f, 0.1f, 9.9f, 9.9f };
            var centroids = new float[] { 0, 0, 10, 10 };
            var assign = new int[2];
            AnnPrimitives.IvfAssign(vectors, centroids, assign, 2, 2, 2, AnnPrimitives.MetricL2);
            Assert.Equal(0, assign[0]);
            Assert.Equal(1, assign[1]);
        }

        [Fact]
        public void Primitives_PqAdc_SumsSubspaceTables()
        {
            // m=2 subspaces, ksub=2. tables[q=0]: subspace0 -> [1,4], subspace1 -> [2,8].
            var tables = new float[] { 1f, 4f, 2f, 8f };
            var codes = new byte[] { 0, 1,   1, 0 }; // vec0 -> 1+8=9 ; vec1 -> 4+2=6
            var dist = new float[2];
            AnnPrimitives.PqAdcScan(codes, tables, dist, numQueries: 1, numCodes: 2, m: 2, ksub: 2);
            Assert.Equal(9f, dist[0], 3);
            Assert.Equal(6f, dist[1], 3);
        }
    }
}
