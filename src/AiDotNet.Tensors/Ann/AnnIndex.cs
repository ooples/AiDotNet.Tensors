// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Ann
{
    /// <summary>The ANN index family. All are backed by <see cref="AnnPrimitives"/> (managed CPU) and,
    /// when a GPU <see cref="AiDotNet.Tensors.Engines.DirectGpu.IAnnBackend"/> is present, the fused kernels.</summary>
    public enum AnnIndexType
    {
        /// <summary>Exact brute-force scan.</summary>
        Flat = 0,
        /// <summary>Inverted file (coarse k-means) — probe the nearest lists only.</summary>
        Ivf = 1,
        /// <summary>Product quantization — compressed codes + ADC search.</summary>
        Pq = 2,
        /// <summary>IVF over PQ-compressed residuals — FAISS's workhorse index.</summary>
        IvfPq = 3,
    }

    /// <summary>
    /// A self-contained, dependency-free approximate-nearest-neighbour index (Flat / IVF / PQ / IVFPQ)
    /// running entirely on the AiDotNet stack — no FAISS / MKL. Compute is delegated to
    /// <see cref="AnnPrimitives"/> (the managed CPU reference and correctness oracle); the same calls map to
    /// the fused GPU kernels via <see cref="AiDotNet.Tensors.Engines.DirectGpu.IAnnBackend"/> when available.
    /// This is the replacement for the external FaissNet backend whose IVF/PQ path was blocked by an
    /// incomplete native MKL redistribution.
    /// </summary>
    public sealed class AnnIndex
    {
        private readonly AnnIndexType _type;
        private readonly int _dim;
        private readonly int _metric;
        private readonly int _nlist;      // IVF coarse lists
        private readonly int _nprobe;     // IVF lists probed at query time
        private readonly int _m;          // PQ subspaces
        private readonly int _ksub;       // PQ sub-centroids per subspace
        private readonly int _dsub;       // PQ subspace dimensionality
        private readonly int _seed;

        private float[]? _coarse;                 // [nlist * dim] IVF centroids
        private float[]? _pqCodebooks;            // [m * ksub * dsub]
        private readonly List<float[]> _flatVectors = new(); // Flat/IVF: raw vectors
        private readonly List<long> _ids = new();
        private readonly List<int> _listOf = new();          // IVF: coarse list per stored vector
        private byte[]? _codes;                              // PQ/IVFPQ: [count * m]
        private int _count;
        private bool _trained;

        public AnnIndexType IndexType => _type;
        public int Dimension => _dim;
        public int Count => _count;
        public bool IsTrained => _trained;

        public AnnIndex(AnnIndexType type, int dim, int metric = AnnPrimitives.MetricL2,
            int nlist = 64, int nprobe = 8, int m = 8, int ksub = 256, int seed = 42)
        {
            if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
            if ((type == AnnIndexType.Pq || type == AnnIndexType.IvfPq) && dim % m != 0)
                throw new ArgumentException($"dim ({dim}) must be divisible by m ({m}) for PQ.", nameof(m));
            _type = type; _dim = dim; _metric = metric;
            _nlist = Math.Max(1, nlist); _nprobe = Math.Max(1, Math.Min(nprobe, _nlist));
            _m = m; _ksub = ksub; _dsub = dim / Math.Max(1, m); _seed = seed;
            _trained = type == AnnIndexType.Flat; // Flat needs no training
        }

        /// <summary>Trains the coarse quantizer (IVF) and/or PQ codebooks on a representative sample.</summary>
        public void Train(float[] trainingVectors, int numVectors)
        {
            if (_type == AnnIndexType.Flat) { _trained = true; return; }
            if (numVectors < 1) throw new ArgumentException("Need training vectors", nameof(numVectors));

            if (_type == AnnIndexType.Ivf || _type == AnnIndexType.IvfPq)
                _coarse = KMeans(trainingVectors, numVectors, _dim, Math.Min(_nlist, numVectors), _metric, _seed);

            if (_type == AnnIndexType.Pq || _type == AnnIndexType.IvfPq)
            {
                // For IVFPQ, PQ is trained on residuals (vector - nearest coarse centroid).
                float[] pqTrain = trainingVectors;
                if (_type == AnnIndexType.IvfPq)
                    pqTrain = ResidualsOf(trainingVectors, numVectors);
                _pqCodebooks = TrainPq(pqTrain, numVectors);
            }
            _trained = true;
        }

        /// <summary>Adds one vector (must already be trained for IVF/PQ/IVFPQ).</summary>
        public void Add(long id, float[] vector)
        {
            if (vector.Length != _dim) throw new ArgumentException("dim mismatch", nameof(vector));
            if (!_trained) throw new InvalidOperationException("Index must be trained before Add.");
            _ids.Add(id);

            if (_type == AnnIndexType.Flat)
            {
                _flatVectors.Add((float[])vector.Clone());
            }
            else if (_type == AnnIndexType.Ivf)
            {
                _flatVectors.Add((float[])vector.Clone());
                _listOf.Add(NearestCoarse(vector));
            }
            else if (_type == AnnIndexType.Pq)
            {
                AppendCodes(EncodePq(vector));
            }
            else // IVFPQ
            {
                int list = NearestCoarse(vector);
                _listOf.Add(list);
                var residual = Subtract(vector, _coarse!, list * _dim);
                AppendCodes(EncodePq(residual));
            }
            _count++;
        }

        /// <summary>Searches for the <paramref name="k"/> nearest ids to <paramref name="query"/>.</summary>
        public (long[] ids, float[] distances) Search(float[] query, int k)
        {
            if (query.Length != _dim) throw new ArgumentException("dim mismatch", nameof(query));
            k = Math.Min(k, _count);
            if (k <= 0) return (Array.Empty<long>(), Array.Empty<float>());

            switch (_type)
            {
                case AnnIndexType.Flat: return SearchFlat(query, k);
                case AnnIndexType.Ivf: return SearchIvf(query, k);
                case AnnIndexType.Pq: return SearchPq(query, k);
                default: return SearchIvfPq(query, k);
            }
        }

        // ---------------- Flat ----------------
        private (long[], float[]) SearchFlat(float[] query, int k)
        {
            var db = FlattenAll();
            var dist = new float[_count];
            AnnPrimitives.ComputeDistances(query, db, dist, 1, _count, _dim, _metric);
            return TopK(dist, k, i => _ids[i]);
        }

        // ---------------- IVF ----------------
        private (long[], float[]) SearchIvf(float[] query, int k)
        {
            var lists = NearestCoarseLists(query, _nprobe);
            var cand = new List<int>();
            for (int i = 0; i < _count; i++)
                if (lists.Contains(_listOf[i])) cand.Add(i);
            if (cand.Count == 0) return SearchFlat(query, k); // fall back if lists empty
            var db = new float[cand.Count * _dim];
            for (int c = 0; c < cand.Count; c++) Array.Copy(_flatVectors[cand[c]], 0, db, c * _dim, _dim);
            var dist = new float[cand.Count];
            AnnPrimitives.ComputeDistances(query, db, dist, 1, cand.Count, _dim, _metric);
            return TopK(dist, Math.Min(k, cand.Count), i => _ids[cand[i]]);
        }

        // ---------------- PQ ----------------
        private (long[], float[]) SearchPq(float[] query, int k)
        {
            var tables = new float[_m * _ksub];
            AnnPrimitives.PqComputeDistanceTables(query, _pqCodebooks!, tables, 1, _m, _ksub, _dsub, _metric);
            var dist = new float[_count];
            AnnPrimitives.PqAdcScan(_codes!, tables, dist, 1, _count, _m, _ksub);
            return TopK(dist, k, i => _ids[i]);
        }

        // ---------------- IVFPQ ----------------
        private (long[], float[]) SearchIvfPq(float[] query, int k)
        {
            var lists = NearestCoarseLists(query, _nprobe);
            var cand = new List<int>();
            for (int i = 0; i < _count; i++)
                if (lists.Contains(_listOf[i])) cand.Add(i);
            if (cand.Count == 0) return (Array.Empty<long>(), Array.Empty<float>());
            // ADC on residual space: table computed per probed list (query residual vs that list's centroid).
            var dist = new float[cand.Count];
            var listTables = new Dictionary<int, float[]>();
            for (int c = 0; c < cand.Count; c++)
            {
                int list = _listOf[cand[c]];
                if (!listTables.TryGetValue(list, out var tbl))
                {
                    var qres = Subtract(query, _coarse!, list * _dim);
                    tbl = new float[_m * _ksub];
                    AnnPrimitives.PqComputeDistanceTables(qres, _pqCodebooks!, tbl, 1, _m, _ksub, _dsub, _metric);
                    listTables[list] = tbl;
                }
                float sum = 0f;
                int codeOff = cand[c] * _m;
                for (int s = 0; s < _m; s++) sum += tbl[s * _ksub + _codes![codeOff + s]];
                dist[c] = sum;
            }
            return TopK(dist, Math.Min(k, cand.Count), i => _ids[cand[i]]);
        }

        // ---------------- helpers ----------------
        private (long[], float[]) TopK(float[] dist, int k, Func<int, long> idOf)
        {
            var idx = Enumerable.Range(0, dist.Length).ToArray();
            // Lower is nearer for L2; higher is nearer for inner product.
            Array.Sort(idx, (a, b) => _metric == AnnPrimitives.MetricInnerProduct
                ? dist[b].CompareTo(dist[a]) : dist[a].CompareTo(dist[b]));
            var ids = new long[k]; var ds = new float[k];
            for (int i = 0; i < k; i++) { ids[i] = idOf(idx[i]); ds[i] = dist[idx[i]]; }
            return (ids, ds);
        }

        private float[] FlattenAll()
        {
            var db = new float[_count * _dim];
            for (int i = 0; i < _count; i++) Array.Copy(_flatVectors[i], 0, db, i * _dim, _dim);
            return db;
        }

        private int NearestCoarse(float[] v)
        {
            var a = new int[1];
            AnnPrimitives.IvfAssign(v, _coarse!, a, 1, _nlist, _dim, _metric);
            return a[0];
        }

        private HashSet<int> NearestCoarseLists(float[] query, int nprobe)
        {
            var dist = new float[_nlist];
            AnnPrimitives.ComputeDistances(query, _coarse!, dist, 1, _nlist, _dim, _metric);
            var order = Enumerable.Range(0, _nlist).ToArray();
            Array.Sort(order, (a, b) => _metric == AnnPrimitives.MetricInnerProduct
                ? dist[b].CompareTo(dist[a]) : dist[a].CompareTo(dist[b]));
            return new HashSet<int>(order.Take(Math.Min(nprobe, _nlist)));
        }

        private float[] ResidualsOf(float[] vectors, int n)
        {
            var res = new float[n * _dim];
            var a = new int[n];
            AnnPrimitives.IvfAssign(vectors, _coarse!, a, n, _nlist, _dim, _metric);
            for (int i = 0; i < n; i++)
                for (int d = 0; d < _dim; d++)
                    res[i * _dim + d] = vectors[i * _dim + d] - _coarse![a[i] * _dim + d];
            return res;
        }

        private static float[] Subtract(float[] v, float[] baseArr, int baseOff)
        {
            var r = new float[v.Length];
            for (int i = 0; i < v.Length; i++) r[i] = v[i] - baseArr[baseOff + i];
            return r;
        }

        private void AppendCodes(byte[] code)
        {
            int newLen = (_count + 1) * _m;
            if (_codes == null || _codes.Length < newLen)
            {
                var grown = new byte[Math.Max(newLen, (_codes?.Length ?? 0) * 2)];
                if (_codes != null) Array.Copy(_codes, grown, _count * _m);
                _codes = grown;
            }
            Array.Copy(code, 0, _codes, _count * _m, _m);
        }

        private byte[] EncodePq(float[] vector)
        {
            var code = new byte[_m];
            for (int s = 0; s < _m; s++)
            {
                int best = 0; float bestScore = AnnPrimitives.WorstScore(_metric);
                int cbOff = s * _ksub * _dsub;
                for (int c = 0; c < _ksub; c++)
                {
                    float score = AnnPrimitives.Distance(vector, s * _dsub, _pqCodebooks!, cbOff + c * _dsub, _dsub, _metric);
                    if (AnnPrimitives.IsBetter(score, bestScore, _metric)) { bestScore = score; best = c; }
                }
                code[s] = (byte)best;
            }
            return code;
        }

        private float[] TrainPq(float[] vectors, int n)
        {
            var cb = new float[_m * _ksub * _dsub];
            for (int s = 0; s < _m; s++)
            {
                // Extract subspace s into a contiguous [n * dsub] buffer, then k-means it.
                var sub = new float[n * _dsub];
                for (int i = 0; i < n; i++)
                    Array.Copy(vectors, i * _dim + s * _dsub, sub, i * _dsub, _dsub);
                var centroids = KMeans(sub, n, _dsub, Math.Min(_ksub, n), _metric, _seed + s);
                // Pad to ksub centroids if fewer training points than ksub.
                Array.Copy(centroids, 0, cb, s * _ksub * _dsub, Math.Min(centroids.Length, _ksub * _dsub));
            }
            return cb;
        }

        /// <summary>Lloyd's k-means (seeded, fixed iterations) using the ANN assignment primitive.</summary>
        private static float[] KMeans(float[] vectors, int n, int dim, int kRaw, int metric, int seed)
        {
            int k = Math.Max(1, Math.Min(kRaw, n));
            var rng = new Random(seed);
            var centroids = new float[k * dim];
            // Init: distinct random samples.
            var picked = new HashSet<int>();
            for (int c = 0; c < k; c++)
            {
                int idx; do { idx = rng.Next(n); } while (!picked.Add(idx) && picked.Count < n);
                Array.Copy(vectors, idx * dim, centroids, c * dim, dim);
            }
            var assign = new int[n];
            for (int iter = 0; iter < 12; iter++)
            {
                AnnPrimitives.IvfAssign(vectors, centroids, assign, n, k, dim, metric);
                var sums = new float[k * dim];
                var counts = new int[k];
                for (int i = 0; i < n; i++)
                {
                    int a = assign[i]; counts[a]++;
                    for (int d = 0; d < dim; d++) sums[a * dim + d] += vectors[i * dim + d];
                }
                for (int c = 0; c < k; c++)
                {
                    if (counts[c] == 0) continue; // keep empty centroid as-is
                    for (int d = 0; d < dim; d++) centroids[c * dim + d] = sums[c * dim + d] / counts[c];
                }
            }
            return centroids;
        }
    }
}
