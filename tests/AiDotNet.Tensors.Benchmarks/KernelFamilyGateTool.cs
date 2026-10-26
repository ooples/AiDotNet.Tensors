// Copyright (c) AiDotNet. All rights reserved.
// --kernel-family-gate: the wall-clock gate (protocol p15) for the families #1440 and #1441 declared -- the row-wise
// scatter family, the sparse kernels, the index family and the attention family. Each case calls a CUDA backend entry
// point with the declared route on and off (the incumbent hand-written kernel), INTERLEAVED sample by sample: every
// sample times a batch of each, in alternating order, so a clock or thermal drift lands on both sides of a pair. The
// declared call must launch its program (a decline fails the case). Run three times as separate processes for the
// three clean runs; a case passes a run when its median paired ratio (incumbent / declared) is at least --min-ratio
// (default 1.0: no worse than the hand-written kernel).
//
// Options: --family <prefix> (only cases whose name starts with it), --samples <n> (default 31), --batch <n> calls per
// sample (default 10), --min-ratio <r>, --deterministic (AiDotNetEngine's deterministic mode, so the incumbents are the
// deterministic kernels), --gated (a case the declared route declines passes: the hand-written kernel serves it, which
// is what shape gating decides).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelFamilyGateTool
{
    private sealed record GateCase(string Name, Func<CudaBackend, (Action Call, IDisposable Buffers)> Setup);

    private sealed class BufferSet : IDisposable
    {
        private readonly List<IGpuBuffer> _buffers = new();
        internal T Add<T>(T buffer) where T : IGpuBuffer { _buffers.Add(buffer); return buffer; }
        public void Dispose() { foreach (var b in _buffers) b.Dispose(); }
    }

    internal static void Run(string[] args)
    {
        string? only = Value(args, "--family");
        int samples = int.Parse(Value(args, "--samples") ?? "31", CultureInfo.InvariantCulture);
        int batch = int.Parse(Value(args, "--batch") ?? "10", CultureInfo.InvariantCulture);
        double minRatio = double.Parse(Value(args, "--min-ratio") ?? "1.0", CultureInfo.InvariantCulture);
        bool gated = Array.IndexOf(args, "--gated") >= 0;
        bool deterministic = Array.IndexOf(args, "--deterministic") >= 0;
        bool previousDeterminism = AiDotNet.Tensors.Engines.AiDotNetEngine.DeterministicMode;
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(deterministic);
        try { RunCases(only, samples, batch, minRatio, gated, deterministic); }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(previousDeterminism); }
    }

    private static void RunCases(string? only, int samples, int batch, double minRatio, bool gated, bool deterministic)
    {

        using var engine = new DirectGpuEngine();
        if (engine.Backend is not CudaBackend backend)
        {
            Console.WriteLine("The declared routes are CUDA-only; no CUDA backend here.");
            return;
        }
        Console.WriteLine("FAMILY GATE (p15) - declared vs hand-written, interleaved paired medians, us per call" +
            (deterministic ? " [deterministic]" : "") + (gated ? " [gated]" : ""));
        Console.WriteLine("case                                            declared us  incumbent us   ratio  routed");
        var failures = new List<string>();
        foreach (var gate in Cases())
        {
            if (only is not null && !gate.Name.StartsWith(only, StringComparison.OrdinalIgnoreCase)) continue;
            var (call, buffers) = gate.Setup(backend);
            using (buffers)
            {
                // Warm both sides (the attention routes build a shape on its second sighting).
                bool routed;
                backend.DeclaredKernelsSwitch = true;
                call(); call(); backend.Synchronize();
                long before = backend.DeclaredProgramLaunches;
                call(); backend.Synchronize();
                routed = backend.DeclaredProgramLaunches > before;
                backend.DeclaredKernelsSwitch = false;
                call(); call(); backend.Synchronize();

                var declared = new double[samples];
                var incumbent = new double[samples];
                var ratios = new double[samples];
                for (int s = 0; s < samples; s++)
                {
                    bool declaredFirst = (s & 1) == 0;
                    for (int side = 0; side < 2; side++)
                    {
                        bool useDeclared = (side == 0) == declaredFirst;
                        backend.DeclaredKernelsSwitch = useDeclared;
                        long start = Stopwatch.GetTimestamp();
                        for (int i = 0; i < batch; i++) call();
                        backend.Synchronize();
                        double us = Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1000.0 / batch;
                        if (useDeclared) declared[s] = us; else incumbent[s] = us;
                    }
                    ratios[s] = incumbent[s] / declared[s];
                }
                backend.DeclaredKernelsSwitch = null;
                double md = Median(declared), mi = Median(incumbent), mr = Median(ratios);
                bool ok = routed ? mr >= minRatio : gated;
                if (!ok) failures.Add(gate.Name + (routed ? " ratio " + mr.ToString("F2", CultureInfo.InvariantCulture) : " not routed"));
                Console.WriteLine(gate.Name.PadRight(46) + md.ToString("F1", CultureInfo.InvariantCulture).PadLeft(13) +
                    mi.ToString("F1", CultureInfo.InvariantCulture).PadLeft(14) + mr.ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) +
                    "x" + (routed ? "  yes" : gated ? "  gated" : "  NO") + (ok ? "" : "  FAIL"));
            }
        }
        Console.WriteLine(failures.Count == 0 ? "GATE PASS" : "GATE FAIL: " + string.Join("; ", failures));
    }

    private static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        return sorted[sorted.Length / 2];
    }

    private static string? Value(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    // ---- data -------------------------------------------------------------------------------------------------

    private static float[] Values(Random random, int count, float magnitude = 3f)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)((random.NextDouble() * 2 - 1) * magnitude);
        return v;
    }

    private static IGpuBuffer Ints(CudaBackend backend, int[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length * sizeof(int));
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static IGpuBuffer Doubles(CudaBackend backend, double[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length * sizeof(double));
        var bytes = new byte[data.Length * sizeof(double)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static int[] GroupIds(Random random, int rows, int groups)
    {
        var ids = new int[rows];
        for (int i = 0; i < rows; i++) ids[i] = i % 11 == 10 ? -1 : random.Next(groups);
        return ids;
    }

    private static (int[] RowPtr, int[] Cols) Csr(Random random, int rows, int cols, double density)
    {
        var rowPtr = new int[rows + 1];
        var list = new List<int>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++) if (random.NextDouble() < density) list.Add(c);
            rowPtr[r + 1] = list.Count;
        }
        return (rowPtr, list.ToArray());
    }

    // ---- cases ------------------------------------------------------------------------------------------------

    private static IEnumerable<GateCase> Cases()
    {
        foreach (var (rows, inner, groups) in new[] { (1000, 16, 37), (4096, 8, 256), (20000, 64, 64), (16384, 128, 256), (65536, 32, 64) })
        {
            string shape = rows + "x" + inner + "->" + groups;
            yield return new GateCase("ScatterMaxArgmaxRows " + shape, b =>
            {
                var r = new Random(1); var set = new BufferSet();
                var src = set.Add(b.AllocateBuffer(Values(r, rows * inner))); var idx = set.Add(Ints(b, GroupIds(r, rows, groups)));
                var o = set.Add(b.AllocateBuffer(groups * inner)); var a = set.Add(b.AllocateBuffer(groups * inner));
                return (() => b.ScatterMaxWithArgmaxRows(src, idx, o, a, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterMeanRows " + shape, b =>
            {
                var r = new Random(31); var set = new BufferSet();
                var src = set.Add(b.AllocateBuffer(Values(r, rows * inner))); var idx = set.Add(Ints(b, GroupIds(r, rows, groups)));
                var o = set.Add(b.AllocateBuffer(groups * inner)); var c = set.Add(b.AllocateBuffer(groups));
                return (() => b.ScatterMeanRowsWithCounts(src, idx, o, c, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterSoftmaxRows " + shape, b =>
            {
                var r = new Random(32); var set = new BufferSet();
                var src = set.Add(b.AllocateBuffer(Values(r, rows * inner))); var idx = set.Add(Ints(b, GroupIds(r, rows, groups)));
                var o = set.Add(b.AllocateBuffer(rows * inner));
                return (() => b.ScatterSoftmaxRows(src, idx, o, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterAddRows " + shape, b =>
            {
                var r = new Random(2); var set = new BufferSet();
                var src = set.Add(b.AllocateBuffer(Values(r, rows * inner))); var idx = set.Add(Ints(b, GroupIds(r, rows, groups)));
                var o = set.Add(b.AllocateBuffer(groups * inner));
                return (() => b.ScatterAddRows(src, idx, o, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterMean " + shape, b =>
            {
                var r = new Random(3); var set = new BufferSet();
                var src = set.Add(b.AllocateBuffer(Values(r, rows * inner))); var idx = set.Add(Ints(b, GroupIds(r, rows, groups)));
                var o = set.Add(b.AllocateBuffer(groups * inner)); var c = set.Add(b.AllocateByteBuffer(groups * sizeof(int)));
                return (() => b.ScatterMean(src, idx, o, c, rows * inner, groups, inner), set);
            });
            yield return new GateCase("ScatterAddBackwardRows " + shape, b =>
            {
                var r = new Random(4); var set = new BufferSet();
                var g = set.Add(b.AllocateBuffer(Values(r, groups * inner))); var idx = set.Add(Ints(b, GroupIds(r, rows, groups)));
                var o = set.Add(b.AllocateBuffer(rows * inner));
                return (() => b.ScatterAddBackwardRows(g, idx, o, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterMeanBackwardRows " + shape, b =>
            {
                var r = new Random(5); var set = new BufferSet();
                var ids = GroupIds(r, rows, groups); var counts = new int[groups];
                foreach (int id in ids) if (id >= 0) counts[id]++;
                var g = set.Add(b.AllocateBuffer(Values(r, groups * inner))); var idx = set.Add(Ints(b, ids));
                var c = set.Add(Ints(b, counts)); var o = set.Add(b.AllocateBuffer(rows * inner));
                return (() => b.ScatterMeanBackwardRows(g, idx, c, o, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterMaxBackwardRows " + shape, b =>
            {
                var r = new Random(6); var set = new BufferSet();
                var argmax = new int[groups * inner];
                for (int i = 0; i < argmax.Length; i++) argmax[i] = r.Next(rows);
                var g = set.Add(b.AllocateBuffer(Values(r, groups * inner))); var a = set.Add(Ints(b, argmax));
                var o = set.Add(b.AllocateBuffer(rows * inner));
                return (() => b.ScatterMaxBackwardRows(g, a, o, rows, inner, groups), set);
            });
            yield return new GateCase("ScatterSoftmaxBackwardRows " + shape, b =>
            {
                var r = new Random(7); var set = new BufferSet();
                var g = set.Add(b.AllocateBuffer(Values(r, rows * inner))); var y = set.Add(b.AllocateBuffer(Values(r, rows * inner, 1f)));
                var idx = set.Add(Ints(b, GroupIds(r, rows, groups))); var o = set.Add(b.AllocateBuffer(rows * inner));
                return (() => b.ScatterSoftmaxBackwardRows(g, y, idx, o, rows, inner, groups), set);
            });
        }

        foreach (var (rows, cols, n, density) in new[] { (64, 48, 16, 0.1), (128, 256, 32, 0.05), (512, 512, 64, 0.01), (2048, 2048, 64, 0.01), (8192, 8192, 128, 0.002), (4096, 1024, 256, 0.05) })
        {
            string shape = rows + "x" + cols + "x" + n;
            Func<CudaBackend, Random, BufferSet, (IGpuBuffer V, IGpuBuffer C, IGpuBuffer R, IGpuBuffer D, int Nnz)> csr = (b, r, set) =>
            {
                var (rowPtr, colsArr) = Csr(r, rows, cols, density);
                return (set.Add(b.AllocateBuffer(Values(r, colsArr.Length))), set.Add(Ints(b, colsArr)), set.Add(Ints(b, rowPtr)),
                    set.Add(b.AllocateBuffer(Values(r, cols * n))), colsArr.Length);
            };
            yield return new GateCase("CsrSpMMWarp " + shape, b =>
            {
                var set = new BufferSet(); var m = csr(b, new Random(8), set); var o = set.Add(b.AllocateBuffer(rows * n));
                return (() => b.CsrSpMMWarp(m.V, m.C, m.R, m.D, o, rows, cols, n, m.Nnz), set);
            });
            yield return new GateCase("CsrSpMMVec4 " + shape, b =>
            {
                var set = new BufferSet(); var m = csr(b, new Random(9), set); var o = set.Add(b.AllocateBuffer(rows * n));
                return (() => b.CsrSpMMVec4(m.V, m.C, m.R, m.D, o, rows, cols, n, m.Nnz), set);
            });
            yield return new GateCase("CsrSpMMBias " + shape, b =>
            {
                var set = new BufferSet(); var m = csr(b, new Random(10), set); var o = set.Add(b.AllocateBuffer(rows * n));
                var bias = set.Add(b.AllocateBuffer(Values(new Random(11), n)));
                return (() => b.CsrSpMMBias(m.V, m.C, m.R, m.D, bias, o, rows, cols, n, m.Nnz), set);
            });
            yield return new GateCase("CsrSpMMDouble " + shape, b =>
            {
                var r = new Random(12); var set = new BufferSet();
                var (rowPtr, colsArr) = Csr(r, rows, cols, density);
                var vals = new double[colsArr.Length]; for (int i = 0; i < vals.Length; i++) vals[i] = r.NextDouble() - 0.5;
                var dense = new double[cols * n]; for (int i = 0; i < dense.Length; i++) dense[i] = r.NextDouble() - 0.5;
                var v = set.Add(Doubles(b, vals)); var c = set.Add(Ints(b, colsArr)); var rp = set.Add(Ints(b, rowPtr));
                var d = set.Add(Doubles(b, dense)); var o = set.Add(b.AllocateByteBuffer(rows * n * sizeof(double)));
                return (() => b.CsrSpMMDouble(v, c, rp, d, o, rows, cols, n, colsArr.Length), set);
            });
            yield return new GateCase("CsrSddmm " + shape, b =>
            {
                var r = new Random(13); var set = new BufferSet();
                var (rowPtr, colsArr) = Csr(r, rows, cols, density);
                var rowsArr = new int[colsArr.Length];
                for (int row = 0; row < rows; row++) for (int p = rowPtr[row]; p < rowPtr[row + 1]; p++) rowsArr[p] = row;
                var ri = set.Add(Ints(b, rowsArr)); var ci = set.Add(Ints(b, colsArr));
                var x = set.Add(b.AllocateBuffer(Values(r, rows * n))); var y = set.Add(b.AllocateBuffer(Values(r, cols * n)));
                var o = set.Add(b.AllocateBuffer(colsArr.Length));
                return (() => b.CsrSddmm(ri, ci, x, y, o, colsArr.Length, n), set);
            });
            foreach (string op in new[] { "Max", "Min", "StdDev" })
                yield return new GateCase("CsrSegmented" + op + " " + shape, b =>
                {
                    var r = new Random(14); var set = new BufferSet();
                    var (rowPtr, colsArr) = Csr(r, rows, cols, density);
                    var ci = set.Add(b.AllocateBuffer(colsArr.Select(c => (float)c).ToArray()));
                    var rp = set.Add(b.AllocateBuffer(rowPtr.Select(p => (float)p).ToArray()));
                    var x = set.Add(b.AllocateBuffer(Values(r, cols * n))); var o = set.Add(b.AllocateBuffer(rows * n));
                    Action call = op switch
                    {
                        "Max" => () => b.CsrSegmentedMax(ci, rp, x, o, rows, cols, n),
                        "Min" => () => b.CsrSegmentedMin(ci, rp, x, o, rows, cols, n),
                        _ => () => b.CsrSegmentedStdDev(ci, rp, x, o, rows, cols, n),
                    };
                    return (call, set);
                });
        }

        foreach (var (nodes, edges, features, weighted) in new[] { (50, 200, 16, false), (300, 1000, 32, true), (1000, 4000, 64, true), (100, 20000, 128, false) })
            yield return new GateCase("ScatterAddEdges " + nodes + "x" + edges + "x" + features + (weighted ? " w" : ""), b =>
            {
                var r = new Random(15); var set = new BufferSet();
                var src = new int[edges]; var tgt = new int[edges];
                for (int e = 0; e < edges; e++) { src[e] = r.Next(nodes); tgt[e] = r.Next(nodes); }
                var x = set.Add(b.AllocateBuffer(Values(r, nodes * features)));
                var s = set.Add(Ints(b, src)); var t = set.Add(Ints(b, tgt));
                IGpuBuffer? w = weighted ? set.Add(b.AllocateBuffer(Values(r, edges, 1f))) : null;
                var o = set.Add(b.AllocateBuffer(nodes * features));
                return (() => b.ScatterAddEdges(x, s, t, w, o, nodes, edges, features), set);
            });

        foreach (var (outer, idxLen, inner, dst) in new[] { (2, 9, 5, 12), (4, 64, 16, 100), (8, 256, 64, 512), (1, 1024, 512, 1024) })
        {
            string shape = outer + "x" + idxLen + "x" + inner + "->" + dst;
            Func<CudaBackend, Random, BufferSet, (IGpuBuffer O, IGpuBuffer I, IGpuBuffer S)> data = (b, r, set) =>
            {
                var ids = new int[idxLen]; for (int j = 0; j < idxLen; j++) ids[j] = r.Next(dst);
                return (set.Add(b.AllocateBuffer(Values(r, outer * dst * inner))), set.Add(Ints(b, ids)),
                    set.Add(b.AllocateBuffer(Values(r, outer * idxLen * inner))));
            };
            yield return new GateCase("IndexWrite " + shape, b =>
            {
                var set = new BufferSet(); var d = data(b, new Random(16), set);
                return (() => b.IndexWrite(d.O, d.I, d.S, 0f, 0, outer, idxLen, inner, dst), set);
            });
            yield return new GateCase("IndexFill " + shape, b =>
            {
                var set = new BufferSet(); var d = data(b, new Random(17), set);
                return (() => b.Parity210IndexFill(d.O, d.I, 1.5f, outer, dst, inner, idxLen), set);
            });
            yield return new GateCase("IndexAdd " + shape, b =>
            {
                var set = new BufferSet(); var d = data(b, new Random(18), set);
                return (() => b.Parity210IndexAdd(d.O, d.I, d.S, outer, dst, inner, idxLen), set);
            });
        }

        foreach (var (outer, srcDim, dstDim, inner) in new[] { (2, 9, 5, 3), (3, 200, 40, 8), (4, 1024, 256, 16), (2, 16384, 32, 64) })
            foreach (int mode in new[] { 0, 1, 2, 3 })
                yield return new GateCase("ScatterReduce" + mode + " " + outer + "x" + srcDim + "->" + dstDim + "x" + inner, b =>
                {
                    var r = new Random(19); var set = new BufferSet();
                    var index = new int[outer * srcDim * inner]; for (int k = 0; k < index.Length; k++) index[k] = r.Next(dstDim);
                    var o = set.Add(b.AllocateBuffer(Values(r, outer * dstDim * inner, 1f)));
                    var s = set.Add(b.AllocateBuffer(Values(r, outer * srcDim * inner, mode == 1 ? 1.0001f : 3f)));
                    var i = set.Add(Ints(b, index));
                    return (() => b.ScatterReduce(o, s, i, outer, srcDim, dstDim, inner, mode), set);
                });

        yield return new GateCase("TakeAlongDimInt 4x64x80<-50", b =>
        {
            var r = new Random(20); var set = new BufferSet();
            var x = set.Add(b.AllocateBuffer(Values(r, 4 * 50 * 80)));
            var ids = new int[4 * 64 * 80]; for (int k = 0; k < ids.Length; k++) ids[k] = r.Next(50);
            var i = set.Add(Ints(b, ids)); var o = set.Add(b.AllocateBuffer(ids.Length));
            return (() => b.Parity210TakeAlongDim(x, i, o, 4, 64, 80, 50), set);
        });
        yield return new GateCase("TakeLinear 65536<-1M", b =>
        {
            var r = new Random(21); var set = new BufferSet();
            var x = set.Add(b.AllocateBuffer(Values(r, 1 << 20)));
            var ids = new int[65536]; for (int k = 0; k < ids.Length; k++) ids[k] = r.Next(1 << 20);
            var i = set.Add(Ints(b, ids)); var o = set.Add(b.AllocateBuffer(ids.Length));
            return (() => b.Parity210TakeLinear(x, i, o, ids.Length, 1 << 20), set);
        });
        yield return new GateCase("IndexSelect 2048x512<-8192", b =>
        {
            var r = new Random(22); var set = new BufferSet();
            var x = set.Add(b.AllocateBuffer(Values(r, 8192 * 512)));
            var ids = new float[2048]; for (int k = 0; k < ids.Length; k++) ids[k] = r.Next(8192);
            var i = set.Add(b.AllocateBuffer(ids)); var o = set.Add(b.AllocateBuffer(2048 * 512));
            return (() => b.IndexSelect(x, i, o, 2048, 512), set);
        });
        yield return new GateCase("ScatterAddBackward 2048x512<-8192", b =>
        {
            var r = new Random(23); var set = new BufferSet();
            var g = set.Add(b.AllocateBuffer(Values(r, 8192 * 512)));
            var ids = new int[2048]; for (int k = 0; k < ids.Length; k++) ids[k] = r.Next(8192);
            var i = set.Add(Ints(b, ids)); var o = set.Add(b.AllocateBuffer(2048 * 512));
            return (() => b.ScatterAddBackward(g, i, o, 2048, 512), set);
        });
        yield return new GateCase("MaskedScatter 1M<-400k", b =>
        {
            var r = new Random(24); var set = new BufferSet();
            int total = 1 << 20; var mask = new byte[total]; var prefix = new int[total]; int run = 0;
            for (int k = 0; k < total; k++) { mask[k] = (byte)(r.NextDouble() < 0.4 ? 1 : 0); prefix[k] = mask[k] != 0 ? run++ : -1; }
            var o = set.Add(b.AllocateBuffer(Values(r, total))); var m = set.Add(b.AllocateByteBuffer(total));
            b.UploadByteBuffer(m, mask);
            var p = set.Add(Ints(b, prefix)); var s = set.Add(b.AllocateBuffer(Values(r, run)));
            return (() => b.Parity210MaskedScatter(o, m, p, s, total), set);
        });

        foreach (var (bt, h, kvh, sq, sk, d, causal) in new[]
                 {
                     (2, 8, 2, 128, 128, 64, false), (2, 8, 2, 128, 128, 64, true), (1, 16, 4, 192, 192, 64, true),
                     (4, 32, 8, 1, 1024, 128, true),
                 })
        {
            string shape = bt + "x" + h + "/" + kvh + "x" + sq + "x" + sk + "x" + d + (causal ? " causal" : "");
            float scale = 1f / MathF.Sqrt(d);
            int qs = bt * h * sq * d, kvs = bt * kvh * sk * d, ps = bt * h * sq * sk;
            yield return new GateCase("GroupedQueryAttention " + shape, b =>
            {
                var r = new Random(25); var set = new BufferSet();
                var q = set.Add(b.AllocateBuffer(Values(r, qs, 1f))); var k = set.Add(b.AllocateBuffer(Values(r, kvs, 1f)));
                var v = set.Add(b.AllocateBuffer(Values(r, kvs, 1f))); var o = set.Add(b.AllocateBuffer(qs));
                return (() => b.GroupedQueryAttention(q, k, v, o, null, bt, h, kvh, sq, sk, d, scale, causal), set);
            });
            yield return new GateCase("GroupedQueryAttentionBackward " + shape, b =>
            {
                var r = new Random(26); var set = new BufferSet();
                var q = set.Add(b.AllocateBuffer(Values(r, qs, 1f))); var k = set.Add(b.AllocateBuffer(Values(r, kvs, 1f)));
                var v = set.Add(b.AllocateBuffer(Values(r, kvs, 1f))); var dout = set.Add(b.AllocateBuffer(Values(r, qs, 1f)));
                var w = set.Add(b.AllocateBuffer(Values(r, ps, 0.01f)));
                var dq = set.Add(b.AllocateBuffer(qs)); var dk = set.Add(b.AllocateBuffer(kvs)); var dv = set.Add(b.AllocateBuffer(kvs));
                return (() => b.GroupedQueryAttentionBackward(dout, q, k, v, w, dq, dk, dv, bt, h, kvh, sq, sk, d, scale, h / kvh), set);
            });
        }
        foreach (var (h, kvh, d, seqLen) in new[] { (32, 8, 128, 1024), (32, 8, 128, 4096), (16, 16, 80, 512) })
            yield return new GateCase("FlashDecode " + h + "/" + kvh + "x" + d + "x" + seqLen, b =>
            {
                var r = new Random(27); var set = new BufferSet();
                var q = set.Add(b.AllocateBuffer(Values(r, h * d, 1f)));
                var k = set.Add(b.AllocateBuffer(Values(r, seqLen * kvh * d, 1f)));
                var v = set.Add(b.AllocateBuffer(Values(r, seqLen * kvh * d, 1f)));
                float scale = 1f / MathF.Sqrt(d);
                return (() => b.FlashDecode(q, k, v, h, kvh, d, seqLen, scale).Dispose(), set);
            });
    }
}