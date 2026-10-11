using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// #653 phase 2 baseline: parallel efficiency of the everyday non-GEMM ops (elementwise, broadcast, activations,
/// norms, reductions, layout) across a DOP sweep. Each call runs eagerly inside its own create-and-dispose TensorArena
/// -- the per-step pattern eager model code uses -- after five warm-up calls, so outputs come from the arena's
/// persistent pool rather than fresh allocations and the numbers measure the kernel and its fan-out. The sweep must
/// include DOP 1, the efficiency baseline.
/// usage: --ops [--dtype float|double] [--dops 1,2,4,8,16,32,64] [--reps 60] [--filter name]
/// </summary>
internal static partial class Program
{
    private static int RunOps(CpuEngine eng, string[] a)
    {
        string dtype = ArgS(a, "--dtype", "float");
        int reps = ArgI(a, "--reps", 60);
        string filter = ArgS(a, "--filter", "");
        if (dtype != "float" && dtype != "double") return OpsUsageError($"--dtype must be float or double, not '{dtype}'.");
        if (reps <= 0) return OpsUsageError($"--reps must be positive, not {reps}.");
        var dopText = ArgS(a, "--dops", "1,2,4,8,16,32,64").Split(',');
        var dops = new int[dopText.Length];
        for (int i = 0; i < dopText.Length; i++)
            if (!int.TryParse(dopText[i], out dops[i]) || dops[i] <= 0)
                return OpsUsageError($"--dops entries must be positive integers, not '{dopText[i]}'.");
        if (Array.IndexOf(dops, 1) < 0) return OpsUsageError("--dops must include 1: it is the efficiency baseline.");
        return dtype == "double" ? RunOpsT<double>(eng, dops, reps, filter) : RunOpsT<float>(eng, dops, reps, filter);
    }

    private static int OpsUsageError(string message)
    {
        Console.Error.WriteLine("--ops: " + message);
        return 2;
    }

    private static int RunOpsT<T>(CpuEngine eng, int[] dops, int reps, string filter)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var rng = new Random(0);
        Tensor<T> R(params int[] shape)
        {
            var t = new Tensor<T>(shape);
            for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble(rng.NextDouble() * 2 - 1);
            return t;
        }

        // [rows, cols] shapes: a small model activation (196K), a mid FFN block (2M), a large one (8M).
        var shapes = new[] { new[] { 6144, 32 }, new[] { 8192, 256 }, new[] { 8192, 1024 } };
        Console.WriteLine($"OPS dtype={typeof(T).Name} procs={Environment.ProcessorCount} reps={reps} (median us; eff = t1 / (dop * t_dop))");
        var header = "op                         elems   " + string.Join("", Array.ConvertAll(dops, d => $"{"dop" + d,10}")) + $"{"eff@8",8}{"eff@max",9}";
        Console.WriteLine(header);
        foreach (var shape in shapes)
        {
            int rows = shape[0], cols = shape[1];
            var x = R(rows, cols); var y = R(rows, cols); var bias = R(cols); var gamma = R(cols); var beta = R(cols);
            var x3 = eng.Reshape(x, new[] { rows / 2, 2, cols });
            var cases = new List<(string, Action)>
            {
                ("add", () => eng.TensorAdd(x, y)),
                ("multiply", () => eng.TensorMultiply(x, y)),
                ("add_rowbias_bcast", () => eng.TensorBroadcastAdd(x, bias)),
                ("relu", () => eng.ReLU(x)),
                ("sigmoid", () => eng.Sigmoid(x)),
                ("tanh", () => eng.Tanh(x)),
                ("gelu", () => eng.GELU(x)),
                ("elu", () => eng.ELU(x)),
                ("exp", () => eng.TensorExp(x)),
                ("layernorm_lastdim", () => eng.LayerNorm(x, gamma, beta, 1e-5, out _, out _)),
                ("softmax_lastdim", () => eng.Softmax(x, -1)),
                ("reducesum_lastdim", () => eng.ReduceSum(x, new[] { 1 }, keepDims: false)),
                ("reducemean_axis0", () => eng.ReduceMean(x, new[] { 0 }, keepDims: false)),
                ("reducemax_mid", () => eng.ReduceMax(x3, new[] { 1 }, keepDims: false)),
                ("transpose_contig", () => eng.TensorTranspose(x).Contiguous()),
                ("concat_axis1", () => eng.Concat(new[] { x, y }, 1)),
            };
            foreach (var (name, run) in cases)
            {
                if (filter.Length > 0 && !name.Contains(filter, StringComparison.OrdinalIgnoreCase)) continue;
                var times = new double[dops.Length];
                for (int di = 0; di < dops.Length; di++)
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = dops[di];
                    for (int w = 0; w < 5; w++) { using var ar = TensorArena.Create(); run(); }
                    var ts = new double[reps];
                    for (int r = 0; r < reps; r++)
                    {
                        using var ar = TensorArena.Create();
                        long t0 = Stopwatch.GetTimestamp();
                        run();
                        ts[r] = (Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency;
                    }
                    Array.Sort(ts);
                    times[di] = ts[reps / 2];
                }
                double t1 = times[Array.IndexOf(dops, 1)];
                int i8 = Array.IndexOf(dops, 8);
                double eff8 = i8 >= 0 ? t1 / (8 * times[i8]) : double.NaN;
                double effMax = t1 / (dops[^1] * times[^1]);
                Console.WriteLine($"{name,-26}{rows * cols,8}   " + string.Join("", Array.ConvertAll(times, t => $"{t,10:F1}")) + $"{eff8,8:P0}{effMax,9:P0}");
            }
        }
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        return 0;
    }
}
