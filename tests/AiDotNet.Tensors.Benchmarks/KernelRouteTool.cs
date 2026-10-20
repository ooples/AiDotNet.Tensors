// Copyright (c) AiDotNet. All rights reserved.
// --kernel-routes: the production A/B of the declared-program route (#890 phase 4). Each case calls the CUDA BACKEND's
// own entry point (what DirectGpuTensorEngine calls) twice on identical inputs -- once with the declared route on and
// once off, so the hand-written kernel serves it. Each route is scored against an fp64 reference of the same
// contract: the declared route must have fired and must sit within the fp32 conditioning bound of the problem --
// the data are offset by 1000, so rounding the mean alone moves y by about u * |x| * rstd * |gamma| -- and the
// hand-written kernel's error is printed beside it. The two paired timings are reported.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelRouteTool
{
    private const double Unit = 1.0 / (1 << 24);   // fp32 unit roundoff

    internal static void Run(string[] args)
    {
        // --dump <rows> <cols> <dir>: write the LayerNorm program's lowered PTX for sm_75, one file per launch.
        if (args.Length == 4 && args[0] == "--dump")
        {
            int r = int.Parse(args[1], CultureInfo.InvariantCulture), c = int.Parse(args[2], CultureInfo.InvariantCulture);
            var lowering = AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx.PtxProgramLowering.Create(
                AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenBackendPrograms.LayerNorm(r, c, 1e-5f),
                CudaBackend.LayerNormKey(r, c, 1e-5f).ToString(), 7, 5);
            System.IO.Directory.CreateDirectory(args[3]);
            foreach (var step in lowering.Steps)
                System.IO.File.WriteAllText(System.IO.Path.Combine(args[3], step.EntryPoint + ".ptx"), step.Ptx);
            Console.WriteLine(lowering.Steps.Count + " kernels written to " + args[3]);
            return;
        }
        using var engine = new DirectGpuEngine();
        if (engine.Backend is not CudaBackend backend)
        {
            Console.WriteLine("The declared-program route is CUDA-only; no CUDA backend here.");
            return;
        }

        var failures = new List<string>();
        Console.WriteLine("ROUTES - backend entry points, declared program vs hand-written kernel");
        Console.WriteLine("case                         routed  declared err  hand-written err   declared us  hand-written us   ratio");

        foreach (var (rows, cols) in new[] { (8192, 1024), (8192, 64), (256, 64), (33, 96), (100, 100), (4, 4096), (128, 1000), (512, 7),
                     (64, 50000), (16, 50001), (8192, 768), (2048, 4000), (1024, 3136), (4096, 1568) })
        {
            string name = "LayerNorm " + rows + "x" + cols;
            int n = rows * cols;
            var x = Fill(n, 1, 1000.0);   // offset data: the cancellation case
            var g = Fill(cols, 2, 1.0);
            var b = Fill(cols, 3, 0.0);
            using var bx = backend.AllocateBuffer(x);
            using var bg = backend.AllocateBuffer(g);
            using var bb = backend.AllocateBuffer(b);
            var outputs = new Dictionary<bool, (float[] Y, float[] Mean, float[] Rstd)>();
            var times = new Dictionary<bool, double>();
            var bytes = new Dictionary<bool, double>();
            bool routed = false;
            foreach (bool declared in new[] { true, false })
            {
                backend.DeclaredKernelsSwitch = declared;
                using var by = backend.AllocateBuffer(n);
                using var bm = backend.AllocateBuffer(rows);
                using var bv = backend.AllocateBuffer(rows);
                long before = backend.DeclaredProgramLaunches;
                void Call() => backend.LayerNorm(bx, by, bg, bb, bm, bv, rows, cols, 1e-5f);
                Call();
                backend.Synchronize();
                if (declared) routed = backend.DeclaredProgramLaunches > before;
                outputs[declared] = (backend.DownloadBuffer(by), backend.DownloadBuffer(bm), backend.DownloadBuffer(bv));
                times[declared] = Time(Call, backend.Synchronize);
                bytes[declared] = AllocatedPerCall(Call, backend.Synchronize);
            }
            backend.DeclaredKernelsSwitch = null;

            var (refY, refMean, refRstd) = LayerNormReference(x, g, b, rows, cols, 1e-5);
            double Error(bool declared) => Math.Max(MaxDev(outputs[declared].Y, refY),
                Math.Max(MaxDev(outputs[declared].Mean, refMean), MaxDev(outputs[declared].Rstd, refRstd)));
            double declaredError = Error(true), handError = Error(false);
            double bound = ConditioningBound(x, g, refRstd, cols);
            bool ok = routed && !double.IsNaN(declaredError) && declaredError <= bound;
            if (!ok) failures.Add(name + (routed ? ": error " + declaredError.ToString("E3", CultureInfo.InvariantCulture) +
                " over the bound " + bound.ToString("E3", CultureInfo.InvariantCulture) : ": not routed"));
            Console.WriteLine(name.PadRight(28) + (routed ? "yes" : "no").PadLeft(7) +
                declaredError.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                handError.ToString("E3", CultureInfo.InvariantCulture).PadLeft(18) +
                times[true].ToString("F1", CultureInfo.InvariantCulture).PadLeft(14) +
                times[false].ToString("F1", CultureInfo.InvariantCulture).PadLeft(17) +
                (times[false] / times[true]).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" +
                (ok ? "" : "  FAIL"));
            var key = CudaBackend.LayerNormKey(rows, cols, 1e-5f);
            var steps = backend.DescribeDeclaredProgram(key);
            if (steps is not null) Console.WriteLine("    " + string.Join(" | ", steps) +
                "   host bytes/call: declared " + bytes[true].ToString("F0", CultureInfo.InvariantCulture) +
                ", hand-written " + bytes[false].ToString("F0", CultureInfo.InvariantCulture));
            else Console.WriteLine("    declined: " + backend.DeclaredDeclineReason(key));
        }

        // Capture: a shape first met under capture falls through; built outside, it captures and replays.
        {
            const int rows = 48, cols = 200;
            var x = Fill(rows * cols, 1, 1000.0);
            var g = Fill(cols, 2, 1.0);
            var b = Fill(cols, 3, 0.0);
            using var bx = backend.AllocateBuffer(x);
            using var bg = backend.AllocateBuffer(g);
            using var bb = backend.AllocateBuffer(b);
            using var by = backend.AllocateBuffer(rows * cols);
            using var bm = backend.AllocateBuffer(rows);
            using var bv = backend.AllocateBuffer(rows);
            void Call() => backend.LayerNorm(bx, by, bg, bb, bm, bv, rows, cols, 1e-5f);
            var (refY, refMean, refRstd) = LayerNormReference(x, g, b, rows, cols, 1e-5);
            double bound = ConditioningBound(x, g, refRstd, cols);
            foreach (bool built in new[] { false, true })
            {
                if (built) Call();
                long before = backend.DeclaredProgramLaunches;
                IntPtr graph = backend.CaptureGraph(Call);
                bool routed = backend.DeclaredProgramLaunches > before;
                backend.LaunchCapturedGraph(graph);
                backend.Synchronize();
                double error = Math.Max(MaxDev(backend.DownloadBuffer(by), refY),
                    Math.Max(MaxDev(backend.DownloadBuffer(bm), refMean), MaxDev(backend.DownloadBuffer(bv), refRstd)));
                backend.DestroyCapturedGraph(graph);
                bool ok = routed == built && error <= bound;
                Console.WriteLine("capture, program " + (built ? "built first" : "unbuilt") + ": routed " + routed +
                    ", replay error " + error.ToString("E3", CultureInfo.InvariantCulture) + (ok ? "" : "  FAIL"));
                if (!ok) failures.Add("capture (" + (built ? "built" : "unbuilt") + ")");
            }
        }

        if (failures.Count != 0)
            throw new InvalidOperationException("Route disagreement: " + string.Join("; ", failures));
    }

    private static (double[] Y, double[] Mean, double[] Rstd) LayerNormReference(
        float[] x, float[] gamma, float[] beta, int rows, int cols, double epsilon)
    {
        var y = new double[x.Length];
        var mean = new double[rows];
        var rstd = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += x[r * cols + c];
            double m = sum / cols, sq = 0;
            for (int c = 0; c < cols; c++) { double d = x[r * cols + c] - m; sq += d * d; }
            double s = 1.0 / Math.Sqrt(sq / cols + epsilon);
            mean[r] = m;
            rstd[r] = s;
            for (int c = 0; c < cols; c++) y[r * cols + c] = (x[r * cols + c] - m) * s * gamma[c] + beta[c];
        }
        return (y, mean, rstd);
    }

    // 16 roundings of the largest input, carried through rstd and gamma, plus the reduction's log2(n) growth.
    private static double ConditioningBound(float[] x, float[] gamma, double[] rstd, int cols)
    {
        double maxX = 0, maxGamma = 1, maxRstd = 0;
        foreach (float v in x) maxX = Math.Max(maxX, Math.Abs(v));
        foreach (float v in gamma) maxGamma = Math.Max(maxGamma, Math.Abs(v));
        foreach (double v in rstd) maxRstd = Math.Max(maxRstd, v);
        return Unit * (16 + Math.Log(cols, 2)) * Math.Max(1.0, maxX * maxRstd * maxGamma);
    }

    private static float[] Fill(int count, int salt, double offset)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(((i * 37L + salt * 101) % 97 - 48) / 64.0 + offset);
        return v;
    }

    private static double MaxDev(float[] a, double[] reference)
    {
        double worst = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (float.IsNaN(a[i])) return double.NaN;
            worst = Math.Max(worst, Math.Abs(a[i] - reference[i]) / Math.Max(1.0, Math.Abs(reference[i])));
        }
        return worst;
    }

    // Managed bytes this thread allocates per call, over 1000 calls (deterministic, unlike time).
    private static double AllocatedPerCall(Action call, Action sync)
    {
        call();
        sync();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) call();
        long after = GC.GetAllocatedBytesForCurrentThread();
        sync();
        return (after - before) / 1000.0;
    }

    // Median of 21 samples of 20 calls each, after warming: microseconds per call.
    private static double Time(Action call, Action sync)
    {
        for (int i = 0; i < 10; i++) call();
        sync();
        var samples = new double[21];
        for (int s = 0; s < samples.Length; s++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 20; i++) call();
            sync();
            samples[s] = Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1000.0 / 20;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }
}
