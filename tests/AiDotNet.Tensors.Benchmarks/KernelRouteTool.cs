// Copyright (c) AiDotNet. All rights reserved.
// --kernel-routes: the production A/B of the declared-program routes (#890 phase 4). Each case calls a CUDA BACKEND
// entry point (what DirectGpuTensorEngine calls) twice on identical inputs -- with the declared route on, and off so
// the incumbent (the hand-written kernel, or cuDNN where the backend prefers it) serves it -- and scores BOTH against
// an fp64 reference of the call's contract. A case passes when the declared route fired and its error is within the
// fp32 conditioning bound of the problem, or no worse than twice the incumbent's own error against the same
// reference. Paired timings and host bytes per call are reported.
//
// Forward normalisations run on data offset by 1000, where rounding the mean alone moves y by about
// u * |x| * rstd * |gamma|; backward cases take their saved statistics from the fp64 forward, rounded to fp32, so
// every route and the reference read the same values.
//
// --dump <rows> <cols> <dir>: write the LayerNorm program's lowered PTX for sm_75, one file per launch.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelRouteTool
{
    private const double Unit = 1.0 / (1 << 24);   // fp32 unit roundoff

    /// <summary>One routed call: its buffers (inputs first), the call, and what is checked against what.</summary>
    private sealed class RouteCase
    {
        internal RouteCase(string name, DeclaredKey key, float[][] inputs, int[] outputSizes,
            Action<CudaBackend, IGpuBuffer[]> call, (string Name, int Buffer, double[] Reference)[] checks,
            double scale, int reduction)
        {
            Name = name; Key = key; Inputs = inputs; OutputSizes = outputSizes; Call = call; Checks = checks;
            Scale = scale; Reduction = reduction;
        }

        internal string Name { get; }
        internal DeclaredKey Key { get; }
        internal float[][] Inputs { get; }
        internal int[] OutputSizes { get; }
        internal Action<CudaBackend, IGpuBuffer[]> Call { get; }
        internal (string Name, int Buffer, double[] Reference)[] Checks { get; }

        /// <summary>The problem's magnitude: the fp32 bound is Unit * (16 + log2 Reduction) * max(1, Scale).</summary>
        internal double Scale { get; }

        internal int Reduction { get; }
    }

    internal static void Run(string[] args)
    {
        if (args.Length == 4 && args[0] == "--dump")
        {
            int r = int.Parse(args[1], CultureInfo.InvariantCulture), c = int.Parse(args[2], CultureInfo.InvariantCulture);
            var lowering = PtxProgramLowering.Create(CodegenBackendPrograms.LayerNorm(r, c, 1e-5f),
                CudaBackend.LayerNormKey(r, c, 1e-5f).ToString(), 7, 5);
            System.IO.Directory.CreateDirectory(args[3]);
            foreach (var step in lowering.Steps)
                System.IO.File.WriteAllText(System.IO.Path.Combine(args[3], step.EntryPoint + ".ptx"), step.Ptx);
            Console.WriteLine(lowering.Steps.Count + " kernels written to " + args[3]);
            return;
        }
        if (args.Length > 0 && args[0] == "--static")
        {
            KernelRouteStaticTool.Run(args);
            return;
        }
        string? only = args.Length >= 2 && args[0] == "--family" ? args[1] : null;

        using var engine = new DirectGpuEngine();
        if (engine.Backend is not CudaBackend backend)
        {
            Console.WriteLine("The declared-program route is CUDA-only; no CUDA backend here.");
            return;
        }

        var failures = new List<string>();
        Console.WriteLine("ROUTES - backend entry points, declared program vs incumbent, each against fp64");
        Console.WriteLine("case                               routed  declared err  incumbent err      bound   declared us  incumbent us   ratio");
        foreach (var route in Cases())
        {
            if (only is not null && !route.Name.StartsWith(only, StringComparison.OrdinalIgnoreCase)) continue;
            RunCase(backend, route, failures);
        }
        if (only is null || only.Equals("capture", StringComparison.OrdinalIgnoreCase)) CaptureCheck(backend, failures);
        if (only is null || only.Equals("resident", StringComparison.OrdinalIgnoreCase)) ResidentChains(backend, failures);
        if (only is null || only.Equals("conv", StringComparison.OrdinalIgnoreCase)) ConvRoutes(backend, failures);

        if (failures.Count != 0)
            throw new InvalidOperationException("Route failures: " + string.Join("; ", failures));
    }

    private static void RunCase(CudaBackend backend, RouteCase route, List<string> failures)
    {
        var errors = new Dictionary<bool, double>();
        var times = new Dictionary<bool, double>();
        var bytes = new Dictionary<bool, double>();
        bool routed = false;
        foreach (bool declared in new[] { true, false })
        {
            backend.DeclaredKernelsSwitch = declared;
            var buffers = new IGpuBuffer[route.Inputs.Length + route.OutputSizes.Length];
            try
            {
                for (int i = 0; i < route.Inputs.Length; i++) buffers[i] = backend.AllocateBuffer(route.Inputs[i]);
                for (int o = 0; o < route.OutputSizes.Length; o++)
                    buffers[route.Inputs.Length + o] = backend.AllocateBuffer(new float[route.OutputSizes[o]]);
                long before = backend.DeclaredProgramLaunches;
                route.Call(backend, buffers);
                backend.Synchronize();
                if (declared) routed = backend.DeclaredProgramLaunches > before;
                double worst = 0;
                foreach (var check in route.Checks)
                    worst = Math.Max(worst, MaxDev(backend.DownloadBuffer(buffers[check.Buffer]), check.Reference));
                errors[declared] = worst;
                void Call() => route.Call(backend, buffers);
                times[declared] = Time(Call, backend.Synchronize);
                bytes[declared] = AllocatedPerCall(Call, backend.Synchronize);
            }
            finally
            {
                foreach (var buffer in buffers) buffer?.Dispose();
                backend.DeclaredKernelsSwitch = null;
            }
        }

        double bound = Unit * (16 + Math.Log(Math.Max(2, route.Reduction), 2)) * Math.Max(1.0, route.Scale);
        double declaredError = errors[true], incumbentError = errors[false];
        bool ok = routed && !double.IsNaN(declaredError) && (declaredError <= bound || declaredError <= 2 * incumbentError);
        if (!ok)
            failures.Add(route.Name + (routed ? ": error " + E(declaredError) + " over bound " + E(bound) : ": not routed"));
        Console.WriteLine(route.Name.PadRight(34) + (routed ? "yes" : "no").PadLeft(7) + E(declaredError).PadLeft(14) +
            E(incumbentError).PadLeft(15) + E(bound).PadLeft(11) +
            times[true].ToString("F1", CultureInfo.InvariantCulture).PadLeft(14) +
            times[false].ToString("F1", CultureInfo.InvariantCulture).PadLeft(14) +
            (times[false] / times[true]).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" + (ok ? "" : "  FAIL"));
        var steps = backend.DescribeDeclaredProgram(route.Key);
        Console.WriteLine("    " + (steps is not null ? string.Join(" | ", steps) : "declined: " + backend.DeclaredDeclineReason(route.Key)) +
            "   host bytes/call " + bytes[true].ToString("F0", CultureInfo.InvariantCulture) + " / " +
            bytes[false].ToString("F0", CultureInfo.InvariantCulture));
    }

    private static string E(double v) => v.ToString("E3", CultureInfo.InvariantCulture);

    // ---- cases --------------------------------------------------------------------------------------------------

    private static IEnumerable<RouteCase> Cases()
    {
        const float Eps = 1e-5f;
        var rows = new[] { (8192, 1024), (8192, 768), (8192, 64), (256, 64), (100, 100), (128, 1000), (512, 7),
                           (16, 50001), (2048, 4000) };
        foreach (var (r, c) in rows) yield return LayerNorm(r, c, Eps);
        foreach (var (r, c) in rows) yield return RmsNorm(r, c, Eps);
        foreach (var (n, ch, g, s) in new[] { (32, 64, 8, 1024), (8, 64, 32, 3136), (16, 32, 32, 49), (4, 6, 3, 10), (2, 64, 8, 100) })
            yield return GroupNorm(n, ch, g, s, Eps);
        foreach (var (n, ch, s) in new[] { (32, 64, 3136), (8, 64, 64), (64, 128, 196), (16, 32, 49), (2, 3, 5) })
            yield return BatchNorm(n, ch, s, Eps, 0.1f);
        foreach (var (r, c) in new[] { (8192, 1024), (8192, 64), (100, 100), (16, 50001), (2048, 4000) })
            yield return LayerNormBackward(r, c, Eps);
        foreach (var (r, c) in new[] { (8192, 1024), (8192, 64), (100, 100), (16, 50001), (2048, 4000) })
            yield return RmsNormBackward(r, c, Eps);
        foreach (var (n, ch, s) in new[] { (32, 64, 3136), (8, 64, 64), (64, 128, 196), (16, 32, 49), (2, 3, 5) })
            yield return BatchNormBackward(n, ch, s, Eps);
        foreach (var (r, c) in new[] { (8192, 1024), (8192, 64), (100, 100), (64, 50257), (512, 7) })
            yield return Softmax(r, c);
        foreach (var (r, c) in new[] { (8192, 1024), (8192, 64), (100, 100), (64, 50257), (512, 7) })
            yield return SoftmaxBackward(r, c);
        foreach (var (b, i, o) in new[] { (64, 128, 96), (5, 33, 17), (256, 512, 512), (512, 768, 768) })
            foreach (var activation in new[] { CodegenActivationKind.ReLU, CodegenActivationKind.Gelu })
            {
                yield return LinearForward(b, i, o, activation);
                yield return LinearBackward(b, i, o, activation);
            }
    }

    // ---- linear layers (#890 phase 4c) ------------------------------------------------------------------------

    private const double GeluInner = 0.7978845608, GeluCubic = 0.044715;

    private static double Activate(CodegenActivationKind activation, double x) => activation == CodegenActivationKind.ReLU
        ? Math.Max(0.0, x)
        : 0.5 * x * (1.0 + Math.Tanh(GeluInner * (x + GeluCubic * x * x * x)));

    private static double ActivationGrad(CodegenActivationKind activation, double dy, double pre)
    {
        if (activation == CodegenActivationKind.ReLU) return pre > 0 ? dy : 0.0;
        double t = Math.Tanh(GeluInner * (pre + GeluCubic * pre * pre * pre));
        double dt = GeluInner * (1.0 + 3.0 * GeluCubic * pre * pre) * (1.0 - t * t);
        return dy * (0.5 * (1.0 + t) + 0.5 * pre * dt);
    }

    private static double[] LinearPre(float[] x, float[] w, float[] bias, int batch, int inFeatures, int outFeatures)
    {
        var pre = new double[batch * outFeatures];
        System.Threading.Tasks.Parallel.For(0, batch, r =>
        {
            for (int j = 0; j < outFeatures; j++)
            {
                double sum = bias[j];
                for (int k = 0; k < inFeatures; k++) sum += (double)x[r * inFeatures + k] * w[k * outFeatures + j];
                pre[r * outFeatures + j] = sum;
            }
        });
        return pre;
    }

    private static RouteCase LinearForward(int batch, int inFeatures, int outFeatures, CodegenActivationKind activation)
    {
        float[] x = Fill(batch * inFeatures, 11, 0), w = Fill(inFeatures * outFeatures, 12, 0), bias = Fill(outFeatures, 13, 0);
        double[] pre = LinearPre(x, w, bias, batch, inFeatures, outFeatures);
        var y = new double[pre.Length];
        for (int i = 0; i < y.Length; i++) y[i] = Activate(activation, pre[i]);
        bool relu = activation == CodegenActivationKind.ReLU;
        return new RouteCase("FusedLinear" + (relu ? "ReLU " : "GELU ") + batch + "x" + inFeatures + "x" + outFeatures,
            CudaBackend.FusedLinearKey(batch, inFeatures, outFeatures, activation),
            new[] { x, w, bias }, new[] { batch * outFeatures },
            (k, a) =>
            {
                if (relu) k.FusedLinearReLU(a[0], a[1], a[2], a[3], batch, inFeatures, outFeatures);
                else k.FusedLinearGELU(a[0], a[1], a[2], a[3], batch, inFeatures, outFeatures);
            },
            new[] { ("y", 3, y) }, Math.Max(1, Max(Abs(pre))), inFeatures);
    }

    private static RouteCase LinearBackward(int batch, int inFeatures, int outFeatures, CodegenActivationKind activation)
    {
        float[] x = Fill(batch * inFeatures, 14, 0), w = Fill(inFeatures * outFeatures, 15, 0), bias = Fill(outFeatures, 16, 0);
        float[] dy = Fill(batch * outFeatures, 17, 0);
        float[] pre = ToFloat(LinearPre(x, w, bias, batch, inFeatures, outFeatures));
        var dpre = new double[pre.Length];
        for (int i = 0; i < dpre.Length; i++) dpre[i] = ActivationGrad(activation, dy[i], pre[i]);
        var dx = new double[x.Length];
        var dw = new double[w.Length];
        var db = new double[outFeatures];
        System.Threading.Tasks.Parallel.For(0, batch, r =>
        {
            for (int k = 0; k < inFeatures; k++)
            {
                double sum = 0;
                for (int j = 0; j < outFeatures; j++) sum += dpre[r * outFeatures + j] * w[k * outFeatures + j];
                dx[r * inFeatures + k] = sum;
            }
        });
        System.Threading.Tasks.Parallel.For(0, inFeatures, k =>
        {
            for (int j = 0; j < outFeatures; j++)
            {
                double sum = 0;
                for (int r = 0; r < batch; r++) sum += x[r * inFeatures + k] * dpre[r * outFeatures + j];
                dw[k * outFeatures + j] = sum;
            }
        });
        for (int r = 0; r < batch; r++)
            for (int j = 0; j < outFeatures; j++) db[j] += dpre[r * outFeatures + j];
        bool relu = activation == CodegenActivationKind.ReLU;
        return new RouteCase("FusedLinear" + (relu ? "ReLU" : "GELU") + "Backward " + batch + "x" + inFeatures + "x" + outFeatures,
            CudaBackend.FusedLinearBackwardKey(batch, inFeatures, outFeatures, activation),
            new[] { dy, x, w, pre }, new[] { x.Length, w.Length, outFeatures },
            (k, a) =>
            {
                if (relu) k.FusedLinearReLUBackward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], batch, inFeatures, outFeatures);
                else k.FusedLinearGELUBackward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], batch, inFeatures, outFeatures);
            },
            new[] { ("dx", 4, dx), ("dw", 5, dw), ("dbias", 6, db) },
            Math.Max(1, Math.Max(Max(Abs(dx)), Math.Max(Max(Abs(dw)), Max(Abs(db))))), Math.Max(batch, outFeatures));
    }

    private static RouteCase LayerNorm(int rows, int cols, float eps)
    {
        var x = Fill(rows * cols, 1, 1000.0);
        var g = Fill(cols, 2, 1.0);
        var b = Fill(cols, 3, 0.0);
        var (y, mean, rstd) = NormRef(x, g, b, rows, cols, eps, rms: false);
        return new RouteCase("LayerNorm " + rows + "x" + cols, CudaBackend.LayerNormKey(rows, cols, eps),
            new[] { x, g, b }, new[] { rows * cols, rows, rows },
            (k, a) => k.LayerNorm(a[0], a[3], a[1], a[2], a[4], a[5], rows, cols, eps),
            new[] { ("y", 3, y), ("mean", 4, mean), ("rstd", 5, rstd) },
            MaxAbs(x) * Max(rstd) * Math.Max(1, MaxAbs(g)), cols);
    }

    private static RouteCase RmsNorm(int rows, int cols, float eps)
    {
        var x = Fill(rows * cols, 4, 0.0);
        var g = Fill(cols, 5, 1.0);
        var (y, _, rstd) = NormRef(x, g, null, rows, cols, eps, rms: true);
        var rms = new double[rows];
        for (int r = 0; r < rows; r++) rms[r] = 1.0 / rstd[r];
        return new RouteCase("RmsNorm " + rows + "x" + cols, CudaBackend.RmsNormKey(rows, cols, eps),
            new[] { x, g }, new[] { rows * cols, rows },
            (k, a) => k.RmsNorm(a[0], a[2], a[1], a[3], rows, cols, eps),
            new[] { ("y", 2, y), ("rms", 3, rms) },
            MaxAbs(x) * Max(rstd) * Math.Max(1, MaxAbs(g)), cols);
    }

    private static RouteCase GroupNorm(int n, int channels, int groups, int spatial, float eps)
    {
        int cg = channels / groups;
        var x = Fill(n * channels * spatial, 6, 1000.0);
        var g = Fill(channels, 7, 1.0);
        var b = Fill(channels, 8, 0.0);
        int rows = n * groups, width = cg * spatial;
        // A group's elements are contiguous in [n, c, s]: rows of cg*spatial, gamma per channel.
        var y = new double[x.Length];
        var (_, mean, rstd) = NormRef(x, new float[width], null, rows, width, eps, rms: false);
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < width; j++)
            {
                int c = r % groups * cg + j / spatial;
                y[r * width + j] = (x[r * width + j] - mean[r]) * rstd[r] * g[c] + b[c];
            }
        return new RouteCase("GroupNorm " + n + "x" + channels + "/" + groups + "x" + spatial,
            CudaBackend.GroupNormKey(n, channels, groups, spatial, eps),
            new[] { x, g, b }, new[] { x.Length, rows, rows },
            (k, a) => k.GroupNorm(a[0], a[3], a[1], a[2], a[4], a[5], n, groups, channels, spatial, eps),
            new[] { ("y", 3, y), ("mean", 4, mean), ("rstd", 5, rstd) },
            MaxAbs(x) * Max(rstd) * Math.Max(1, MaxAbs(g)), width);
    }

    private static RouteCase BatchNorm(int n, int channels, int spatial, float eps, float momentum)
    {
        var x = Fill(n * channels * spatial, 9, 1000.0);
        var g = Fill(channels, 10, 1.0);
        var b = Fill(channels, 11, 0.0);
        var rm = Fill(channels, 12, 999.0);
        var rv = Fill(channels, 13, 2.0);
        var (mean, rstd, variance) = ChannelStats(x, n, channels, spatial, eps);
        var y = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            int c = i / spatial % channels;
            y[i] = (x[i] - mean[c]) * rstd[c] * g[c] + b[c];
        }
        long count = (long)n * spatial;
        var rmOut = new double[channels];
        var rvOut = new double[channels];
        for (int c = 0; c < channels; c++)
        {
            rmOut[c] = (1 - (double)momentum) * rm[c] + momentum * mean[c];
            double unbiased = count > 1 ? variance[c] * count / (count - 1) : variance[c];
            rvOut[c] = (1 - (double)momentum) * rv[c] + momentum * unbiased;
        }
        return new RouteCase("BatchNorm " + n + "x" + channels + "x" + spatial,
            CudaBackend.BatchNormKey(n, channels, spatial, eps, momentum),
            new[] { x, g, b, rm, rv }, new[] { x.Length, channels, channels },
            (k, a) => k.BatchNorm(a[0], a[5], a[1], a[2], a[3], a[4], a[6], a[7], n, channels, spatial, eps, momentum, true),
            new[] { ("y", 5, y), ("mean", 6, mean), ("rstd", 7, rstd), ("running_mean", 3, rmOut), ("running_var", 4, rvOut) },
            MaxAbs(x) * Max(rstd) * Math.Max(1, MaxAbs(g)), n * spatial);
    }

    // Backward cases: saved statistics from the fp64 forward, rounded to fp32 and used by every party.
    private static RouteCase LayerNormBackward(int rows, int cols, float eps)
    {
        var x = Fill(rows * cols, 14, 1000.0);
        var g = Fill(cols, 15, 1.0);
        var dy = Fill(rows * cols, 16, 0.0);
        var (_, meanD, rstdD) = NormRef(x, g, null, rows, cols, eps, rms: false);
        var mean = ToFloat(meanD);
        var rstd = ToFloat(rstdD);
        var (dx, dgamma, dbeta) = RowNormBackwardRef(dy, x, g, mean, rstd, rows, cols, rms: false);
        return new RouteCase("LayerNormBackward " + rows + "x" + cols, CudaBackend.LayerNormBackwardKey(rows, cols),
            new[] { dy, x, g, mean, rstd }, new[] { rows * cols, cols, cols },
            (k, a) => k.LayerNormBackward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], rows, cols, eps),
            new[] { ("dx", 5, dx), ("dgamma", 6, dgamma), ("dbeta", 7, dbeta) },
            MaxAbs(x) * Max(rstdD) * Math.Max(1, MaxAbs(g)) * Math.Max(1, MaxAbs(dy)) * Math.Sqrt(rows), Math.Max(rows, cols));
    }

    private static RouteCase RmsNormBackward(int rows, int cols, float eps)
    {
        var x = Fill(rows * cols, 17, 0.0);
        var g = Fill(cols, 18, 1.0);
        var dy = Fill(rows * cols, 19, 0.0);
        var (_, _, rstdD) = NormRef(x, g, null, rows, cols, eps, rms: true);
        var rms = new float[rows];
        for (int r = 0; r < rows; r++) rms[r] = (float)(1.0 / rstdD[r]);
        var rstd = new float[rows];
        for (int r = 0; r < rows; r++) rstd[r] = (float)(1.0 / rms[r]);
        var (dx, dgamma, _) = RowNormBackwardRef(dy, x, g, new float[rows], rstd, rows, cols, rms: true);
        return new RouteCase("RmsNormBackward " + rows + "x" + cols, CudaBackend.RmsNormBackwardKey(rows, cols),
            new[] { dy, x, g, rms }, new[] { rows * cols, cols },
            (k, a) => k.RmsNormBackward(a[0], a[1], a[2], a[3], a[4], a[5], rows, cols, eps),
            new[] { ("dx", 4, dx), ("dgamma", 5, dgamma) },
            MaxAbs(x) * Max(rstdD) * Math.Max(1, MaxAbs(g)) * Math.Max(1, MaxAbs(dy)) * Math.Sqrt(rows), Math.Max(rows, cols));
    }

    private static RouteCase BatchNormBackward(int n, int channels, int spatial, float eps)
    {
        var x = Fill(n * channels * spatial, 20, 1000.0);
        var g = Fill(channels, 21, 1.0);
        var dy = Fill(x.Length, 22, 0.0);
        var (meanD, rstdD, _) = ChannelStats(x, n, channels, spatial, eps);
        var mean = ToFloat(meanD);
        var rstd = ToFloat(rstdD);
        long count = (long)n * spatial;
        var dbeta = new double[channels];
        var dgamma = new double[channels];
        for (int i = 0; i < x.Length; i++)
        {
            int c = i / spatial % channels;
            dbeta[c] += dy[i];
            dgamma[c] += dy[i] * ((x[i] - (double)mean[c]) * rstd[c]);
        }
        var dx = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            int c = i / spatial % channels;
            double xhat = (x[i] - (double)mean[c]) * rstd[c];
            dx[i] = g[c] * (double)rstd[c] * (dy[i] - dbeta[c] / count - xhat * dgamma[c] / count);
        }
        return new RouteCase("BatchNormBackward " + n + "x" + channels + "x" + spatial,
            CudaBackend.BatchNormBackwardKey(n, channels, spatial),
            new[] { dy, x, g, mean, rstd }, new[] { x.Length, channels, channels },
            (k, a) => k.BatchNormBackward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], n, channels, spatial, eps),
            new[] { ("dx", 5, dx), ("dgamma", 6, dgamma), ("dbeta", 7, dbeta) },
            MaxAbs(x) * Max(rstdD) * Math.Max(1, MaxAbs(g)) * Math.Sqrt(count), (int)Math.Min(int.MaxValue, count));
    }

    private static RouteCase Softmax(int rows, int cols)
    {
        var x = Fill(rows * cols, 23, 0.0, 8.0);
        var y = SoftmaxRef(x, rows, cols);
        return new RouteCase("Softmax " + rows + "x" + cols, CudaBackend.SoftmaxKey(rows, cols),
            new[] { x }, new[] { rows * cols },
            (k, a) => k.Softmax(a[0], a[1], rows, cols),
            new[] { ("y", 1, y) }, MaxAbs(x), cols);
    }

    private static RouteCase SoftmaxBackward(int rows, int cols)
    {
        var y = ToFloat(SoftmaxRef(Fill(rows * cols, 24, 0.0, 8.0), rows, cols));
        var dy = Fill(rows * cols, 25, 0.0);
        var dx = new double[y.Length];
        for (int r = 0; r < rows; r++)
        {
            double dot = 0;
            for (int c = 0; c < cols; c++) dot += (double)dy[r * cols + c] * y[r * cols + c];
            for (int c = 0; c < cols; c++) dx[r * cols + c] = y[r * cols + c] * (dy[r * cols + c] - dot);
        }
        return new RouteCase("SoftmaxBackward " + rows + "x" + cols, CudaBackend.SoftmaxBackwardKey(rows, cols),
            new[] { dy, y }, new[] { rows * cols },
            (k, a) => k.SoftmaxBackward(a[0], a[1], a[2], rows, cols),
            new[] { ("dx", 2, dx) }, MaxAbs(dy), cols);
    }

    // ---- references -----------------------------------------------------------------------------------------------

    // Row normalisation: LayerNorm (x - mean) * rstd, or RMSNorm x * rstd, times gamma plus beta.
    private static (double[] Y, double[] Mean, double[] Rstd) NormRef(float[] x, float[] gamma, float[]? beta,
        int rows, int cols, double eps, bool rms)
    {
        var y = new double[x.Length];
        var mean = new double[rows];
        var rstd = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double m = 0;
            if (!rms)
            {
                for (int c = 0; c < cols; c++) m += x[r * cols + c];
                m /= cols;
            }
            double sq = 0;
            for (int c = 0; c < cols; c++) { double d = x[r * cols + c] - m; sq += d * d; }
            double s = 1.0 / Math.Sqrt(sq / cols + eps);
            mean[r] = m;
            rstd[r] = s;
            for (int c = 0; c < cols; c++)
                y[r * cols + c] = (x[r * cols + c] - m) * s * gamma[c] + (beta is null ? 0 : beta[c]);
        }
        return (y, mean, rstd);
    }

    private static (double[] Dx, double[] DGamma, double[] DBeta) RowNormBackwardRef(float[] dy, float[] x, float[] gamma,
        float[] mean, float[] rstd, int rows, int cols, bool rms)
    {
        var dx = new double[x.Length];
        var dgamma = new double[cols];
        var dbeta = new double[cols];
        for (int r = 0; r < rows; r++)
        {
            double m1 = 0, m2 = 0;
            for (int c = 0; c < cols; c++)
            {
                double xhat = (x[r * cols + c] - (rms ? 0.0 : mean[r])) * rstd[r];
                double gdy = (double)dy[r * cols + c] * gamma[c];
                m1 += gdy;
                m2 += gdy * xhat;
                dgamma[c] += dy[r * cols + c] * xhat;
                dbeta[c] += dy[r * cols + c];
            }
            m1 /= cols;
            m2 /= cols;
            for (int c = 0; c < cols; c++)
            {
                double xhat = (x[r * cols + c] - (rms ? 0.0 : mean[r])) * rstd[r];
                double gdy = (double)dy[r * cols + c] * gamma[c];
                dx[r * cols + c] = rstd[r] * (gdy - (rms ? 0.0 : m1) - xhat * m2);
            }
        }
        return (dx, dgamma, dbeta);
    }

    private static (double[] Mean, double[] Rstd, double[] Variance) ChannelStats(float[] x, int n, int channels, int spatial, double eps)
    {
        var mean = new double[channels];
        var variance = new double[channels];
        var rstd = new double[channels];
        long count = (long)n * spatial;
        for (int i = 0; i < x.Length; i++) mean[i / spatial % channels] += x[i];
        for (int c = 0; c < channels; c++) mean[c] /= count;
        for (int i = 0; i < x.Length; i++) { double d = x[i] - mean[i / spatial % channels]; variance[i / spatial % channels] += d * d; }
        for (int c = 0; c < channels; c++)
        {
            variance[c] /= count;
            rstd[c] = 1.0 / Math.Sqrt(variance[c] + eps);
        }
        return (mean, rstd, variance);
    }

    private static double[] SoftmaxRef(float[] x, int rows, int cols)
    {
        var y = new double[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < cols; c++) max = Math.Max(max, x[r * cols + c]);
            for (int c = 0; c < cols; c++) sum += Math.Exp(x[r * cols + c] - max);
            for (int c = 0; c < cols; c++) y[r * cols + c] = Math.Exp(x[r * cols + c] - max) / sum;
        }
        return y;
    }

    // ---- resident chains ------------------------------------------------------------------------------------------

    // Declared stages as ONE cooperative launch (#890 phase 4 on #1430): bit-identical to the same stages launched
    // separately, within the fp32 bound of an fp64 reference, and timed both ways.
    private static void ResidentChains(CudaBackend backend, List<string> failures)
    {
        if (!backend.SupportsResidentProgram)
        {
            Console.WriteLine("resident: this device has no cooperative launch");
            return;
        }
        Console.WriteLine();
        Console.WriteLine("RESIDENT - declared stages as one cooperative launch vs the same stages launched separately");
        Console.WriteLine("chain                                   grid  bit-identical   error vs fp64     separate us  resident us   ratio");
        using var barrier = backend.AllocateResidentBarrier();
        const float Eps = 1e-5f;

        foreach (var (r, c) in new[] { (64, 1024), (16, 256), (256, 64) })
        {
            var x = Fill(r * c, 31, 0.0);
            var g = Fill(c, 32, 1.0);
            var b = Fill(c, 33, 0.0);
            var (yRef, _, _) = NormRef(x, g, b, r, c, Eps, rms: false);
            var softRef = SoftmaxRef(ToFloat(yRef), r, c);
            RunChain(backend, barrier, failures, "LayerNorm -> Softmax " + r + "x" + c,
                new[] { x, g, b }, new[] { r * c, r, r, r * c },
                a => new[]
                {
                    CudaBackend.LayerNormStage(a[0], a[3], a[1], a[2], a[4], a[5], r, c, Eps),
                    CudaBackend.SoftmaxStage(a[3], a[6], r, c),
                },
                (6, softRef), MaxAbs(x) * 4, c);
        }

        foreach (var (r, c) in new[] { (8192, 64), (512, 256) })
        {
            var x = Fill(r * c, 34, 0.0);
            var g = Fill(c, 35, 1.0);
            var b = Fill(c, 36, 0.0);
            var dy = Fill(r * c, 37, 0.0);
            var (_, meanD, rstdD) = NormRef(x, g, b, r, c, Eps, rms: false);
            var (dxRef, _, _) = RowNormBackwardRef(dy, x, g, ToFloat(meanD), ToFloat(rstdD), r, c, rms: false);
            RunChain(backend, barrier, failures, "LayerNorm -> LayerNormBackward " + r + "x" + c,
                new[] { x, g, b, dy }, new[] { r * c, r, r, r * c, c, c },
                a => new[]
                {
                    CudaBackend.LayerNormStage(a[0], a[4], a[1], a[2], a[5], a[6], r, c, Eps),
                    CudaBackend.LayerNormBackwardStage(a[3], a[0], a[1], a[5], a[6], a[7], a[8], a[9], r, c),
                },
                (7, dxRef), MaxAbs(x) * Max(rstdD) * Math.Max(1, MaxAbs(g)) * Math.Sqrt(r), Math.Max(r, c));
        }

        // Elementwise stages (#1437's pointwise family): LayerNorm, then gelu on its output, then leaky_relu(0.1),
        // then softmax -- the pointwise stages calling the same derived library functions the routed calls do.
        foreach (var (r, c) in new[] { (64, 1024), (256, 64) })
        {
            var x = Fill(r * c, 41, 0.0);
            var g = Fill(c, 42, 1.0);
            var b = Fill(c, 43, 0.0);
            var (yRef, _, _) = NormRef(x, g, b, r, c, Eps, rms: false);
            var act = new double[r * c];
            for (int i = 0; i < act.Length; i++)
            {
                double v = yRef[i];
                double gelu = 0.5 * v * (1 + Math.Tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
                act[i] = gelu > 0 ? gelu : 0.1 * gelu;
            }
            var softRef = SoftmaxRef(ToFloat(act), r, c);
            RunChain(backend, barrier, failures, "LayerNorm -> gelu -> leaky_relu -> Softmax " + r + "x" + c,
                new[] { x, g, b }, new[] { r * c, r, r, r * c, r * c, r * c },
                a => new[]
                {
                    CudaBackend.LayerNormStage(a[0], a[3], a[1], a[2], a[4], a[5], r, c, Eps),
                    backend.ElementwiseStage("gelu", a[6], r * c, a[3]),
                    backend.ElementwiseStage("leaky_relu", a[7], r * c, a[6], scalars: 0.1f),
                    CudaBackend.SoftmaxStage(a[7], a[8], r, c),
                },
                (8, softRef), MaxAbs(x) * 4, c);
        }

        {
            const int r = 128, c = 512;
            var x = Fill(r * c, 38, 0.0);
            var g = Fill(c, 39, 1.0);
            var dy = Fill(r * c, 40, 0.0);
            var (yRef, _, _) = NormRef(x, g, null, r, c, Eps, rms: true);
            var sRef = SoftmaxRef(ToFloat(yRef), r, c);
            var dxRef = new double[r * c];
            for (int row = 0; row < r; row++)
            {
                double dot = 0;
                for (int j = 0; j < c; j++) dot += dy[row * c + j] * sRef[row * c + j];
                for (int j = 0; j < c; j++) dxRef[row * c + j] = sRef[row * c + j] * (dy[row * c + j] - dot);
            }
            RunChain(backend, barrier, failures, "RmsNorm -> Softmax -> SoftmaxBackward " + r + "x" + c,
                new[] { x, g, dy }, new[] { r * c, r, r * c, r * c },
                a => new[]
                {
                    CudaBackend.RmsNormStage(a[0], a[3], a[1], a[4], r, c, Eps),
                    CudaBackend.SoftmaxStage(a[3], a[5], r, c),
                    CudaBackend.SoftmaxBackwardStage(a[2], a[5], a[6], r, c),
                },
                (6, dxRef), MaxAbs(x) * 4, c);
        }
    }

    private static void RunChain(CudaBackend backend, IGpuBuffer barrier, List<string> failures, string name,
        float[][] inputs, int[] outputSizes, Func<IGpuBuffer[], DeclaredResidentStage[]> stagesOf,
        (int Buffer, double[] Reference) check, double scale, int reduction)
    {
        var results = new float[2][][];
        var times = new double[2];
        int grid = 0;
        for (int mode = 0; mode < 2; mode++)   // 0: separate, 1: resident
        {
            var buffers = new IGpuBuffer[inputs.Length + outputSizes.Length];
            try
            {
                for (int i = 0; i < inputs.Length; i++) buffers[i] = backend.AllocateBuffer(inputs[i]);
                for (int o = 0; o < outputSizes.Length; o++) buffers[inputs.Length + o] = backend.AllocateBuffer(new float[outputSizes[o]]);
                var stages = stagesOf(buffers);
                void Separate() => backend.RunDeclaredStagesSeparately(stages);
                void Resident() => grid = backend.RunDeclaredResidentProgram(stages, barrier);
                Action call = mode == 0 ? Separate : Resident;
                call();
                backend.Synchronize();
                results[mode] = new float[buffers.Length][];
                for (int i = 0; i < buffers.Length; i++) results[mode][i] = backend.DownloadBuffer(buffers[i]);
                times[mode] = Time(call, backend.Synchronize);
            }
            catch (NotSupportedException ex)
            {
                failures.Add(name + ": " + ex.Message);
                Console.WriteLine(name.PadRight(40) + "  not composed: " + ex.Message);
                return;
            }
            finally
            {
                foreach (var buffer in buffers) buffer?.Dispose();
            }
        }
        bool identical = true;
        for (int i = 0; i < results[0].Length && identical; i++)
            for (int j = 0; j < results[0][i].Length; j++)
                if (BitConverter.SingleToInt32Bits(results[0][i][j]) != BitConverter.SingleToInt32Bits(results[1][i][j])) { identical = false; break; }
        double error = MaxDev(results[1][check.Buffer], check.Reference);
        double bound = Unit * (16 + Math.Log(Math.Max(2, reduction), 2)) * Math.Max(1.0, scale);
        bool ok = identical && error <= bound;
        if (!ok) failures.Add(name + (identical ? ": error " + E(error) + " over " + E(bound) : ": not bit-identical"));
        Console.WriteLine(name.PadRight(40) + grid.ToString(CultureInfo.InvariantCulture).PadLeft(5) +
            (identical ? "yes" : "NO").PadLeft(15) + E(error).PadLeft(16) +
            times[0].ToString("F1", CultureInfo.InvariantCulture).PadLeft(16) +
            times[1].ToString("F1", CultureInfo.InvariantCulture).PadLeft(13) +
            (times[0] / times[1]).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" + (ok ? "" : "  FAIL"));
    }

    // ---- capture ------------------------------------------------------------------------------------------------

    // A shape first met under capture falls through; built outside, it captures and replays.
    private static void CaptureCheck(CudaBackend backend, List<string> failures)
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
        var (refY, refMean, refRstd) = NormRef(x, g, b, rows, cols, 1e-5, rms: false);
        double bound = Unit * (16 + Math.Log(cols, 2)) * Math.Max(1.0, MaxAbs(x) * Max(refRstd) * Math.Max(1, MaxAbs(g)));
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
                ", replay error " + E(error) + (ok ? "" : "  FAIL"));
            if (!ok) failures.Add("capture (" + (built ? "built" : "unbuilt") + ")");
        }
    }

    // ---- helpers ------------------------------------------------------------------------------------------------

    // Deterministic values in offset + [-spread/2, spread/2).
    private static float[] Fill(int count, int salt, double offset, double spread = 1.5)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(((i * 37L + salt * 101) % 97 - 48) / 96.0 * spread + offset);
        return v;
    }

    private static float[] ToFloat(double[] v)
    {
        var f = new float[v.Length];
        for (int i = 0; i < v.Length; i++) f[i] = (float)v[i];
        return f;
    }

    private static double MaxAbs(float[] v) { double m = 0; foreach (float f in v) m = Math.Max(m, Math.Abs(f)); return m; }

    private static double Max(double[] v) { double m = 0; foreach (double d in v) m = Math.Max(m, d); return m; }

    // Worst error relative to max(1, |reference|); NaN anywhere is NaN.
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
    // ---- convolutions ---------------------------------------------------------------------------------------------

    /// <summary>
    /// Each declared convolution schedule (the slots' Generated candidates) against the backend's own convolution with
    /// the declared route off -- whatever its tuned slot serves -- both against an fp64 direct convolution, timed in the
    /// same process.
    /// </summary>
    private static void ConvRoutes(CudaBackend backend, List<string> failures)
    {
        Console.WriteLine();
        Console.WriteLine("CONV - declared schedules vs the backend's own route (declared off), each against fp64");
        Console.WriteLine("case                                       schedule          err        incumbent err    us      incumbent us  ratio");
        var shapes = new[]
        {
            ("s2 8x64->128 56", 8, 64, 128, 56, 56, 3, 2, 1, 1),
            ("3x3 8x32->64 28", 8, 32, 64, 28, 28, 3, 1, 1, 1),
            ("1x1 16x64->64 28", 16, 64, 64, 28, 28, 1, 1, 0, 1),
            ("d2 8x32->32 28", 8, 32, 32, 28, 28, 3, 1, 2, 2),
        };
        foreach (var (label, n, c, k, h, w, kernel, stride, pad, dilation) in shapes)
        {
            int oh = (h + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
            int ow = (w + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
            float[] x = Fill(n * c * h * w, 1, 0), wt = Fill(k * c * kernel * kernel, 2, 0), dy = Fill(n * k * oh * ow, 3, 0);
            var (y, dx, dw) = ConvRef(x, wt, dy, n, c, k, h, w, kernel, stride, pad, dilation, oh, ow);
            foreach (DeclaredConvPass pass in new[] { DeclaredConvPass.Forward, DeclaredConvPass.BackwardData, DeclaredConvPass.BackwardWeights })
            {
                double[] reference = pass == DeclaredConvPass.Forward ? y : pass == DeclaredConvPass.BackwardData ? dx : dw;
                using var first = backend.AllocateBuffer(pass == DeclaredConvPass.BackwardData ? dy : x);
                using var second = backend.AllocateBuffer(pass == DeclaredConvPass.Forward || pass == DeclaredConvPass.BackwardData ? wt : dy);
                using var output = backend.AllocateBuffer(reference.Length);
                void Incumbent()
                {
                    if (pass == DeclaredConvPass.Forward)
                        backend.Conv2D(first, second, output, n, c, h, w, k, oh, ow, kernel, kernel, stride, stride, pad, pad, dilation, dilation);
                    else if (pass == DeclaredConvPass.BackwardData)
                        backend.Conv2DBackwardInput(first, second, output, n, c, h, w, k, oh, ow, kernel, kernel, stride, stride, pad, pad, dilation, dilation);
                    else
                        backend.Conv2DBackwardKernel(first, second, output, n, c, h, w, k, oh, ow, kernel, kernel, stride, stride, pad, pad, dilation, dilation);
                }
                double incumbentUs, incumbentErr;
                backend.DeclaredKernelsSwitch = false;
                try
                {
                    Incumbent();
                    backend.Synchronize();
                    incumbentErr = MaxDev(backend.DownloadBuffer(output), reference);
                    incumbentUs = Time(Incumbent, backend.Synchronize);
                }
                finally { backend.DeclaredKernelsSwitch = null; }

                double scale = Math.Max(1.0, Max(Abs(reference)));
                foreach (var schedule in CudaBackend.DeclaredConvSchedules(pass))
                {
                    var forwardArgs = new CudaConv2DArgs(first, second, output, n, c, h, w, k, oh, ow, kernel, kernel,
                        stride, stride, pad, pad, dilation, dilation);
                    var backwardArgs = new TunedConv2DBackwardArgs(first, second, output, pass == DeclaredConvPass.BackwardWeights,
                        n, c, h, w, k, oh, ow, kernel, kernel, stride, stride, pad, pad, dilation, dilation);
                    bool Declared() => pass == DeclaredConvPass.Forward
                        ? backend.TryDeclaredConv2DForward(forwardArgs, schedule)
                        : backend.TryDeclaredConv2DBackward(backwardArgs, schedule);
                    string name = (label + " " + pass).PadRight(43) + schedule.Suffix.PadRight(16);
                    if (!Declared())
                    {
                        Console.WriteLine(name + "declined: " + backend.DeclaredConvDeclineReason(new DeclaredConvKey(pass,
                            n, c, k, h, w, kernel, stride, pad, dilation, false, schedule)));
                        continue;
                    }
                    backend.Synchronize();
                    double err = MaxDev(backend.DownloadBuffer(output), reference);
                    double us = Time(() => Declared(), backend.Synchronize);
                    bool ok = err <= Math.Max(2 * incumbentErr, 1e-5 * scale);
                    if (!ok) failures.Add(label + " " + pass + " " + schedule.Suffix + ": error " + E(err));
                    Console.WriteLine(name + E(err).PadLeft(10) + E(incumbentErr).PadLeft(14) +
                        us.ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                        incumbentUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(14) +
                        (incumbentUs / us).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" + (ok ? "" : "  FAIL"));
                }
            }
        }
    }

    private static double[] Abs(double[] v)
    {
        var r = new double[v.Length];
        for (int i = 0; i < v.Length; i++) r[i] = Math.Abs(v[i]);
        return r;
    }

    // y = conv(x, w), dx = conv^T(dy, w), dw = correlation of dy with x, in fp64, by direct loops.
    private static (double[] Y, double[] Dx, double[] Dw) ConvRef(float[] x, float[] w, float[] dy, int n, int c, int k,
        int h, int wd, int kernel, int stride, int pad, int dilation, int oh, int ow)
    {
        var y = new double[n * k * oh * ow];
        var dx = new double[x.Length];
        var dw = new double[w.Length];
        System.Threading.Tasks.Parallel.For(0, n * k, nk =>
        {
            int b = nk / k, o = nk % k;
            for (int p = 0; p < oh; p++)
            for (int q = 0; q < ow; q++)
            {
                double sum = 0;
                for (int ci = 0; ci < c; ci++)
                for (int r = 0; r < kernel; r++)
                {
                    int ih = p * stride - pad + r * dilation;
                    if (ih < 0 || ih >= h) continue;
                    for (int s = 0; s < kernel; s++)
                    {
                        int iw = q * stride - pad + s * dilation;
                        if (iw < 0 || iw >= wd) continue;
                        sum += (double)x[((b * c + ci) * h + ih) * wd + iw] * w[((o * c + ci) * kernel + r) * kernel + s];
                    }
                }
                y[((b * k + o) * oh + p) * ow + q] = sum;
            }
        });
        // dx and dw are scatters; partitioning by the written index keeps each thread's writes its own.
        System.Threading.Tasks.Parallel.For(0, n * c, nc =>
        {
            int b = nc / c, ci = nc % c;
            for (int o = 0; o < k; o++)
            for (int p = 0; p < oh; p++)
            for (int q = 0; q < ow; q++)
            {
                double g = dy[((b * k + o) * oh + p) * ow + q];
                for (int r = 0; r < kernel; r++)
                {
                    int ih = p * stride - pad + r * dilation;
                    if (ih < 0 || ih >= h) continue;
                    for (int s = 0; s < kernel; s++)
                    {
                        int iw = q * stride - pad + s * dilation;
                        if (iw < 0 || iw >= wd) continue;
                        dx[((b * c + ci) * h + ih) * wd + iw] += g * w[((o * c + ci) * kernel + r) * kernel + s];
                    }
                }
            }
        });
        System.Threading.Tasks.Parallel.For(0, k * c, kc =>
        {
            int o = kc / c, ci = kc % c;
            for (int b = 0; b < n; b++)
            for (int p = 0; p < oh; p++)
            for (int q = 0; q < ow; q++)
            {
                double g = dy[((b * k + o) * oh + p) * ow + q];
                for (int r = 0; r < kernel; r++)
                {
                    int ih = p * stride - pad + r * dilation;
                    if (ih < 0 || ih >= h) continue;
                    for (int s = 0; s < kernel; s++)
                    {
                        int iw = q * stride - pad + s * dilation;
                        if (iw < 0 || iw >= wd) continue;
                        dw[((o * c + ci) * kernel + r) * kernel + s] += g * x[((b * c + ci) * h + ih) * wd + iw];
                    }
                }
            }
        });
        return (y, dx, dw);
    }

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
