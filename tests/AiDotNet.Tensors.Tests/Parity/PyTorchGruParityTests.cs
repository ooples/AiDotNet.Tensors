#if NET5_0_OR_GREATER
// Net-core only: it drives an external Python process (ProcessStartInfo.ArgumentList), as HeadToHeadNetworkHarness does.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// GruSequenceForward against PyTorch's nn.GRU itself, with the same weights (#1568 acceptance: forward and gradient
/// parity, gradient cosine at least 0.99999). The reference runs tools/parity/torch_gru_reference.py; without Python
/// and torch the case is skipped with the reason, never passed. On a CUDA machine the GPU engine's case is the CUDA
/// parity check.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class PyTorchGruParityTests
{
    private const int B = 4, T = 7, In = 6, H = 10;

    private static double[] Values(int seed, int n, double scale)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = (rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private sealed class Case
    {
        public double[] X = Values(1, B * T * In, 1.0), H0 = Values(2, B * H, 0.5);
        public double[] WIh = Values(3, 3 * H * In, 0.4), WHh = Values(4, 3 * H * H, 0.4);
        public double[] BIh = Values(5, 3 * H, 0.3), BHh = Values(6, 3 * H, 0.3), Projection = Values(7, B * T * H, 1.0);
    }

    private static Dictionary<string, double[]>? Torch(Case c, string dtype, out string? skipReason)
    {
        skipReason = null;
        string? root = PyTorchParityInventory.FindRepositoryRoot();
        if (root is null) { skipReason = "tools/parity is missing from this checkout."; return null; }
        string script = Path.Combine(root, "tools", "parity", "torch_gru_reference.py");
        string dir = Path.Combine(Path.GetTempPath(), "aidotnet-gru-parity-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string casePath = Path.Combine(dir, "case.json"), resultPath = Path.Combine(dir, "result.json");
            File.WriteAllText(casePath, JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["batch"] = B, ["seq"] = T, ["input"] = In, ["hidden"] = H,
                ["x"] = c.X, ["h0"] = c.H0, ["w_ih"] = c.WIh, ["w_hh"] = c.WHh, ["b_ih"] = c.BIh, ["b_hh"] = c.BHh,
                ["projection"] = c.Projection,
            }));
            var psi = new ProcessStartInfo(HeadToHeadNetworkHarness.PythonExecutable)
            {
                RedirectStandardError = true, RedirectStandardOutput = true, UseShellExecute = false,
            };
            foreach (var a in new[] { script, casePath, resultPath, "--dtype", dtype }) psi.ArgumentList.Add(a);
            Process? p;
            try { p = Process.Start(psi); }
            catch (System.ComponentModel.Win32Exception) { skipReason = $"{psi.FileName} is not on PATH (set PARITY_PYTHON)."; return null; }
            if (p is null) { skipReason = "python did not start."; return null; }
            using (p)
            {
                string err = p.StandardError.ReadToEnd();
                p.WaitForExit(120_000);
                if (p.ExitCode != 0)
                {
                    skipReason = err.Contains("No module named 'torch'", StringComparison.Ordinal)
                        ? "torch is not installed for " + psi.FileName + "."
                        : null;
                    if (skipReason is null) throw new InvalidOperationException($"torch_gru_reference.py failed ({p.ExitCode}): {err}");
                    return null;
                }
            }
            using var doc = JsonDocument.Parse(File.ReadAllText(resultPath));
            var result = new Dictionary<string, double[]>();
            foreach (var key in new[] { "y", "x", "h0", "w_ih", "w_hh", "b_ih", "b_hh" })
                result[key] = doc.RootElement.GetProperty(key).EnumerateArray().Select(v => v.GetDouble()).ToArray();
            return result;
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }

    private static Dictionary<string, double[]> Ours<TT>(IEngine e, Case c, Func<double[], int[], Tensor<TT>> make)
    {
        var x = make(c.X, new[] { B, T, In }); var h0 = make(c.H0, new[] { B, H });
        var wih = make(c.WIh, new[] { 3 * H, In }); var whh = make(c.WHh, new[] { 3 * H, H });
        var bih = make(c.BIh, new[] { 3 * H }); var bhh = make(c.BHh, new[] { 3 * H });
        var projection = make(c.Projection, new[] { B, T, H });
        var sources = new[] { x, h0, wih, whh, bih, bhh };
        using var tape = new GradientTape<TT>();
        var y = e.GruSequenceForward(x, h0, wih, whh, bih, bhh, returnSequences: true);   // PyTorch's reset-after form
        var g = tape.ComputeGradients(e.ReduceMean(e.TensorMultiply(y, projection), new[] { 0, 1, 2 }, keepDims: false), sources);
        double[] D(Tensor<TT> t) => Array.ConvertAll(t.ToArray(), v => Convert.ToDouble(v));
        return new Dictionary<string, double[]>
        {
            ["y"] = D(y), ["x"] = D(g[x]), ["h0"] = D(g[h0]), ["w_ih"] = D(g[wih]), ["w_hh"] = D(g[whh]),
            ["b_ih"] = D(g[bih]), ["b_hh"] = D(g[bhh]),
        };
    }

    private static void AssertParity(Dictionary<string, double[]> torch, Dictionary<string, double[]> ours, double rel, double abs)
    {
        foreach (var key in torch.Keys)
        {
            var a = torch[key]; var b = ours[key];
            Assert.Equal(a.Length, b.Length);
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i];
                Assert.True(Math.Abs(a[i] - b[i]) <= abs + rel * Math.Abs(a[i]), $"{key}[{i}]: torch {a[i]}, ours {b[i]}.");
            }
            if (key != "y")
            {
                double cosine = ab / Math.Sqrt(aa * bb);
                Assert.True(cosine >= 0.99999, $"{key}: gradient cosine {cosine} against nn.GRU.");
            }
        }
    }

    [SkippableFact]
    public void Cpu_Double_MatchesNnGru()
    {
        var c = new Case();
        var torch = Torch(c, "float64", out var skip);
        Skip.If(torch is null, skip);
        AssertParity(torch!, Ours<double>(new CpuEngine(), c, (v, s) => new Tensor<double>((double[])v.Clone(), s)), 1e-9, 1e-12);
    }

    [SkippableFact]
    public void Cpu_Float_MatchesNnGru()
    {
        var c = new Case();
        var torch = Torch(c, "float64", out var skip);
        Skip.If(torch is null, skip);
        AssertParity(torch!, Ours<float>(new CpuEngine(), c, (v, s) => new Tensor<float>(Array.ConvertAll(v, d => (float)d), s)), 1e-3, 1e-5);
    }

    [SkippableFact]
    public void Gpu_MatchesNnGru()
    {
        var c = new Case();
        var torch = Torch(c, "float64", out var skip);
        Skip.If(torch is null, skip);
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend.");
        using var _ = gpu;
        AssertParity(torch!, Ours<float>(gpu!, c, (v, s) => new Tensor<float>(Array.ConvertAll(v, d => (float)d), s)), 1e-3, 1e-5);
    }
}
#endif
