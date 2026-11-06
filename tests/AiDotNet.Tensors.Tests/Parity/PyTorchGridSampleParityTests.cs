#if NET5_0_OR_GREATER
// Net-core only: it drives an external Python process (ProcessStartInfo.ArgumentList), as HeadToHeadNetworkHarness does.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// The mode-aware GridSample forward, input gradient and grid gradient against torch.nn.functional.grid_sample, for
/// every mode, padding and align_corners combination. The reference runs tools/parity/torch_grid_sample_reference.py;
/// without Python and torch the case is skipped with the reason, never passed.
/// </summary>
/// <remarks>
/// The finite-difference tests check that the backward is the derivative of the forward; only a reference can check
/// that the forward itself follows PyTorch's padding rules. Reflection without align_corners used to reflect each
/// integer tap about pixel 0 (the align_corners=true rule), so the forward and both gradients differed from PyTorch
/// for every mode while the finite differences agreed. Besides random points in [-1.5, 1.5], the grid holds exact
/// pixel centres, half-pixels and the image edges, where PyTorch picks a one-sided gradient.
/// </remarks>
[Collection("DirectGpuSerial")]
public sealed class PyTorchGridSampleParityTests
{
    private const int N = 2, C = 2, H = 3, W = 4, RandomPoints = 40;

    private static readonly double[] Special =
    {
        -1.0, -1.25, -1.5, -0.75, -0.5, -2.0, 1.0, 1.25, 1.5, 0.5, 0.0, -1.0 / 3, 1.0 / 3, 2.0 / 3,
        -1.0 + 1.0 / W, 1.0 - 1.0 / W,
    };

    private sealed class Case
    {
        public readonly int Points;
        public double[] Input = Values(1, N * C * H * W, 1.0);
        public double[] Grid;
        public double[] GradOutput;

        /// <param name="withEdges">Adds the exact pixel centres and edges. Only a double run can hold them exactly:
        /// in float a point a hair off a pixel centre flips the side PyTorch's one-sided gradient takes.</param>
        public Case(bool withEdges)
        {
            int special = withEdges ? Special.Length : 0;
            Points = RandomPoints + special;
            var rng = RandomHelper.CreateSeededRandom(2);
            Grid = new double[N * Points * 2];
            for (int n = 0; n < N; n++)
            {
                for (int p = 0; p < RandomPoints * 2; p++) Grid[n * Points * 2 + p] = rng.NextDouble() * 3 - 1.5;
                for (int s = 0; s < special; s++)
                {
                    int at = n * Points * 2 + (RandomPoints + s) * 2;
                    Grid[at] = Special[s];                           // x
                    Grid[at + 1] = Special[Special.Length - 1 - s];  // y
                }
            }
            GradOutput = Values(3, N * C * Points, 1.0);
        }
    }

    private static double[] Values(int seed, int n, double scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = (rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static readonly (string Name, GridSampleMode Mode)[] Modes =
        { ("bilinear", GridSampleMode.Bilinear), ("nearest", GridSampleMode.Nearest), ("bicubic", GridSampleMode.Bicubic) };

    private static readonly (string Name, GridSamplePadding Padding)[] Paddings =
        { ("zeros", GridSamplePadding.Zeros), ("border", GridSamplePadding.Border), ("reflection", GridSamplePadding.Reflection) };

    private static Dictionary<string, Dictionary<string, double[]>>? Torch(Case c, out string? skipReason)
    {
        skipReason = null;
        string? root = PyTorchParityInventory.FindRepositoryRoot();
        if (root is null) { skipReason = "tools/parity is missing from this checkout."; return null; }
        string script = Path.Combine(root, "tools", "parity", "torch_grid_sample_reference.py");
        string dir = Path.Combine(Path.GetTempPath(), "aidotnet-gridsample-parity-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string casePath = Path.Combine(dir, "case.json"), resultPath = Path.Combine(dir, "result.json");
            File.WriteAllText(casePath, JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["input"] = c.Input, ["input_shape"] = new[] { N, C, H, W },
                ["grid"] = c.Grid, ["grid_shape"] = new[] { N, 1, c.Points, 2 },
                ["grad_output"] = c.GradOutput, ["grad_output_shape"] = new[] { N, C, 1, c.Points },
            }));
            var psi = new ProcessStartInfo(HeadToHeadNetworkHarness.PythonExecutable)
            {
                // stdout is not read (the result is a file), so it is not redirected: an unread pipe that fills blocks the child.
                RedirectStandardError = true, RedirectStandardOutput = false, UseShellExecute = false,
            };
            foreach (var a in new[] { script, casePath, resultPath }) psi.ArgumentList.Add(a);
            Process? p;
            try { p = Process.Start(psi); }
            catch (System.ComponentModel.Win32Exception) { skipReason = $"{psi.FileName} is not on PATH (set PARITY_PYTHON)."; return null; }
            if (p is null) { skipReason = "python did not start."; return null; }
            using (p)
            {
                // Read stderr while waiting: reading it to the end first would never return from a child that hangs.
                var errRead = p.StandardError.ReadToEndAsync();
                if (!p.WaitForExit(120_000))
                {
                    try { p.Kill(); } catch (InvalidOperationException) { }   // it exited between the wait and the kill
                    throw new TimeoutException($"{script} did not finish within 120 s.");
                }
                string err = errRead.GetAwaiter().GetResult();
                if (p.ExitCode != 0)
                {
                    skipReason = err.Contains("No module named 'torch'", StringComparison.Ordinal)
                        ? "torch is not installed for " + psi.FileName + "."
                        : null;
                    if (skipReason is null)
                        throw new InvalidOperationException($"torch_grid_sample_reference.py failed ({p.ExitCode}): {err}");
                    return null;
                }
            }
            using var doc = JsonDocument.Parse(File.ReadAllText(resultPath));
            var result = new Dictionary<string, Dictionary<string, double[]>>();
            foreach (var combination in doc.RootElement.EnumerateObject())
            {
                var parts = new Dictionary<string, double[]>();
                foreach (var key in new[] { "output", "grad_input", "grad_grid" })
                    parts[key] = combination.Value.GetProperty(key).EnumerateArray().Select(v => v.GetDouble()).ToArray();
                result[combination.Name] = parts;
            }
            return result;
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }

    private static Dictionary<string, double[]> Ours<TT>(IEngine e, Case c, GridSampleMode mode,
        GridSamplePadding padding, bool align, Func<double[], int[], Tensor<TT>> make)
    {
        var input = make(c.Input, new[] { N, C, H, W });
        var grid = make(c.Grid, new[] { N, 1, c.Points, 2 });
        var gradOut = make(c.GradOutput, new[] { N, C, 1, c.Points });
        double[] D(Tensor<TT> t) => Array.ConvertAll(t.ToArray(), v => Convert.ToDouble(v));
        return new Dictionary<string, double[]>
        {
            ["output"] = D(e.GridSample(input, grid, mode, padding, align)),
            ["grad_input"] = D(e.GridSampleBackwardInput(gradOut, grid, new[] { N, C, H, W }, mode, padding, align)),
            ["grad_grid"] = D(e.GridSampleBackwardGrid(gradOut, input, grid, mode, padding, align)),
        };
    }

    private static void AssertAllCombinations(IEngine engine, Dictionary<string, Dictionary<string, double[]>> torch,
        Case c, double rel, double abs, Func<double[], int[], Tensor<double>>? makeDouble,
        Func<double[], int[], Tensor<float>>? makeFloat)
    {
        var failures = new List<string>();
        foreach (var (modeName, mode) in Modes)
        foreach (var (padName, padding) in Paddings)
        foreach (bool align in new[] { false, true })
        {
            string key = $"{modeName}/{padName}/{(align ? 1 : 0)}";
            var ours = makeDouble is not null
                ? Ours(engine, c, mode, padding, align, makeDouble)
                : Ours(engine, c, mode, padding, align, makeFloat ?? throw new ArgumentNullException(nameof(makeFloat)));
            foreach (var part in torch[key])
            {
                var a = part.Value;
                var b = ours[part.Key];
                Assert.Equal(a.Length, b.Length);
                for (int i = 0; i < a.Length; i++)
                {
                    bool finiteMatch = double.IsFinite(a[i]) == double.IsFinite(b[i]);
                    if (!finiteMatch || Math.Abs(a[i] - b[i]) > abs + rel * Math.Abs(a[i]))
                    {
                        failures.Add($"{key} {part.Key}[{i}]: torch {a[i]:G10}, ours {b[i]:G10}");
                        break;
                    }
                }
            }
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [SkippableFact]
    public void Cpu_Double_MatchesTorchGridSample()
    {
        var c = new Case(withEdges: true);
        var torch = Torch(c, out var skip);
        Skip.If(torch is null, skip);
        if (torch is null) return;
        AssertAllCombinations(new CpuEngine(), torch, c, 1e-9, 1e-12,
            (v, s) => new Tensor<double>((double[])v.Clone(), s), null);
    }

    [SkippableFact]
    public void Gpu_MatchesTorchGridSample()
    {
        var c = new Case(withEdges: false);
        var torch = Torch(c, out var skip);
        Skip.If(torch is null, skip);
        if (torch is null) return;
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpu backend.");
        AssertAllCombinations(gpu, torch, c, 1e-3, 1e-4, null,
            (v, s) => new Tensor<float>(Array.ConvertAll(v, d => (float)d), s));
    }
}
#endif
