using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source audit: every CUDA/HIP launch that builds a <c>void**</c> argument array for a cached kernel must fill
/// exactly as many slots as the kernel declares parameters.
/// </summary>
/// <remarks>
/// cuLaunchKernel / hipModuleLaunchKernel read parameters from the array by position and cannot check the count.
/// Fewer slots: the kernel reads its trailing parameters from whatever follows the array (HIP im2col read outW that
/// way; the loss kernels read size after a missing epsilon). More slots: values land in the wrong parameters. SELU,
/// SELU-backward and hardtanh passed alpha/scale or min/max to kernels that take (input, output, size), so size
/// held a float's bit pattern (~1e9) and every thread of the last block wrote past its buffers; SELU corrupted
/// the engine's cached ones vectors that way deep into the op-parity suite. Unfilled spare slots at the end of an
/// over-sized stackalloc are harmless and not reported.
/// Kernel names that two modules define with different parameter lists are also reported: the backend's kernel
/// cache keeps one function per name, so a launch built for the other definition runs the wrong kernel (the
/// scalar MseLoss/HuberLoss overloads did).
/// </remarks>
public class KernelLaunchArityAuditTests
{
    private static string RepoRoot()
    {
        var probe = Path.Combine("src", "AiDotNet.Tensors", "Engines", "CpuEngine.cs");
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, probe)))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException(
            $"could not locate repo root by walking up from {AppContext.BaseDirectory} looking for {probe}");
    }

    private static readonly Regex Kernel = new(
        @"__global__\s+(?:__launch_bounds__\([^)]*\)\s*)?void\s+(\w+)\s*\(([^)]*)\)", RegexOptions.Compiled);

    private static string StripComments(string s)
        => Regex.Replace(Regex.Replace(s, @"/\*.*?\*/", " ", RegexOptions.Singleline), @"//[^\n]*", " ");

    /// <summary>Kernel name → (parameter count, file) for every definition under the backend folder.</summary>
    private static Dictionary<string, List<(int Count, string File)>> KernelParams(string backendDir)
    {
        var result = new Dictionary<string, List<(int, string)>>(StringComparer.Ordinal);
        foreach (var file in Directory.EnumerateFiles(backendDir, "*.cs", SearchOption.AllDirectories))
        {
            var src = StripComments(File.ReadAllText(file));
            foreach (Match m in Kernel.Matches(src))
            {
                if (m.Groups[1].Value == "name") continue; // macro template parameter, not a kernel
                string body = Regex.Replace(m.Groups[2].Value, @"\s+", " ").Trim();
                int count = body.Length == 0 || body == "void" ? 0 : body.Split(',').Length;
                if (!result.TryGetValue(m.Groups[1].Value, out var list))
                    result[m.Groups[1].Value] = list = new List<(int, string)>();
                list.Add((count, Path.GetFileName(file)));
            }
        }
        return result;
    }

    /// <summary>(kernel, filled slots, location) for each launch whose kernel and argument array can be resolved.</summary>
    private static IEnumerable<(string Kernel, int Filled, string Where)> Launches(string backendDir)
    {
        var bind1 = new Regex(@"_kernelCache\.TryGetValue\(\s*""(\w+)""\s*,\s*out\s+(?:var|IntPtr|\w+)\s+(\w+)\)");
        var bind2 = new Regex(@"(?:var|IntPtr)\s+(\w+)\s*=\s*_kernelCache\[\s*""(\w+)""\s*\]");
        var launch = new Regex(@"Launch\w*\(\s*(\w+)\s*,[^;]*?,\s*(\w+)\s*\)\s*;");
        var method = new Regex(@"^\s{4}(?:public|private|internal|protected)\b[^\n;=]*\(", RegexOptions.Multiline);
        foreach (var file in Directory.EnumerateFiles(backendDir, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains(Path.DirectorySeparatorChar + "Kernels" + Path.DirectorySeparatorChar)) continue;
            var src = File.ReadAllText(file);
            var starts = method.Matches(src).Cast<Match>().Select(m => m.Index).Append(src.Length).ToList();
            for (int s = 0; s + 1 < starts.Count; s++)
            {
                string body = src.Substring(starts[s], starts[s + 1] - starts[s]);
                var binds = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (Match m in bind1.Matches(body)) binds[m.Groups[2].Value] = m.Groups[1].Value;
                foreach (Match m in bind2.Matches(body)) binds[m.Groups[1].Value] = m.Groups[2].Value;
                if (binds.Count == 0) continue;
                foreach (Match m in launch.Matches(body))
                {
                    if (!binds.TryGetValue(m.Groups[1].Value, out var kernel)) continue;
                    string argsVar = m.Groups[2].Value;
                    // The array this launch passes: the nearest preceding stackalloc of that name.
                    var decls = Regex.Matches(body.Substring(0, m.Index),
                        @"void\*\*\s+" + Regex.Escape(argsVar) + @"\s*=\s*stackalloc\s+void\*\s*\[\s*\d+\s*\]");
                    if (decls.Count == 0) continue;
                    int from = decls[decls.Count - 1].Index;
                    string region = body.Substring(from, m.Index - from);
                    var slots = Regex.Matches(region, Regex.Escape(argsVar) + @"\[\s*(\d+)\s*\]\s*=")
                        .Cast<Match>().Select(x => int.Parse(x.Groups[1].Value)).ToList();
                    if (slots.Count == 0) continue;
                    int line = src.Take(starts[s] + m.Index).Count(c => c == '\n') + 1;
                    yield return (kernel, slots.Max() + 1, $"{Path.GetFileName(file)}:{line}");
                }
            }
        }
    }

    [Theory]
    [InlineData("CUDA")]
    [InlineData("HIP")]
    public void EveryLaunchFillsExactlyTheKernelsParameters(string backend)
    {
        string dir = Path.Combine(RepoRoot(), "src", "AiDotNet.Tensors", "Engines", "DirectGpu", backend);
        var kernels = KernelParams(dir);
        var launches = Launches(dir).ToList();
        Assert.True(launches.Count > 100, $"Only {launches.Count} {backend} launches resolved: the audit's patterns no longer match the sources.");

        var bad = launches
            .Where(l => kernels.TryGetValue(l.Kernel, out var defs) && defs.All(d => d.Count != l.Filled))
            .Select(l => $"{l.Where}: {l.Kernel} fills {l.Filled} argument slot(s), the kernel declares "
                + string.Join(" or ", kernels[l.Kernel].Select(d => d.Count).Distinct()))
            .ToList();
        Assert.True(bad.Count == 0, $"{backend} launch/kernel arity mismatches:\n" + string.Join("\n", bad));
    }

    [Theory]
    [InlineData("CUDA")]
    [InlineData("HIP")]
    public void LaunchedKernelNamesHaveOneParameterList(string backend)
    {
        string dir = Path.Combine(RepoRoot(), "src", "AiDotNet.Tensors", "Engines", "DirectGpu", backend);
        var kernels = KernelParams(dir);
        var launched = new HashSet<string>(Launches(dir).Select(l => l.Kernel), StringComparer.Ordinal);
        var conflicts = kernels
            .Where(k => launched.Contains(k.Key) && k.Value.Select(d => d.Count).Distinct().Count() > 1)
            .Select(k => $"{k.Key}: " + string.Join(", ", k.Value.Select(d => $"{d.Count} params in {d.File}")))
            .ToList();
        Assert.True(conflicts.Count == 0,
            $"{backend} kernels defined twice with different parameter lists (the cache keeps only one):\n"
            + string.Join("\n", conflicts));
    }
}
