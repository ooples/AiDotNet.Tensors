using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Records what a competitive run measured next to its BenchmarkDotNet reports: the competitor package versions this
/// app resolved, the commit and the runtime. The README claims gate (check_benchmark_claims.py) compares those versions
/// with the benchmark project's PackageReferences and with the versions the README names, so bumping a competitor
/// without re-running fails CI instead of publishing old numbers under a new version (#1094).
/// </summary>
internal static class RunManifest
{
    internal static void Write(string suite, params string[] packages)
    {
        string results = Path.Combine("BenchmarkDotNet.Artifacts", "results");
        Directory.CreateDirectory(results);
        string path = Path.Combine(results, "manifest.json");
        var root = File.Exists(path) ? JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? new JsonObject() : new JsonObject();
        var runs = root["runs"] as JsonObject ?? new JsonObject();
        root["runs"] = runs;

        var resolved = ResolvedPackageVersions();
        var versions = new JsonObject();
        foreach (string package in packages)
        {
            if (!resolved.TryGetValue(package, out string? version))
                throw new InvalidOperationException($"{package} is not in this app's deps.json; the manifest would not say what ran.");
            versions[package] = version;
        }

        runs[suite] = new JsonObject
        {
            ["date"] = DateTime.UtcNow.ToString("yyyy-MM-dd"),
            ["commit"] = GitCommit(),
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["os"] = RuntimeInformation.OSDescription,
            ["logicalProcessors"] = Environment.ProcessorCount,
            ["packages"] = versions,
        };
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Run manifest: {Path.GetFullPath(path)} -- copy it into Results/<run>/ with the reports.");
    }

    // deps.json names the exact NuGet version the build resolved ("NumSharp/0.70.0"); assembly versions do not.
    private static Dictionary<string, string> ResolvedPackageVersions()
    {
        string name = Assembly.GetEntryAssembly()?.GetName().Name ?? "AiDotNet.Tensors.Benchmarks";
        string deps = Path.Combine(AppContext.BaseDirectory, name + ".deps.json");
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (JsonNode.Parse(File.ReadAllText(deps))?["libraries"] is not JsonObject libraries) return map;
        foreach (var entry in libraries)
        {
            int slash = entry.Key.IndexOf('/');
            if (slash > 0 && entry.Value?["type"]?.GetValue<string>() == "package")
                map[entry.Key.Substring(0, slash)] = entry.Key.Substring(slash + 1);
        }
        return map;
    }

    private static string GitCommit()
    {
        try
        {
            using var git = Process.Start(new ProcessStartInfo("git", "rev-parse --short=9 HEAD")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            });
            if (git is null) return "unknown";
            string sha = git.StandardOutput.ReadToEnd().Trim();
            git.StandardError.ReadToEnd();
            git.WaitForExit();
            return git.ExitCode == 0 && sha.Length > 0 ? sha : "unknown";
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            Console.Error.WriteLine($"Run manifest: git is not available ({ex.NativeErrorCode}); commit recorded as unknown.");
            return "unknown";
        }
    }
}
