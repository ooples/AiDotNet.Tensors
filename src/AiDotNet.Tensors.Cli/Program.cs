// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Serialization.Pickle;
using AiDotNet.Tensors.Serialization.Safetensors;

namespace AiDotNet.Tensors.Cli;

/// <summary>
/// Entry point for the <c>ai-tensors</c> dotnet tool. Currently
/// supports one verb — <c>convert</c> — that translates between the
/// formats AiDotNet.Tensors reads and writes:
/// PyTorch <c>.pt</c> → safetensors, safetensors ↔ safetensors
/// (sharded re-pack), and a placeholder for safetensors → GGUF
/// quantisation that the GGUF writer (separate follow-up) will fill.
/// </summary>
internal static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "-h" or "help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        try
        {
            return args[0] switch
            {
                "convert" => RunConvert(args.Skip(1).ToArray()),
                "inspect" => RunInspect(args.Skip(1).ToArray()),
                _ => UnknownCommand(args[0]),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: {ex.GetType().Name}: {ex.Message}");
            return 1;
        }
    }

    private static int UnknownCommand(string verb)
    {
        Console.Error.WriteLine($"error: unknown command '{verb}'.");
        PrintUsage();
        return 1;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("ai-tensors — checkpoint conversion + inspection.");
        Console.WriteLine();
        Console.WriteLine("USAGE:");
        Console.WriteLine("  ai-tensors convert --from <fmt> --to <fmt> --input <path> --output <path>");
        Console.WriteLine("  ai-tensors inspect <path>");
        Console.WriteLine();
        Console.WriteLine("FORMATS:");
        Console.WriteLine("  pt          PyTorch .pt / .pth (zip-format or legacy pickle)");
        Console.WriteLine("  safetensors safetensors single-file");
        Console.WriteLine("  safetensors-sharded   sharded safetensors directory + index.json");
        Console.WriteLine();
        Console.WriteLine("EXAMPLES:");
        Console.WriteLine("  ai-tensors convert --from pt --to safetensors \\");
        Console.WriteLine("    --input model.pt --output model.safetensors");
        Console.WriteLine();
        Console.WriteLine("  ai-tensors convert --from safetensors --to safetensors-sharded \\");
        Console.WriteLine("    --input model.safetensors --output checkpoints/sharded \\");
        Console.WriteLine("    --shard-size 5GB");
        Console.WriteLine();
        Console.WriteLine("  ai-tensors inspect model.safetensors");
    }

    private static int RunConvert(string[] args)
    {
        var parsed = ParseFlags(args, "from", "to", "input", "output", "shard-size");
        string from = Require(parsed, "from");
        string to = Require(parsed, "to");
        string input = Require(parsed, "input");
        string output = Require(parsed, "output");
        long shardSize = parsed.TryGetValue("shard-size", out var ss)
            ? ParseSize(ss)
            : ShardedSafetensorsWriter.DefaultShardSizeBytes;

        Console.WriteLine($"convert {from} → {to}");
        Console.WriteLine($"  in : {input}");
        Console.WriteLine($"  out: {output}");

        switch ($"{from}->{to}")
        {
            case "pt->safetensors":
                ConvertPtToSafetensors(input, output);
                break;
            case "safetensors->safetensors-sharded":
                ConvertSafetensorsToSharded(input, output, shardSize);
                break;
            case "safetensors-sharded->safetensors":
                ConvertShardedToSafetensors(input, output);
                break;
            case "safetensors-sharded->safetensors-sharded":
                ConvertReshard(input, output, shardSize);
                break;
            default:
                throw new InvalidOperationException(
                    $"Unsupported conversion: {from} → {to}. Run --help for the supported pairs.");
        }
        return 0;
    }

    private static int RunInspect(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("error: inspect requires one path argument.");
            return 1;
        }
        string path = args[0];
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            using var r = SafetensorsReader.Open(path);
            Console.WriteLine($"safetensors  {path}");
            Console.WriteLine($"  tensors: {r.Entries.Count}");
            Console.WriteLine($"  metadata: {r.Metadata.Count} entries");
            foreach (var kv in r.Entries)
            {
                Console.WriteLine($"    {kv.Key}  {kv.Value.Dtype,-8}  [{string.Join(",", kv.Value.Shape)}]");
            }
        }
        else if (path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase)
              || path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
        {
            var r = PtReader.Open(path);
            Console.WriteLine($".pt  {path}");
            Console.WriteLine($"  tensors: {r.Tensors.Count}");
            foreach (var kv in r.Tensors)
                Console.WriteLine($"    {kv.Key}  {kv.Value.DtypeStorage,-15}  [{string.Join(",", kv.Value.Shape)}]");
        }
        else if (path.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase))
        {
            using var r = ShardedSafetensorsReader.Open(path);
            Console.WriteLine($"safetensors-sharded  {path}");
            Console.WriteLine($"  tensors: {r.Entries.Count}");
        }
        else
        {
            Console.Error.WriteLine(
                $"error: don't know how to inspect '{path}'. Expected .safetensors, .pt, .pth, or .safetensors.index.json.");
            return 1;
        }
        return 0;
    }

    private static void ConvertPtToSafetensors(string input, string output)
    {
        var pt = PtReader.Open(input);
        using var w = SafetensorsWriter.Create(output);
        int count = 0;
        foreach (var kv in pt.Tensors)
        {
            var t = kv.Value;
            switch (t.DtypeStorage)
            {
                case "FloatStorage":
                    w.Add(kv.Key, PtReader.ToTensor<float>(t));
                    break;
                case "DoubleStorage":
                    w.Add(kv.Key, PtReader.ToTensor<double>(t));
                    break;
                case "LongStorage":
                    w.Add(kv.Key, PtReader.ToTensor<long>(t));
                    break;
                case "IntStorage":
                    w.Add(kv.Key, PtReader.ToTensor<int>(t));
                    break;
                case "ShortStorage":
                    w.Add(kv.Key, PtReader.ToTensor<short>(t));
                    break;
                case "ByteStorage":
                    w.Add(kv.Key, PtReader.ToTensor<byte>(t));
                    break;
                case "CharStorage":
                    w.Add(kv.Key, PtReader.ToTensor<sbyte>(t));
                    break;
                case "BoolStorage":
                    w.Add(kv.Key, PtReader.ToTensor<bool>(t));
                    break;
                default:
                    Console.Error.WriteLine(
                        $"warn: skipping {kv.Key} — unsupported PyTorch storage type {t.DtypeStorage}.");
                    continue;
            }
            count++;
        }
        w.Save();
        Console.WriteLine($"  wrote {count} tensor(s)");
    }

    private static void ConvertSafetensorsToSharded(string input, string output, long shardSize)
    {
        using var r = SafetensorsReader.Open(input);
        var w = new ShardedSafetensorsWriter(output, "model", shardSize);
        foreach (var kv in r.Entries)
        {
            var bytes = r.ReadRawBytes(kv.Key);
            w.AddRaw(kv.Key, kv.Value.Dtype, kv.Value.Shape, bytes);
        }
        foreach (var meta in r.Metadata) w.Metadata[meta.Key] = meta.Value;
        int n = w.Save();
        Console.WriteLine($"  wrote {r.Entries.Count} tensor(s) across {n} shard(s)");
    }

    private static void ConvertShardedToSafetensors(string input, string output)
    {
        using var r = ShardedSafetensorsReader.Open(input);
        using var w = SafetensorsWriter.Create(output);
        foreach (var kv in r.Entries)
        {
            var bytes = r.ReadRawBytes(kv.Key);
            w.AddRaw(kv.Key, kv.Value.Dtype, kv.Value.Shape, bytes);
        }
        foreach (var meta in r.Metadata)
            if (meta.Key != "total_size") w.Metadata[meta.Key] = meta.Value;
        w.Save();
        Console.WriteLine($"  wrote {r.Entries.Count} tensor(s) into a single safetensors file");
    }

    private static void ConvertReshard(string input, string output, long shardSize)
    {
        using var r = ShardedSafetensorsReader.Open(input);
        var w = new ShardedSafetensorsWriter(output, "model", shardSize);
        foreach (var kv in r.Entries)
        {
            var bytes = r.ReadRawBytes(kv.Key);
            w.AddRaw(kv.Key, kv.Value.Dtype, kv.Value.Shape, bytes);
        }
        foreach (var meta in r.Metadata)
            if (meta.Key != "total_size") w.Metadata[meta.Key] = meta.Value;
        int n = w.Save();
        Console.WriteLine($"  resharded {r.Entries.Count} tensor(s) into {n} shard(s)");
    }

    private static long ParseSize(string s)
    {
        if (string.IsNullOrEmpty(s)) throw new ArgumentException("Empty size.");
        long mult = 1;
        string num = s;
        if (s.EndsWith("KB", StringComparison.OrdinalIgnoreCase)) { mult = 1024L; num = s[..^2]; }
        else if (s.EndsWith("MB", StringComparison.OrdinalIgnoreCase)) { mult = 1024L * 1024; num = s[..^2]; }
        else if (s.EndsWith("GB", StringComparison.OrdinalIgnoreCase)) { mult = 1024L * 1024 * 1024; num = s[..^2]; }
        else if (s.EndsWith("TB", StringComparison.OrdinalIgnoreCase)) { mult = 1024L * 1024 * 1024 * 1024; num = s[..^2]; }
        else if (s.EndsWith("B", StringComparison.OrdinalIgnoreCase)) { num = s[..^1]; }
        if (!double.TryParse(num, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double n))
            throw new ArgumentException($"Could not parse size '{s}'.");
        return (long)(n * mult);
    }

    private static Dictionary<string, string> ParseFlags(string[] args, params string[] knownFlags)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a.StartsWith("--", StringComparison.Ordinal))
            {
                string key = a[2..];
                if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                    throw new ArgumentException($"Flag --{key} expects a value.");
                dict[key] = args[i + 1];
                i++;
            }
        }
        foreach (var k in dict.Keys.ToList())
            if (Array.IndexOf(knownFlags, k.ToLowerInvariant()) < 0)
                Console.Error.WriteLine($"warn: unknown flag --{k}");
        return dict;
    }

    private static string Require(Dictionary<string, string> flags, string name)
    {
        if (!flags.TryGetValue(name, out var v))
            throw new ArgumentException($"Missing required flag --{name}.");
        return v;
    }
}
