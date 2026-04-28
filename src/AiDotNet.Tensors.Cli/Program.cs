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
        // Format names are case-insensitive — `--from PT` or `--from
        // SafeTensors` should hit the same dispatch as the lower-case
        // versions documented in --help. Normalise once at the entry.
        string from = Require(parsed, "from").ToLowerInvariant();
        string to = Require(parsed, "to").ToLowerInvariant();
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
        else if (path.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase))
        {
            // Restrict to the canonical HF sharded-index suffix so an
            // unrelated *.index.json (HF model card index, dataset
            // index, etc.) doesn't get misclassified as a sharded
            // safetensors checkpoint and confuse the reader.
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
                // FP16 / BF16 — PyTorch's HalfStorage / BFloat16Storage
                // don't have a CLR mapping that PtReader.ToTensor<T> can
                // satisfy (System.Half and BFloat16 are .NET 5+/9+ types
                // with awkward cross-tfm support), so route the raw byte
                // payload through the writer's AddRaw using the matching
                // safetensors dtype tag. Most public HuggingFace
                // checkpoints (Llama / Mistral / Whisper / Stable
                // Diffusion / etc.) are F16 or BF16; without this branch
                // a `convert pt → safetensors` of those checkpoints
                // would drop almost every weight with a "skipping"
                // warning. Now: lossless byte-for-byte transfer.
                case "HalfStorage":
                {
                    var shape = new long[t.Shape.Length];
                    for (int i = 0; i < shape.Length; i++) shape[i] = t.Shape[i];
                    w.AddRaw(kv.Key, AiDotNet.Tensors.Serialization.Safetensors.SafetensorsDtype.F16,
                        shape, ExtractContiguousBytes(t));
                    break;
                }
                case "BFloat16Storage":
                {
                    var shape = new long[t.Shape.Length];
                    for (int i = 0; i < shape.Length; i++) shape[i] = t.Shape[i];
                    w.AddRaw(kv.Key, AiDotNet.Tensors.Serialization.Safetensors.SafetensorsDtype.BF16,
                        shape, ExtractContiguousBytes(t));
                    break;
                }
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

    /// <summary>
    /// Returns the contiguous byte payload of <paramref name="t"/>
    /// suitable for direct emission to a safetensors writer's AddRaw.
    /// For row-major-contiguous tensors this is the raw storage at the
    /// declared StorageOffset; for non-contiguous (strided) tensors it
    /// gathers via stride walk into a fresh contiguous buffer so the
    /// emitted file always represents the tensor's logical shape, not
    /// its on-disk layout.
    /// </summary>
    private static byte[] ExtractContiguousBytes(AiDotNet.Tensors.Serialization.Pickle.PtTensorRef t)
    {
        // All arithmetic is in `checked` blocks so a malformed
        // checkpoint with absurd shape / strides / offset surfaces as
        // an OverflowException-wrapped InvalidDataException instead of
        // a downstream out-of-range Span.Slice or wrapped negative
        // size. Shape entries are also validated as non-negative
        // since PtTensorRef accepts them as long but downstream code
        // assumes a positive product.
        long elemCount = 1;
        for (int i = 0; i < t.Shape.Length; i++)
        {
            if (t.Shape[i] < 0)
                throw new InvalidDataException(
                    $"Tensor shape dim {i} = {t.Shape[i]} is negative; refusing to materialise.");
            try { elemCount = checked(elemCount * t.Shape[i]); }
            catch (OverflowException ex)
            {
                throw new InvalidDataException(
                    $"Tensor shape product overflows long for storage '{t.DtypeStorage}'.", ex);
            }
        }
        // Inline the size lookup rather than calling
        // PtReader.StorageElementSize (which is `internal` for test-
        // project access only). Mirror the same set the reader knows.
        int eltSize = t.DtypeStorage switch
        {
            "FloatStorage" => 4,
            "DoubleStorage" => 8,
            "LongStorage" => 8,
            "IntStorage" => 4,
            "ShortStorage" => 2,
            "CharStorage" or "ByteStorage" or "BoolStorage" => 1,
            "HalfStorage" or "BFloat16Storage" => 2,
            _ => throw new InvalidOperationException(
                $"Unknown PyTorch storage type '{t.DtypeStorage}'."),
        };
        long byteCount;
        try { byteCount = checked(elemCount * eltSize); }
        catch (OverflowException ex)
        {
            throw new InvalidDataException(
                $"Tensor element count × element size overflows long.", ex);
        }
        if (byteCount > int.MaxValue)
            throw new InvalidOperationException(
                $"Tensor {byteCount} bytes — exceeds int.MaxValue and cannot fit in a single byte[].");

        if (t.IsContiguous)
        {
            // StorageOffset and the resulting byte range must fit
            // inside the storage. Without these guards a negative
            // StorageOffset would pass to ReadOnlySpan and throw
            // ArgumentOutOfRangeException with no diagnostic context.
            long startByte;
            try { startByte = checked(t.StorageOffset * eltSize); }
            catch (OverflowException ex)
            {
                throw new InvalidDataException(
                    $"Tensor storage-offset × element-size overflows long.", ex);
            }
            long endByte;
            try { endByte = checked(startByte + byteCount); }
            catch (OverflowException ex)
            {
                throw new InvalidDataException(
                    $"Tensor storage range overflows long.", ex);
            }
            if (startByte < 0 || endByte > t.Bytes.Length)
                throw new InvalidDataException(
                    $"Tensor storage range [{startByte}, {endByte}) is outside the {t.Bytes.Length}-byte storage.");
            var span = new ReadOnlySpan<byte>(t.Bytes, (int)startByte, (int)byteCount);
            return span.ToArray();
        }

        // Strided gather — walk the multi-index, copy element-by-
        // element into a fresh row-major buffer with bounds-checks
        // on every source-byte computation. Strides come from
        // checkpoint metadata so a malicious file could otherwise
        // wrap the multiply or land outside Bytes.Length.
        var buf = new byte[byteCount];
        var idx = new long[t.Shape.Length];
        for (long flat = 0; flat < elemCount; flat++)
        {
            long src = t.StorageOffset;
            for (int d = 0; d < t.Shape.Length; d++)
            {
                long step;
                try { step = checked(idx[d] * t.Strides[d]); }
                catch (OverflowException ex)
                {
                    throw new InvalidDataException(
                        $"Stride × index overflows at flat={flat}, dim={d}.", ex);
                }
                try { src = checked(src + step); }
                catch (OverflowException ex)
                {
                    throw new InvalidDataException(
                        $"Source-element accumulation overflows at flat={flat}, dim={d}.", ex);
                }
            }
            long srcByte;
            try { srcByte = checked(src * eltSize); }
            catch (OverflowException ex)
            {
                throw new InvalidDataException(
                    $"Source-byte computation overflows at flat={flat}.", ex);
            }
            if (srcByte < 0 || srcByte + eltSize > t.Bytes.Length)
                throw new InvalidDataException(
                    $"Strided read out of bounds at flat={flat}: srcByte={srcByte}, eltSize={eltSize}, " +
                    $"storage bytes={t.Bytes.Length}.");
            new ReadOnlySpan<byte>(t.Bytes, (int)srcByte, eltSize)
                .CopyTo(new Span<byte>(buf, (int)(flat * eltSize), eltSize));
            for (int d = t.Shape.Length - 1; d >= 0; d--)
            {
                idx[d]++;
                if (idx[d] < t.Shape[d]) break;
                idx[d] = 0;
            }
        }
        return buf;
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
        if (double.IsNaN(n) || double.IsInfinity(n) || n <= 0)
            throw new ArgumentException(
                $"Shard size '{s}' must be a positive finite number; got {n}.");
        // Multiply at double precision and bounds-check before the long
        // cast — `(long)(n * mult)` was silently truncating overflow
        // results to a wrap-around long, so '8000PB' parsed to a small
        // positive number that downstream consumers happily accepted.
        double bytes = n * mult;
        if (bytes <= 0 || bytes > long.MaxValue)
            throw new ArgumentException(
                $"Shard size '{s}' is out of range (parsed {bytes} bytes; must be in (0, {long.MaxValue}]).");
        return (long)bytes;
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
