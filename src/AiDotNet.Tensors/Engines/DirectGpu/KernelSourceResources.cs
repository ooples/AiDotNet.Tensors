// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Kernel source text shipped as embedded resources (Engines/DirectGpu/KernelSources) rather than string literals. The
/// assembly's user-string heap is capped at 16 MB (CS8103) and the inline kernel families already fill nearly all of it,
/// so new kernel families keep their source text here. Each file is embedded under the logical name
/// <c>AiDotNet.Tensors.KernelSources.&lt;file name&gt;</c> and read once (UTF-8, line endings normalised to LF).
/// </summary>
internal static class KernelSourceResources
{
    private const string Prefix = "AiDotNet.Tensors.KernelSources.";

    private static readonly ConcurrentDictionary<string, string> Cache = new(StringComparer.Ordinal);

    /// <summary>The text of the embedded kernel source <paramref name="fileName"/>.</summary>
    internal static string Read(string fileName) => Cache.GetOrAdd(fileName, Load);

    private static string Load(string fileName)
    {
        var assembly = typeof(KernelSourceResources).Assembly;
        using Stream? stream = assembly.GetManifestResourceStream(Prefix + fileName);
        if (stream is null)
            throw new InvalidOperationException($"The embedded kernel source '{fileName}' is missing from the assembly.");
        using var reader = new StreamReader(stream, new UTF8Encoding(false), detectEncodingFromByteOrderMarks: true);
        return reader.ReadToEnd().Replace("\r\n", "\n");
    }
}
