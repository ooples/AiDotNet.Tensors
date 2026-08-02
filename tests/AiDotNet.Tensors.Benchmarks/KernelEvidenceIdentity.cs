// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Content identity for the exact generated dispatch measured by evidence lanes.</summary>
internal static class KernelEvidenceIdentity
{
    /// <summary>
    /// Hashes device, target, semantic spec, emitted candidate set, and selected winner for
    /// every catalog row. A competitor file with another hash measured another program.
    /// </summary>
    internal static string CurrentDispatch()
    {
        using var runtime = new DirectPtxRuntime();
        return CurrentDispatch(runtime);
    }

    /// <summary>Hashes the current dispatch using an already-open device runtime.</summary>
    internal static string CurrentDispatch(DirectPtxRuntime runtime)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        var text = new StringBuilder();
        foreach (CodegenCatalogEntry entry in CodegenKernelCatalog.All)
        {
            var identity = CodegenAutotuneIdentity.Create(
                entry.Bench, runtime.DeviceFingerprint,
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            string winner = CodegenAutotuneCache.WinnerFor(entry.Name, identity) ?? "modelled";
            text.Append(entry.Name).Append('|')
                .Append(identity.DeviceFingerprint).Append('|')
                .Append(identity.Target).Append('|')
                .Append(identity.SpecFingerprint).Append('|')
                .Append(identity.EmitterFingerprint).Append('|')
                .Append(winner).Append(';');
        }

        using var sha = SHA256.Create();
        byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(text.ToString()));
        var result = new StringBuilder(digest.Length * 2);
        for (int i = 0; i < digest.Length; i++)
            result.Append(digest[i].ToString("x2", CultureInfo.InvariantCulture));
        return "sha256-" + result;
    }
}
