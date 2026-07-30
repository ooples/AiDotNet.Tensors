// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Prints the exact per-kernel autotune identity without requiring a timing run.</summary>
internal static class KernelIdentityTool
{
    internal static void Run(string[] args)
    {
        string selector = KernelToolArgs.Selector(args);
        var entries = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All
            : new[] { CodegenKernelCatalog.Find(selector)! }.Where(e => e is not null).ToList();
        KernelToolArgs.RequireNonEmptySelection(selector, entries.Count, "kernel-identity");

        if (!DirectPtxRuntime.IsAvailable)
        {
            Console.WriteLine(
                "NVIDIA CUDA Driver API is unavailable; kernel identity needs a device fingerprint.");
            return;
        }
        using var runtime = new DirectPtxRuntime();
        Console.WriteLine("kernel\tdevice\ttarget\tspec\temitter");
        foreach (CodegenCatalogEntry entry in entries)
        {
            CodegenAutotuneIdentity identity = CodegenAutotuneIdentity.Create(
                entry.Bench, runtime.DeviceFingerprint,
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            Console.WriteLine(string.Join("\t", entry.Name, identity.DeviceFingerprint,
                identity.Target, identity.SpecFingerprint, identity.EmitterFingerprint));
        }
    }
}
