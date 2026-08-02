// Copyright (c) AiDotNet. All rights reserved.
// Shared argument parsing for the kernel tools.
//
// Every one of them picked the kernel selector with
//
//     args.FirstOrDefault(a => !a.StartsWith("--"))
//
// which cannot tell a positional argument from a FLAG'S VALUE. So
//
//     --kernel-limiter --ncu "C:\...\ncu.exe"
//
// took the path as the kernel name, matched no catalog entry, and profiled nothing --
// while still printing its header and its "0 at a named roofline" summary. A gate that
// silently measures an empty set is worse than one that fails, because it looks like a
// clean result. The same shape would hit --kernel-autotune --out, --kernel-splitk --out
// and every conveyor stage's --coarsen.
//
// Fixed by naming the flags that take a value, so a selector is a token that is neither a
// flag nor one of their values.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelToolArgs
{
    /// <summary>
    /// Flags whose NEXT token is a value rather than a kernel selector.
    /// </summary>
    /// <remarks>
    /// An explicit set rather than "anything after a flag", because boolean flags exist
    /// too: <c>--no-coarsen depthwise_conv2d_3x3</c> has to keep selecting the kernel.
    /// </remarks>
    private static readonly HashSet<string> ValueFlags = new(StringComparer.Ordinal)
    {
        "--ncu", "--out", "--nvdisasm", "--coarsen", "--max-lanes",
        "--competitor", "--limiter", "--runner-python", "--max-spread-pct",
        "--competitor-python", "--ptxas", "--coverage-out", "--candidate",
        "--profile-candidate", "--evidence-dir",
    };

    /// <summary>The kernel selector, or "all" when none was given.</summary>
    public static string Selector(string[] args)
    {
        if (args is null) return "all";
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                if (ValueFlags.Contains(args[i])) i++;   // skip its value
                continue;
            }
            return args[i];
        }
        return "all";
    }

    /// <summary>The value of a flag, or null when it was not given.</summary>
    public static string? ValueOf(string[] args, string flag)
    {
        if (!ValueFlags.Contains(flag))
            throw new ArgumentException(
                "Value flag '" + flag + "' is not registered in KernelToolArgs.ValueFlags.",
                nameof(flag));
        if (args is null) return null;
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], flag, StringComparison.Ordinal)) return args[i + 1];
        return null;
    }

    /// <summary>
    /// Throws when a selector names nothing, so an empty run cannot be mistaken for a
    /// clean one.
    /// </summary>
    /// <param name="selector">The selector that was parsed.</param>
    /// <param name="matched">How many catalog entries it matched.</param>
    /// <param name="tool">Tool name, for the message.</param>
    public static void RequireNonEmptySelection(string selector, int matched, string tool)
    {
        if (matched > 0) return;
        throw new ArgumentException(
            "[" + tool + "] selector '" + selector + "' matched no catalog kernel, so " +
            "there is nothing to measure. Pass a kernel name, or 'all'.");
    }
}
