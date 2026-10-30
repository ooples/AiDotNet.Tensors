// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Folds a one-dimensional dispatch that needs more workgroups than the device allows per dimension (65,535 by
/// default, so 16.7M invocations at 256 per group) onto a 2-D grid, for every WGSL entry point and without touching
/// its source.
/// </summary>
/// <remarks>
/// The folded copy of an entry point reads its builtins from the flattened workgroup index
/// <c>workgroup_id.x + workgroup_id.y * num_workgroups.x</c>. For the dispatch it replaces, (groups, 1, 1), that gives
/// every builtin exactly the value it had: <c>global_invocation_id = (flat * size.x + local.x, local.y, local.z)</c>,
/// <c>workgroup_id = (flat, 0, 0)</c> and <c>num_workgroups = (groups, 1, 1)</c>. The 2-D grid can hold a few groups
/// more than <c>groups</c>; those return first, which is uniform control flow (the condition depends only on the
/// workgroup), so a later barrier stays valid. The true group count comes from a uniform at <c>@group(1)</c>, which no
/// shader in this backend uses.
/// </remarks>
internal static class WebGpuDispatchFold
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private static readonly Regex BuiltinParameter = new(
        @"^\s*@builtin\s*\(\s*(\w+)\s*\)\s*(\w+)\s*:\s*(\S(?:.*\S)?)\s*$", RegexOptions.Singleline, RegexTimeout);

    /// <summary>
    /// The group count's uniform, appended to a folded module (after any <c>enable</c> directive). @binding comes
    /// first so webgpu-tensors.js, which collects <c>@binding(n) var&lt;uniform&gt;</c> as group 0's uniform
    /// candidates, does not count it.
    /// </summary>
    internal const string GroupsDeclaration = "@binding(0) @group(1) var<uniform> aidn_fold_groups: vec4<u32>;";

    /// <summary>
    /// The grid a dispatch of <paramref name="groups"/> workgroups folds onto: as few rows as fit, each as even as
    /// possible, so at most <c>y - 1</c> groups are surplus.
    /// </summary>
    internal static (int X, int Y) Grid(long groups, int limit)
    {
        if (groups < 1) throw new ArgumentOutOfRangeException(nameof(groups), "A dispatch needs at least one workgroup.");
        if (limit < 1) throw new ArgumentOutOfRangeException(nameof(limit));
        long y = (groups + limit - 1) / limit;
        if (y > limit)
            throw new GpuCapabilityUnavailableException(
                $"A dispatch of {groups} workgroups does not fit {limit} x {limit}, the most a folded 1-D dispatch can hold.");
        long x = (groups + y - 1) / y;
        return ((int)x, (int)y);
    }

    /// <summary>
    /// The WGSL of <paramref name="source"/> with entry point <paramref name="entryPoint"/> folded, or null when that
    /// entry point cannot be folded (it is not found, or takes a parameter other than a builtin).
    /// </summary>
    internal static string? FoldSource(string source, string entryPoint)
    {
        int fn = FindEntryFunction(source, entryPoint);
        if (fn < 0) return null;
        int open = source.IndexOf('(', fn);
        int close = MatchingParen(source, open);
        if (open < 0 || close < 0) return null;
        int bodyOpen = source.IndexOf('{', close);
        if (bodyOpen < 0) return null;
        string? sizeX = WorkgroupSizeX(source, fn);
        if (sizeX is null) return null;

        string? gid = null, wid = null, nwg = null, lid = null;
        var kept = new List<string>();
        foreach (var parameter in SplitTopLevel(source.Substring(open + 1, close - open - 1)))
        {
            if (parameter.Trim().Length == 0) continue;
            var m = BuiltinParameter.Match(parameter);
            if (!m.Success) return null;
            string name = m.Groups[2].Value;
            switch (m.Groups[1].Value)
            {
                case "global_invocation_id": gid = name; break;
                case "workgroup_id": wid = name; break;
                case "num_workgroups": nwg = name; break;
                case "local_invocation_id": lid = name; kept.Add(parameter.Trim()); break;
                default: kept.Add(parameter.Trim()); break;
            }
        }
        string local = lid ?? "aidn_fold_lid";
        if (lid is null) kept.Add("@builtin(local_invocation_id) aidn_fold_lid: vec3<u32>");
        kept.Add("@builtin(workgroup_id) aidn_fold_wid: vec3<u32>");
        kept.Add("@builtin(num_workgroups) aidn_fold_nwg: vec3<u32>");

        var prologue = new StringBuilder();
        prologue.Append("\n    let aidn_fold_flat = aidn_fold_wid.x + aidn_fold_wid.y * aidn_fold_nwg.x;");
        prologue.Append("\n    if (aidn_fold_flat >= aidn_fold_groups.x) { return; }");
        if (gid is not null)
            prologue.Append($"\n    let {gid} = vec3<u32>(aidn_fold_flat * u32({sizeX}) + {local}.x, {local}.y, {local}.z);");
        if (wid is not null) prologue.Append($"\n    let {wid} = vec3<u32>(aidn_fold_flat, 0u, 0u);");
        if (nwg is not null) prologue.Append($"\n    let {nwg} = vec3<u32>(aidn_fold_groups.x, 1u, 1u);");

        return source.Substring(0, open + 1) + string.Join(", ", kept) + source.Substring(close, bodyOpen + 1 - close)
            + prologue + source.Substring(bodyOpen + 1) + "\n" + GroupsDeclaration + "\n";
    }

    // The `fn` of the @compute function named entryPoint (the name alone could also match a helper's prefix).
    private static int FindEntryFunction(string source, string entryPoint)
    {
        var m = Regex.Match(source, @"\bfn\s+" + Regex.Escape(entryPoint) + @"\s*\(", RegexOptions.None, RegexTimeout);
        while (m.Success)
        {
            int attributesStart = Math.Max(source.LastIndexOf('}', m.Index), source.LastIndexOf(';', m.Index)) + 1;
            if (source.IndexOf("@compute", attributesStart, m.Index - attributesStart, StringComparison.Ordinal) >= 0)
                return m.Index;
            m = m.NextMatch();
        }
        return -1;
    }

    // The first argument of the entry point's @workgroup_size, as written (a literal or a module-scope constant).
    private static string? WorkgroupSizeX(string source, int fn)
    {
        int attributesStart = Math.Max(source.LastIndexOf('}', fn), source.LastIndexOf(';', fn)) + 1;
        int at = source.IndexOf("@workgroup_size", attributesStart, fn - attributesStart, StringComparison.Ordinal);
        if (at < 0) return null;
        int open = source.IndexOf('(', at);
        int close = MatchingParen(source, open);
        if (open < 0 || close < 0) return null;
        var args = SplitTopLevel(source.Substring(open + 1, close - open - 1));
        return args.Count == 0 || args[0].Trim().Length == 0 ? null : args[0].Trim();
    }

    private static int MatchingParen(string s, int open)
    {
        if (open < 0 || open >= s.Length || s[open] != '(') return -1;
        int depth = 0;
        for (int i = open; i < s.Length; i++)
        {
            if (s[i] == '(') depth++;
            else if (s[i] == ')' && --depth == 0) return i;
        }
        return -1;
    }

    // Splits on commas outside (), <> and [] (vec3<u32>, array<f32, 4>).
    private static List<string> SplitTopLevel(string s)
    {
        var parts = new List<string>();
        int depth = 0, start = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '(' || c == '<' || c == '[') depth++;
            else if (c == ')' || c == '>' || c == ']') depth--;
            else if (c == ',' && depth == 0)
            {
                parts.Add(s.Substring(start, i - start));
                start = i + 1;
            }
        }
        parts.Add(s.Substring(start));
        return parts;
    }
}
#endif
