// Copyright (c) AiDotNet. All rights reserved.
// One global->shared tile copy, lowered for the target: cp.async on sm_80+, and on Turing (sm_75), which has no
// cp.async, a register round trip.
//
// The tiled emitters were written around cp.async and refused anything below sm_80, which left the GTX 1660 Ti with
// only the untiled affine kernels -- 0.54x cuDNN on dense 3x3 and 0.35x on its data gradient. What those emitters buy
// is the TILING (operand reuse through shared memory, register blocking), and that does not depend on the copy being
// asynchronous. The double-buffered schedule stays correct with synchronous copies: each copy completes before the
// barrier that publishes it, so the only thing lost on Turing is the overlap of the next copy with this compute.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

internal static class PtxSharedCopy
{
    /// <summary>Whether the target has cp.async (Ampere, sm_80, and later).</summary>
    internal static bool HasAsyncCopy(int computeMajor) => computeMajor >= 8;

    /// <summary>
    /// Emits one copy of <paramref name="bytes"/> (4, 8 or 16) from global <paramref name="source"/> to shared
    /// <paramref name="destination"/> (bracket contents, e.g. <c>%rd5+128</c>), under <paramref name="valid"/>.
    /// With <paramref name="zeroFill"/> set the destination receives zeros instead of the source, as cp.async's
    /// ignore-source operand does. On sm_80+ the emitted text is exactly the cp.async the emitters always produced,
    /// so Ampere PTX -- and every cubin and autotune identity keyed on it -- is unchanged.
    /// </summary>
    internal static void Emit(
        bool async, Action<string> line, Func<string> nextF, Func<string> nextP,
        string destination, string source, int bytes, string valid, string? zeroFill = null)
    {
        if (async)
        {
            line(zeroFill is null
                ? $"@{valid} cp.async.ca.shared.global [{destination}], [{source}], {bytes};"
                : $"@{valid} cp.async.ca.shared.global [{destination}], [{source}], {bytes}, {zeroFill};");
            return;
        }

        int words = bytes / 4;
        if (words != 1 && words != 2 && words != 4)
            throw new ArgumentOutOfRangeException(nameof(bytes), "A shared copy moves 4, 8 or 16 bytes.");
        var value = new string[words];
        for (int w = 0; w < words; w++) value[w] = nextF();
        string read = valid;
        if (zeroFill is not null)
        {
            // Zeros unless the source is both valid and not ignored.
            foreach (string v in value) line($"mov.f32 {v}, 0f00000000;");
            string keep = nextP(), load = nextP();
            line($"not.pred {keep}, {zeroFill};");
            line($"and.pred {load}, {valid}, {keep};");
            read = load;
        }
        string type = words == 1 ? ".f32" : ".v" + words + ".f32";
        string operands = words == 1 ? value[0] : "{" + string.Join(", ", value) + "}";
        line($"@{read} ld.global{type} {operands}, [{source}];");
        line($"@{valid} st.shared{type} [{destination}], {operands};");
    }

    /// <summary>Emits a cp.async group boundary, which a synchronous copy does not need.</summary>
    internal static void Commit(bool async, Action<string> line)
    {
        if (async) line("cp.async.commit_group;");
    }

    /// <summary>Emits a cp.async wait, which a synchronous copy does not need.</summary>
    internal static void Wait(bool async, Action<string> line)
    {
        if (async) line("cp.async.wait_group 0;");
    }
}
