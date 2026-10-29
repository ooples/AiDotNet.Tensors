// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Register renaming for the own SASS backend (#1442): rewrites every vector register operand of a kernel through one
// map, the primitive a register allocator's assignment is applied with. Renaming is only sound when the opcode table
// names every register field of every instruction -- --sass-learn checks exactly that against nvdisasm (each printed
// register operand decoded, and no decoded field that nvdisasm does not print) -- so a kernel with an unknown opcode is
// refused. A multi-register operand (a 64-bit address, a 128-bit load) names its first register and occupies the next
// ones implicitly: the map must carry the whole group to an equally aligned consecutive group, or the rename is refused.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>Applies a vector register renaming to a kernel's instructions.</summary>
public static class SassRegisterRenamer
{
    /// <summary>Why the last Rename call returned null (diagnostics).</summary>
    [ThreadStatic] public static string? LastDecline;

    /// <summary>
    /// <paramref name="code"/> with every vector register r (other than RZ) replaced by <paramref name="map"/>[r], or
    /// null when an instruction's opcode is unknown, a register lies outside the map, or the map splits or misaligns a
    /// multi-register operand.
    /// </summary>
    public static SassInstruction[]? Rename(IReadOnlyList<SassInstruction> code, IReadOnlyList<int> map)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        if (map is null) throw new ArgumentNullException(nameof(map));
        var result = new SassInstruction[code.Count];
        for (int i = 0; i < code.Count; i++)
        {
            var instruction = code[i];
            if (SassOpcodeTable.Decode(instruction) is not { } effects)
            {
                LastDecline = "instruction " + i + " has unknown opcode 0x" + instruction.Opcode.ToString("x3");
                return null;
            }
            // Each field once: a field that is both read and written (none today) must not be renamed twice.
            var renamed = new HashSet<int>();
            foreach (var operand in Operands(effects))
            {
                if (operand.Uniform || !renamed.Add(operand.Field)) continue;
                int r = code[i].Register(operand.Field);
                if (r == SassInstruction.ZeroRegister) continue;
                if (r + operand.Width > map.Count)
                {
                    LastDecline = "instruction " + i + " names R" + r + ", outside the map";
                    return null;
                }
                int target = map[r];
                if (target % operand.Width != 0 || target + operand.Width > SassInstruction.ZeroRegister)
                {
                    LastDecline = "instruction " + i + ": R" + r + " (width " + operand.Width + ") maps to misaligned R" + target;
                    return null;
                }
                for (int k = 1; k < operand.Width; k++)
                    if (map[r + k] != target + k)
                    {
                        LastDecline = "instruction " + i + ": the map splits R" + r + "..R" + (r + operand.Width - 1);
                        return null;
                    }
                instruction.SetRegister(operand.Field, target);
            }
            result[i] = instruction;
        }
        return result;
    }

    private static IEnumerable<SassRegisterOperand> Operands(SassEffects effects)
    {
        foreach (var operand in effects.Writes) yield return operand;
        foreach (var operand in effects.Reads) yield return operand;
    }
}
