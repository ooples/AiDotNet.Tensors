// Copyright (c) CrowdTrainer LLC. All rights reserved.
// A ptxas cubin opened for rewriting by the own SASS backend (#1442): an ELF64 image whose kernels' code lives in
// ".text.<entry>" sections of 16-byte instructions. A kernel's register count is recorded twice and both must agree
// with the code: in the high byte of the .text section header's sh_info (the low 24 bits are the kernel's symbol
// index), and as the EIATTR_REGCOUNT attribute in the ".nv.info" section. Rewriting keeps every section's size, so
// no offset in the image moves.

using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>A cubin image whose kernels' instructions and register counts can be rewritten in place.</summary>
public sealed class SassCubin
{
    private const int InstructionBytes = 16;
    private const byte EiattrRegCount = 0x2f;
    private const byte EifmtSval = 0x04;   // an attribute with a 16-bit size and a value payload

    private readonly byte[] _image;
    private readonly List<Section> _sections = new();

    private readonly record struct Section(string Name, uint Type, long InfoOffset, long Offset, long Size, uint Info);

    /// <summary>Opens a copy of <paramref name="image"/>; the input array is not modified.</summary>
    public SassCubin(byte[] image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Length < 64 || image[0] != 0x7f || image[1] != (byte)'E' || image[2] != (byte)'L' || image[3] != (byte)'F')
            throw new ArgumentException("Not an ELF image.", nameof(image));
        if (image[4] != 2 || image[5] != 1) throw new ArgumentException("Not a little-endian ELF64 image.", nameof(image));
        _image = (byte[])image.Clone();
        long sectionHeaders = (long)ReadU64(0x28);
        int entrySize = ReadU16(0x3a), count = ReadU16(0x3c), names = ReadU16(0x3e);
        if (sectionHeaders <= 0 || entrySize < 64 || sectionHeaders + (long)entrySize * count > _image.Length)
            throw new ArgumentException("The section header table lies outside the image.", nameof(image));
        if (names >= count) throw new ArgumentException("The section-name table index is out of range.", nameof(image));
        long nameTable = (long)ReadU64(sectionHeaders + (long)names * entrySize + 0x18);
        for (int i = 0; i < count; i++)
        {
            long header = sectionHeaders + (long)i * entrySize;
            long nameAt = nameTable + ReadU32(header);
            if (nameTable < 0 || nameAt >= _image.Length)
                throw new ArgumentException("Section " + i + "'s name lies outside the image.", nameof(image));
            string name = ReadString(nameAt);
            _sections.Add(new Section(name, ReadU32(header + 4), header + 0x2c, (long)ReadU64(header + 0x18),
                (long)ReadU64(header + 0x20), ReadU32(header + 0x2c)));
        }
    }

    /// <summary>The rewritten image.</summary>
    public byte[] ToArray() => (byte[])_image.Clone();

    /// <summary>The kernels (entry names) the image holds code for.</summary>
    public IReadOnlyList<string> Kernels
    {
        get
        {
            var names = new List<string>();
            foreach (var section in _sections)
                if (section.Name.StartsWith(".text.", StringComparison.Ordinal)) names.Add(section.Name.Substring(6));
            return names;
        }
    }

    /// <summary>The instructions of <paramref name="kernel"/>.</summary>
    public SassInstruction[] Instructions(string kernel)
    {
        var text = Text(kernel);
        var result = new SassInstruction[text.Size / InstructionBytes];
        for (int i = 0; i < result.Length; i++)
            result[i] = new SassInstruction(ReadU64(text.Offset + (long)i * InstructionBytes),
                ReadU64(text.Offset + (long)i * InstructionBytes + 8));
        return result;
    }

    /// <summary>Replaces <paramref name="kernel"/>'s instructions; the count must be unchanged.</summary>
    public void SetInstructions(string kernel, IReadOnlyList<SassInstruction> instructions)
    {
        if (instructions is null) throw new ArgumentNullException(nameof(instructions));
        var text = Text(kernel);
        if ((long)instructions.Count * InstructionBytes != text.Size)
            throw new ArgumentException(kernel + " holds " + text.Size / InstructionBytes + " instructions; " +
                instructions.Count + " were given.", nameof(instructions));
        for (int i = 0; i < instructions.Count; i++)
        {
            WriteU64(text.Offset + (long)i * InstructionBytes, instructions[i].Low);
            WriteU64(text.Offset + (long)i * InstructionBytes + 8, instructions[i].High);
        }
    }

    /// <summary>The register count recorded for <paramref name="kernel"/> (the .text section header's).</summary>
    public int RegisterCount(string kernel) => (int)(Text(kernel).Info >> 24);

    /// <summary>
    /// Records <paramref name="registers"/> as <paramref name="kernel"/>'s register count, in the .text section header
    /// and in the .nv.info EIATTR_REGCOUNT attribute.
    /// </summary>
    public void SetRegisterCount(string kernel, int registers)
    {
        if (registers is < 1 or > 255) throw new ArgumentOutOfRangeException(nameof(registers));
        int index = _sections.FindIndex(s => s.Name == ".text." + kernel);
        if (index < 0) throw new ArgumentException("No code for kernel " + kernel + ".", nameof(kernel));
        var text = _sections[index];
        uint symbol = text.Info & 0x00ffffffu;
        uint info = ((uint)registers << 24) | symbol;
        WriteU32(text.InfoOffset, info);
        _sections[index] = text with { Info = info };

        bool found = false;
        foreach (var section in _sections)
        {
            if (section.Name != ".nv.info") continue;
            if (section.Offset < 0 || section.Size < 0 || section.Offset + section.Size > _image.Length)
                throw new InvalidOperationException("Malformed .nv.info section.");
            // Attributes: format byte, attribute byte, then for EIFMT_SVAL a 16-bit size and the payload
            // (symbol index, value) for REGCOUNT.
            for (long at = section.Offset; at + 4 <= section.Offset + section.Size;)
            {
                byte format = _image[at], attribute = _image[at + 1];
                int size = format == EifmtSval ? ReadU16(at + 2) : 0;
                if (format == EifmtSval && attribute == EiattrRegCount && size >= 8 &&
                    at + 4 + size <= section.Offset + section.Size && ReadU32(at + 4) == symbol)
                {
                    WriteU32(at + 8, (uint)registers);
                    found = true;
                }
                at += format == EifmtSval ? 4 + size : 4;
            }
        }
        if (!found)
            throw new InvalidOperationException("The image records no EIATTR_REGCOUNT for kernel " + kernel + ".");
    }

    private Section Text(string kernel)
    {
        foreach (var section in _sections)
            if (section.Name == ".text." + kernel)
            {
                if (section.Size % InstructionBytes != 0 || section.Offset < 0 || section.Size < 0 ||
                    section.Offset + section.Size > _image.Length)
                    throw new InvalidOperationException("Malformed code section for " + kernel + ".");
                return section;
            }
        throw new ArgumentException("No code for kernel " + kernel + ".", nameof(kernel));
    }

    private ushort ReadU16(long at) => (ushort)(_image[at] | _image[at + 1] << 8);

    private uint ReadU32(long at) => (uint)(_image[at] | _image[at + 1] << 8 | _image[at + 2] << 16 | _image[at + 3] << 24);

    private ulong ReadU64(long at) => ReadU32(at) | (ulong)ReadU32(at + 4) << 32;

    private void WriteU32(long at, uint value)
    {
        for (int b = 0; b < 4; b++) _image[at + b] = (byte)(value >> (8 * b));
    }

    private void WriteU64(long at, ulong value)
    {
        for (int b = 0; b < 8; b++) _image[at + b] = (byte)(value >> (8 * b));
    }

    private string ReadString(long at)
    {
        long end = at;
        while (end < _image.Length && _image[end] != 0) end++;
        return Encoding.ASCII.GetString(_image, (int)at, (int)(end - at));
    }
}
