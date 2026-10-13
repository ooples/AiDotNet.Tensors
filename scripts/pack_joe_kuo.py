#!/usr/bin/env python3
"""Packs the Joe-Kuo Sobol direction numbers into the binary the library embeds.

Input:  new-joe-kuo-6.21201 from https://web.maths.unsw.edu.au/~fkuo/sobol/ (S. Joe and F. Y. Kuo, "Constructing
        Sobol sequences with better two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654, 2008). One line
        per dimension d >= 2: d, the degree s of the primitive polynomial, its coefficient word a, and m_1..m_s.
Output: src/AiDotNet.Tensors/Engines/DevicePrimitives/Cpu/Data/joe-kuo-6.bin

Format: "JK6\\0", then a little-endian uint32 count of dimensions stored (the dimensions 2 .. count + 1; dimension 1
is the van der Corput sequence and needs no data), then for each: one byte s, a as a LEB128 varint, and m_1..m_s as
LEB128 varints. SobolDirectionNumbers reads it.

    python scripts/pack_joe_kuo.py new-joe-kuo-6.21201 [dimensions]
"""
import struct
import sys


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def main():
    source = sys.argv[1]
    dimensions = int(sys.argv[2]) if len(sys.argv) > 2 else 20000   # cuRAND's limit
    target = 'src/AiDotNet.Tensors/Engines/DevicePrimitives/Cpu/Data/joe-kuo-6.bin'
    body = bytearray()
    stored = 0
    with open(source, encoding='ascii') as lines:
        next(lines)                                   # the header line
        for line in lines:
            fields = line.split()
            if not fields:
                continue
            d, s, a = int(fields[0]), int(fields[1]), int(fields[2])
            if d > dimensions:
                break
            m = [int(x) for x in fields[3:]]
            assert d == stored + 2, (d, stored)
            assert len(m) == s and 1 <= s <= 32, (d, s, m)
            for i, value in enumerate(m, start=1):
                assert value & 1 and value < (1 << i), (d, i, value)
            body.append(s)
            body += varint(a)
            for value in m:
                body += varint(value)
            stored += 1
    assert stored == dimensions - 1, (stored, dimensions)
    with open(target, 'wb') as out:
        out.write(b'JK6\x00')
        out.write(struct.pack('<I', stored))
        out.write(body)
    print(f'{target}: dimensions 1..{dimensions}, {8 + len(body)} bytes')


if __name__ == '__main__':
    main()
