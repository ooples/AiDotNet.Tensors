// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Serialization.Pickle;

/// <summary>
/// Subset of CPython pickle opcodes — only the ones the
/// <see cref="PickleInterpreter"/> needs to recover PyTorch tensors.
/// Full opcode set is in <c>cpython/Lib/pickle.py</c>; protocols 2/3
/// (PyTorch's defaults) are well-covered here.
/// </summary>
internal static class PickleOpcode
{
    public const byte MARK = (byte)'(';
    public const byte STOP = (byte)'.';
    public const byte POP = (byte)'0';
    public const byte POP_MARK = (byte)'1';
    public const byte DUP = (byte)'2';

    public const byte EMPTY_DICT = (byte)'}';
    public const byte EMPTY_LIST = (byte)']';
    public const byte EMPTY_TUPLE = (byte)')';
    public const byte EMPTY_SET = 0x8f;

    public const byte NONE = (byte)'N';
    public const byte NEWTRUE = 0x88;
    public const byte NEWFALSE = 0x89;

    public const byte BININT = (byte)'J';
    public const byte BININT1 = (byte)'K';
    public const byte BININT2 = (byte)'M';
    public const byte LONG1 = 0x8a;
    public const byte LONG4 = 0x8b;
    public const byte BINFLOAT = (byte)'G';

    public const byte SHORT_BINSTRING = (byte)'U';
    public const byte BINUNICODE = (byte)'X';
    public const byte SHORT_BINUNICODE = 0x8c;
    public const byte BINUNICODE8 = 0x8d;

    public const byte APPEND = (byte)'a';
    public const byte APPENDS = (byte)'e';
    public const byte SETITEM = (byte)'s';
    public const byte SETITEMS = (byte)'u';

    public const byte TUPLE = (byte)'t';
    public const byte TUPLE1 = 0x85;
    public const byte TUPLE2 = 0x86;
    public const byte TUPLE3 = 0x87;

    public const byte LIST = (byte)'l';
    public const byte DICT = (byte)'d';

    public const byte BINGET = (byte)'h';
    public const byte LONG_BINGET = (byte)'j';
    public const byte BINPUT = (byte)'q';
    public const byte LONG_BINPUT = (byte)'r';
    public const byte MEMOIZE = 0x94;

    public const byte GLOBAL = (byte)'c';
    public const byte STACK_GLOBAL = 0x93;
    public const byte REDUCE = (byte)'R';
    public const byte BUILD = (byte)'b';
    public const byte INST = (byte)'i';
    public const byte OBJ = (byte)'o';
    public const byte NEWOBJ = 0x81;
    public const byte NEWOBJ_EX = 0x92;

    public const byte PROTO = 0x80;
    public const byte FRAME = 0x95;
    public const byte PERSID = (byte)'P';
    public const byte BINPERSID = (byte)'Q';
}
