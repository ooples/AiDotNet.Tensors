// Copyright (c) AiDotNet. All rights reserved.

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AiDotNet.Tensors.Serialization.Pickle;

/// <summary>
/// Minimal CPython pickle protocol 2/3 interpreter. Handles enough
/// opcodes to walk a PyTorch <c>data.pkl</c> stream and recover a
/// state_dict (typically a <c>dict</c> of <c>str</c> →
/// <c>torch.Tensor</c>).
/// </summary>
/// <remarks>
/// <para><b>Why we don't pull in a full pickle library:</b></para>
/// <para>
/// The .NET ecosystem doesn't ship one — third-party packages exist
/// but bring transitive deps and pin to specific protocol versions.
/// PyTorch checkpoints use a small-and-stable subset of the protocol;
/// hand-implementing it lets us guarantee no Python runtime is
/// required and keeps the dependency surface zero.
/// </para>
/// <para><b>Security:</b> the interpreter does NOT execute arbitrary
/// constructors. <see cref="PickleOpcode.REDUCE"/> /
/// <see cref="PickleOpcode.NEWOBJ"/> opcodes route through
/// <see cref="ReduceFunction"/> — a caller-provided dispatch keyed on
/// the global's qualified name. The default dispatch only knows the
/// PyTorch tensor-rebuild functions; everything else throws. This is
/// in line with HuggingFace's <c>transformers</c> safety stance —
/// pickle is dangerous in general but a tightly-scoped interpreter
/// covering only known-safe symbols is the best we can do for ecosystem
/// compatibility.</para>
/// </remarks>
internal sealed class PickleInterpreter
{
    private readonly Stream _stream;
    private readonly Stack<object?> _stack = new();
    private readonly List<object?> _memo = new();
    // Sentinel marker. Used by MARK / TUPLE / LIST / DICT to find
    // where a sub-collection started.
    private static readonly object MarkObject = new();

    /// <summary>
    /// Looks up a (module, name) pair from a <c>GLOBAL</c> or
    /// <c>STACK_GLOBAL</c> opcode and returns either a class
    /// representation (for use later in REDUCE) or null to skip.
    /// </summary>
    public Func<string, string, object?> ResolveGlobal { get; set; } = (_, _) => null;

    /// <summary>
    /// Resolves a persistent ID from <c>BINPERSID</c> back to the
    /// referenced object — PyTorch uses persistent IDs for tensor
    /// storages.
    /// </summary>
    public Func<object, object?> ResolvePersistentId { get; set; } = _ => null;

    /// <summary>
    /// Invoked for <c>REDUCE</c> opcodes — caller decides what
    /// callable+args means. PyTorch's <c>_rebuild_tensor_v2</c>,
    /// <c>_rebuild_qtensor</c>, <c>OrderedDict</c> all flow here.
    /// </summary>
    public Func<object?, object?, object?> ReduceFunction { get; set; } = (_, _) => null;

    /// <summary>Constructs an interpreter reading from <paramref name="stream"/>.</summary>
    public PickleInterpreter(Stream stream)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    }

    /// <summary>
    /// Runs the interpreter to completion, returning the value that
    /// was on the stack when <see cref="PickleOpcode.STOP"/> was
    /// encountered.
    /// </summary>
    public object? Load()
    {
        while (true)
        {
            int op = _stream.ReadByte();
            if (op < 0) throw new InvalidDataException("Pickle stream ended without STOP opcode.");
            byte opcode = (byte)op;
            switch (opcode)
            {
                case PickleOpcode.PROTO: ReadByteChecked(); break;             // protocol number — informational
                case PickleOpcode.FRAME: ReadBytes(8); break;                  // frame length — informational
                case PickleOpcode.STOP: return _stack.Count > 0 ? _stack.Pop() : null;

                case PickleOpcode.MARK: _stack.Push(MarkObject); break;
                case PickleOpcode.POP: _stack.Pop(); break;
                case PickleOpcode.POP_MARK: PopUntilMark(); break;
                case PickleOpcode.DUP: _stack.Push(_stack.Peek()); break;

                case PickleOpcode.NONE: _stack.Push(null); break;
                case PickleOpcode.NEWTRUE: _stack.Push(true); break;
                case PickleOpcode.NEWFALSE: _stack.Push(false); break;

                case PickleOpcode.EMPTY_DICT: _stack.Push(new Dictionary<object, object?>()); break;
                case PickleOpcode.EMPTY_LIST: _stack.Push(new List<object?>()); break;
                case PickleOpcode.EMPTY_TUPLE: _stack.Push(new object?[0]); break;
                case PickleOpcode.EMPTY_SET: _stack.Push(new HashSet<object>()); break;

                case PickleOpcode.BININT: _stack.Push((long)System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(ReadBytes(4))); break;
                case PickleOpcode.BININT1: _stack.Push((long)ReadByteChecked()); break;
                case PickleOpcode.BININT2: _stack.Push((long)System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(ReadBytes(2))); break;
                case PickleOpcode.LONG1:
                {
                    int len = ReadByteChecked();
                    _stack.Push(ReadLong(len));
                    break;
                }
                case PickleOpcode.LONG4:
                {
                    int len = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(ReadBytes(4));
                    _stack.Push(ReadLong(len));
                    break;
                }
                case PickleOpcode.BINFLOAT:
                {
                    // Big-endian double per pickle spec.
                    var b = ReadBytes(8);
                    Array.Reverse(b);
                    _stack.Push(BitConverter.ToDouble(b, 0));
                    break;
                }

                case PickleOpcode.SHORT_BINSTRING:
                {
                    int len = ReadByteChecked();
                    _stack.Push(Encoding.ASCII.GetString(ReadBytes(len)));
                    break;
                }
                case PickleOpcode.BINUNICODE:
                {
                    uint ulen = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4));
                    if (ulen > int.MaxValue)
                        throw new InvalidDataException(
                            $"BINUNICODE length {ulen} exceeds int.MaxValue.");
                    _stack.Push(Encoding.UTF8.GetString(ReadBytes((int)ulen)));
                    break;
                }
                case PickleOpcode.SHORT_BINUNICODE:
                {
                    int len = ReadByteChecked();
                    _stack.Push(Encoding.UTF8.GetString(ReadBytes(len)));
                    break;
                }
                case PickleOpcode.BINUNICODE8:
                {
                    ulong ulen = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(ReadBytes(8));
                    // Reject lengths that would overflow int — both
                    // because allocations are int-sized and because a
                    // forged 64-bit length would otherwise wrap and
                    // succeed with the wrong data.
                    if (ulen > int.MaxValue)
                        throw new InvalidDataException(
                            $"BINUNICODE8 length {ulen} exceeds int.MaxValue; refusing to allocate.");
                    _stack.Push(Encoding.UTF8.GetString(ReadBytes((int)ulen)));
                    break;
                }

                case PickleOpcode.APPEND:
                {
                    var v = _stack.Pop();
                    var list = _stack.Peek() as List<object?>
                        ?? throw new InvalidDataException("APPEND target is not a list.");
                    list.Add(v);
                    break;
                }
                case PickleOpcode.APPENDS:
                {
                    var items = PopUntilMark();
                    var list = _stack.Peek() as List<object?>
                        ?? throw new InvalidDataException("APPENDS target is not a list.");
                    foreach (var i in items) list.Add(i);
                    break;
                }
                case PickleOpcode.SETITEM:
                {
                    var v = _stack.Pop();
                    var k = _stack.Pop();
                    var dict = _stack.Peek() as IDictionary
                        ?? throw new InvalidDataException("SETITEM target is not a dict.");
                    dict[k!] = v;
                    break;
                }
                case PickleOpcode.SETITEMS:
                {
                    var items = PopUntilMark();
                    var dict = _stack.Peek() as IDictionary
                        ?? throw new InvalidDataException("SETITEMS target is not a dict.");
                    for (int i = 0; i < items.Count; i += 2)
                        dict[items[i]!] = items[i + 1];
                    break;
                }

                case PickleOpcode.TUPLE:
                {
                    var items = PopUntilMark();
                    _stack.Push(items.ToArray());
                    break;
                }
                case PickleOpcode.TUPLE1:
                {
                    var a = _stack.Pop();
                    _stack.Push(new[] { a });
                    break;
                }
                case PickleOpcode.TUPLE2:
                {
                    var b = _stack.Pop();
                    var a = _stack.Pop();
                    _stack.Push(new[] { a, b });
                    break;
                }
                case PickleOpcode.TUPLE3:
                {
                    var c = _stack.Pop();
                    var b = _stack.Pop();
                    var a = _stack.Pop();
                    _stack.Push(new[] { a, b, c });
                    break;
                }

                case PickleOpcode.BINGET: _stack.Push(_memo[ReadByteChecked()]); break;
                case PickleOpcode.LONG_BINGET: _stack.Push(_memo[(int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4))]); break;
                case PickleOpcode.BINPUT:
                {
                    int idx = ReadByteChecked();
                    while (_memo.Count <= idx) _memo.Add(null);
                    _memo[idx] = _stack.Peek();
                    break;
                }
                case PickleOpcode.LONG_BINPUT:
                {
                    int idx = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4));
                    while (_memo.Count <= idx) _memo.Add(null);
                    _memo[idx] = _stack.Peek();
                    break;
                }
                case PickleOpcode.MEMOIZE:
                    _memo.Add(_stack.Peek());
                    break;

                case PickleOpcode.GLOBAL:
                {
                    string module = ReadLine();
                    string name = ReadLine();
                    _stack.Push(ResolveGlobal(module, name));
                    break;
                }
                case PickleOpcode.STACK_GLOBAL:
                {
                    string name = (string)_stack.Pop()!;
                    string module = (string)_stack.Pop()!;
                    _stack.Push(ResolveGlobal(module, name));
                    break;
                }

                case PickleOpcode.REDUCE:
                {
                    var args = _stack.Pop();
                    var func = _stack.Pop();
                    _stack.Push(ReduceFunction(func, args));
                    break;
                }
                case PickleOpcode.NEWOBJ:
                {
                    var args = _stack.Pop();
                    var cls = _stack.Pop();
                    _stack.Push(ReduceFunction(cls, args));
                    break;
                }
                case PickleOpcode.NEWOBJ_EX:
                {
                    _stack.Pop();   // kwargs
                    var args = _stack.Pop();
                    var cls = _stack.Pop();
                    _stack.Push(ReduceFunction(cls, args));
                    break;
                }
                case PickleOpcode.BUILD:
                {
                    var state = _stack.Pop();
                    // BUILD merges `state` into the top-of-stack
                    // object. For the PyTorch state-dict cases we
                    // handle, the rebuilt tensor object already
                    // contains everything from REDUCE, so BUILD's
                    // state is ignored. (Fully implementing BUILD
                    // requires running __setstate__ on the object,
                    // which is opaque without execution.)
                    break;
                }
                case PickleOpcode.BINPERSID:
                {
                    var pid = _stack.Pop();
                    _stack.Push(ResolvePersistentId(pid!));
                    break;
                }

                default:
                    throw new InvalidDataException(
                        $"Unsupported pickle opcode 0x{opcode:X2} ('{(char)opcode}'). " +
                        $"PtReader covers protocol 2/3 PyTorch checkpoints; non-tensor pickles may include opcodes outside that subset.");
            }
        }
    }

    private byte[] ReadBytes(int n)
    {
        var buf = new byte[n];
        int off = 0;
        while (off < n)
        {
            int r = _stream.Read(buf, off, n - off);
            if (r == 0) throw new EndOfStreamException("Unexpected EOF in pickle stream.");
            off += r;
        }
        return buf;
    }

    private long ReadLong(int len)
    {
        // pickle.LONG1/LONG4 store little-endian two's-complement.
        if (len == 0) return 0;
        // We materialise into a long, so any width above 8 bytes can't
        // be represented faithfully. Worse, the shift below would
        // wrap because C# shift counts are mod 64 on long, so a
        // 9-byte input would shift by 72 = 8 mod 64 and silently
        // alias the wrong byte. Reject up front.
        if (len < 0 || len > 8)
            throw new InvalidDataException(
                $"Pickle LONG opcode width {len} cannot fit in a 64-bit integer; refusing to decode.");
        var b = ReadBytes(len);
        long v = 0;
        for (int i = 0; i < len; i++) v |= ((long)b[i]) << (8 * i);
        // Sign-extend if the high bit of the high byte is set.
        if ((b[len - 1] & 0x80) != 0 && len < 8)
            v |= -1L << (8 * len);
        return v;
    }

    /// <summary>
    /// Reads exactly one byte from the stream. Throws
    /// <see cref="EndOfStreamException"/> on EOF instead of silently
    /// returning <c>-1</c> the way the raw <see cref="Stream.ReadByte"/>
    /// does — opcode handlers that pass the result through to a
    /// <c>byte[]</c> indexer or a stack push otherwise propagate the
    /// invalid <c>-1</c> as data and produce non-deterministic decode
    /// errors much later.
    /// </summary>
    private byte ReadByteChecked()
    {
        int b = _stream.ReadByte();
        if (b < 0) throw new EndOfStreamException("Unexpected EOF in pickle stream — needed one more byte.");
        return (byte)b;
    }

    private string ReadLine()
    {
        var sb = new StringBuilder();
        while (true)
        {
            int c = _stream.ReadByte();
            if (c < 0 || c == '\n') break;
            sb.Append((char)c);
        }
        return sb.ToString();
    }

    private List<object?> PopUntilMark()
    {
        var items = new List<object?>();
        while (true)
        {
            var top = _stack.Pop();
            if (ReferenceEquals(top, MarkObject)) break;
            items.Insert(0, top);
        }
        return items;
    }
}
