using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>A program file is malformed, from a newer format, or names something this runtime will not load.</summary>
public sealed class ProgramFormatException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ProgramFormatException(string message) : base(message) { }
}

/// <summary>
/// The program wire format. Little-endian, deterministic, and closed: the reader accepts only catalog op ids, the
/// element types in <see cref="ElementTypes"/>, and a fixed set of literal tags, so nothing in a file can choose a type
/// or method to load. Every length is checked against the bytes remaining before anything is allocated.
/// </summary>
internal static class ProgramSerializer
{
    internal const uint Magic = 0x504E5441; // "ATNP"
    internal const ushort Version = 1;

    internal const byte SectionProgram = 1;
    internal const byte SectionTensors = 2;
    internal const byte SectionCompiled = 3;

    private const byte LitNull = 0, LitBool = 1, LitInt = 2, LitLong = 3, LitFloat = 4, LitDouble = 5, LitString = 6,
        LitEnum = 7, LitArray = 8, LitObject = 9;

    // ── Program section ────────────────────────────────────────────────────────────────────────────────────

    internal static byte[] WriteProgram(TensorProgram p)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write(p.TracedElementType);
        w.Write(p.OpSetHash);
        w.Write(p.Values.Count);
        foreach (var v in p.Values)
        {
            w.Write((byte)v.Role);
            w.Write(v.ElementType);
            w.Write(v.Container);
            WriteInts(w, v.Shape);
            w.Write(v.Name ?? "");
        }
        WriteInts(w, p.Inputs.ToArray());
        WriteInts(w, p.Outputs.ToArray());
        w.Write(p.Symbols.Count);
        foreach (var s in p.Symbols) w.Write(s);
        w.Write(p.InputDimensions.Count);
        foreach (var dims in p.InputDimensions)
        {
            w.Write(dims.Count);
            foreach (var d in dims) WriteSym(w, d);
        }
        w.Write(p.Nodes.Count);
        foreach (var n in p.Nodes)
        {
            w.Write(n.Op.Id);
            w.Write(n.TypeArguments.Count);
            foreach (var t in n.TypeArguments) w.Write(t);
            w.Write(n.Args.Count);
            foreach (var a in n.Args)
            {
                w.Write(a.Name);
                w.Write((byte)a.Kind);
                switch (a.Kind)
                {
                    case ProgramArgKind.Value or ProgramArgKind.ValueList: WriteInts(w, a.Values.ToArray()); break;
                    case ProgramArgKind.Literal: WriteLiteral(w, a.Literal); break;
                    case ProgramArgKind.Symbolic:
                        w.Write(a.SymbolicIsArray);
                        w.Write(a.Symbolic!.Count);
                        foreach (var s in a.Symbolic) WriteSym(w, s);
                        break;
                }
            }
            WriteInts(w, n.Outputs.ToArray());
            w.Write(n.InPlaceInput);
        }
        w.Write(p.Diagnostics.Count);
        foreach (var d in p.Diagnostics) w.Write(d);
        w.Flush();
        return ms.ToArray();
    }

    internal static TensorProgram ReadProgram(byte[] data, Dictionary<int, object> stored)
    {
        var r = new Reader(data);
        string elementType = r.String();
        if (ElementTypes.TypeOf(elementType) is null) throw new ProgramFormatException($"Unknown element type '{elementType}'.");
        string opSetHash = r.String();
        int valueCount = r.Count(4);
        var values = new List<ProgramValue>(valueCount);
        for (int i = 0; i < valueCount; i++)
        {
            var role = (ProgramValueRole)r.Byte();
            if (role > ProgramValueRole.Intermediate) throw new ProgramFormatException($"Value {i} has an unknown role.");
            string et = r.String();
            if (et != "$T" && ElementTypes.TypeOf(et) is null) throw new ProgramFormatException($"Value {i} has unknown element type '{et}'.");
            string container = r.String();
            if (container is not ("Tensor" or "Vector" or "Matrix")) throw new ProgramFormatException($"Value {i} has unknown container '{container}'.");
            var shape = r.Ints();
            var v = new ProgramValue(i, role, et, shape, container);
            var name = r.String();
            if (name.Length > 0) v.Name = name;
            values.Add(v);
        }
        int Id(int id) => id >= 0 && id < valueCount ? id : throw new ProgramFormatException($"Value id {id} is out of range.");

        var inputs = r.Ints().Select(Id).ToArray();
        var outputs = r.Ints().Select(Id).ToArray();
        int symbolCount = r.Count(1);
        var symbols = new string[symbolCount];
        for (int i = 0; i < symbolCount; i++) symbols[i] = r.String();
        int dimInputs = r.Count(4);
        if (dimInputs != inputs.Length) throw new ProgramFormatException("Input dimension table does not match the inputs.");
        var dims = new SymbolicInt[dimInputs][];
        for (int i = 0; i < dimInputs; i++)
        {
            int rank = r.Count(1);
            dims[i] = new SymbolicInt[rank];
            for (int d = 0; d < rank; d++) dims[i][d] = r.Sym(symbolCount);
        }

        int nodeCount = r.Count(8);
        var nodes = new List<ProgramNode>(nodeCount);
        for (int n = 0; n < nodeCount; n++)
        {
            string opId = r.String();
            var op = EngineOpCatalog.Find(opId)
                     ?? throw new ProgramFormatException($"The program uses engine op '{opId}', which this version of AiDotNet.Tensors does not have. Update the package.");
            if (!op.IsTraceable) throw new ProgramFormatException($"Engine op '{opId}' cannot be replayed: {op.UntraceableReason}.");
            int typeArgCount = r.Count(1);
            if (typeArgCount != op.GenericArity) throw new ProgramFormatException($"Node {n} ({op.Name}) has {typeArgCount} type arguments; the op takes {op.GenericArity}.");
            var typeArgs = new string[typeArgCount];
            for (int i = 0; i < typeArgCount; i++)
            {
                typeArgs[i] = r.String();
                if (typeArgs[i] != "$T" && ElementTypes.TypeOf(typeArgs[i]) is null)
                    throw new ProgramFormatException($"Node {n} ({op.Name}) names unknown element type '{typeArgs[i]}'.");
            }
            int argCount = r.Count(2);
            if (argCount != op.Parameters.Count) throw new ProgramFormatException($"Node {n} ({op.Name}) has {argCount} arguments; the op takes {op.Parameters.Count}.");
            var args = new ProgramArg[argCount];
            for (int i = 0; i < argCount; i++)
            {
                string name = r.String();
                if (name != op.Parameters[i].Name) throw new ProgramFormatException($"Node {n} ({op.Name}) argument {i} is '{name}'; the op's parameter is '{op.Parameters[i].Name}'.");
                var kind = (ProgramArgKind)r.Byte();
                args[i] = kind switch
                {
                    ProgramArgKind.Value => new ProgramArg(name, kind, OneValue(r.Ints().Select(Id).ToArray(), n)),
                    ProgramArgKind.ValueList => new ProgramArg(name, kind, r.Ints().Select(Id).ToArray()),
                    ProgramArgKind.Literal => new ProgramArg(name, kind, literal: r.Literal(0)),
                    ProgramArgKind.Symbolic => ReadSymbolicArg(r, name, symbolCount),
                    ProgramArgKind.Null or ProgramArgKind.Out => new ProgramArg(name, kind),
                    _ => throw new ProgramFormatException($"Node {n} ({op.Name}) argument '{name}' has an unknown kind."),
                };
            }
            var outs = r.Ints().Select(Id).ToArray();
            int inPlace = r.Int();
            if (inPlace != -1) Id(inPlace);
            nodes.Add(new ProgramNode(op, typeArgs, args, outs, inPlace));
        }
        int diagCount = r.Count(1);
        var diagnostics = new List<string>(diagCount);
        for (int i = 0; i < diagCount; i++) diagnostics.Add(r.String());

        foreach (var v in values.Where(v => v.Role is ProgramValueRole.Constant or ProgramValueRole.Parameter))
            if (!stored.ContainsKey(v.Id)) throw new ProgramFormatException($"Value %{v.Id} ({v.Role}) has no stored tensor.");

        return new TensorProgram(elementType, values, nodes, inputs, outputs, stored, dims, symbols, diagnostics) { OpSetHash = opSetHash };
    }

    private static int[] OneValue(int[] ids, int node) =>
        ids.Length == 1 ? ids : throw new ProgramFormatException($"Node {node} has a single-value argument with {ids.Length} values.");

    private static ProgramArg ReadSymbolicArg(Reader r, string name, int symbolCount)
    {
        bool isArray = r.Bool();
        int count = r.Count(9);
        var syms = new SymbolicInt[count];
        for (int i = 0; i < count; i++) syms[i] = r.Sym(symbolCount);
        return new ProgramArg(name, ProgramArgKind.Symbolic, symbolic: syms, symbolicIsArray: isArray);
    }

    private static void WriteSym(BinaryWriter w, SymbolicInt s)
    {
        w.Write(s.Symbol);
        w.Write(s.Scale);
        w.Write(s.Offset);
    }

    private static void WriteInts(BinaryWriter w, int[] values)
    {
        w.Write(values.Length);
        foreach (var v in values) w.Write(v);
    }

    private static void WriteLiteral(BinaryWriter w, object? v)
    {
        switch (v)
        {
            case null: w.Write(LitNull); break;
            case bool b: w.Write(LitBool); w.Write(b); break;
            case int i: w.Write(LitInt); w.Write(i); break;
            case long l: w.Write(LitLong); w.Write(l); break;
            case float f: w.Write(LitFloat); w.Write(f); break;
            case double d: w.Write(LitDouble); w.Write(d); break;
            case string s: w.Write(LitString); w.Write(s); break;
            case EnumLiteral e: w.Write(LitEnum); w.Write(e.Member); break;
            case Array arr:
                w.Write(LitArray);
                w.Write(ElementTag(arr.GetType().GetElementType()!));
                w.Write(arr.Length);
                foreach (var item in arr) WriteLiteral(w, item);
                break;
            default: throw new NotSupportedException($"A literal of type {v.GetType().Name} cannot be saved.");
        }
    }

    private static byte ElementTag(Type t) =>
        t == typeof(bool) ? LitBool : t == typeof(int) ? LitInt : t == typeof(long) ? LitLong : t == typeof(float) ? LitFloat
        : t == typeof(double) ? LitDouble : t == typeof(string) ? LitString : t.IsArray ? LitArray : LitObject;

    private static Type TagType(byte tag) => tag switch
    {
        LitBool => typeof(bool), LitInt => typeof(int), LitLong => typeof(long), LitFloat => typeof(float),
        LitDouble => typeof(double), LitString => typeof(string), LitArray or LitObject => typeof(object),
        _ => throw new ProgramFormatException($"Unknown array element tag {tag}."),
    };

    // ── Tensor section ─────────────────────────────────────────────────────────────────────────────────────

    internal static byte[] WriteTensors(IReadOnlyDictionary<int, object> stored)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write(stored.Count);
        foreach (var kv in stored.OrderBy(k => k.Key))
        {
            var element = kv.Value.GetType().GetGenericArguments()[0];
            var key = ElementTypes.KeyOf(element) ?? throw new NotSupportedException($"Element type {element.Name} cannot be saved.");
            w.Write(kv.Key);
            w.Write(key);
            var (bytes, shape) = ((byte[], int[]))Generic(nameof(TensorBytes), element).Invoke(null, new[] { kv.Value })!;
            WriteInts(w, shape);
            w.Write(bytes.Length);
            w.Write(bytes);
        }
        w.Flush();
        return ms.ToArray();
    }

    internal static Dictionary<int, object> ReadTensors(byte[] data)
    {
        var r = new Reader(data);
        int count = r.Count(12);
        var result = new Dictionary<int, object>(count);
        for (int i = 0; i < count; i++)
        {
            int id = r.Int();
            string key = r.String();
            var element = ElementTypes.TypeOf(key) ?? throw new ProgramFormatException($"Stored tensor %{id} has unknown element type '{key}'.");
            var shape = r.Ints();
            long elements = 1;
            foreach (var d in shape)
            {
                if (d < 0) throw new ProgramFormatException($"Stored tensor %{id} has a negative dimension.");
                elements = checked(elements * d);
            }
            int length = r.Count(1);
            var bytes = r.Bytes(length);
            if (elements * StoredSize(element) != length) throw new ProgramFormatException($"Stored tensor %{id} has {length} bytes for {elements} elements.");
            result[id] = Generic(nameof(TensorFromBytes), element).Invoke(null, new object[] { bytes, shape })!;
        }
        return result;
    }

    // Half is stored as float32 everywhere so files are identical across target frameworks (the .NET Framework Half
    // shim is float-backed).
    private static int StoredSize(Type t) => t == typeof(Half) ? 4 : t == typeof(bool) ? 1 : Marshal.SizeOf(t);

    private static (byte[], int[]) TensorBytes<T>(Tensor<T> t)
    {
        var span = t.Contiguous().AsSpan();
        if (typeof(T) == typeof(Half))
        {
            var floats = new float[span.Length];
            var halves = (Half[])(object)span.ToArray();
            for (int i = 0; i < floats.Length; i++) floats[i] = (float)halves[i];
            return (MemoryMarshal.AsBytes(floats.AsSpan()).ToArray(), t._shape.ToArray());
        }
        return (BytesOf(span), t._shape.ToArray());
    }

    private static byte[] BytesOf<T>(ReadOnlySpan<T> span) =>
        (byte[])Generic(nameof(BytesOfStruct), typeof(T)).Invoke(null, new object[] { span.ToArray() })!;

    private static byte[] BytesOfStruct<T>(T[] data) where T : struct => MemoryMarshal.AsBytes(data.AsSpan()).ToArray();

    private static T[] StructsOf<T>(byte[] bytes) where T : struct => MemoryMarshal.Cast<byte, T>(bytes.AsSpan()).ToArray();

    private static Tensor<T> TensorFromBytes<T>(byte[] bytes, int[] shape)
    {
        if (typeof(T) == typeof(Half))
        {
            var floats = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            var halves = new Half[floats.Length];
            for (int i = 0; i < halves.Length; i++) halves[i] = (Half)floats[i];
            return new Tensor<T>((T[])(object)halves, shape);
        }
        var data = (T[])Generic(nameof(StructsOf), typeof(T)).Invoke(null, new object[] { bytes })!;
        return new Tensor<T>(data, shape);
    }

    private static MethodInfo Generic(string name, Type t) =>
        typeof(ProgramSerializer).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!.MakeGenericMethod(t);

    // ── Bounded reader ─────────────────────────────────────────────────────────────────────────────────────

    internal sealed class Reader
    {
        private readonly byte[] _data;
        private int _pos;
        private const int MaxDepth = 4;

        internal Reader(byte[] data) => _data = data;

        private void Need(long n)
        {
            if (n < 0 || _pos + n > _data.Length) throw new ProgramFormatException("The program file is truncated or a length is corrupt.");
        }

        internal byte Byte() { Need(1); return _data[_pos++]; }
        internal bool Bool() => Byte() != 0;
        internal int Int() { Need(4); int v = BitConverter.ToInt32(_data, _pos); _pos += 4; return v; }
        internal long Long() { Need(8); long v = BitConverter.ToInt64(_data, _pos); _pos += 8; return v; }
        internal float Float() { Need(4); float v = BitConverter.ToSingle(_data, _pos); _pos += 4; return v; }
        internal double Double() { Need(8); double v = BitConverter.ToDouble(_data, _pos); _pos += 8; return v; }

        /// <summary>A count whose items take at least <paramref name="minItemBytes"/> each, checked against what is left.</summary>
        internal int Count(int minItemBytes)
        {
            int n = Int();
            if (n < 0 || (long)n * minItemBytes > _data.Length - _pos) throw new ProgramFormatException("A count in the program file is corrupt.");
            return n;
        }

        internal byte[] Bytes(int n) { Need(n); var b = new byte[n]; Buffer.BlockCopy(_data, _pos, b, 0, n); _pos += n; return b; }

        internal string String()
        {
            // BinaryWriter's 7-bit encoded length prefix.
            int len = 0, shift = 0;
            byte b;
            do
            {
                if (shift > 28) throw new ProgramFormatException("A string length in the program file is corrupt.");
                b = Byte();
                len |= (b & 0x7F) << shift;
                shift += 7;
            } while ((b & 0x80) != 0);
            Need(len);
            var s = Encoding.UTF8.GetString(_data, _pos, len);
            _pos += len;
            return s;
        }

        internal int[] Ints()
        {
            int n = Count(4);
            var a = new int[n];
            for (int i = 0; i < n; i++) a[i] = Int();
            return a;
        }

        internal SymbolicInt Sym(int symbolCount)
        {
            int s = Int();
            if (s < -1 || s >= symbolCount) throw new ProgramFormatException($"Symbol {s} is out of range.");
            return new SymbolicInt(s, Long(), Long());
        }

        internal object? Literal(int depth)
        {
            if (depth > MaxDepth) throw new ProgramFormatException("Literals are nested too deeply.");
            byte tag = Byte();
            switch (tag)
            {
                case LitNull: return null;
                case LitBool: return Bool();
                case LitInt: return Int();
                case LitLong: return Long();
                case LitFloat: return Float();
                case LitDouble: return Double();
                case LitString: return String();
                case LitEnum: return new EnumLiteral(String());
                case LitArray:
                    var elemType = TagType(Byte());
                    int n = Count(1);
                    var arr = Array.CreateInstance(elemType, n);
                    for (int i = 0; i < n; i++)
                    {
                        var item = Literal(depth + 1);
                        if (item is not null && elemType != typeof(object) && item.GetType() != elemType)
                            throw new ProgramFormatException("An array literal mixes element types.");
                        arr.SetValue(item, i);
                    }
                    return arr;
                default: throw new ProgramFormatException($"Unknown literal tag {tag}.");
            }
        }
    }
}
