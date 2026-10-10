using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Named dimensions (torch named tensors): a name per dimension, and the operations that set, check and use them —
/// <see cref="Rename(string?[])"/>, <see cref="RenameInPlace(string?[])"/>, <see cref="RefineNames"/>,
/// <see cref="AlignTo"/>, <see cref="AlignAs"/> and <see cref="AlignTensors"/>.
/// </summary>
/// <remarks>
/// <para>Names are metadata of the tensor object. The naming operations return views that share the storage (and are
/// tracked by the gradient tape), as torch's do. Other operations return unnamed results: names are not propagated
/// through arithmetic, reductions or layers the way torch's prototype named-tensor support propagates them.</para>
/// <para>"..." in a name list stands for the dimensions not otherwise listed, in order.</para>
/// </remarks>
public partial class Tensor<T>
{
    private const string Ellipsis = "...";
    private string?[]? _names;

    /// <summary>The name of each dimension; null entries are unnamed (torch <c>Tensor.names</c>).</summary>
    public IReadOnlyList<string?> Names => _names ?? new string?[Rank];

    /// <summary>True when any dimension has a name (torch <c>has_names</c>).</summary>
    public bool HasNames() => _names is not null && _names.Any(n => n is not null);

    /// <summary>
    /// A view of this tensor with its dimensions named <paramref name="names"/> (torch <c>rename</c>); no names, or a
    /// single null, removes them. "..." keeps the names of the dimensions it covers.
    /// </summary>
    public Tensor<T> Rename(params string?[] names)
    {
        var alias = NamedAlias();
        alias._names = ResolveRename(names);
        return alias;
    }

    /// <summary>A view of this tensor with names renamed through <paramref name="mapping"/> (torch <c>rename(**rename_map)</c>).</summary>
    public Tensor<T> Rename(IReadOnlyDictionary<string, string> mapping)
    {
        var alias = NamedAlias();
        alias._names = ResolveMapping(mapping);
        return alias;
    }

    /// <summary>Names this tensor's dimensions in place and returns it (torch <c>rename_</c>); see <see cref="Rename(string?[])"/>.</summary>
    public Tensor<T> RenameInPlace(params string?[] names)
    {
        _names = ResolveRename(names);
        return this;
    }

    /// <summary>Renames this tensor's dimensions in place through <paramref name="mapping"/> and returns it (torch <c>rename_(**rename_map)</c>).</summary>
    public Tensor<T> RenameInPlace(IReadOnlyDictionary<string, string> mapping)
    {
        _names = ResolveMapping(mapping);
        return this;
    }

    /// <summary>
    /// A view with unnamed dimensions given names (torch <c>refine_names</c>): a named dimension must keep its name,
    /// so refining checks the names already there.
    /// </summary>
    public Tensor<T> RefineNames(params string[] names)
    {
        var current = Names;
        var refined = ExpandEllipsis(names.Cast<string?>().ToArray(), current.ToArray());
        for (int d = 0; d < Rank; d++)
            if (current[d] is not null && current[d] != refined[d])
                throw new InvalidOperationException($"Cannot refine dimension {d} from '{current[d]}' to '{refined[d]}': a named dimension keeps its name.");
        var alias = NamedAlias();
        alias._names = CheckNames(refined);
        return alias;
    }

    /// <summary>
    /// This tensor with its dimensions permuted to the order of <paramref name="names"/>, a size-1 dimension inserted
    /// for each name it lacks (torch <c>align_to</c>). Every dimension must be named; "..." places the remaining ones.
    /// </summary>
    public Tensor<T> AlignTo(params string[] names)
    {
        if (names is null) throw new ArgumentNullException(nameof(names));
        var current = Names;
        if (current.Any(n => n is null)) throw new InvalidOperationException("align_to needs every dimension of the tensor named.");
        // "..." covers the tensor's dimensions not listed, in their order.
        var listed = new HashSet<string>(names.Where(n => n != Ellipsis));
        var order = new List<string>();
        foreach (var n in names)
        {
            if (n == Ellipsis) order.AddRange(current.Where(c => !listed.Contains(c!)).Select(c => c!));
            else order.Add(n);
        }
        foreach (var c in current)
            if (!order.Contains(c!)) throw new InvalidOperationException($"align_to would drop dimension '{c}'; list it or use \"...\".");
        if (order.Distinct().Count() != order.Count) throw new ArgumentException("align_to names must be distinct.", nameof(names));

        IEngine engine = AiDotNetEngine.Current;
        var present = order.Where(n => current.Contains(n)).ToList();
        var perm = present.Select(n => current.ToList().IndexOf(n)).ToArray();
        var permuted = perm.SequenceEqual(Enumerable.Range(0, Rank)) ? this : engine.TensorPermute(this, perm);
        var shape = order.Select(n => current.Contains(n) ? _shape[current.ToList().IndexOf(n)] : 1).ToArray();
        var aligned = shape.SequenceEqual(permuted._shape) ? permuted : engine.Reshape(permuted, shape);
        var result = ReferenceEquals(aligned, this) ? NamedAlias() : aligned;
        result._names = order.Cast<string?>().ToArray();
        return result;
    }

    /// <summary><see cref="AlignTo"/> the names of <paramref name="other"/> (torch <c>align_as</c>).</summary>
    public Tensor<T> AlignAs(Tensor<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (other.Names.Any(n => n is null)) throw new InvalidOperationException("align_as needs every dimension of the other tensor named.");
        return AlignTo(other.Names.Select(n => n!).ToArray());
    }

    /// <summary>
    /// Aligns every tensor to the names of the one with the most dimensions (torch <c>align_tensors</c>); each
    /// tensor's names must appear among them.
    /// </summary>
    public static Tensor<T>[] AlignTensors(params Tensor<T>[] tensors)
    {
        if (tensors is null || tensors.Length == 0) throw new ArgumentException("align_tensors needs at least one tensor.", nameof(tensors));
        var widest = tensors.OrderByDescending(t => t.Rank).First();
        if (widest.Names.Any(n => n is null)) throw new InvalidOperationException("align_tensors needs every dimension named.");
        var order = widest.Names.Select(n => n!).ToArray();
        return tensors.Select(t => t.AlignTo(order)).ToArray();
    }

    // A tape-tracked view over the same storage, the result of every naming operation.
    private Tensor<T> NamedAlias()
    {
        var alias = IsSparse || !IsContiguous && IsGpuResident
            ? (Tensor<T>)Clone()
            : new Tensor<T>(DataVector, _shape, _strides, _storageOffset, _storage);
        DifferentiableOps.RecordUnary("Rename", alias, this, BackwardFunctions<T>.InPlaceAssignBackward);
        return alias;
    }

    private string?[]? ResolveRename(string?[]? names)
    {
        if (names is null || names.Length == 0 || names.Length == 1 && names[0] is null && Rank != 1) return null;
        return CheckNames(ExpandEllipsis(names, Names.ToArray()));
    }

    private string?[] ResolveMapping(IReadOnlyDictionary<string, string> mapping)
    {
        if (mapping is null) throw new ArgumentNullException(nameof(mapping));
        var current = Names.ToArray();
        foreach (var key in mapping.Keys)
            if (!current.Contains(key)) throw new ArgumentException($"No dimension is named '{key}'.", nameof(mapping));
        return CheckNames(current.Select(n => n is not null && mapping.TryGetValue(n, out var to) ? to : n).ToArray());
    }

    // Expands "..." to the dimensions it stands for, keeping their current names.
    private string?[] ExpandEllipsis(string?[] names, string?[] current)
    {
        int at = Array.IndexOf(names, Ellipsis);
        if (at < 0)
        {
            if (names.Length != Rank) throw new ArgumentException($"{names.Length} names for a tensor of rank {Rank}.", nameof(names));
            return names;
        }
        if (Array.IndexOf(names, Ellipsis, at + 1) >= 0) throw new ArgumentException("At most one \"...\" per name list.", nameof(names));
        int covered = Rank - (names.Length - 1);
        if (covered < 0) throw new ArgumentException($"{names.Length - 1} names for a tensor of rank {Rank}.", nameof(names));
        var r = new string?[Rank];
        for (int i = 0; i < at; i++) r[i] = names[i];
        for (int i = 0; i < covered; i++) r[at + i] = current[at + i];
        for (int i = at + 1; i < names.Length; i++) r[covered + i - 1] = names[i];
        return r;
    }

    private static string?[] CheckNames(string?[] names)
    {
        var seen = new HashSet<string>();
        foreach (var n in names)
        {
            if (n is null) continue;
            if (n.Length == 0 || n == Ellipsis) throw new ArgumentException($"'{n}' is not a dimension name.");
            if (!seen.Add(n)) throw new ArgumentException($"The name '{n}' is used for two dimensions.");
        }
        return names;
    }
}
