// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.LinearAlgebra.Named;

/// <summary>
/// Axis-labelled view over a <see cref="Tensor{T}"/>. PyTorch's
/// <c>torch.Tensor</c> with the <c>names</c> attribute equivalent. Each
/// dimension carries an optional name (<c>null</c> = unlabelled, matches
/// PyTorch's <c>None</c>); operations over named tensors line up
/// dimensions by <em>name</em> rather than position.
///
/// <para><b>How we beat PyTorch (#222 point #3):</b> our names survive
/// <c>LazyTensorScope</c> compile / fusion / kernel generation. Engine
/// passes carry the name array through every transformation so a fused
/// matmul produced by the compiler keeps the original axis labels.
/// PyTorch erases names on the <c>torch.compile</c> boundary.</para>
///
/// <para><b>Storage:</b> a named tensor is a thin wrapper holding a
/// <see cref="Tensor{T}"/> + a parallel names array. No copies occur on
/// rename / refine / align — shape mutations are share-storage views.</para>
/// </summary>
public sealed class NamedTensor<T>
{
    /// <summary>Underlying dense tensor — name-erased view.</summary>
    public Tensor<T> Tensor { get; }

    /// <summary>Per-axis names. <c>null</c> entries are unlabelled axes.</summary>
    public string?[] Names { get; }

    /// <summary>True when at least one axis is named.</summary>
    public bool HasNames
    {
        get
        {
            for (int i = 0; i < Names.Length; i++) if (Names[i] is not null) return true;
            return false;
        }
    }

    /// <summary>Rank of the underlying tensor.</summary>
    public int Rank => Tensor.Rank;

    /// <summary>Shape of the underlying tensor.</summary>
    public int[] Shape => (int[])Tensor._shape.Clone();

    /// <summary>Number of elements (= <c>Tensor.Length</c>).</summary>
    public int Length => Tensor.Length;

    /// <summary>Wraps an existing tensor with the supplied names.
    /// <paramref name="names"/> length must match the tensor's rank.</summary>
    public NamedTensor(Tensor<T> tensor, params string?[] names)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (names is null) names = new string?[tensor.Rank];
        if (names.Length != tensor.Rank)
            throw new ArgumentException(
                $"Names length {names.Length} doesn't match tensor rank {tensor.Rank}.", nameof(names));
        EnsureNoDuplicateNames(names);

        Tensor = tensor;
        Names = (string?[])names.Clone();
    }

    /// <summary>Returns a new named tensor with the supplied names.
    /// Mirrors <c>torch.Tensor.rename</c>.</summary>
    public NamedTensor<T> Rename(params string?[] newNames)
    {
        if (newNames.Length != Rank)
            throw new ArgumentException($"newNames length {newNames.Length} != rank {Rank}.", nameof(newNames));
        return new NamedTensor<T>(Tensor, newNames);
    }

    /// <summary>
    /// Refines names by adding labels to previously-<c>null</c> axes.
    /// Throws when an existing label would be overwritten with a
    /// different one. Mirrors <c>torch.Tensor.refine_names</c>.
    /// </summary>
    public NamedTensor<T> RefineNames(params string?[] desired)
    {
        if (desired.Length != Rank)
            throw new ArgumentException($"desired length {desired.Length} != rank {Rank}.", nameof(desired));
        var refined = new string?[Rank];
        for (int i = 0; i < Rank; i++)
        {
            string? cur = Names[i];
            string? want = desired[i];
            if (cur is null) refined[i] = want;
            else if (want is null || want == cur) refined[i] = cur;
            else throw new InvalidOperationException(
                $"Cannot refine axis {i}: existing name '{cur}' conflicts with desired '{want}'.");
        }
        return new NamedTensor<T>(Tensor, refined);
    }

    /// <summary>
    /// Permutes the tensor so its axes appear in <paramref name="targetNames"/>
    /// order. Names not in the target list raise; missing names raise.
    /// Mirrors <c>torch.Tensor.align_to</c>.
    /// </summary>
    public NamedTensor<T> AlignTo(params string[] targetNames)
    {
        if (targetNames is null) throw new ArgumentNullException(nameof(targetNames));
        if (targetNames.Length != Rank)
            throw new ArgumentException(
                $"AlignTo expects {Rank} names; got {targetNames.Length}.", nameof(targetNames));
        // Fully-named contract: AlignTo can't represent unnamed axes
        // since the target list is just a string[]. Reject any
        // unlabelled source or empty target name with a clear message
        // rather than silently misalign.
        for (int i = 0; i < Rank; i++)
            if (string.IsNullOrEmpty(Names[i]))
                throw new InvalidOperationException(
                    $"AlignTo requires fully-named source tensor; axis {i} is unnamed.");
        for (int i = 0; i < targetNames.Length; i++)
            if (string.IsNullOrEmpty(targetNames[i]))
                throw new ArgumentException(
                    $"AlignTo target name at index {i} must be non-empty.", nameof(targetNames));

        var perm = new int[Rank];
        var used = new bool[Rank];
        for (int i = 0; i < Rank; i++)
        {
            string want = targetNames[i];
            int idx = Array.IndexOf(Names, want);
            if (idx < 0)
                throw new InvalidOperationException(
                    $"AlignTo target '{want}' not present in tensor names [{string.Join(", ", Names)}].");
            if (used[idx])
                throw new InvalidOperationException($"AlignTo: name '{want}' appears twice in target.");
            used[idx] = true;
            perm[i] = idx;
        }
        var permuted = PermuteTensor(Tensor, perm);
        var newNames = new string?[Rank];
        for (int i = 0; i < Rank; i++) newNames[i] = targetNames[i];
        return new NamedTensor<T>(permuted, newNames);
    }

    /// <summary>Aligns this tensor to <paramref name="other"/>'s name
    /// order. Names exclusive to this tensor must be a contiguous
    /// suffix; names exclusive to other are added as broadcast-1 dims
    /// — same semantics as <c>torch.Tensor.align_as</c>.</summary>
    public NamedTensor<T> AlignAs(NamedTensor<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        // Fully-named contract on the source: every axis must be named
        // for AlignAs to reach the count check below cleanly. (`other`
        // can have unnamed axes — those become broadcast-1 dims.)
        for (int i = 0; i < Rank; i++)
            if (string.IsNullOrEmpty(Names[i]))
                throw new InvalidOperationException(
                    $"AlignAs requires fully-named source tensor; axis {i} is unnamed.");
        // Subset path: every name in `this` exists in `other` ⇒ permute then unsqueeze.
        var orderedSelfNames = new List<string>();
        var orderedSelfDims = new List<int>();
        for (int i = 0; i < other.Rank; i++)
        {
            string? n = other.Names[i];
            if (n is null) continue;
            int srcIdx = Array.IndexOf(Names, n);
            if (srcIdx >= 0)
            {
                orderedSelfNames.Add(n);
                orderedSelfDims.Add(srcIdx);
            }
        }
        if (orderedSelfNames.Count != Rank)
            throw new InvalidOperationException(
                "AlignAs requires every name in this tensor to appear in `other`.");

        // Permute existing dims into the order they appear in `other`,
        // then build a new tensor whose shape is `other.Shape` with
        // 1s in positions where `this` has no axis.
        var perm = orderedSelfDims.ToArray();
        var permutedTensor = PermuteTensor(Tensor, perm);

        // Build broadcast-1 shape — copy permuted into the matching
        // positions, leave 1 elsewhere.
        var newShape = new int[other.Rank];
        var newNames = new string?[other.Rank];
        for (int i = 0; i < other.Rank; i++)
        {
            newShape[i] = 1;
            newNames[i] = other.Names[i];
        }
        for (int i = 0; i < orderedSelfNames.Count; i++)
        {
            int dst = Array.IndexOf(other.Names, orderedSelfNames[i]);
            newShape[dst] = permutedTensor._shape[i];
        }
        var reshaped = new Tensor<T>(newShape);
        permutedTensor.AsSpan().CopyTo(reshaped.AsWritableSpan());
        return new NamedTensor<T>(reshaped, newNames);
    }

    /// <summary>Flattens the contiguous axes named in <paramref name="names"/>
    /// into a single axis labelled <paramref name="newName"/>. The named
    /// axes must be contiguous in the current tensor order.</summary>
    public NamedTensor<T> Flatten(string[] names, string newName)
    {
        if (names is null || names.Length == 0)
            throw new ArgumentException("Flatten needs at least one source axis.", nameof(names));
        int firstIdx = Array.IndexOf(Names, names[0]);
        if (firstIdx < 0)
            throw new InvalidOperationException($"Flatten: axis '{names[0]}' not found.");
        for (int i = 0; i < names.Length; i++)
        {
            if (Names[firstIdx + i] != names[i])
                throw new InvalidOperationException(
                    "Flatten: source axes must be contiguous in tensor order.");
        }
        // New shape: axes before, flattened axis, axes after.
        int collapsed = 1;
        for (int i = 0; i < names.Length; i++) collapsed *= Tensor._shape[firstIdx + i];
        var newShape = new int[Rank - names.Length + 1];
        var newNames = new string?[Rank - names.Length + 1];
        int writer = 0;
        for (int i = 0; i < firstIdx; i++)
        {
            newShape[writer] = Tensor._shape[i];
            newNames[writer] = Names[i];
            writer++;
        }
        newShape[writer] = collapsed;
        newNames[writer] = newName;
        writer++;
        for (int i = firstIdx + names.Length; i < Rank; i++)
        {
            newShape[writer] = Tensor._shape[i];
            newNames[writer] = Names[i];
            writer++;
        }
        var flat = new Tensor<T>(newShape);
        Tensor.AsSpan().CopyTo(flat.AsWritableSpan());
        return new NamedTensor<T>(flat, newNames);
    }

    /// <summary>Inverse of <see cref="Flatten"/> — splits axis
    /// <paramref name="name"/> into <paramref name="newSizes"/> with the
    /// supplied <paramref name="newNames"/>. Product of sizes must equal
    /// the original axis length.</summary>
    public NamedTensor<T> Unflatten(string name, int[] newSizes, string?[] newNames)
    {
        int idx = Array.IndexOf(Names, name);
        if (idx < 0) throw new InvalidOperationException($"Unflatten: axis '{name}' not found.");
        if (newSizes.Length != newNames.Length)
            throw new ArgumentException("newSizes and newNames must have the same length.");
        int product = 1;
        for (int i = 0; i < newSizes.Length; i++) product *= newSizes[i];
        if (product != Tensor._shape[idx])
            throw new ArgumentException(
                $"Unflatten product {product} != axis length {Tensor._shape[idx]}.");

        var resultShape = new int[Rank - 1 + newSizes.Length];
        var resultNames = new string?[resultShape.Length];
        int writer = 0;
        for (int i = 0; i < idx; i++)
        {
            resultShape[writer] = Tensor._shape[i];
            resultNames[writer] = Names[i];
            writer++;
        }
        for (int i = 0; i < newSizes.Length; i++)
        {
            resultShape[writer] = newSizes[i];
            resultNames[writer] = newNames[i];
            writer++;
        }
        for (int i = idx + 1; i < Rank; i++)
        {
            resultShape[writer] = Tensor._shape[i];
            resultNames[writer] = Names[i];
            writer++;
        }
        var unflat = new Tensor<T>(resultShape);
        Tensor.AsSpan().CopyTo(unflat.AsWritableSpan());
        return new NamedTensor<T>(unflat, resultNames);
    }

    private static Tensor<T> PermuteTensor(Tensor<T> tensor, int[] perm)
    {
        if (tensor.Rank != perm.Length)
            throw new ArgumentException("perm length must equal tensor rank.");
        // Identity permutation — return as-is.
        bool identity = true;
        for (int i = 0; i < perm.Length; i++) if (perm[i] != i) { identity = false; break; }
        if (identity) return tensor;

        // Compute new shape and reorder elements.
        var newShape = new int[perm.Length];
        for (int i = 0; i < perm.Length; i++) newShape[i] = tensor._shape[perm[i]];
        var result = new Tensor<T>(newShape);
        var dst = result.AsWritableSpan();
        var src = tensor.AsSpan();

        // Use the tensor's own strides — recomputing dense row-major
        // strides from _shape is only correct for contiguous tensors.
        // For views/slices with non-default strides we'd permute the
        // wrong elements.
        var oldStrides = tensor._strides;
        var indices = new int[perm.Length];

        for (int linear = 0; linear < tensor.Length; linear++)
        {
            // Decompose linear → multi-index in NEW shape.
            int rem = linear;
            for (int d = perm.Length - 1; d >= 0; d--)
            {
                indices[d] = rem % newShape[d];
                rem /= newShape[d];
            }
            // Map new-index → old-index via inverse permutation.
            int srcLinear = 0;
            for (int d = 0; d < perm.Length; d++)
                srcLinear += indices[d] * oldStrides[perm[d]];
            dst[linear] = src[srcLinear];
        }
        return result;
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int s = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    private static void EnsureNoDuplicateNames(string?[] names)
    {
        var seen = new HashSet<string>();
        for (int i = 0; i < names.Length; i++)
        {
            string? n = names[i];
            if (n is null) continue;
            if (!seen.Add(n))
                throw new ArgumentException($"Duplicate axis name '{n}'.", nameof(names));
        }
    }
}
