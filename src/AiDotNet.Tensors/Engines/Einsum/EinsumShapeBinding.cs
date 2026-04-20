using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AiDotNet.Tensors.Engines.Einsum;

/// <summary>
/// Resolves an <see cref="EinsumEquation"/> against a set of operand shapes:
/// binds every label to a concrete dim size, expands ellipsis to its
/// broadcast batch shape, and computes the output shape.
/// </summary>
/// <remarks>
/// <para>Rules enforced here:</para>
/// <list type="bullet">
///   <item><description>
///     Every operand has the minimum rank required by its labels
///     (<see cref="OperandLabels.MinimumRank"/>). Operands with an ellipsis may
///     have strictly more dims (the surplus is the ellipsis batch shape).
///   </description></item>
///   <item><description>
///     A label that appears more than once — whether within one operand
///     (diagonal) or across operands — must have the same size at every
///     occurrence. <b>Labeled dimensions do not broadcast.</b>
///   </description></item>
///   <item><description>
///     Ellipsis batch shapes <b>do</b> broadcast (numpy alignment-right,
///     size-1 is compatible with any size).
///   </description></item>
/// </list>
/// </remarks>
public sealed class EinsumShapeBinding
{
    /// <summary>The equation the binding was computed against.</summary>
    public EinsumEquation Equation { get; }

    /// <summary>Shapes of the original operands (same order as the equation).</summary>
    public IReadOnlyList<int[]> OperandShapes { get; }

    /// <summary>
    /// Per-operand ellipsis dim sizes as they appear in that operand (empty
    /// array for operands with no ellipsis).
    /// </summary>
    public IReadOnlyList<int[]> OperandEllipsisDims { get; }

    /// <summary>
    /// Broadcast ellipsis batch shape (aligned right across all operands'
    /// ellipsis dims). Empty if the equation has no ellipsis.
    /// </summary>
    public int[] BatchDims { get; }

    /// <summary>Resolved per-label dim size.</summary>
    public IReadOnlyDictionary<char, int> LabelSizes { get; }

    /// <summary>Final output shape (ellipsis-expanded, label-sized).</summary>
    public int[] OutputShape { get; }

    private EinsumShapeBinding(
        EinsumEquation equation,
        IReadOnlyList<int[]> operandShapes,
        IReadOnlyList<int[]> operandEllipsisDims,
        int[] batchDims,
        IReadOnlyDictionary<char, int> labelSizes,
        int[] outputShape)
    {
        Equation = equation;
        OperandShapes = operandShapes;
        OperandEllipsisDims = operandEllipsisDims;
        BatchDims = batchDims;
        LabelSizes = labelSizes;
        OutputShape = outputShape;
    }

    /// <summary>
    /// Binds <paramref name="eq"/> against <paramref name="operandShapes"/>.
    /// Throws <see cref="EinsumShapeException"/> with a descriptive message on
    /// any rank, label-size, or broadcast mismatch.
    /// </summary>
    public static EinsumShapeBinding Bind(EinsumEquation eq, int[][] operandShapes)
    {
        if (eq is null) throw new ArgumentNullException(nameof(eq));
        if (operandShapes is null) throw new ArgumentNullException(nameof(operandShapes));
        if (operandShapes.Length != eq.Operands.Count)
            throw new EinsumShapeException(
                $"Equation expects {eq.Operands.Count} operand(s), got {operandShapes.Length}");

        var perOperandEllipsisDims = new int[eq.Operands.Count][];
        var labelSizes = new Dictionary<char, int>();

        for (int k = 0; k < eq.Operands.Count; k++)
        {
            var op = eq.Operands[k];
            int[] shape = operandShapes[k] ?? throw new EinsumShapeException(
                $"Operand {k} has a null shape");

            int rank = shape.Length;
            int min = op.MinimumRank;
            if (rank < min)
                throw new EinsumShapeException(
                    $"Operand {k} has rank {rank} but equation requires at least {min} " +
                    $"(labels: \"{op}\")");

            int ellipsisDimCount = op.HasEllipsis ? rank - min : 0;
            if (!op.HasEllipsis && rank != min)
                throw new EinsumShapeException(
                    $"Operand {k} has rank {rank}, but labels \"{op}\" describe exactly " +
                    $"{min} dim(s). Use '...' to represent extra batch dims.");

            // Split the operand shape into: before-ellipsis labels / ellipsis dims /
            // after-ellipsis labels.
            int cursor = 0;
            for (int i = 0; i < op.CountBeforeEllipsis; i++)
            {
                BindLabel(labelSizes, op.Labels[i], shape[cursor++], operandIndex: k);
            }

            var ellipsisDims = new int[ellipsisDimCount];
            for (int i = 0; i < ellipsisDimCount; i++)
            {
                ellipsisDims[i] = shape[cursor++];
            }
            perOperandEllipsisDims[k] = ellipsisDims;

            int afterStart = op.Labels.Count - op.CountAfterEllipsis;
            for (int i = 0; i < op.CountAfterEllipsis; i++)
            {
                BindLabel(labelSizes, op.Labels[afterStart + i], shape[cursor++], operandIndex: k);
            }
        }

        // Broadcast ellipsis batch dims across operands (numpy right-align).
        int[] batchDims = eq.HasEllipsis
            ? BroadcastBatchDims(perOperandEllipsisDims)
            : Array.Empty<int>();

        // Build output shape from Output labels + (optionally) batch dims at
        // the ellipsis position.
        int[] outputShape = BuildOutputShape(eq.Output, labelSizes, batchDims);

        return new EinsumShapeBinding(
            eq,
            operandShapes,
            perOperandEllipsisDims,
            batchDims,
            labelSizes,
            outputShape);
    }

    private static void BindLabel(
        Dictionary<char, int> labelSizes,
        char label,
        int size,
        int operandIndex)
    {
        if (size < 0)
            throw new EinsumShapeException(
                $"Operand {operandIndex}: dim for label '{label}' is negative ({size})");
        if (labelSizes.TryGetValue(label, out int existing))
        {
            if (existing != size)
                throw new EinsumShapeException(
                    $"Label '{label}' has inconsistent sizes: " +
                    $"previously {existing}, now {size} in operand {operandIndex}. " +
                    "Labeled dimensions do not broadcast; use '...' for batch dims.");
        }
        else
        {
            labelSizes[label] = size;
        }
    }

    private static int[] BroadcastBatchDims(int[][] perOperandEllipsisDims)
    {
        int maxRank = perOperandEllipsisDims.Max(e => e.Length);
        var result = new int[maxRank];
        // Default every slot to 1 so absent ellipsis dims behave like broadcastable 1s.
        for (int i = 0; i < maxRank; i++) result[i] = 1;

        foreach (var dims in perOperandEllipsisDims)
        {
            int offset = maxRank - dims.Length; // right-align
            for (int i = 0; i < dims.Length; i++)
            {
                int d = dims[i];
                int acc = result[offset + i];
                if (acc == 1)
                {
                    result[offset + i] = d;
                }
                else if (d != 1 && d != acc)
                {
                    throw new EinsumShapeException(
                        $"Ellipsis batch dim mismatch: cannot broadcast {d} against {acc} " +
                        $"at position {offset + i}");
                }
            }
        }
        return result;
    }

    private static int[] BuildOutputShape(
        OperandLabels output,
        Dictionary<char, int> labelSizes,
        int[] batchDims)
    {
        int total = output.Labels.Count + (output.HasEllipsis ? batchDims.Length : 0);
        var shape = new int[total];
        int cursor = 0;

        // Labels before ellipsis (or all labels if no ellipsis).
        int before = output.HasEllipsis ? output.EllipsisPosition : output.Labels.Count;
        for (int i = 0; i < before; i++)
            shape[cursor++] = labelSizes[output.Labels[i]];

        // Ellipsis batch dims.
        if (output.HasEllipsis)
        {
            for (int i = 0; i < batchDims.Length; i++)
                shape[cursor++] = batchDims[i];
        }

        // Labels after ellipsis.
        if (output.HasEllipsis)
        {
            for (int i = output.EllipsisPosition; i < output.Labels.Count; i++)
                shape[cursor++] = labelSizes[output.Labels[i]];
        }

        return shape;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append(Equation.Source);
        sb.Append(" -> [");
        for (int i = 0; i < OutputShape.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(OutputShape[i]);
        }
        sb.Append(']');
        return sb.ToString();
    }
}

/// <summary>
/// Thrown when an einsum binding fails to reconcile an equation against a
/// specific set of operand shapes (rank mismatch, inconsistent label size,
/// non-broadcastable ellipsis dims).
/// </summary>
public sealed class EinsumShapeException : ArgumentException
{
    /// <summary>Constructs a shape-binding error.</summary>
    public EinsumShapeException(string message) : base(message) { }
}
