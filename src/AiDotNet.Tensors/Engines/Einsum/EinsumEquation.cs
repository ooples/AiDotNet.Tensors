using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AiDotNet.Tensors.Engines.Einsum;

/// <summary>
/// Parsed einsum equation: per-operand labels + output labels.
/// </summary>
/// <remarks>
/// <para>
/// Einsum notation is a compact way to describe tensor contractions and
/// reshapes. Every axis in every operand gets a single-letter label
/// (<c>a</c>-<c>z</c>, <c>A</c>-<c>Z</c>); repeated labels between operands
/// are summed, labels that appear in the output are kept, labels that appear
/// only on the input side are summed out.
/// </para>
/// <para>The parser accepts three forms:</para>
/// <list type="bullet">
///   <item><description><c>"ij,jk-&gt;ik"</c> — explicit output.</description></item>
///   <item><description><c>"ij,jk"</c> — implicit output; labels that appear
///     exactly once across all operands are included, sorted alphabetically.</description></item>
///   <item><description><c>"...ij,...jk-&gt;...ik"</c> — ellipsis (<c>...</c>)
///     stands in for zero or more leading batch dims. Each operand may contain
///     at most one <c>...</c>.</description></item>
/// </list>
/// <para>
/// This class performs syntactic parsing only. Dimension agreement (same
/// label ⇒ same size) is validated later, once operand shapes are bound.
/// </para>
/// </remarks>
public sealed class EinsumEquation
{
    /// <summary>Per-operand label lists (one entry per operand).</summary>
    public IReadOnlyList<OperandLabels> Operands { get; }

    /// <summary>Output labels.</summary>
    public OperandLabels Output { get; }

    /// <summary>True if the equation contained <c>-&gt;</c>.</summary>
    public bool HasExplicitOutput { get; }

    /// <summary>True if any operand or the output contains <c>...</c>.</summary>
    public bool HasEllipsis { get; }

    /// <summary>Union of every label appearing anywhere in the equation.</summary>
    /// <remarks>
    /// Backed by a <see cref="HashSet{T}"/>; LINQ <c>.Contains</c> takes the
    /// <see cref="ICollection{T}"/> fast path, so lookup is O(1) despite the
    /// <see cref="IReadOnlyCollection{T}"/> surface (we need a net471-compatible
    /// type — <c>IReadOnlySet&lt;T&gt;</c> was added in .NET 5).
    /// </remarks>
    public IReadOnlyCollection<char> AllLabels { get; }

    /// <summary>
    /// Labels that appear on the input side but not on the output side
    /// (i.e. labels that get summed over).
    /// </summary>
    public IReadOnlyCollection<char> ContractionLabels { get; }

    /// <summary>The original equation string the parser was given.</summary>
    public string Source { get; }

    private EinsumEquation(
        string source,
        IReadOnlyList<OperandLabels> operands,
        OperandLabels output,
        bool hasExplicitOutput,
        bool hasEllipsis,
        IReadOnlyCollection<char> allLabels,
        IReadOnlyCollection<char> contractionLabels)
    {
        Source = source;
        Operands = operands;
        Output = output;
        HasExplicitOutput = hasExplicitOutput;
        HasEllipsis = hasEllipsis;
        AllLabels = allLabels;
        ContractionLabels = contractionLabels;
    }

    /// <summary>
    /// Parses an einsum equation string. Throws
    /// <see cref="EinsumSyntaxException"/> on malformed input with the
    /// character position of the offending token.
    /// </summary>
    public static EinsumEquation Parse(string equation)
    {
        if (equation is null) throw new ArgumentNullException(nameof(equation));

        // Strip whitespace but keep a position map so error messages can point
        // back at the original equation.
        var (stripped, positionMap) = StripWhitespace(equation);

        if (stripped.Length == 0)
            throw new EinsumSyntaxException(equation, 0, "equation is empty");

        // Locate "->" (at most one occurrence).
        int arrow = IndexOfArrow(stripped, positionMap, equation);
        bool hasExplicit = arrow >= 0;

        string inputSide = hasExplicit ? stripped.Substring(0, arrow) : stripped;
        string outputSide = hasExplicit ? stripped.Substring(arrow + 2) : string.Empty;
        int inputSideOffset = 0;
        int outputSideOffset = hasExplicit ? arrow + 2 : -1;

        // Split operand side on ','.
        var operands = new List<OperandLabels>();
        bool hasEllipsis = false;
        var allLabels = new HashSet<char>();
        int cursor = 0;
        int operandIndex = 0;
        while (cursor <= inputSide.Length)
        {
            int comma = inputSide.IndexOf(',', cursor);
            int end = comma < 0 ? inputSide.Length : comma;
            string operandStr = inputSide.Substring(cursor, end - cursor);
            var labels = ParseOperand(
                operandStr,
                positionMap,
                equation,
                cursor + inputSideOffset,
                operandIndex,
                isOutput: false);
            operands.Add(labels);
            if (labels.HasEllipsis) hasEllipsis = true;
            foreach (var c in labels.Labels) allLabels.Add(c);

            operandIndex++;
            if (comma < 0) break;
            cursor = comma + 1;
        }

        if (operands.Count == 0)
            throw new EinsumSyntaxException(equation, 0, "equation has no operands");

        // Parse / infer output.
        OperandLabels output;
        if (hasExplicit)
        {
            output = ParseOperand(
                outputSide,
                positionMap,
                equation,
                outputSideOffset,
                operandIndex: -1,
                isOutput: true);

            // Every output label must appear somewhere on the input side.
            foreach (var c in output.Labels)
            {
                if (!allLabels.Contains(c))
                    throw new EinsumSyntaxException(
                        equation,
                        FindOutputLabelPosition(positionMap, outputSideOffset, outputSide, c),
                        $"output label '{c}' does not appear in any operand");
            }

            // Output may not repeat labels — report the position of the
            // duplicate (the second occurrence), not the first.
            var outputSeen = new HashSet<char>();
            for (int i = 0; i < output.Labels.Count; i++)
            {
                char c = output.Labels[i];
                if (!outputSeen.Add(c))
                {
                    int labelOffset = FindNthLabelPosition(outputSide, i);
                    throw new EinsumSyntaxException(
                        equation,
                        MapPosition(positionMap, outputSideOffset + labelOffset),
                        $"output label '{c}' appears more than once");
                }
            }

            if (output.HasEllipsis && !hasEllipsis)
                throw new EinsumSyntaxException(
                    equation,
                    FindEllipsisPosition(positionMap, outputSideOffset, outputSide),
                    "output contains '...' but no operand does");
        }
        else
        {
            output = InferImplicitOutput(operands, hasEllipsis);
        }

        // Contraction labels = (union of operand labels) \ (output labels).
        var outputSet = new HashSet<char>(output.Labels);
        var contractionLabels = new HashSet<char>();
        foreach (var c in allLabels)
        {
            if (!outputSet.Contains(c)) contractionLabels.Add(c);
        }

        return new EinsumEquation(
            source: equation,
            operands: operands,
            output: output,
            hasExplicitOutput: hasExplicit,
            hasEllipsis: hasEllipsis,
            allLabels: allLabels,
            contractionLabels: contractionLabels);
    }

    private static OperandLabels ParseOperand(
        string operand,
        int[] positionMap,
        string original,
        int baseOffset,
        int operandIndex,
        bool isOutput)
    {
        var labels = new List<char>();
        int ellipsisPos = -1;
        int countBefore = 0;
        int countAfter = 0;

        int i = 0;
        while (i < operand.Length)
        {
            char c = operand[i];
            if (c == '.')
            {
                // Must be exactly three consecutive dots.
                if (i + 2 >= operand.Length || operand[i + 1] != '.' || operand[i + 2] != '.')
                    throw new EinsumSyntaxException(
                        original,
                        MapPosition(positionMap, baseOffset + i),
                        "lone '.' is not allowed; use '...' for ellipsis");

                if (ellipsisPos >= 0)
                    throw new EinsumSyntaxException(
                        original,
                        MapPosition(positionMap, baseOffset + i),
                        isOutput
                            ? "output contains more than one '...'"
                            : $"operand {operandIndex} contains more than one '...'");

                ellipsisPos = labels.Count;
                countBefore = labels.Count;
                i += 3;
                continue;
            }

            if (!IsLabelChar(c))
                throw new EinsumSyntaxException(
                    original,
                    MapPosition(positionMap, baseOffset + i),
                    $"invalid character '{c}'; einsum labels must be ASCII letters");

            labels.Add(c);
            i++;
        }

        if (ellipsisPos >= 0)
        {
            countAfter = labels.Count - ellipsisPos;
        }
        else
        {
            countBefore = labels.Count;
        }

        return new OperandLabels(labels, ellipsisPos, countBefore, countAfter);
    }

    private static OperandLabels InferImplicitOutput(
        IReadOnlyList<OperandLabels> operands,
        bool hasEllipsis)
    {
        // Count total occurrences of each label across all operands.
        var counts = new Dictionary<char, int>();
        foreach (var op in operands)
        {
            foreach (var c in op.Labels)
            {
                counts[c] = counts.TryGetValue(c, out var n) ? n + 1 : 1;
            }
        }

        // In implicit mode, labels that appear exactly once across the input
        // side become output labels, sorted alphabetically. If any operand
        // contains an ellipsis, the output contains a leading '...'.
        var outputLabels = counts
            .Where(kv => kv.Value == 1)
            .Select(kv => kv.Key)
            .OrderBy(c => c)
            .ToList();

        if (hasEllipsis)
        {
            // Ellipsis is prepended.
            return new OperandLabels(
                labels: outputLabels,
                ellipsisPosition: 0,
                countBefore: 0,
                countAfter: outputLabels.Count);
        }

        return new OperandLabels(
            labels: outputLabels,
            ellipsisPosition: -1,
            countBefore: outputLabels.Count,
            countAfter: 0);
    }

    private static (string stripped, int[] positionMap) StripWhitespace(string equation)
    {
        var sb = new StringBuilder(equation.Length);
        var map = new int[equation.Length];
        int j = 0;
        for (int i = 0; i < equation.Length; i++)
        {
            char c = equation[i];
            if (char.IsWhiteSpace(c)) continue;
            sb.Append(c);
            map[j++] = i;
        }
        // Truncate map to actual length used.
        if (j != map.Length)
        {
            var trimmed = new int[j];
            Array.Copy(map, trimmed, j);
            map = trimmed;
        }
        return (sb.ToString(), map);
    }

    private static int MapPosition(int[] positionMap, int strippedIndex)
    {
        if (strippedIndex < 0) return 0;
        if (strippedIndex >= positionMap.Length)
            return positionMap.Length == 0 ? 0 : positionMap[positionMap.Length - 1] + 1;
        return positionMap[strippedIndex];
    }

    private static int IndexOfArrow(string stripped, int[] positionMap, string original)
    {
        int first = stripped.IndexOf("->", StringComparison.Ordinal);
        if (first < 0) return -1;
        int second = stripped.IndexOf("->", first + 2, StringComparison.Ordinal);
        if (second >= 0)
            throw new EinsumSyntaxException(
                original,
                MapPosition(positionMap, second),
                "einsum equation contains more than one '->'");
        return first;
    }

    private static int FindOutputLabelPosition(
        int[] positionMap,
        int outputSideOffset,
        string outputSide,
        char label)
    {
        // Walks the (stripped) outputSide ignoring "..." tokens to find the
        // first stripped-index where `label` occurs.
        int i = 0;
        while (i < outputSide.Length)
        {
            if (i + 2 < outputSide.Length
                && outputSide[i] == '.'
                && outputSide[i + 1] == '.'
                && outputSide[i + 2] == '.')
            {
                i += 3;
                continue;
            }
            if (outputSide[i] == label)
                return MapPosition(positionMap, outputSideOffset + i);
            i++;
        }
        return MapPosition(positionMap, outputSideOffset);
    }

    /// <summary>
    /// Given an operand/output string, returns the stripped-index of the
    /// <paramref name="n"/>-th label character (skipping any "..." tokens).
    /// </summary>
    private static int FindNthLabelPosition(string operandStr, int n)
    {
        int labelCount = 0;
        int i = 0;
        while (i < operandStr.Length)
        {
            if (i + 2 < operandStr.Length
                && operandStr[i] == '.'
                && operandStr[i + 1] == '.'
                && operandStr[i + 2] == '.')
            {
                i += 3;
                continue;
            }
            if (labelCount == n) return i;
            labelCount++;
            i++;
        }
        return operandStr.Length;
    }

    private static int FindEllipsisPosition(
        int[] positionMap,
        int outputSideOffset,
        string outputSide)
    {
        int idx = outputSide.IndexOf("...", StringComparison.Ordinal);
        if (idx < 0) return 0;
        return MapPosition(positionMap, outputSideOffset + idx);
    }

    private static bool IsLabelChar(char c)
        => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

/// <summary>
/// Labels for one operand (or the output) of an einsum equation.
/// </summary>
public sealed class OperandLabels
{
    /// <summary>Sequence of non-ellipsis labels, in original order.</summary>
    public IReadOnlyList<char> Labels { get; }

    /// <summary>
    /// Position of the ellipsis in the label sequence, or <c>-1</c> if there
    /// is no ellipsis. If non-negative, the ellipsis "sits between"
    /// <c>Labels[EllipsisPosition - 1]</c> and <c>Labels[EllipsisPosition]</c>
    /// (or at the start/end for boundary positions).
    /// </summary>
    public int EllipsisPosition { get; }

    /// <summary>True if this operand contains <c>...</c>.</summary>
    public bool HasEllipsis => EllipsisPosition >= 0;

    /// <summary>Number of explicit labels that appear before the ellipsis.</summary>
    public int CountBeforeEllipsis { get; }

    /// <summary>Number of explicit labels that appear after the ellipsis.</summary>
    public int CountAfterEllipsis { get; }

    internal OperandLabels(
        IReadOnlyList<char> labels,
        int ellipsisPosition,
        int countBefore,
        int countAfter)
    {
        Labels = labels;
        EllipsisPosition = ellipsisPosition;
        CountBeforeEllipsis = countBefore;
        CountAfterEllipsis = countAfter;
    }

    /// <summary>
    /// Minimum number of dims an operand must have to satisfy these labels.
    /// When the operand has an ellipsis, the operand may have strictly more
    /// dims (the extra dims are the "batch" dims represented by <c>...</c>).
    /// </summary>
    public int MinimumRank => CountBeforeEllipsis + CountAfterEllipsis;

    /// <inheritdoc/>
    public override string ToString()
    {
        if (!HasEllipsis) return new string(Labels.ToArray());
        var sb = new StringBuilder(Labels.Count + 3);
        for (int i = 0; i < EllipsisPosition; i++) sb.Append(Labels[i]);
        sb.Append("...");
        for (int i = EllipsisPosition; i < Labels.Count; i++) sb.Append(Labels[i]);
        return sb.ToString();
    }
}

/// <summary>
/// Thrown when an einsum equation string is syntactically malformed.
/// The <see cref="Position"/> is the 0-based character offset in the original
/// equation at which the error was detected.
/// </summary>
public sealed class EinsumSyntaxException : ArgumentException
{
    /// <summary>The original equation string.</summary>
    public string Equation { get; }

    /// <summary>0-based character position in <see cref="Equation"/>.</summary>
    public int Position { get; }

    /// <summary>Constructs a parse error.</summary>
    public EinsumSyntaxException(string equation, int position, string reason)
        : base(BuildMessage(equation, position, reason))
    {
        Equation = equation;
        Position = position;
    }

    private static string BuildMessage(string equation, int position, string reason)
    {
        // ASCII caret line under the offending column.
        var sb = new StringBuilder();
        sb.Append("Einsum equation syntax error at column ");
        sb.Append(position + 1);
        sb.Append(": ");
        sb.AppendLine(reason);
        sb.Append("    ");
        sb.AppendLine(equation);
        sb.Append("    ");
        if (position > 0) sb.Append(' ', position);
        sb.Append('^');
        return sb.ToString();
    }
}
