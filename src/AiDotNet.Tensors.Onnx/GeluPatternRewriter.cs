using System.Runtime.InteropServices;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Detects the pre-opset-20 GELU-via-Erf decomposition that BERT / XLNet /
/// RoBERTa / T5 exports emit and rewrites it to a single <c>Gelu</c> node.
///
/// <para>The typical 5-node chain (numbers are ONNX op types):</para>
/// <code>
/// t0 = Div(x, √2)              // constant 1.41421356…
///    | Mul(x, 1/√2)            // equivalent form, 0.70710678…
/// t1 = Erf(t0)
/// t2 = Add(t1, 1.0)            // constant 1.0
/// t3 = Mul(x, t2)              // commuted forms allowed
/// y  = Mul(t3, 0.5)            // commuted forms allowed
/// </code>
///
/// <para>Our A-S 7.1.26 Erf approximation has max error 1.5e-7 which is
/// bit-accurate enough in isolation, but compounds through 12 transformer
/// layers × 768+ hidden dims. The engine's fused <c>Gelu</c> uses the exact
/// same kernel ORT does, so a pattern rewrite gives bit-exact logit
/// parity instead of ~5-6 absolute divergence.</para>
///
/// <para>Rewriting is safe when:</para>
/// <list type="bullet">
///   <item>Each intermediate tensor (t0, t1, t2, t3) has exactly one
///   consumer — otherwise another op would see the wrong value after
///   collapse.</item>
///   <item>Every constant in the chain matches its expected numeric value
///   (√2 or 1/√2, 1.0, 0.5) within a tight FP tolerance. Validating the
///   constants stops us from collapsing e.g. a general polynomial
///   activation into <c>Gelu</c> (silent miscompile).</item>
///   <item>The non-constant operand of each Mul resolves to the SAME
///   tensor <c>x</c> that feeds the Div/Mul-by-1/√2. Without this
///   cross-check, a commuted <c>Mul(const, x)</c> can mis-route
///   <c>xName</c> onto the constant tensor.</item>
/// </list>
/// </summary>
internal static class GeluPatternRewriter
{
    // Float tolerance for constant matching. 1e-5 is ~ULP × 10 at √2,
    // generous enough to accept fp32 round-trips through exporters that
    // store doubles and fp16 serialisations that widen to fp32, tight
    // enough to reject any reasonable non-GELU constant.
    private const double ConstTol = 1e-5;

    private static readonly double Sqrt2        = Math.Sqrt(2.0);          // ≈ 1.4142135624
    private static readonly double InvSqrt2     = 1.0 / Math.Sqrt(2.0);    // ≈ 0.7071067812
    private const double Half                   = 0.5;
    private const double One                    = 1.0;

    internal static IReadOnlyList<NodeProto> Rewrite(
        IReadOnlyList<NodeProto> nodes,
        IReadOnlyList<TensorProto>? initializers = null)
    {
        // Build a name → scalar-float lookup for constants referenced by
        // the chain. Non-scalar initializers are ignored (a GELU constant
        // is always a scalar). Missing initializers are treated as "not a
        // known constant" which will fail validation below.
        var scalarConsts = BuildScalarConstants(initializers);

        // Build consumer counts once: for each tensor name, how many nodes
        // consume it (strict > 1 disqualifies the chain from rewrite).
        var consumers = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < nodes.Count; i++)
        {
            var n = nodes[i];
            for (int j = 0; j < n.Input.Count; j++)
            {
                var name = n.Input[j];
                if (string.IsNullOrEmpty(name)) continue;
                consumers.TryGetValue(name, out int c);
                consumers[name] = c + 1;
            }
        }

        // Map output-name → node index for neighbor lookup.
        var producedBy = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < nodes.Count; i++)
            foreach (var o in nodes[i].Output)
                if (!string.IsNullOrEmpty(o))
                    producedBy[o] = i;

        // First pass: find the first node in each detected pattern and
        // record (startIndex → Gelu replacement + indices to skip).
        var replacements = new Dictionary<int, NodeProto>();
        var toSkip = new HashSet<int>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var n = nodes[i];
            if (n.OpType != "Erf") continue;
            if (n.Input.Count == 0 || n.Output.Count == 0) continue;
            if (!producedBy.TryGetValue(n.Input[0], out int divIdx)) continue;
            if (toSkip.Contains(divIdx)) continue;

            var divNode = nodes[divIdx];

            // Step 1: divNode is either Div(x, √2) or Mul(x, 1/√2).
            // Resolve xName (the non-constant operand) and verify the
            // constant. If either check fails we leave the chain alone.
            if (!TryResolveDivStage(divNode, scalarConsts, out string? xName))
                continue;
            if (GetCount(consumers, divNode.Output[0]) != 1) continue;

            // Step 2: Add(erfOut, 1.0) or Add(1.0, erfOut). Non-erf input
            // must be the constant 1.0.
            if (!TryFindSoleConsumer(nodes, consumers, n.Output[0], out int addIdx)) continue;
            var addNode = nodes[addIdx];
            if (addNode.OpType != "Add") continue;
            if (!TryValidateAddStage(addNode, n.Output[0], scalarConsts)) continue;

            // Step 3: mul1Node = Mul(x, addOut) or Mul(addOut, x). The
            // non-addOut input must be the same xName we resolved above.
            if (!TryFindSoleConsumer(nodes, consumers, addNode.Output[0], out int mul1Idx)) continue;
            var mul1Node = nodes[mul1Idx];
            if (mul1Node.OpType != "Mul") continue;
            if (!TryValidateMul1Stage(mul1Node, addNode.Output[0], xName!)) continue;

            // Step 4: mul2Node = Mul(mul1Out, 0.5) or Mul(0.5, mul1Out).
            // The non-mul1Out input must be the constant 0.5.
            if (!TryFindSoleConsumer(nodes, consumers, mul1Node.Output[0], out int mul2Idx)) continue;
            var mul2Node = nodes[mul2Idx];
            if (mul2Node.OpType != "Mul") continue;
            if (!TryValidateMul2Stage(mul2Node, mul1Node.Output[0], scalarConsts)) continue;

            // All stages validated. Register replacement at the earliest
            // source index so the emitted Gelu lands where the chain
            // started; later topo-sort handles ordering regardless.
            int earliest = Math.Min(Math.Min(divIdx, i), Math.Min(Math.Min(addIdx, mul1Idx), mul2Idx));
            var geluNode = new NodeProto { OpType = "Gelu" };
            geluNode.Input.Add(xName!);
            geluNode.Output.Add(mul2Node.Output[0]);
            replacements[earliest] = geluNode;
            foreach (var idx in new[] { divIdx, i, addIdx, mul1Idx, mul2Idx })
            {
                if (idx != earliest) toSkip.Add(idx);
            }
        }

        // Second pass: emit in source order, substituting Gelu at the
        // earliest-index position of each match and skipping the rest.
        var result = new List<NodeProto>(nodes.Count - toSkip.Count);
        for (int i = 0; i < nodes.Count; i++)
        {
            if (toSkip.Contains(i)) continue;
            if (replacements.TryGetValue(i, out var gelu)) result.Add(gelu);
            else result.Add(nodes[i]);
        }
        return result;
    }

    /// <summary>
    /// Resolves the <c>x</c> operand of the chain's entry op (Div or Mul
    /// by a constant) and verifies the constant is √2 for Div or 1/√2 for
    /// Mul. Returns false if the node shape, constant value, or operand
    /// role doesn't match the expected GELU form.
    /// </summary>
    private static bool TryResolveDivStage(
        NodeProto divNode,
        Dictionary<string, double> scalarConsts,
        out string? xName)
    {
        xName = null;
        if (divNode.Input.Count < 2) return false;

        if (divNode.OpType == "Div")
        {
            // Div is non-commutative: Div(numerator, denominator).
            // x must be the numerator, √2 the denominator.
            if (!scalarConsts.TryGetValue(divNode.Input[1], out double denom))
                return false;
            if (!NearlyEqual(denom, Sqrt2)) return false;
            // Guard against Div(const, x) where Input[0] is itself a
            // constant — that's not the GELU form.
            if (scalarConsts.ContainsKey(divNode.Input[0])) return false;
            xName = divNode.Input[0];
            return true;
        }

        if (divNode.OpType == "Mul")
        {
            // Mul is commutative — pick whichever input is the 1/√2
            // constant; the OTHER input must be the non-constant x.
            if (TryGetConstInput(divNode, scalarConsts, out int constIdx, out double constVal))
            {
                if (!NearlyEqual(constVal, InvSqrt2)) return false;
                xName = divNode.Input[1 - constIdx];
                // Reject if "non-const" operand is also a scalar constant
                // (can't happen with a well-formed lookup — this is a
                // belt-and-suspenders check against degenerate graphs).
                if (scalarConsts.ContainsKey(xName)) return false;
                return true;
            }
            return false;
        }

        return false;
    }

    /// <summary>
    /// Verifies that <paramref name="addNode"/> is <c>Add(erfOut, 1.0)</c>
    /// in either operand order: one input must equal <paramref name="erfOutName"/>
    /// and the other must be a scalar constant ≈ 1.0.
    /// </summary>
    private static bool TryValidateAddStage(
        NodeProto addNode,
        string erfOutName,
        Dictionary<string, double> scalarConsts)
    {
        if (addNode.Input.Count < 2) return false;
        string a = addNode.Input[0], b = addNode.Input[1];
        if (a == erfOutName)
        {
            return scalarConsts.TryGetValue(b, out double val) && NearlyEqual(val, One);
        }
        if (b == erfOutName)
        {
            return scalarConsts.TryGetValue(a, out double val) && NearlyEqual(val, One);
        }
        return false;
    }

    /// <summary>
    /// Verifies that <paramref name="mul1Node"/> is <c>Mul(addOut, x)</c>
    /// in either operand order: one input must equal <paramref name="addOutName"/>
    /// and the OTHER must equal <paramref name="xName"/>. This is the
    /// check that stops a commuted <c>Mul(const, something)</c> from
    /// sneaking through.
    /// </summary>
    private static bool TryValidateMul1Stage(
        NodeProto mul1Node, string addOutName, string xName)
    {
        if (mul1Node.Input.Count < 2) return false;
        string a = mul1Node.Input[0], b = mul1Node.Input[1];
        return (a == addOutName && b == xName) || (a == xName && b == addOutName);
    }

    /// <summary>
    /// Verifies that <paramref name="mul2Node"/> is <c>Mul(mul1Out, 0.5)</c>
    /// in either operand order: one input must equal
    /// <paramref name="mul1OutName"/> and the other must be a scalar
    /// constant ≈ 0.5.
    /// </summary>
    private static bool TryValidateMul2Stage(
        NodeProto mul2Node, string mul1OutName, Dictionary<string, double> scalarConsts)
    {
        if (mul2Node.Input.Count < 2) return false;
        string a = mul2Node.Input[0], b = mul2Node.Input[1];
        if (a == mul1OutName)
        {
            return scalarConsts.TryGetValue(b, out double val) && NearlyEqual(val, Half);
        }
        if (b == mul1OutName)
        {
            return scalarConsts.TryGetValue(a, out double val) && NearlyEqual(val, Half);
        }
        return false;
    }

    /// <summary>
    /// For a commutative binary op, returns whichever input is a scalar
    /// initializer constant and the output index (0 or 1) of that input.
    /// Returns false if neither or both inputs are constant (the "both
    /// constant" case is already degenerate — the producer should have
    /// constant-folded it).
    /// </summary>
    private static bool TryGetConstInput(
        NodeProto node,
        Dictionary<string, double> scalarConsts,
        out int constIdx, out double constVal)
    {
        bool a = scalarConsts.TryGetValue(node.Input[0], out double va);
        bool b = scalarConsts.TryGetValue(node.Input[1], out double vb);
        if (a && !b) { constIdx = 0; constVal = va; return true; }
        if (!a && b) { constIdx = 1; constVal = vb; return true; }
        constIdx = -1; constVal = 0; return false;
    }

    /// <summary>
    /// Extracts scalar (single-element) float/double initializers into a
    /// name → double lookup. Non-scalar initializers are skipped —
    /// GELU constants are always scalars, so a broadcast tensor named
    /// like a scalar would still be rejected via the scalar-check here.
    /// </summary>
    private static Dictionary<string, double> BuildScalarConstants(
        IReadOnlyList<TensorProto>? initializers)
    {
        var dict = new Dictionary<string, double>(StringComparer.Ordinal);
        if (initializers is null) return dict;
        for (int i = 0; i < initializers.Count; i++)
        {
            var init = initializers[i];
            if (string.IsNullOrEmpty(init.Name)) continue;
            if (!IsScalarShape(init)) continue;
            if (TryReadScalarAsDouble(init, out double val))
                dict[init.Name] = val;
        }
        return dict;
    }

    private static bool IsScalarShape(TensorProto proto)
    {
        // Dims.Count == 0 is the canonical scalar. Count == 1 with dim[0]==1
        // is the common rank-1 singleton that exporters emit for broadcast
        // constants. Both collapse to a single element.
        if (proto.Dims.Count == 0) return true;
        long total = 1;
        for (int i = 0; i < proto.Dims.Count; i++) total *= proto.Dims[i];
        return total == 1;
    }

    // ONNX TensorProto.DataType enum values (mirror InitializerLoader's
    // private constants; duplicated here to avoid widening visibility).
    private const int DT_FLOAT = 1;
    private const int DT_DOUBLE = 11;

    private static bool TryReadScalarAsDouble(TensorProto proto, out double value)
    {
        value = 0;
        if (proto.DataType == DT_FLOAT)
        {
            if (!proto.RawData.IsEmpty && proto.RawData.Length >= sizeof(float))
            {
                value = MemoryMarshal.Cast<byte, float>(proto.RawData.Span)[0];
                return true;
            }
            if (proto.FloatData.Count >= 1)
            {
                value = proto.FloatData[0];
                return true;
            }
        }
        else if (proto.DataType == DT_DOUBLE)
        {
            if (!proto.RawData.IsEmpty && proto.RawData.Length >= sizeof(double))
            {
                value = MemoryMarshal.Cast<byte, double>(proto.RawData.Span)[0];
                return true;
            }
            if (proto.DoubleData.Count >= 1)
            {
                value = proto.DoubleData[0];
                return true;
            }
        }
        return false;
    }

    private static bool NearlyEqual(double a, double b)
        => Math.Abs(a - b) <= ConstTol * Math.Max(1.0, Math.Abs(b));

    private static int GetCount(Dictionary<string, int> counts, string name)
    {
        return counts.TryGetValue(name, out int c) ? c : 0;
    }

    private static bool TryFindSoleConsumer(
        IReadOnlyList<NodeProto> nodes,
        Dictionary<string, int> consumers,
        string outputName,
        out int consumerIdx)
    {
        consumerIdx = -1;
        if (GetCount(consumers, outputName) != 1) return false;
        for (int i = 0; i < nodes.Count; i++)
        {
            var n = nodes[i];
            for (int j = 0; j < n.Input.Count; j++)
            {
                if (n.Input[j] == outputName)
                {
                    consumerIdx = i;
                    return true;
                }
            }
        }
        return false;
    }
}
