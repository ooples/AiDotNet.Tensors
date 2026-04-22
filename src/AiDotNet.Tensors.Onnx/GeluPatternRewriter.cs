using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Detects the pre-opset-20 GELU-via-Erf decomposition that BERT / XLNet /
/// RoBERTa / T5 exports emit and rewrites it to a single <c>Gelu</c> node.
///
/// <para>The typical 5-node chain (numbers are ONNX op types):</para>
/// <code>
/// t0 = Div(x, √2)              // constant 1.41421356…
/// t1 = Erf(t0)
/// t2 = Add(t1, 1.0)            // constant 1.0
/// t3 = Mul(x, t2) OR Mul(x, 0.5) first
/// y  = Mul(t3, 0.5) OR Mul(t3, t_half)
/// </code>
///
/// <para>Our A-S 7.1.26 Erf approximation has max error 1.5e-7 which is
/// bit-accurate enough in isolation, but compounds through 12 transformer
/// layers × 768+ hidden dims. The engine's fused <c>Gelu</c> uses the exact
/// same kernel ORT does, so a pattern rewrite gives bit-exact logit
/// parity instead of ~5-6 absolute divergence.</para>
///
/// <para>Rewriting is safe when the intermediate tensors (t0, t1, t2, t3)
/// have exactly one consumer — the next link in the chain. If any
/// intermediate is also consumed by an unrelated op, the rewrite is
/// suppressed so downstream consumers still see the correct value.</para>
/// </summary>
internal static class GeluPatternRewriter
{
    /// <summary>
    /// Look up the scalar value of an initializer by tensor name. Returns
    /// <c>null</c> when the name doesn't name an initializer or the
    /// initializer isn't a single-element scalar.
    /// </summary>
    internal delegate double? ScalarInitLookup(string tensorName);

    internal static IReadOnlyList<NodeProto> Rewrite(IReadOnlyList<NodeProto> nodes)
        => Rewrite(nodes, static _ => null);

    /// <summary>
    /// Rewrite the Div/Mul → Erf → Add → Mul → Mul chain into a single
    /// <c>Gelu</c> op only when the three embedded constants match the
    /// GELU formula (√2 in the Div/Mul, 1.0 in the Add, 0.5 in one of the
    /// Muls). Without the constant check we were rewriting structurally
    /// identical but semantically different subgraphs to Gelu — silently
    /// changing numerics. Callers pass a <paramref name="scalarLookup"/>
    /// that returns the initializer scalar by name or <c>null</c> if the
    /// tensor isn't a known scalar constant.
    /// </summary>
    internal static IReadOnlyList<NodeProto> Rewrite(
        IReadOnlyList<NodeProto> nodes,
        ScalarInitLookup scalarLookup)
    {
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
            if (divNode.OpType != "Div" && divNode.OpType != "Mul") continue;
            if (divNode.Input.Count < 2) continue;
            if (GetCount(consumers, divNode.Output[0]) != 1) continue;

            // The Div/Mul carries the √2 (or 1/√2 for the Mul form). Identify
            // x = the non-constant operand, verify the constant is √2 ±1e-3
            // (Div) or 1/√2 ±1e-3 (Mul). Either operand position is valid.
            string xName;
            if (!TryIdentifyX(divNode, scalarLookup, out xName, out double divConst)) continue;
            const double Sqrt2 = 1.41421356237309504880;
            const double InvSqrt2 = 0.70710678118654752440;
            if (divNode.OpType == "Div")
            {
                if (Math.Abs(divConst - Sqrt2) > 1e-3) continue;
            }
            else
            {
                // Mul(x, 1/√2) is mathematically the same as Div(x, √2).
                if (Math.Abs(divConst - InvSqrt2) > 1e-3) continue;
            }

            if (!TryFindSoleConsumer(nodes, consumers, n.Output[0], out int addIdx)) continue;
            var addNode = nodes[addIdx];
            if (addNode.OpType != "Add") continue;
            if (addNode.Input.Count < 2) continue;
            // Add(Erf(…), 1.0) — the non-Erf operand must be scalar 1.0.
            string erfOut = n.Output[0];
            string addConstName = addNode.Input[0] == erfOut ? addNode.Input[1] :
                                  addNode.Input[1] == erfOut ? addNode.Input[0] : null!;
            if (addConstName is null) continue;
            double? addConst = scalarLookup(addConstName);
            if (addConst is null || Math.Abs(addConst.Value - 1.0) > 1e-3) continue;

            if (!TryFindSoleConsumer(nodes, consumers, addNode.Output[0], out int mul1Idx)) continue;
            var mul1Node = nodes[mul1Idx];
            if (mul1Node.OpType != "Mul") continue;

            if (!TryFindSoleConsumer(nodes, consumers, mul1Node.Output[0], out int mul2Idx)) continue;
            var mul2Node = nodes[mul2Idx];
            if (mul2Node.OpType != "Mul") continue;

            // Across the final two Muls, exactly one operand must be the 0.5
            // constant and the pair of non-constant operands must be {x, Add_out}.
            if (!HasGeluFinalMuls(mul1Node, mul2Node, xName, addNode.Output[0], scalarLookup)) continue;

            // Pattern matched. Register replacement at the earliest source
            // index so the emitted Gelu lands where Div was in the original
            // stream; later topo-sort handles ordering correctly regardless.
            int earliest = Math.Min(Math.Min(divIdx, i), Math.Min(Math.Min(addIdx, mul1Idx), mul2Idx));
            var geluNode = new NodeProto { OpType = "Gelu" };
            geluNode.Input.Add(xName);
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

    /// <summary>
    /// Identify the non-constant input of a Div/Mul node where the other
    /// operand is a scalar initializer. Returns false (+ zeros) when
    /// neither operand is a recognizable scalar (safe: rewrite skips).
    /// </summary>
    private static bool TryIdentifyX(NodeProto divMul, ScalarInitLookup scalarLookup, out string xName, out double constVal)
    {
        xName = string.Empty;
        constVal = 0.0;
        double? lhs = scalarLookup(divMul.Input[0]);
        double? rhs = scalarLookup(divMul.Input[1]);
        if (lhs is null && rhs is not null)
        {
            xName = divMul.Input[0];
            constVal = rhs.Value;
            return true;
        }
        if (rhs is null && lhs is not null)
        {
            // For Div, this means Div(const, x) which is NOT the GELU pattern
            // (GELU needs x as the dividend). For Mul it's symmetric and OK.
            if (divMul.OpType == "Div") return false;
            xName = divMul.Input[1];
            constVal = lhs.Value;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Verify the last two Muls of the GELU chain match the structure
    /// <c>Mul(Add_out, 0.5) → Mul(x, that)</c> OR <c>Mul(x, 0.5) →
    /// Mul(that, Add_out)</c> — order is not fixed by ONNX exporters.
    /// </summary>
    private static bool HasGeluFinalMuls(
        NodeProto mul1, NodeProto mul2, string xName, string addOut,
        ScalarInitLookup scalarLookup)
    {
        if (mul1.Input.Count < 2 || mul2.Input.Count < 2) return false;

        // Gather the pair of non-constant operands across both Muls.
        // One Mul must have a scalar 0.5 input; the other must not.
        double? half1 = FindHalfOperand(mul1, scalarLookup, out string? mul1Other);
        double? half2 = FindHalfOperand(mul2, scalarLookup, out string? mul2Other);
        bool mul1HasHalf = half1 is not null && Math.Abs(half1.Value - 0.5) < 1e-3;
        bool mul2HasHalf = half2 is not null && Math.Abs(half2.Value - 0.5) < 1e-3;
        // Exactly one of the two Muls should carry the 0.5 constant.
        if (mul1HasHalf == mul2HasHalf) return false;

        // The non-0.5 operand of the "half" Mul must be either x or addOut.
        // The other Mul's pair of non-constant inputs must be {mul1.Output, the remaining of {x, addOut}}.
        string halfMulOther = mul1HasHalf ? mul1Other! : mul2Other!;
        if (halfMulOther != xName && halfMulOther != addOut) return false;

        string expectedPartner = halfMulOther == xName ? addOut : xName;
        NodeProto otherMul = mul1HasHalf ? mul2 : mul1;
        NodeProto halfMul = mul1HasHalf ? mul1 : mul2;
        // otherMul's inputs must be { halfMul.Output, expectedPartner }.
        bool a = otherMul.Input[0] == halfMul.Output[0] && otherMul.Input[1] == expectedPartner;
        bool b = otherMul.Input[1] == halfMul.Output[0] && otherMul.Input[0] == expectedPartner;
        return a || b;
    }

    private static double? FindHalfOperand(NodeProto mul, ScalarInitLookup scalarLookup, out string? otherInput)
    {
        double? lhs = scalarLookup(mul.Input[0]);
        double? rhs = scalarLookup(mul.Input[1]);
        if (lhs is not null && rhs is null) { otherInput = mul.Input[1]; return lhs; }
        if (rhs is not null && lhs is null) { otherInput = mul.Input[0]; return rhs; }
        otherInput = null;
        return null;
    }
}
