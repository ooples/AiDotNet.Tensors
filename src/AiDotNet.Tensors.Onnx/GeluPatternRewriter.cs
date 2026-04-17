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
    internal static IReadOnlyList<NodeProto> Rewrite(IReadOnlyList<NodeProto> nodes)
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

            string xName = divNode.Input[0];

            if (!TryFindSoleConsumer(nodes, consumers, n.Output[0], out int addIdx)) continue;
            var addNode = nodes[addIdx];
            if (addNode.OpType != "Add") continue;

            if (!TryFindSoleConsumer(nodes, consumers, addNode.Output[0], out int mul1Idx)) continue;
            var mul1Node = nodes[mul1Idx];
            if (mul1Node.OpType != "Mul") continue;

            if (!TryFindSoleConsumer(nodes, consumers, mul1Node.Output[0], out int mul2Idx)) continue;
            var mul2Node = nodes[mul2Idx];
            if (mul2Node.OpType != "Mul") continue;

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
}
