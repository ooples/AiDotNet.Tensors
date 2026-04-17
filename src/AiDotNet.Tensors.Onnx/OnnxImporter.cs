using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Top-level ONNX import entry point. Parses an ONNX protobuf model and
/// produces a compiled inference plan that runs through the AiDotNet.Tensors
/// fusion + autotune stack.
///
/// <para>Usage:</para>
/// <code>
/// using var stream = File.OpenRead("model.onnx");
/// var engine = new CpuEngine();
/// var result = OnnxImporter.Import&lt;float&gt;(stream, engine);
/// if (result.Plan is null)
///     throw new Exception($"Unsupported ops: {string.Join(", ", result.UnsupportedOperators)}");
/// // Fill result.Inputs["input_name"] with input data, then:
/// var output = result.Plan.Execute();
/// </code>
///
/// <para>Phase 1 operator coverage (Issue #169): MatMul, Gemm, Add, Mul, Relu,
/// Sigmoid, Tanh, Softmax, GELU, LayerNormalization, BatchNormalization, Conv,
/// ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, Reshape, Transpose,
/// Slice, Concat, Split, Gather, Squeeze, Unsqueeze, and com.microsoft.Attention.
/// Unsupported operators are collected into
/// <see cref="OnnxImportResult{T}.UnsupportedOperators"/> unless
/// <see cref="OnnxImportOptions.StrictMode"/> is set.</para>
/// </summary>
public static class OnnxImporter
{
    /// <summary>
    /// Parses an ONNX model from a stream and produces a compiled inference
    /// plan. Currently supports <c>T = float</c> end-to-end; <c>T = double</c>
    /// converts float initializers on load.
    /// </summary>
    public static OnnxImportResult<T> Import<T>(
        Stream onnxModel,
        IEngine engine,
        OnnxImportOptions? options = null)
        where T : unmanaged
    {
        if (onnxModel is null) throw new ArgumentNullException(nameof(onnxModel));
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        options ??= new OnnxImportOptions();

        // ── Parse protobuf ────────────────────────────────────────────────
        var model = ModelProto.Parser.ParseFrom(onnxModel);
        var graph = model.Graph ?? throw new InvalidDataException(
            "ONNX model has no graph. The file may be a weights-only export rather than a full model.");

        // ── Materialize initializers as Tensor<T> ─────────────────────────
        var tensorsByName = new Dictionary<string, Tensor<T>>(graph.Initializer.Count + graph.Input.Count);
        for (int i = 0; i < graph.Initializer.Count; i++)
        {
            var init = graph.Initializer[i];
            tensorsByName[init.Name] = InitializerLoader.Load<T>(init);
        }

        // ── Resolve graph inputs (skip any that are also initializers;
        //    older ONNX specs duplicated them) ──────────────────────────
        var namedInputs = new Dictionary<string, ShapeProfile>(graph.Input.Count);
        var inputTensors = new Dictionary<string, Tensor<T>>(graph.Input.Count);
        // dynamicTensorNames tracks tensors whose values are only known at
        // Execute time (graph inputs). They're eager storage-wise but
        // constant-folding MUST NOT walk through them — downstream ops
        // that depend on graph inputs have to stay under GraphMode.
        var dynamicTensorNames = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < graph.Input.Count; i++)
        {
            var vi = graph.Input[i];
            if (tensorsByName.ContainsKey(vi.Name))
                continue; // Graph input that's also an initializer — skip.

            var shape = ResolveShape(vi, options);
            var elemType = ElementTypeName(vi.Type?.TensorType?.ElemType ?? 0);
            namedInputs[vi.Name] = new ShapeProfile(vi.Name, shape, elemType);

            // Allocate a placeholder tensor for this graph input. Users write
            // data into it before Execute() — the compiled plan references
            // this tensor's storage.
            var placeholder = new Tensor<T>(shape);
            tensorsByName[vi.Name] = placeholder;
            inputTensors[vi.Name] = placeholder;
            dynamicTensorNames.Add(vi.Name);
        }

        // ── Resolve graph outputs (shape info only; not actually allocated) ─
        // Graph outputs don't always carry declared shape metadata —
        // intermediate-tensor names promoted by diagnostic tools (or
        // ONNX-Runtime output-naming conventions) often omit it. We
        // tolerate null shapes here since output shape is known at
        // Execute time from the producing step.
        var namedOutputs = new Dictionary<string, ShapeProfile>(graph.Output.Count);
        for (int i = 0; i < graph.Output.Count; i++)
        {
            var vo = graph.Output[i];
            int[] shape;
            try { shape = ResolveShape(vo, options); }
            catch (InvalidDataException)
            {
                // Missing shape info on outputs is tolerated — set empty,
                // caller reads the actual shape off the produced tensor.
                shape = Array.Empty<int>();
            }
            var elemType = ElementTypeName(vo.Type?.TensorType?.ElemType ?? 0);
            namedOutputs[vo.Name] = new ShapeProfile(vo.Name, shape, elemType);
        }

        // ── Pre-pass: rewrite BERT-style GELU erf decomposition to fused
        //    Gelu. Exports from tf2onnx / torch <= opset 19 emit
        //    GELU(x) = 0.5·x·(1 + erf(x/√2)) as 5 ops (Div → Erf → Add → Mul
        //    → Mul). Our A-S Erf approximation has only 1.5e-7 absolute
        //    error but the engine's fused Gelu uses the same rational
        //    approximation as ORT — rewriting gives bit-exact parity on
        //    BERT logits instead of compounding through 12 transformer
        //    layers. Only fires when the input to Div is a regular tensor
        //    (not a constant) and the output of the final Mul has no other
        //    consumers, so rewrite is semantically equivalent.
        var rewrittenNodes = GeluPatternRewriter.Rewrite(graph.Node);

        // ── Topologically sort nodes ──────────────────────────────────────
        var sortedNodes = TopologicalSort(graph, tensorsByName.Keys, rewrittenNodes);

        // ── Dispatch translators for each node under GraphMode ────────────
        var registry = OnnxOpTranslatorRegistry<T>.BuildDefault();
        var unsupported = new List<string>();

        // First pass: catalog unsupported operators. If any, we return early
        // (unless the caller wants a partial trace, which we don't offer).
        for (int i = 0; i < sortedNodes.Count; i++)
        {
            var node = sortedNodes[i];
            var t = registry.Find(node.OpType, string.IsNullOrEmpty(node.Domain) ? null : node.Domain);
            if (t is null)
            {
                string key = string.IsNullOrEmpty(node.Domain) ? node.OpType : $"{node.Domain}.{node.OpType}";
                if (!unsupported.Contains(key))
                    unsupported.Add(key);
            }
        }

        if (unsupported.Count > 0)
        {
            if (options.StrictMode)
                throw new NotSupportedException(
                    "ONNX model contains operators with no registered translator: " +
                    string.Join(", ", unsupported) +
                    ". Register a custom IOnnxOpTranslator<T> or disable StrictMode to get a null Plan with diagnostics.");
            return new OnnxImportResult<T>(
                plan: null,
                unsupportedOperators: unsupported,
                namedInputs: namedInputs,
                namedOutputs: namedOutputs,
                inputs: inputTensors,
                outputs: new Dictionary<string, Tensor<T>>(),
                producerName: model.ProducerName ?? string.Empty,
                irVersion: model.IrVersion);
        }

        // ── Trace the graph under GraphMode and compile ───────────────────
        //
        // ONNX models come from every ML toolchain (PyTorch, TF, JAX, ORT,
        // Hugging Face optimum) — we can't assume graph shapes look like the
        // in-house compiler's fusion patterns expect. Disable dataflow fusion
        // and spectral decomposition for imported plans; they silently
        // coalesce multi-op chains in ways that break when the graph
        // producer doesn't match the exact shape the pass was written for.
        // Individual op-level translators still route through all the fast
        // paths inside CpuEngine (SIMD, BLAS-backed MatMul, fused Conv).
        CompiledInferencePlan<T> plan;
        var prevOpts = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions
        {
            EnableDataflowFusion = false,
            EnableSpectralDecomposition = false,
        });
        try
        {
            using var scope = GraphMode.Enable();
            var ctx = new OnnxTranslationContext<T>(engine, tensorsByName, options);
            for (int i = 0; i < sortedNodes.Count; i++)
            {
                var node = sortedNodes[i];
                var translator = registry.Find(node.OpType,
                    string.IsNullOrEmpty(node.Domain) ? null : node.Domain)!;

                // Constant-folding: if every input to this node is a
                // *static* eager tensor — an initializer or an already-
                // folded op's output, but NOT a graph-input placeholder —
                // execute the translator outside GraphMode so its result
                // is also static. This is how BERT-family shape-arithmetic
                // subgraphs (Shape → Gather → Concat → Reshape) become
                // constant tensors at import time, letting downstream ops
                // like Reshape / OneHot / ConstantOfShape read their
                // integer values via AsSpan without triggering Realize.
                //
                // Graph input placeholders are storage-wise eager but are
                // tracked in dynamicTensorNames and break the fold — any
                // subgraph that reaches a graph input has to stay under
                // GraphMode.
                if (AllInputsEagerAndStatic(node, tensorsByName, dynamicTensorNames))
                {
                    var savedScope = GraphMode.Current;
                    GraphMode.SetCurrent(null);
                    try
                    {
                        translator.Translate(ctx, node);
                        // Mark every output as static so further nodes in
                        // the fold-chain see it as foldable.
                        foreach (var outName in node.Output)
                            if (!string.IsNullOrEmpty(outName) && tensorsByName.TryGetValue(outName, out var t))
                                t.LazySource = null; // belt-and-suspenders; eager ops already produce LazySource=null
                    }
                    finally { GraphMode.SetCurrent(savedScope); }
                }
                else
                {
                    translator.Translate(ctx, node);
                }
            }
            // Every declared graph output needs to survive to the end of
            // Plan.Execute(). Two problems the wrap handles:
            //
            //  1. Constant-folded outputs have no plan step, so Execute()
            //     would return the empty fallback tensor.
            //  2. Memory planning reuses buffers across non-overlapping op
            //     lifetimes. If a node's output is consumed early by
            //     downstream steps, the planner is free to reuse the same
            //     storage for a later step — and the "output" tensor the
            //     importer handed to result.Outputs then reads whatever
            //     happens to be there at the end of execution.
            //
            // Adding a TensorAdd(x, 0) at the very end binds the output to
            // a fresh buffer whose lifetime definitionally extends through
            // the last plan step. This matches how ONNX Runtime handles
            // output liveness (outputs are always live).
            foreach (var outputVi in graph.Output)
            {
                if (!tensorsByName.TryGetValue(outputVi.Name, out var outTensor)) continue;
                var zero = new Tensor<T>(outTensor._shape);
                var wrapped = engine.TensorAdd(outTensor, zero);
                tensorsByName[outputVi.Name] = wrapped;
            }

            plan = scope.CompileInference<T>();
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prevOpts);
        }

        // Collect the declared graph outputs' tensors for the result.
        // After plan.Execute() runs, each of these holds the final computed
        // value for the corresponding ONNX output name.
        var outputTensors = new Dictionary<string, Tensor<T>>(graph.Output.Count);
        foreach (var vo in graph.Output)
            if (tensorsByName.TryGetValue(vo.Name, out var t))
                outputTensors[vo.Name] = t;

        return new OnnxImportResult<T>(
            plan: plan,
            unsupportedOperators: Array.Empty<string>(),
            namedInputs: namedInputs,
            namedOutputs: namedOutputs,
            inputs: inputTensors,
            outputs: outputTensors,
            producerName: model.ProducerName ?? string.Empty,
            irVersion: model.IrVersion);
    }

    /// <summary>
    /// Returns true when every non-empty input of the node is a STATIC
    /// eager tensor — its LazySource is null AND it isn't a graph-input
    /// placeholder. Initializers and outputs of previously-folded ops
    /// qualify; graph inputs do not because their values are only known
    /// at Execute time.
    /// </summary>
    private static bool AllInputsEagerAndStatic<T>(
        NodeProto node,
        Dictionary<string, Tensor<T>> tensorsByName,
        HashSet<string> dynamicTensorNames)
    {
        for (int i = 0; i < node.Input.Count; i++)
        {
            var name = node.Input[i];
            if (string.IsNullOrEmpty(name)) continue;
            if (dynamicTensorNames.Contains(name)) return false;
            if (!tensorsByName.TryGetValue(name, out var t)) return false;
            if (t.LazySource is not null) return false;
        }
        return true;
    }

    // ─── Shape resolution ────────────────────────────────────────────────

    private static int[] ResolveShape(ValueInfoProto vi, OnnxImportOptions options)
    {
        // Full-shape override wins if present.
        if (options.OverrideInputShapes is not null &&
            options.OverrideInputShapes.TryGetValue(vi.Name, out var full))
        {
            return (int[])full.Clone();
        }

        var ttype = vi.Type?.TensorType;
        if (ttype?.Shape is null)
            throw new InvalidDataException(
                $"ONNX value '{vi.Name}' has no shape metadata. " +
                "Phase 1 import requires declared input shapes.");

        var dims = ttype.Shape.Dim;
        var shape = new int[dims.Count];
        for (int i = 0; i < dims.Count; i++)
        {
            var d = dims[i];
            if (d.ValueCase == TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue)
            {
                shape[i] = checked((int)d.DimValue);
                continue;
            }
            // Parametric dim — look up via DimensionOverrides.
            if (d.ValueCase == TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam &&
                options.DimensionOverrides is not null &&
                options.DimensionOverrides.TryGetValue(d.DimParam, out var resolved))
            {
                shape[i] = resolved;
                continue;
            }
            // Unresolved parametric dim. Fall back to DefaultParametricDim
            // when the caller opted in — common for Hugging Face exports
            // that autogenerate "unk__NNN" names for dims that would be
            // obvious to a human (batch=1, etc.).
            if (options.DefaultParametricDim.HasValue)
            {
                shape[i] = options.DefaultParametricDim.Value;
                continue;
            }
            throw new InvalidOperationException(
                $"ONNX input '{vi.Name}' dim {i} is parametric ('{d.DimParam}') and was not resolved " +
                "via OnnxImportOptions.DimensionOverrides, OverrideInputShapes, or DefaultParametricDim.");
        }
        return shape;
    }

    private static string ElementTypeName(int elemType) => elemType switch
    {
        0 => "undefined",
        1 => "float",
        2 => "uint8",
        3 => "int8",
        4 => "uint16",
        5 => "int16",
        6 => "int32",
        7 => "int64",
        8 => "string",
        9 => "bool",
        10 => "float16",
        11 => "double",
        12 => "uint32",
        13 => "uint64",
        14 => "complex64",
        15 => "complex128",
        16 => "bfloat16",
        _ => $"unknown({elemType})",
    };

    // ─── Topological sort ────────────────────────────────────────────────

    private static List<NodeProto> TopologicalSort(GraphProto graph, IEnumerable<string> seeded, IReadOnlyList<NodeProto>? nodesOverride = null)
    {
        var nodeList = nodesOverride ?? (IReadOnlyList<NodeProto>)graph.Node;
        return TopologicalSortCore(nodeList, seeded);
    }

    private static List<NodeProto> TopologicalSortCore(IReadOnlyList<NodeProto> nodeList, IEnumerable<string> seeded)
    {
        // Seeded = tensor names already in the registry (initializers +
        // graph inputs). Any node whose inputs are all seeded can run first.
        var producedBy = new Dictionary<string, int>(); // output-name → node index
        for (int i = 0; i < nodeList.Count; i++)
        {
            var n = nodeList[i];
            for (int j = 0; j < n.Output.Count; j++)
                producedBy[n.Output[j]] = i;
        }

        var available = new HashSet<string>(seeded, StringComparer.Ordinal);
        var visited = new bool[nodeList.Count];
        var sorted = new List<NodeProto>(nodeList.Count);

        // Iterative Kahn's-like sweep. Each pass emits every node whose inputs
        // are all available and marks its outputs as available for the next
        // pass. Guarded against infinite loops by requiring forward progress.
        while (sorted.Count < nodeList.Count)
        {
            bool progress = false;
            for (int i = 0; i < nodeList.Count; i++)
            {
                if (visited[i]) continue;
                var n = nodeList[i];
                bool ready = true;
                for (int j = 0; j < n.Input.Count; j++)
                {
                    // Empty input names are legal in ONNX — they mark an
                    // optional input that the node is skipping (e.g. Conv
                    // without bias).
                    if (string.IsNullOrEmpty(n.Input[j])) continue;
                    if (!available.Contains(n.Input[j])) { ready = false; break; }
                }
                if (!ready) continue;

                visited[i] = true;
                sorted.Add(n);
                for (int j = 0; j < n.Output.Count; j++)
                    if (!string.IsNullOrEmpty(n.Output[j]))
                        available.Add(n.Output[j]);
                progress = true;
            }
            if (!progress)
            {
                // Cycle or dangling input. List the first stuck node's
                // missing input for diagnostics.
                for (int i = 0; i < nodeList.Count; i++)
                {
                    if (visited[i]) continue;
                    var n = nodeList[i];
                    for (int j = 0; j < n.Input.Count; j++)
                    {
                        if (string.IsNullOrEmpty(n.Input[j])) continue;
                        if (!available.Contains(n.Input[j]))
                        {
                            throw new InvalidDataException(
                                $"ONNX graph has a dangling input: node '{n.Name}' (op {n.OpType}) " +
                                $"references tensor '{n.Input[j]}' which is neither a graph input, " +
                                $"an initializer, nor an upstream node's output.");
                        }
                    }
                }
                throw new InvalidDataException(
                    "ONNX graph topological sort failed without identifying a dangling input — the graph may contain a cycle.");
            }
        }
        return sorted;
    }
}
