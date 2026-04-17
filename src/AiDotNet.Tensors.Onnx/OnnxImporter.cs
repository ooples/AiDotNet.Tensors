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
        }

        // ── Resolve graph outputs (shape info only; not actually allocated) ─
        var namedOutputs = new Dictionary<string, ShapeProfile>(graph.Output.Count);
        for (int i = 0; i < graph.Output.Count; i++)
        {
            var vo = graph.Output[i];
            var shape = ResolveShape(vo, options);
            var elemType = ElementTypeName(vo.Type?.TensorType?.ElemType ?? 0);
            namedOutputs[vo.Name] = new ShapeProfile(vo.Name, shape, elemType);
        }

        // ── Topologically sort nodes ──────────────────────────────────────
        var sortedNodes = TopologicalSort(graph, tensorsByName.Keys);

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
                translator.Translate(ctx, node);
            }
            plan = scope.CompileInference<T>();
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prevOpts);
        }

        return new OnnxImportResult<T>(
            plan: plan,
            unsupportedOperators: Array.Empty<string>(),
            namedInputs: namedInputs,
            namedOutputs: namedOutputs,
            inputs: inputTensors,
            producerName: model.ProducerName ?? string.Empty,
            irVersion: model.IrVersion);
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

    private static List<NodeProto> TopologicalSort(GraphProto graph, IEnumerable<string> seeded)
    {
        // Seeded = tensor names already in the registry (initializers +
        // graph inputs). Any node whose inputs are all seeded can run first.
        var producedBy = new Dictionary<string, int>(); // output-name → node index
        for (int i = 0; i < graph.Node.Count; i++)
        {
            var n = graph.Node[i];
            for (int j = 0; j < n.Output.Count; j++)
                producedBy[n.Output[j]] = i;
        }

        var available = new HashSet<string>(seeded, StringComparer.Ordinal);
        var visited = new bool[graph.Node.Count];
        var sorted = new List<NodeProto>(graph.Node.Count);

        // Iterative Kahn's-like sweep. Each pass emits every node whose inputs
        // are all available and marks its outputs as available for the next
        // pass. Guarded against infinite loops by requiring forward progress.
        while (sorted.Count < graph.Node.Count)
        {
            bool progress = false;
            for (int i = 0; i < graph.Node.Count; i++)
            {
                if (visited[i]) continue;
                var n = graph.Node[i];
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
                for (int i = 0; i < graph.Node.Count; i++)
                {
                    if (visited[i]) continue;
                    var n = graph.Node[i];
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
