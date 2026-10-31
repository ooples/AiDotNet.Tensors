using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Exports a compiled AiDotNet.Tensors plan (<see cref="ICompiledPlan{T}"/>) as an ONNX model, for Tensors-only users
/// who need to hand a model to ONNX Runtime, TensorRT or any other ONNX consumer.
/// </summary>
/// <remarks>
/// <para>Each recorded engine operation of the plan becomes one ONNX node (or a short, exact decomposition when ONNX
/// has no single operator for it); every tensor the plan reads that no step produces and that is not an input becomes
/// an initializer holding its current value. The operator set covers what <see cref="OnnxImporter"/> lowers ONNX
/// models to, so an imported model exports back to an equivalent ONNX graph; an operation outside it is reported by
/// name in a <see cref="NotSupportedException"/>.</para>
/// <para>Element types: <c>float</c> and <c>double</c>.</para>
/// <code>
/// var imported = OnnxImporter.Import&lt;float&gt;(File.OpenRead("model.onnx"));
/// ModelProto model = OnnxExporter.Export(imported);
/// using var file = File.Create("roundtrip.onnx");
/// model.WriteTo(file);
/// </code>
/// </remarks>
public static class OnnxExporter
{
    /// <summary>Exports an imported model's plan with its original input and output names.</summary>
    public static ModelProto Export<T>(OnnxImportResult<T> imported, OnnxExportOptions? options = null) where T : unmanaged
    {
        if (imported is null) throw new ArgumentNullException(nameof(imported));
        if (imported.Plan is null)
            throw new InvalidOperationException(
                "The import produced no plan (unsupported operators: " + string.Join(", ", imported.UnsupportedOperators) + ").");
        return Export(imported.Plan, imported.Inputs, imported.Outputs, options);
    }

    /// <summary>
    /// Exports <paramref name="plan"/> with named inputs and outputs: <paramref name="inputs"/> are the tensors the plan
    /// reads its per-call data from, <paramref name="outputs"/> the tensors that hold its results after
    /// <see cref="ICompiledPlan{T}.Execute"/>.
    /// </summary>
    public static ModelProto Export<T>(
        ICompiledPlan<T> plan,
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        IReadOnlyDictionary<string, Tensor<T>> outputs,
        OnnxExportOptions? options = null)
        where T : unmanaged
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (outputs.Count == 0) throw new ArgumentException("At least one output is required.", nameof(outputs));
        if (plan is not CompiledInferencePlan<T> inference)
            throw new NotSupportedException(
                $"Only inference plans can be exported to ONNX; got {plan.GetType().Name}.");
        options ??= new OnnxExportOptions();
        var writer = new OnnxGraphWriter<T>(options);
        return writer.Write(inference.ExportSteps, inputs, outputs);
    }

    /// <summary>
    /// Exports <paramref name="plan"/>, naming its traced inputs <c>input</c>, <c>input_1</c>, ... and its result
    /// <c>output</c>.
    /// </summary>
    public static ModelProto Export<T>(ICompiledPlan<T> plan, OnnxExportOptions? options = null) where T : unmanaged
    {
        if (plan is not CompiledInferencePlan<T> inference)
            throw new NotSupportedException(
                $"Only inference plans can be exported to ONNX; got {plan?.GetType().Name ?? "null"}.");
        var inputs = new Dictionary<string, Tensor<T>>();
        var traced = inference.CompiledInputTensors;
        for (int i = 0; i < traced.Length; i++)
            inputs[i == 0 ? "input" : "input_" + i] = traced[i];
        var outputs = new Dictionary<string, Tensor<T>> { ["output"] = inference.FinalOutputBuffer };
        return Export(plan, inputs, outputs, options);
    }

    /// <summary>Exports and writes the model to <paramref name="destination"/> (the stream is left open).</summary>
    public static void Export<T>(OnnxImportResult<T> imported, Stream destination, OnnxExportOptions? options = null) where T : unmanaged
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        Export(imported, options).WriteTo(destination);
    }
}
