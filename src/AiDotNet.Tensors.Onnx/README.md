# AiDotNet.Tensors.Onnx

ONNX model importer for AiDotNet.Tensors. Parses ONNX protobuf files and
produces an `ICompiledPlan<T>` that runs through AiDotNet.Tensors'
fusion + autotune stack — zero runtime dependency on ONNX Runtime.

## Status

Phase 1 (Issue #169). The operator set below is numerically bit-exact
versus `Microsoft.ML.OnnxRuntime` on representative models from the
ONNX model zoo (MNIST verified; ResNet-50 / BERT-base / ViT-B/16
scaffolded — run locally by staging the `.onnx` files per
`EndToEndModelTests`).

## Usage

```csharp
using var stream = File.OpenRead("model.onnx");
var engine = new CpuEngine();
var result = OnnxImporter.Import<float>(stream, engine);
if (result.Plan is null)
    throw new Exception(
        "Unsupported ops: " + string.Join(", ", result.UnsupportedOperators));

// Fill each named input with your per-inference data
var input = result.Inputs["input_name"];
myData.AsSpan().CopyTo(input.AsWritableSpan());

// Execute
var output = result.Plan.Execute();
```

### Resolving parametric dims

Models exported with dynamic axes (e.g. `batch_size`, `sequence_length`,
`N`) must have concrete sizes before the plan can execute:

```csharp
var options = new OnnxImportOptions
{
    DimensionOverrides = new Dictionary<string, int>
    {
        ["batch_size"] = 1,
        ["sequence_length"] = 256,
    },
};
var result = OnnxImporter.Import<float>(stream, engine, options);
```

### Handling unsupported operators

By default the importer collects unsupported ops into
`result.UnsupportedOperators` and returns `Plan = null`. Set
`OnnxImportOptions.StrictMode = true` to make the import throw
instead, with the same diagnostic message.

To extend coverage at runtime:

```csharp
var registry = OnnxOpTranslatorRegistry<float>.BuildDefault();
registry.Register(new MyCustomOpTranslator());
// (Note: the importer builds its own registry per import today; full
// registry-override plumbing is scheduled as a follow-up.)
```

## Operator coverage

| Family | Ops |
|---|---|
| Arithmetic | `MatMul` (incl. 4D×4D), `Gemm`, `Add`, `Sub`, `Mul`, `Div` — all with NumPy broadcasting |
| Activation | `Relu`, `Sigmoid`, `Tanh`, `Softmax` (opset-aware axis default), `Gelu` (both `approximate=none` and `=tanh`), `LeakyRelu` |
| Normalization | `LayerNormalization` (2- and 3-input forms), `BatchNormalization` (inference) |
| Convolution | `Conv` (incl. grouped / depthwise, 2D + 3D, asymmetric pads, `ceil_mode`), `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Tensor manipulation | `Reshape`, `Transpose`, `Slice`, `Concat`, `Split`, `Gather`, `Squeeze`, `Unsqueeze`, `Constant`, `Flatten`, `Identity`, `Shape`, `Cast` |
| Math | `Sqrt`, `Pow`, `Abs`, `Neg`, `Exp`, `Log`, `Erf`, `Reciprocal`, `ReduceSum`, `ReduceMean`, `Min`, `Max`, `OneHot`, `Not`, `Where`, `ConstantOfShape`, `Equal`, `Expand` |
| Recurrent | `LSTM`, `GRU` (forward direction, seq-first layout, default activations) |
| Quantized | `QuantizeLinear`, `DequantizeLinear`, `QLinearMatMul`, `QLinearConv` (per-tensor and per-axis) |
| Attention | `com.microsoft.Attention` (packed 3-in / 1-out form) |

Supported attributes cover what ResNet-50 / BERT-base / ViT-B/16 and the
typical HF / tf2onnx export pipelines emit: `auto_pad` (NOTSET / VALID /
SAME_UPPER / SAME_LOWER), symmetric + asymmetric `pads`, `strides`,
`dilations`, `kernel_shape`, `group`, Gemm `alpha`/`beta`/`transA`/`transB`,
opset-aware Softmax `axis` default (1 pre-opset-13, -1 opset-13+), LeakyRelu
`alpha`, per-axis `QuantizeLinear`/`DequantizeLinear` `axis`, LSTM `direction`,
GRU `direction`, LayerNorm last-axis.

## Known limitations (follow-up)

- Non-unit `Slice` strides.
- Custom recurrent activations, `clip`, LSTM `input_forget`, GRU
  `linear_before_reset` — rejected with `NotSupportedException`.
- Bidirectional / reverse-direction recurrent (`direction != forward`).
- `DynamicQuantizeLinear` (requires runtime min/max reduction; Phase 3).
- Grouped / per-axis `QLinearConv` with `group > 1`.
- Mask round-trip in `ScaledDotProductAttention` — requires extending
  `SavedStateSerializer` with a `Tensor<bool>` tag; plans re-load with
  `mask = null`.

## End-to-end validation

`AiDotNet.Tensors.Onnx.Tests.EndToEndModelTests` validates against
ResNet-50, BERT-base, and ViT-B/16 from the ONNX model zoo. Tests
skip in CI unless the model files are staged locally:

```bash
mkdir -p ~/.aidotnet/onnx-models
curl -L -o ~/.aidotnet/onnx-models/resnet50-v1-7.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx
```

`AIDOTNET_ONNX_MODELS` overrides the cache directory. The MNIST test
(`MnistEndToEndTest`) downloads its 26 KB model on demand and runs in
every CI execution.
