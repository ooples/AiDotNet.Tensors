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

## Phase 1 operator coverage

| Family | Ops |
|---|---|
| Arithmetic | `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div` |
| Activation | `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `Gelu`, `LeakyRelu` |
| Normalization | `LayerNormalization`, `BatchNormalization` (inference) |
| Convolution | `Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Tensor manipulation | `Reshape`, `Transpose`, `Slice`, `Concat`, `Split`, `Gather`, `Squeeze`, `Unsqueeze`, `Constant`, `Flatten`, `Identity`, `Shape`, `Cast` (same-type) |
| Attention | `com.microsoft.Attention` |

Supported attributes cover the common set used by ResNet-50 / BERT-base
/ ViT-B/16 exports: `auto_pad` (NOTSET / VALID / SAME_UPPER / SAME_LOWER),
symmetric `pads`, `strides`, `dilations`, `kernel_shape`, `group=1`,
Gemm `alpha`/`beta`/`transA`/`transB`, Softmax `axis`, LeakyRelu
`alpha`, Slice with unit step, LayerNorm last-axis.

## Known limitations (Phase 2 / 3)

- Grouped / depthwise convolution (`group > 1`) — MobileNet support
  follows in Phase 2.
- Asymmetric padding, `ceil_mode = 1`.
- Non-unit `Slice` strides.
- Bare `Erf` op (outside the opset-20 `Gelu` fused form).
- Quantized ops (`QLinearConv`, `QLinearMatMul`) — Phase 2.
- Recurrent ops (`LSTM`, `GRU`) — Phase 2.
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
