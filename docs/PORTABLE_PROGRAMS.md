# Portable tensor programs

`TensorProgram` records what a function computes as calls to public `IEngine` ops with named, typed arguments.
`ProgramPackage` saves that program together with compiled plans for any number of devices. Both live in
`AiDotNet.Tensors.Engines.Export`.

```csharp
var w = /* Tensor<float> [784, 10] */;
var program = TensorProgram.Trace(
    x => AiDotNetEngine.Current.Softmax(AiDotNetEngine.Current.TensorMatMul(x, w)),
    example,                                            // Tensor<float> [32, 784]
    new TraceOptions { Parameters = new object[] { w } });

Console.WriteLine(program);   // readable listing; the batch axis is the symbol s0
var y = program.Run(batch);   // any batch size; any engine; any element type: program.Run(doubleBatch)

var package = new ProgramPackage(program);
package.Precompile(AiDotNetEngine.Current, new[] { new[] { example } });   // optional: start warm
package.Save("classifier.atnp");

var loaded = ProgramPackage.Load("classifier.atnp");
var output = loaded.Execute(AiDotNetEngine.Current, new[] { batch });      // compiled plan or eager replay
if (loaded.IsDirty) loaded.Save("classifier.atnp");                        // it learned a new device
```

## How it compares with torch.export and AOTInductor

| | PyTorch | AiDotNet.Tensors |
|---|---|---|
| Portable graph | `torch.export` → `ExportedProgram` | `TensorProgram` |
| Compiled artifact | AOTInductor `.pt2`, compiled ahead of time for the device it targets | Compiled-plan sections inside the same file, any number of targets |
| New hardware | Re-run AOTInductor for the new target and ship the new package | `GetOrCompile` compiles from the graph, adds the target, and marks the package dirty to re-save (self-healing) |
| Element type | Fixed when exported | Replays at any element type; captured tensors and scalar arguments convert |
| Training a loaded graph | `ExportedProgram` is not a module; `.module()` unlifts it into a `GraphModule` to train | `Run` records on the gradient tape; `GetParameters<T>()` are tape leaves; `BindParameters` attaches live model weights |
| Dynamic shapes | Declared up front with `Dim` | Inferred by probing: integer arguments that follow a dimension become symbolic, and a dimension the graph structure depends on stays static with a diagnostic |
| Operator coverage | Hand-maintained ATen op set | Generated from `IEngine`: every public op is in the catalog the day it is added (1,093 of 1,182 traceable; the rest take spans, delegates, or return host scalars) |
| Loading untrusted files | `torch.load` unpickles; safe only with `weights_only` | No code, type names or delegates in the file: replay binds only to generated catalog entries, element types and literal tags are allowlisted, lengths are bounds-checked, and an xxHash64 checksum covers the file |
| Silent mistranslation | Graph breaks or tracing errors | `Trace` replays the program on the example and compares it with the function's own result, so work done outside the engine (element writes, LINQ over tensor data) fails the trace instead of being dropped |

## Pieces

- **`EngineSurfaceGenerator`** (in `AiDotNet.Tensors.Analyzers`) reads `IEngine` and emits the op catalog (stable ids
  `Name`arity(param types)`, parameter names and kinds, in-place flags), the `TracingEngine` decorator, and a typed
  replay invoker per element type. Replay calls the interface method directly, with no reflection.
- **`TracingEngine`** forwards every call to the engine it wraps and records only outer calls. Nested calls an engine makes
  through `AiDotNetEngine.Current` are suppressed, so the program is the logical graph before fusion. It is an
  `IEngineWrapper`, so `engine.Is<T>()` / `As<T>()` (analyzer rule TENS0020) see through it.
- **Tracing is thread-local.** `Trace` installs the tracer with a per-thread override of `AiDotNetEngine.Current`. Other
  threads, and the engine's own worker threads, keep the process engine.
- **`ProgramPackage`** layout: a header, then the program section, the tensor section, one section per compiled target,
  and the checksum. A target is keyed by engine name, hardware fingerprint, element type, input shapes, and plan
  format and codec versions. A stale target is dropped and rebuilt from the program.

## Limits

- A function must do its tensor work through `IEngine`. Reading values back to the host (`TensorSum` returning a
  scalar, element access) makes the result data-dependent, and the trace fails.
- Programs with several outputs replay eagerly; compiled plans cover single-output programs.
- A compiled target stores the weights it was compiled with. Re-save the package after training so the targets match.
