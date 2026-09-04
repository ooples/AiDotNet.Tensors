# Evolutionary kernel autotuning

`AiDotNet.Tensors` exposes `EvolutionKernelAutotuner<TConfiguration>` for search spaces that are too large for a
fixed first-run sweep. The configuration is a caller-owned immutable type. Finite decisions belong in enums and
numeric launch properties; string dictionaries are confined to the legacy cache serialization boundary.

The benchmark delegate is backend-neutral. Unit tests use deterministic CPU measurements, while production code can
launch and time CUDA/PTX, Vulkan, Metal, HIP, OpenCL, WebGPU, or CPU kernels through the same delegate. The tuning
engine runs during an offline job, startup warmup, or a background task. A completed run atomically publishes the
typed winner through `KernelTuningDeployment<TConfiguration>`.

The serving path calls `TryGet` on a deployment dedicated to one `KernelTuningIdentity`. A hit is one volatile
reference read and a typed assignment. It performs no disk access, parsing, reflection, hashing, or evolution work.
The identity covers kernel, shape, physical device, driver, search-space version, and benchmark-protocol version, so
incompatible evidence cannot be reused accidentally.

`ConvTileAutotune.TypedCandidates` and `ConvTileAutotune.CreateEvolutionTuner` are the first concrete adapter. They
keep tile geometry typed until the old cache compatibility adapter explicitly serializes it. Existing exhaustive
first-run and community-cache APIs remain compatible; larger PTX/codegen schedule spaces can migrate incrementally
without adding search overhead to kernel dispatch.

The evolution package's run-local evaluation memoization and Tensors' deployment cache have different jobs and stay
separate. Evolution avoids evaluating the same canonical candidate twice in one run. The deployment/cache layer
decides whether evidence applies to a hardware and compiler identity across runs.
