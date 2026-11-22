# Breaking change: four new abstract `IEngine` members (PR #1784)

PR #1784 adds four members to `AiDotNet.Tensors.Engines.IEngine` without default bodies:

- `GatedDeltaNetScanForward<T>(qProj, kProj, vProj, alpha, beta, numHeads, GatedDeltaRule rule)` (new overload)
- `GatedDeltaNetStep<T>(state, query, key, value, alpha, beta, GatedDeltaRule rule)`
- `CausalConv1dStep<T>(history, input, weight, bias, silu)`
- `KvCacheAppend<T>(cache, position, value)`

Third-party types that implement `IEngine` directly (instead of deriving from `CpuEngine` or
`DirectGpuTensorEngine`, which implement all four) no longer compile, and binaries built against an
earlier version throw `TypeLoadException` / `EntryPointNotFoundException` when loaded. The 20 `CP0006`
entries in `src/AiDotNet.Tensors/CompatibilitySuppressions.xml` (4 members x 5 TFMs) record this break.

Why there are no default interface bodies: the package targets net471, net462 and netstandard2.0, which
do not support default interface members (see the comment on `IEngine.DevicePrimitives`), and `IEngine`
has been extended as abstract members throughout (185 earlier `IEngine` CP0006 baseline suppressions).
Bodies under `#if NET` would make the interface differ per TFM (an implementer compiling for net8.0
would silently get a throwing member, the same source for net471 would not compile).

Migration: derive from `CpuEngine` (or `DirectGpuTensorEngine`), or implement the four members.
Note also that `CausalConv1dStep` takes its kernel newest-first (`weight[:, 0]` weights the current
input); reverse the tap axis of PyTorch/HF `conv1d` or `DepthwiseConv1D` weights before passing them.
