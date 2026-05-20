# Sub-issue F Implementation Plan — BlasProvider routing shim

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans

**Issue:** #374
**Parent:** #368

## Why this is the biggest lever

Sub-A through Sub-D delivered real perf wins (2 outright wins, median 23.5×→21.5×, ResNet50_layer4 14×, ViT 5×). But **none of those wins are visible to production code** — every caller still goes through `BlasProvider.TryGemm` which routes to libopenblas. Sub-F is the wire that connects A/B/C/D to the 144 call sites.

## Minimum viable Sub-F (this PR)

**One static toggle:** `BlasManaged.PreferManaged`. When true, every `BlasProvider.TryGemm*` routes to `BlasManaged.Gemm<T>` instead of the native cblas path. When false (default), unchanged behavior.

This is the simplest version that delivers value:
- Zero changes to the 144 caller files
- One flag flip exercises all of Sub-A/B/C/D
- Tests can verify managed path correctness across the codebase
- Supply-chain-conscious deployments (the original #368 motivation) just set this true at startup

## Out-of-scope for Sub-F (deferred to F.2 follow-up)

- **`PrefersManagedCache` autotune** — measures both paths per-shape, caches the winner. Useful when BlasManaged wins on some shapes and loses on others (current state). Can land as a follow-up PR; the `PreferManaged` global is the prerequisite.
- **`TryGemmWithBeta` / `TryGemmExBeta`** — these have `beta != 0` (accumulating GEMM) which `BlasManaged.Gemm` doesn't support (it always does `C := AB`, not `C := αAB + βC`). Initial Sub-F only routes the `beta=0` overloads (TryGemm and TryGemmEx). The Beta variants stay native.
- **Batched GEMM** (`TryGemmBatchSameShape*`) — different signature; stays native for now.

## Tasks

### F.1 — `BlasManaged.PreferManaged` static toggle + route TryGemm/TryGemmEx

**Files:**
- Modify: ``src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs`` (add static)
- Modify: ``src/AiDotNet.Tensors/Helpers/BlasProvider.cs`` (add route check to 4 entry points)
- Test: ``tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/RoutingShimTest.cs``

**Acceptance:**
- Setting `BlasManaged.PreferManaged = true` routes TryGemm/TryGemmEx through `BlasManaged.Gemm`
- Output matches native dispatch bit-exact for shapes both can compute (most shapes; Streaming/PackBoth in deterministic mode)
- Full existing test suite continues to pass with `PreferManaged = false` (the default)
- A subset of tests with `PreferManaged = true` confirms managed path is exercised end-to-end

### F.2 — Re-baseline with PreferManaged=true (verify production-visible perf)

**Acceptance:**
- Re-run baseline harness with `PreferManaged = true`
- ResNet50_layer4 and LSTM_cell wins must be visible
- Tied/winning shapes count must match Sub-D baseline (or better; autotune sub-F.3 could add more)
