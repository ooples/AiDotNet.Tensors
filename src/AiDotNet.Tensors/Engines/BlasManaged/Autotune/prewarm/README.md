# Shipped pre-warm autotune entries (#375 Phase 4)

Files here are per-fingerprint strategy caches produced by

```bash
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -- --prewarm-autotune
```

on representative hardware in CI, named `{fingerprint}.prewarm.json` (e.g.
`x64-amd-avx2-cpu16.prewarm.json`). Each line is:

```text
M N K fp64 transA transB strategy mc nc kc threadCount
```

(`fp64`/`transA`/`transB` are `0`/`1`; `strategy` is a `PackingMode` name.)

They are embedded as resources (see the `<EmbeddedResource>` glob in
`AiDotNet.Tensors.csproj`) and loaded once at first autotune access by
`BlasManagedAutotune.EnsurePrewarmLoaded`, seeding `PersistentStrategyCache` only
where no local **learned** entry exists (`SeedFromShippedIfAbsent`). Local learning
always wins; `KernelVersion` mismatches are ignored on read, so a kernel change
invalidates stale shipped entries automatically.

The directory ships empty (only this README) until CI runs the sweep on a target
arch and commits the resulting `*.prewarm.json` — the glob is empty until then, so
the build is unaffected.
