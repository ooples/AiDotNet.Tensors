#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Direct-PTX discipline guard.
#
# A local + CI harness that mechanically enforces the quality rules the
# hand-emitted PTX kernel family must follow, so quality cannot silently
# regress ("drift") between kernels. Run from the repository root:
#
#     bash tools/ptx-discipline/check-ptx-discipline.sh
#
# Exit code 0 = all checks pass. Non-zero = a violation was found; the
# offending file:line and the rule are printed. See README.md in this folder
# for the rationale behind each rule.
#
# The rules encode concrete defects previously found in review (over-broad
# architecture gates, a non-ISA warp shuffle, an unreachable promotion gate).
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PTX_DIR="$ROOT/src/AiDotNet.Tensors/Engines/DirectGpu/CUDA/Ptx"
CUDA_DIR="$ROOT/src/AiDotNet.Tensors/Engines/DirectGpu/CUDA"
BLUEPRINT="$PTX_DIR/DirectPtxKernelBlueprint.cs"
BASELINE_FILE="$ROOT/tools/ptx-discipline/family-gate-baseline.txt"

fail=0
note() { printf '  %s\n' "$1"; }
violation() { printf '\n[FAIL] %s\n' "$1"; fail=1; }
ok() { printf '[ OK ] %s\n' "$1"; }

# ---------------------------------------------------------------------------
# HARD RULE 1 — warp shuffles must be ISA-correct.
# shfl.sync.bfly.b32 is a bit-manipulation instruction: its operands are .b32
# registers. Passing a .f32 register relies on assembler leniency and diverges
# from the house idiom (reinterpret through a .b32 register via mov.b32).
# ---------------------------------------------------------------------------
hits="$(grep -rnE 'shfl\.sync\.bfly\.b32 %f' "$PTX_DIR" 2>/dev/null || true)"
if [ -n "$hits" ]; then
  violation "warp shuffle uses an .f32 register on a .b32 instruction; reinterpret through a .b32 register (mov.b32) first:"
  printf '%s\n' "$hits" | sed 's/^/       /'
else
  ok "Rule 1: no shfl.sync.bfly.b32 on an .f32 register."
fi

# ---------------------------------------------------------------------------
# HARD RULE 2 — every HasValidated* predicate pins an exact (major, minor).
# The measured/promoted device is a specific SM; a whole-family compare
# (== DirectPtxArchitectureFamily.Xxx) would let a kernel engage on hardware it
# was never benchmarked on.
# ---------------------------------------------------------------------------
bad_pred=0
while IFS= read -r line; do
  # line is "<lineno>:    internal static bool HasValidatedXxx(int major, int minor) =>"
  lineno="${line%%:*}"
  body="$(sed -n "$((lineno + 1))p" "$BLUEPRINT")"
  if ! printf '%s' "$body" | grep -qE '\(major, minor\) == \('; then
    violation "HasValidated predicate at $BLUEPRINT:$lineno does not pin an exact (major, minor):"
    note "$body"
    bad_pred=1
  fi
done < <(grep -nE 'internal static bool HasValidated[A-Za-z0-9]+\(int major, int minor\)' "$BLUEPRINT" 2>/dev/null)
[ "$bad_pred" -eq 0 ] && ok "Rule 2: all HasValidated* predicates pin an exact (major, minor)."

# ---------------------------------------------------------------------------
# HARD RULE 3 — kernels fail closed: IsPromotedShape must not hardcode true.
# Promotion is earned by clearing the release gate across clean runs, never by
# a literal in the kernel.
# ---------------------------------------------------------------------------
hits="$(grep -rnE 'IsPromotedShape\([^)]*\)\s*=>\s*true' "$PTX_DIR" 2>/dev/null || true)"
if [ -n "$hits" ]; then
  violation "IsPromotedShape hardcodes true; promotion must be gated by DirectPtxReleaseGate, not a literal:"
  printf '%s\n' "$hits" | sed 's/^/       /'
else
  ok "Rule 3: no kernel hardcodes IsPromotedShape => true (fail-closed)."
fi

# ---------------------------------------------------------------------------
# HARD RULE 4 — kernels must not emit local memory.
# The register-resident contract forbids .local; the JIT budget also enforces
# it at load time, but catch it in source too.
# ---------------------------------------------------------------------------
hits="$(grep -rnE 'AppendLine\(.*\.local' "$PTX_DIR" 2>/dev/null || true)"
if [ -n "$hits" ]; then
  violation "a kernel emits .local memory; the family is register-resident:"
  printf '%s\n' "$hits" | sed 's/^/       /'
else
  ok "Rule 4: no kernel emits .local memory."
fi

# ---------------------------------------------------------------------------
# BASELINE RULE 5 — no NEW whole-family architecture gate.
# Existing family-level gates (decode/paged/attention-backward/rmsnorm) are
# tracked debt to migrate to exact-SM HasValidated* predicates. The guard
# fails only if the count INCREASES, so new kernels cannot add more drift.
# ---------------------------------------------------------------------------
current="$(grep -rnE 'runtime\.ArchitectureFamily != DirectPtxArchitectureFamily\.Ampere|Classify\(_ccMajor, _ccMinor\) [!=]= DirectPtxArchitectureFamily\.Ampere' "$CUDA_DIR" 2>/dev/null | wc -l | tr -d ' ')"
baseline="$(tr -d ' \n' < "$BASELINE_FILE" 2>/dev/null || echo 0)"
if [ "$current" -gt "$baseline" ]; then
  violation "whole-family architecture gates increased from $baseline to $current. New direct-PTX gates must use an exact-SM HasValidated* predicate, not '== DirectPtxArchitectureFamily.Ampere'."
  grep -rnE 'runtime\.ArchitectureFamily != DirectPtxArchitectureFamily\.Ampere|Classify\(_ccMajor, _ccMinor\) [!=]= DirectPtxArchitectureFamily\.Ampere' "$CUDA_DIR" | sed 's/^/       /'
elif [ "$current" -lt "$baseline" ]; then
  ok "Rule 5: family-gate debt dropped from $baseline to $current — please lower the baseline in family-gate-baseline.txt."
else
  ok "Rule 5: family-gate debt unchanged at $baseline (no new drift)."
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "Direct-PTX discipline check FAILED. Fix the violations above before pushing."
  exit 1
fi
echo "Direct-PTX discipline check passed."
