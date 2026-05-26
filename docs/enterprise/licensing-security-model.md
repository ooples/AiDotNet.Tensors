# AiDotNet.Tensors — licensing security model & threat analysis

This is the honest, end-to-end picture of how enterprise/commercial licensing is
enforced, **what it does and does not protect against**, and the remaining
hardening work with realistic feasibility. Read the "Ceiling" section first — it
sets the expectations everything else is measured against.

## The ceiling (non-negotiable truth)

Enforcement that runs **client-side**, in **managed .NET bytecode**, in a
**source-available (BSL)** library **cannot be made un-jailbreakable.** The check
executes on hardware the attacker controls; the IL can be decompiled and patched
(dnSpy/ILSpy) in minutes; and with source they can simply rebuild without the
gate. This is true of *all* client-side DRM. The realistic, achievable goal is:

> **Unforgeable trust anchors + tamper-evidence + raise the cost of a bypass high
> enough that it's cheaper to buy a license — backed by the contract/audit
> relationship that actually protects enterprise revenue.**

Shipping anything that *looks* locked-down but isn't (e.g. a runtime self-integrity
check whose "expected" value lives in the same patchable DLL) is **worse than
nothing** — it creates false confidence. We do not do that here.

## Layers (what's implemented)

| Layer | Trust anchor | Forgeable? | Bypassable? | Status |
|---|---|---|---|---|
| **Build-time flag gate** (`build/AiDotNet.Tensors.Enterprise.targets`) | RSA-2048 signed license file + `keyHash(key)` | No (needs private key) | Yes, with source (remove the gate) | shipped (#439) |
| **Runtime offline entitlement** (`SignedEntitlement` → `PersistenceGuard`) | RSA-2048 signed token, embedded public key, **offline** | **No** (needs private key) | Yes, by IL-patching the check | shipped (this work) |
| **Runtime online validation** (`LicenseValidator`) | License server response | **Yes — response is unsigned → server-spoofable** | Yes (spoof server or patch) | shipped, low-trust tier |
| **Trial** (`TrialState`) | local JSON file | n/a | Yes (edit the file) | shipped, lowest tier |

**Key design point:** high-assurance customers use the **offline signed
entitlement**, whose trust anchor is an RSA signature an attacker cannot forge —
*not* the online server. The online path's spoofability therefore only risks a
**trial bypass** (low tier), not an enterprise bypass. Enterprise = offline
entitlement, full stop.

### Why the offline entitlement matters

`PersistenceGuard.EnforceCore` checks a signed entitlement **first**
(`AIDOTNET_LICENSE_TOKEN` env var, or `~/.aidotnet/tensors-entitlement.json`):

- The token's RSA-2048 PKCS#1 SHA-256 signature is verified against an embedded
  public key, **fully offline** (no server to spoof).
- Marker/expiry/capabilities are read from the **signed** payload only.
- A valid entitlement authorises the op with no network call and no trial tick.
- A **present-but-invalid** token hard-fails (and is logged as a possible forged
  token) — it does *not* silently degrade to the free trial.
- Ships with a **placeholder** public key, so the feature is **inert** until the
  real key is embedded (no behavior change on release).

## Issuing an entitlement (operator guide)

Mirror `build/LicenseValidator/README.md`. Generate the keypair once; embed the
**public** key in `SignedEntitlement.EntitlementPublicKey` as
`{modulusBase64}:{exponentBase64}` (RSAParameters form — cross-TFM, no
BouncyCastle). Keep the **private** key in the HSM/secret manager.

```bash
# Sign the entitlement payload (private key never leaves the HSM):
cat > ent.json <<'EOF'
{ "marker":"AiDotNet-Tensors-Entitlement-v1","tenant":"ACME Corp.",
  "expires":"2027-05-22","capabilities":["tensors:save","tensors:load"] }
EOF
PAYLOAD=$(openssl enc -base64 -A < ent.json)
SIG=$(openssl dgst -sha256 -sign ent-signing.key ent.json | openssl enc -base64 -A)
printf '{"payload":"%s","signature":"%s"}' "$PAYLOAD" "$SIG" > tensors-entitlement.json
# Customer sets AIDOTNET_LICENSE_TOKEN to this JSON (or a path to it).
```

## Hardening roadmap (NOT yet done — honest feasibility)

These raise the bypass cost; **none make it absolute**, and several are
infra/tooling, not code:

1. **Strong-name the assembly** *(release-eng, real on net471 only)*. A patched
   strong-named assembly fails CLR signature verification on .NET Framework
   (re-signing needs the SN private key). **Caveat:** .NET Core / net10.0 does
   **not** verify strong names at load → no protection there. **Caveat:** every
   `InternalsVisibleTo` friend (incl. the test assemblies) must then also be
   strong-named / use public-key IVT, or the build breaks. Deferred because it's
   a cross-cutting change with real breakage risk and net471-only value.
2. **Sign the published NuGet packages** *(release-eng, real)* — **SCAFFOLDED.**
   A patched/repacked `.nupkg` fails NuGet signature verification. The release
   pipeline (`.github/workflows/automated-release.yml`) now signs every package
   via `dotnet nuget sign` + timestamp; it's a **no-op until** the
   `CODE_SIGNING_CERT_BASE64` / `CODE_SIGNING_CERT_PASSWORD` secrets are set, so
   it never breaks a build. **To activate:** provision an Ooples code-signing
   certificate and add those two secrets. (Authenticode-signing the individual
   DLLs is a further step — `signtool`/`osslsigncode` — once a cert exists.)
3. **Sign the online validation response + client nonce** *(client DONE, needs
   server change to activate)*. The client (`LicenseResponseVerifier`,
   `RequireSignedResponse`) is implemented and tested; it sends a per-request
   nonce and verifies an RSA signature over `(nonce|status|tier|expires|caps)`.
   It activates once (a) the Supabase Edge Function signs per
   `license-server-response-signing.md` and (b) the customer enables
   `RequireSignedResponse`. **Note:** the offline entitlement already covers the
   high-assurance case, so this only matters if trial-bypass is a concern.
4. **NativeAOT the enforcement path** *(architecture initiative)*. Compiling
   enforcement to native code turns "minutes in dnSpy" into serious reverse
   engineering. Flipping the *whole* tensor library to AOT is infeasible
   (reflection/generics); the real path is extracting enforcement into a small,
   separately-AOT-compiled component. Scoped as its own project.
5. **Commercial obfuscation / anti-tamper** *(tooling)*. ConfuserEx (OSS) or a
   commercial SDK (e.g. Dotfuscator) integrated into the publish step. Raises
   reverse-engineering cost; still crackable. Requires pipeline integration; not
   something a source edit delivers.

## What actually protects enterprise revenue

For enterprise/federal/air-gapped customers, the technical guard is the
*deterrent and the audit trail*; the **contract + audit** is the enforcement.
Those customers are the most contractually bound and the least likely to crack a
license — which is exactly why the perpetual air-gapped build
(`AIDOTNET_DISABLE_LICENSE_GUARD`, gated by the build-time RSA check) is the
*least* runtime-enforced tier by design.
