# Enterprise licensing — activation runbook (operator)

The code is shipped and inert-by-default. This runbook is the **operator checklist**
to turn each layer on. Everything here is done **once** by Ooples in a secure
environment; none of it can (or should) be automated by a code change, because it
involves secrets and CA-issued certificates.

> Order doesn't matter between sections — each layer is independent.

---

## 1. Embed the signing keys (entitlement + response)

Generate **two separate** RSA-2048 keypairs (compromise isolation) on a secure
operator machine:

```pwsh
pwsh build/keys/generate-signing-keys.ps1 -Purpose entitlement -OutDir <secure-dir>
pwsh build/keys/generate-signing-keys.ps1 -Purpose response    -OutDir <secure-dir>
```

Each run prints the **public** key in `{modulusBase64}:{exponentBase64}` form.
Paste them into the two consts:

| Const | File |
|---|---|
| `SignedEntitlement.EntitlementPublicKey` | `src/AiDotNet.Tensors/Licensing/SignedEntitlement.cs` |
| `LicenseResponseVerifier.ResponseSigningPublicKey` | `src/AiDotNet.Tensors/Licensing/LicenseResponseVerifier.cs` |

Move each **private** key into the HSM / secret manager and delete the local
file. The `build/keys/.gitignore` blocks accidental commits — but verify with
`git status` before committing the public-key edits.

> Until you replace the placeholders, both features are **inert** (the runtime
> falls through to the existing key/trial path) — so this is safe to defer.

---

## 2. NuGet package signing (release pipeline)

The release pipeline already has the signing step (`automated-release.yml` →
"Sign NuGet packages"). It's a **no-op until** these secrets exist:

```bash
# Acquire an OV (or EV) code-signing certificate from a CA (DigiCert, Sectigo, …)
# issued to Ooples Finance LLC. Export it as a password-protected .pfx, then:
base64 -w0 ooples-codesign.pfx > cert.b64          # macOS: base64 -i ooples-codesign.pfx | tr -d '\n'

gh secret set CODE_SIGNING_CERT_BASE64    < cert.b64
gh secret set CODE_SIGNING_CERT_PASSWORD               # paste the .pfx password
rm cert.b64
```

Once set, every release signs + timestamps every `.nupkg`; a tampered/repacked
package then fails `dotnet nuget verify`. (EV certs give the strongest trust /
SmartScreen reputation but require a hardware token — see your CA's guidance.)

### Optional: Authenticode-sign the DLLs too

NuGet signing protects the *package*; Authenticode protects each *DLL*. Add a
step before pack to `signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 ...`
(Windows) or `osslsigncode` (Linux) over `src/**/bin/Release/**/AiDotNet.Tensors.dll`,
guarded by the same secret. Deferred — wire it in if your deployment policy
enforces signed-binary loading.

---

## 3. Online response signing (Supabase Edge Function)

Activates the anti-spoof online path. Requires both halves:

1. **Server:** add `build/license-server/sign-validation-response.ts` to your
   `validate-license` Edge Function (integration sketch is in that file), and
   set the private key secret:
   ```bash
   supabase secrets set RESPONSE_SIGNING_PRIVATE_KEY_PEM="$(cat <secure-dir>/aidotnet-response-signing-PRIVATE.pem)"
   supabase functions deploy validate-license
   ```
   The server signs over the canonical form defined in
   `license-server-response-signing.md`.
2. **Client:** the customer (or you, for managed deployments) sets
   `AiDotNetTensorsLicenseKey.RequireSignedResponse = true`. With the embedded
   response public key from §1, the client now rejects any unsigned/forged
   response.

> High-assurance customers should prefer the fully-offline **entitlement**
> (§1 + issue a token, see `licensing-security-model.md`) — it needs no server
> at all, so it's not subject to server availability or spoofing.

---

## 4. Verify activation

- **Entitlement:** issue a test token (private key), set `AIDOTNET_LICENSE_TOKEN`,
  confirm a persistence op succeeds with no license key/trial; corrupt one byte
  of the token and confirm it's rejected (and audit-logged).
- **NuGet signing:** after a release, `dotnet nuget verify <pkg>.nupkg --all`.
- **Response signing:** with `RequireSignedResponse=true`, point the client at a
  server *without* signing and confirm validation returns Invalid; then at the
  signed server and confirm Active.

## What remains genuinely out of scope (not code)

NativeAOT extraction of the enforcement path and commercial obfuscation /
anti-tamper are architecture/tooling initiatives — see the roadmap in
`licensing-security-model.md`. They raise reverse-engineering cost; none make
client-side enforcement absolute.
