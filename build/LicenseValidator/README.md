# AiDotNet.Tensors.LicenseValidator — keypair generation + license issuance

This directory hosts the MSBuild Task DLL that gates the `DISABLE_TELEMETRY` /
`DISABLE_LICENSE_GUARD` compile-time flags on a valid Ooples enterprise license
via RSA-2048 PKCS#1 v1.5 SHA-256 signature verification.

This README is the procedure for the **Ooples-side** operator that generates the
signing keypair, replaces the placeholder public key embedded in the validator
source, and issues real enterprise licenses to customers.

---

## 1. Generate the signing keypair (one-time setup)

```bash
# Private key — keep in Ooples HSM / 1Password / password manager. NEVER commit.
openssl genrsa -out enterprise-license-signing.key 2048

# Public key in PKCS#1 RSAPublicKey ASN.1 SEQUENCE form (no SubjectPublicKeyInfo
# wrapper, no PEM markers). This is what the validator's
# ImportPkcs1RsaPublicKey() expects.
openssl rsa -in enterprise-license-signing.key -RSAPublicKey_out -outform DER \
  | base64 -w0 > enterprise-license-signing.pub.b64

cat enterprise-license-signing.pub.b64
# → single long base64 string. Copy verbatim.
```

## 2. Embed the public key in the validator

Edit `ValidateAiDotNetEnterpriseLicense.cs`:

```csharp
public const string EnterpriseLicensePublicKeyPkcs1Base64 =
    "<paste contents of enterprise-license-signing.pub.b64 here>";
```

Rebuild the validator:

```bash
dotnet build build/LicenseValidator/AiDotNet.Tensors.LicenseValidator.csproj -c Release
```

The compiled DLL lands at `build/LicenseValidator/bin/AiDotNet.Tensors.LicenseValidator.dll`
and is what `build/AiDotNet.Tensors.Enterprise.targets` loads via `<UsingTask>`.

Commit the patched source. The unit tests in `build/LicenseValidator.Tests/`
will continue to pass because they override the embedded key via the
`s_overridePublicKey` internal test hook — they don't depend on the literal
value of the production constant.

## 3. Issue a license to a customer

Compose the license JSON (substitute customer tenant + expiry date):

```bash
cat > inner.json <<'EOF'
{
  "marker": "AiDotNet-Enterprise-License-v1",
  "tenant": "ACME Corp.",
  "issued": "2026-05-22",
  "expires": "2027-05-22",
  "scope":   ["DISABLE_TELEMETRY", "DISABLE_LICENSE_GUARD"]
}
EOF
```

Sign the canonical bytes with the private key:

```bash
PAYLOAD_B64=$(base64 -w0 < inner.json)
SIG_B64=$(openssl dgst -sha256 -sign enterprise-license-signing.key inner.json \
          | base64 -w0)
```

Compose the final envelope and deliver to the customer over a secure channel
(encrypted email, signed Slack DM, customer-portal download, etc.):

```bash
cat > acme-corp-license.json <<EOF
{
  "marker":    "AiDotNet-Enterprise-License-v1",
  "tenant":    "ACME Corp.",
  "issued":    "2026-05-22",
  "expires":   "2027-05-22",
  "scope":     ["DISABLE_TELEMETRY", "DISABLE_LICENSE_GUARD"],
  "payload":   "$PAYLOAD_B64",
  "signature": "$SIG_B64"
}
EOF
```

The customer points
`AiDotNetEnterpriseLicensePath` / `AIDOTNET_ENTERPRISE_LICENSE_PATH` at the
file's location and adds `-p:DefineConstants=DISABLE_TELEMETRY` to their
`dotnet build` command. The validator picks up the license, verifies the
RSA signature against the embedded public key, checks the expiry and scope,
and authorises the compile.

## 4. Rotate the signing key

When you suspect compromise or simply on the annual cycle:

1. Generate a new keypair (Step 1 with a different filename to keep both around).
2. Update `EnterpriseLicensePublicKeyPkcs1Base64` to the **new** public key.
3. Cut a Tensors NuGet release with the bumped validator DLL.
4. Re-sign every active customer's license with the new private key, deliver.
5. Old licenses signed with the previous private key will fail with
   `AIDOTNET008: invalid RSA signature` — customers can either upgrade the
   Tensors package (which carries the new public key) or stay on the older
   pinned version that still trusts the old key. Either path is fine; document
   both in the rotation announcement.

## 5. Security notes

- **Private key**: NEVER commit to git. NEVER put on a shared developer
  workstation. Store in Ooples HSM or a password manager with two-person
  integrity.
- **Public key**: it IS committed (in the validator source). Customers, OSS
  contributors, and review agents will see it. This is fine — public keys
  are by definition publishable. Just don't accidentally swap the private
  one in.
- **Signature algorithm**: RSA-2048 PKCS#1 v1.5 SHA-256. PSS would also be
  acceptable; we picked PKCS#1 because `openssl dgst -sign` produces it by
  default and `RSACryptoServiceProvider.VerifyData` defaults to it on the
  legacy code path too (we don't use that path, but it's convenient for
  customers who want to roll their own signing tool).
- **Why RSA and not Ed25519**: Ed25519 would give smaller keys (32 vs 270
  bytes), faster verification, and no padding-mode questions. The reason we
  stuck with RSA-2048: every MSBuild host (full-framework, .NET MSBuild,
  legacy SDK-style projects, etc.) has `RSACryptoServiceProvider` built in
  via `System.Security.Cryptography`. Ed25519 needs a NuGet pull — and we
  don't want the validator to fail closed because a transitive NuGet
  resolve broke. Worth revisiting once netstandard2.1+ becomes the minimum
  task TFM.

## 6. Reference: validator diagnostic codes

| Code        | Meaning                                                         |
|-------------|-----------------------------------------------------------------|
| AIDOTNET001 | DISABLE_* flag requested but no license key set                 |
| AIDOTNET002 | DISABLE_* flag requested but no license path set                |
| AIDOTNET003 | License file path doesn't exist on disk                         |
| AIDOTNET004 | License file isn't valid JSON, or wrong / missing marker        |
| AIDOTNET005 | License missing or expired 'expires' field                      |
| AIDOTNET006 | Requested flag not in license's granted scope                   |
| AIDOTNET007 | License missing 'payload' or 'signature' field, or bad base64   |
| AIDOTNET008 | RSA signature verification failed (or placeholder key in build) |
| AIDOTNET009 | Unexpected validator exception (see message for details)        |
| AIDOTNET010 | Validator DLL missing and bootstrap-build failed                |
