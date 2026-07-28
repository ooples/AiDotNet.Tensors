#!/usr/bin/env python3
"""
Tensors entitlement RSA signing keypair bootstrap.

Generates ONE RSA-2048 keypair and prints everything needed to activate AiDotNet.Tensors' offline
entitlement path (SignedEntitlement). RUN THIS ON YOUR OWN MACHINE — the private key is written to a local
PEM and never leaves your terminal; do not paste it into chat, commit it, or store it unencrypted.

Why RSA (not the Ed25519/aidn2 key): the Tensors runtime verifies entitlements with .NET's BUILT-IN
System.Security.Cryptography.RSA, so it works on net471 + net8 + net10 with NO extra crypto dependency
(no BouncyCastle) — a deliberate design choice for the low-level Tensors package. This is a SEPARATE trust
root from the main SDK's Ed25519 aidn2 key.

Emits:
  1. EntitlementPublicKey  — "{modulusBase64}:{exponentBase64}" (RSAParameters form). Paste it in place of
     the SignedEntitlement.PublicKeyPlaceholderMarker in
     src/AiDotNet.Tensors/Licensing/SignedEntitlement.cs (public — safe to commit). This ACTIVATES the
     feature (until then it's inert and every entitlement falls through to the online/trial path).
  2. private_key.pem       — RSA private key. Keep it in your signing secret store; sign per-tenant
     entitlements with rsa_entitlement_issuer.py. NEVER commit.

Requires: pip install cryptography
"""
import argparse
import base64
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def i2b(n: int) -> bytes:
    return n.to_bytes((n.bit_length() + 7) // 8, "big")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Tensors entitlement RSA signing keypair.")
    ap.add_argument("--bits", type=int, default=2048, help="RSA key size (default 2048; minimum 2048)")
    ap.add_argument("--out-dir", default=".", help="where to write private_key.pem (default: cwd)")
    args = ap.parse_args()

    # Enforce the documented RSA-2048 trust-root minimum — a weaker key must never be minted.
    if args.bits < 2048:
        ap.error("--bits must be at least 2048 (RSA-2048 is the documented trust-root minimum)")

    priv = rsa.generate_private_key(public_exponent=65537, key_size=args.bits)
    nums = priv.public_key().public_numbers()

    mod_b64 = base64.b64encode(i2b(nums.n)).decode("ascii")
    exp_b64 = base64.b64encode(i2b(nums.e)).decode("ascii")
    embed = f"{mod_b64}:{exp_b64}"

    os.makedirs(args.out_dir, exist_ok=True)
    priv_path = os.path.join(args.out_dir, "private_key.pem")
    # Create the private key file owner-readable/writable ONLY (0600) so a leaked copy can't be world-read.
    # os.open honours the mode on creation; fchmod additionally forces 0600 on an existing file. Both are
    # effectively no-ops on Windows (which lacks POSIX permission bits / fchmod), so this is safe there.
    fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        try:
            os.fchmod(f.fileno(), 0o600)
        except (AttributeError, OSError):
            pass  # Windows / platforms without fchmod — best effort.
        f.write(priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    print("=== Tensors entitlement RSA keypair ===")
    print(f"key size: {args.bits} bits\n")
    print("--- 1. EntitlementPublicKey (PUBLIC — commit) ---")
    print("Replace SignedEntitlement.EntitlementPublicKey (currently the placeholder) with:\n")
    print(f'    public const string EntitlementPublicKey = "{embed}";\n')
    print(f"--- 2. private key -> {priv_path}  (SECRET — never commit) ---")
    print("Sign per-tenant entitlements with:")
    print(f"    python rsa_entitlement_issuer.py --private-key-pem {priv_path} --tenant \"ACME Corp\" \\")
    print("        --caps tensors:save,tensors:load,model:save,model:load --days 365")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
