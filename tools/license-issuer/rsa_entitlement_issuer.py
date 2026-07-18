#!/usr/bin/env python3
"""
Tensors entitlement issuer — mints an offline, RSA-signed entitlement file for one tenant.

Output envelope (must match src/AiDotNet.Tensors/Licensing/SignedEntitlement.cs):
    { "payload": "<base64(canonical payload JSON)>",
      "signature": "<base64(RSA-2048 PKCS#1 v1.5 SHA-256 over the RAW payload bytes)>" }
Signed payload:
    { "marker": "AiDotNet-Tensors-Entitlement-v1", "tenant": "...", "expires": "YYYY-MM-DD",
      "capabilities": ["tensors:save", ...], "scope"?: "...", "jti"?: "..." }

The signature is computed over the RAW bytes that base64-decoding `payload` yields — exactly what the
runtime re-verifies — so JSON field order/whitespace is irrelevant. Only `payload` is signed; every
privilege field is read from it, never from the envelope.

Deliver the resulting JSON to the tenant to install as ~/.aidotnet/tensors-entitlement.json (or point
AIDOTNET_LICENSE_TOKEN at it). It verifies fully OFFLINE against the embedded RSA public key.

Requires: pip install cryptography
"""
import argparse
import base64
import datetime as dt
import json
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

MARKER = "AiDotNet-Tensors-Entitlement-v1"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mint a signed Tensors entitlement.")
    ap.add_argument("--private-key-pem", required=True, help="RSA private key PEM from rsa_keygen.py")
    ap.add_argument("--tenant", required=True, help="tenant / customer name (recorded in the signed payload)")
    ap.add_argument("--caps", default="tensors:save,tensors:load,model:save,model:load",
                    help="comma-separated capabilities")
    ap.add_argument("--days", type=int, default=365, help="validity in days (omit --days 0 for perpetual)")
    ap.add_argument("--scope", default=None, help="optional audience binding; host must set AIDOTNET_LICENSE_SCOPE to match")
    ap.add_argument("--jti", default=None, help="optional token id for CRL revocation")
    ap.add_argument("--out", default="tensors-entitlement.json", help="output path")
    args = ap.parse_args()

    with open(args.private_key_pem, "rb") as f:
        priv = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(priv, RSAPrivateKey):
        print("ERROR: provided PEM is not an RSA private key", file=sys.stderr)
        return 2

    payload = {
        "marker": MARKER,
        "tenant": args.tenant,
        "capabilities": [c.strip() for c in args.caps.split(",") if c.strip()],
    }
    if args.days > 0:
        exp = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=args.days)
        payload["expires"] = exp.strftime("%Y-%m-%d")
    if args.scope:
        payload["scope"] = args.scope
    if args.jti:
        payload["jti"] = args.jti

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = priv.sign(payload_bytes, padding.PKCS1v15(), hashes.SHA256())

    # Self-verify (fail closed) before writing anything.
    priv.public_key().verify(signature, payload_bytes, padding.PKCS1v15(), hashes.SHA256())

    envelope = {
        "payload": base64.b64encode(payload_bytes).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(envelope, f, separators=(",", ":"))

    print("=== Tensors entitlement issued ===")
    print(f"tenant:       {args.tenant}")
    print(f"capabilities: {payload['capabilities']}")
    print(f"expires:      {payload.get('expires', 'perpetual')}")
    if args.scope:
        print(f"scope:        {args.scope}")
    if args.jti:
        print(f"jti:          {args.jti}")
    print(f"\nwrote -> {args.out}  (deliver to the tenant; install as ~/.aidotnet/tensors-entitlement.json)")
    print("self-verify: OK (RSA-SHA256 signature valid over the exact payload bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
