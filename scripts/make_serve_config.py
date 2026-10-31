#!/usr/bin/env python3
"""Write the serve config: the pack's full config.json, plus the lane's overlay.

The pack's config.json is the model definition (architectures, text_config,
tokenizer metadata, auto_map); the overlay is only the quantization block.
Mounting the overlay alone over config.json would erase the model. With
--overlay this writes every key from the pack, with ``quantization_config``
replaced by the overlay's block. Without it (a lane whose pack config is
complete, e.g. 2.15bpw) the pack config passes through unchanged.

  python3 scripts/make_serve_config.py --base <pack>/config.json \
      --overlay <lane overlay json> \
      --out ~/m31/serve-config.json

  ... --check existing.json     # exit 1 on drift

Fail-closed: the base must look like the MiniMax-M3.1 pack, the overlay must be
the EXL3/MXFP8 split, and a base that already carries non-routed coverage is
refused (it would be applied twice).
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the pack's own config.json")
    ap.add_argument("--overlay", help="the lane's quantization overlay (omit: pass the pack config through)")
    ap.add_argument("--out", help="write the merged config here")
    ap.add_argument("--check", metavar="FILE", help="compare against an existing merged config")
    args = ap.parse_args()

    with open(args.base) as f:
        base = json.load(f)
    if "architectures" not in base or "text_config" not in base:
        raise SystemExit(f"{args.base} is not a full model config")
    if base.get("model_type") != "minimax_m3_vl":
        raise SystemExit(f"unexpected model_type {base.get('model_type')!r}")

    if args.overlay is None:
        if (base.get("quantization_config") or {}).get("quant_method") != "exl3":
            raise SystemExit(f"{args.base} has no exl3 quantization_config")
        merged = base
    else:
        with open(args.overlay) as f:
            overlay = json.load(f)
        if "non_routed_quantization" in (base.get("quantization_config") or {}):
            raise SystemExit(f"{args.base} already carries non-routed coverage; refusing to apply the overlay twice")

        quant = overlay.get("quantization_config")
        if not isinstance(quant, dict) or quant.get("quant_method") != "exl3":
            raise SystemExit("overlay has no exl3 quantization_config")
        if "non_routed_quantization" not in quant or "non_routed_exl3" not in quant:
            raise SystemExit("overlay is missing the non-routed blocks")

        # The overlay may add keys but must keep every key the pack declared, with
        # the same value: the plugin's EXL3 path reads them.
        original = base.get("quantization_config") or {}
        for key, value in original.items():
            if quant.get(key) != value:
                raise SystemExit(f"overlay changes or drops quantization_config.{key} ({value!r} -> {quant.get(key)!r})")

        merged = dict(base)
        merged["quantization_config"] = quant
    payload = json.dumps(merged, indent=1, sort_keys=True) + "\n"

    if args.check:
        with open(args.check) as f:
            current = f.read()
        if current != payload:
            print(f"serve config drift: {args.check}", file=sys.stderr)
            return 1
        print(f"serve config ok: {args.check}")
        return 0
    if not args.out:
        sys.stdout.write(payload)
        return 0
    with open(args.out, "w") as f:
        f.write(payload)
    print(f"wrote {args.out} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
