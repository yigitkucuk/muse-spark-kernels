#!/usr/bin/env python3
"""Assemble each kernel into its single-file module.

<kernel>/py/*.py are consecutive sections of one module; every __cuda__("name") in them is replaced by the
contents of <kernel>/cuda/name as a raw string literal. Output: build/<kernel>.py.

usage: python3 build.py [kernel ...] [--check]
  --check   compare each output against the recorded sha256
"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SHA256 = {
    "muse_decode_layer": "5788a56fbe587810358a4b92ee87b5b2779fa824fda85fff46c764ef704e36d4",
    "muse_moe_w4a16": "c6b137ace683ebbcb049f13182ab4e578c56a0bda691170927387cc22ea580a0",
    "muse_attention_decode": "aa2a11c894d7b83aeabf4d25511a04a2083ee07d161807e7678af2bd326c6b72",
    "muse_prefill_layer": "e122ffcbd4611b3c588bef60a3d6259a1fe3e943144e646edf03f370d5aff122",
    "muse_bf16_proj": "aea77e9bbff7b3b1f747e8c2d28c60b3881ce4197170f9659b852fa7ccd9893f",
}
SIZE_LIMIT = 1_000_000  # single-file submission limit, bytes


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def assemble(kernel):
    d = os.path.join(ROOT, kernel)
    py = "".join(read(os.path.join(d, "py", f)) for f in sorted(os.listdir(os.path.join(d, "py"))) if f.endswith(".py"))
    return re.sub(r'__cuda__\("([^"]+)"\)', lambda m: 'r"""' + read(os.path.join(d, "cuda", m.group(1))) + '"""', py)


def main(argv):
    check = "--check" in argv
    kernels = [a for a in argv if not a.startswith("--")] or list(SHA256)
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    bad = 0
    for k in kernels:
        src = assemble(k)
        compile(src, k + ".py", "exec")
        out = os.path.join(ROOT, "build", k + ".py")
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        n = len(src.encode())
        h = hashlib.sha256(src.encode()).hexdigest()
        status = ""
        if check:
            status = "ok" if h == SHA256[k] else "MISMATCH"
            bad += h != SHA256[k]
        if n >= SIZE_LIMIT:
            status += " over the 1,000,000 B limit"
            bad += 1
        print(f"build/{k + '.py':<26} {n:>9,} B  {h[:16]}  {status}".rstrip())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
