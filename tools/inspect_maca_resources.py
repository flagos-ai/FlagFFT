#!/usr/bin/env python3
"""Read compiled MACA resources without changing the cache or loading a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def device_elf(blob):
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    if not blob.startswith(magic):
        raise ValueError("not a clang offload bundle")
    offset = len(magic)
    count = struct.unpack_from("<Q", blob, offset)[0]
    offset += 8
    for _ in range(count):
        start, size, name_size = struct.unpack_from("<QQQ", blob, offset)
        offset += 24
        name = blob[offset:offset + name_size]
        offset += name_size
        payload = blob[start:start + size]
        if name.startswith(b"maca-") and payload.startswith(b"\x7fELF"):
            return payload
    raise ValueError("no MACA device ELF")


def resource_integer(blob, key):
    encoded = key.encode()
    if blob.count(encoded) != 1:
        raise ValueError(f"expected one resource note: {key}")
    start = blob.index(encoded) + len(encoded)
    marker = blob[start]
    if marker <= 127:
        return marker
    return struct.unpack_from(
        {0xcc: ">B", 0xcd: ">H", 0xce: ">I", 0xcf: ">Q"}[marker], blob, start + 1
    )[0]


def inspect(root):
    records = []
    for path in sorted(root.rglob("*.mcfatbin")):
        record = {"artifact": str(path.relative_to(root))}
        try:
            blob = path.read_bytes()
            elf = device_elf(blob)
            meta = json.loads(path.with_suffix(".json").read_text())
            llir = path.with_suffix(".llir").read_text()
            record.update(
                kernel=meta["name"], sha256=hashlib.sha256(blob).hexdigest(),
                shared_bytes=meta["shared"], num_warps=meta["num_warps"],
                llir_static_barriers=len(re.findall(r"call void @llvm.mxc.barrier\(", llir)),
                elf_resources={key: resource_integer(elf, key) for key in (
                    ".mtreg_count", ".streg_count", ".private_memory_size",
                    ".share_memory_size", ".max_block_size",
                )},
            )
        except (ValueError, KeyError, OSError, struct.error) as error:
            record["inspection_error"] = str(error)
        records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    if not args.cache.is_dir():
        parser.error("cache directory does not exist")
    print(json.dumps({
        "caveat": "LLIR static sites are not dynamic instruction counts; ELF static shared excludes dynamic shared.",
        "kernels": inspect(args.cache),
    }, indent=2))
