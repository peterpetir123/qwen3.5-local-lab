#!/usr/bin/env python3
"""Shrink a GGUF model by quantizing ONLY `output.weight` (BF16 -> Q6_K)
using libggml-base.so's native ggml_quantize_chunk, while copying ALL metadata
and ALL other tensors bit-for-bit untouched (zero double-quantization).
"""
import argparse
import ctypes
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30
Q6_K_BLOCK_SIZE = 256
Q6_K_TYPE_SIZE = 210  # bytes per 256 elements (verified by ggml_row_size)

GGUF_VAL_SCALAR_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8
}


def skip_gguf_value(f, vtype: int):
    if vtype == 8:  # STRING
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(n, os.SEEK_CUR)
    elif vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in GGUF_VAL_SCALAR_SIZES:
            f.seek(count * GGUF_VAL_SCALAR_SIZES[elem_type], os.SEEK_CUR)
        else:
            for _ in range(count):
                skip_gguf_value(f, elem_type)
    elif vtype in GGUF_VAL_SCALAR_SIZES:
        f.seek(GGUF_VAL_SCALAR_SIZES[vtype], os.SEEK_CUR)
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def bf16_bytes_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_gguf", type=Path)
    ap.add_argument("output_gguf", type=Path)
    ap.add_argument("--libggml", type=Path, default=Path("/home/hengkerprotzy/coding/llama.cpp/build/bin/libggml-base.so"))
    args = ap.parse_args()

    t0 = time.time()
    lib = ctypes.CDLL(str(args.libggml))
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    in_size = args.input_gguf.stat().st_size
    with open(args.input_gguf, "rb") as fin:
        magic = fin.read(4)
        assert magic == b"GGUF", f"Not a GGUF file: {magic!r}"
        version, n_tensors, n_kv = struct.unpack("<IQQ", fin.read(20))
        assert version == 3, f"Unsupported GGUF version: {version}"

        alignment = 32
        for _ in range(n_kv):
            klen = struct.unpack("<Q", fin.read(8))[0]
            key = fin.read(klen).decode("utf-8", errors="replace")
            vtype = struct.unpack("<I", fin.read(4))[0]
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack("<I", fin.read(4))[0]
            else:
                skip_gguf_value(fin, vtype)

        kv_end = fin.tell()

        tensors = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", fin.read(8))[0]
            name = fin.read(nlen).decode("utf-8")
            n_dims = struct.unpack("<I", fin.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", fin.read(8 * n_dims))
            ttype, offset = struct.unpack("<IQ", fin.read(12))
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

        header_end = fin.tell()
        data_start = align_up(header_end, alignment)

        # Compute exact byte size of each tensor from sorted offsets
        by_off = sorted(range(n_tensors), key=lambda i: tensors[i]["offset"])
        for pos, idx in enumerate(by_off):
            start_off = tensors[idx]["offset"]
            if pos + 1 < n_tensors:
                next_off = tensors[by_off[pos + 1]]["offset"]
                tensors[idx]["alloc_bytes"] = next_off - start_off
            else:
                tensors[idx]["alloc_bytes"] = (in_size - data_start) - start_off

        target_idx = next((i for i, t in enumerate(tensors) if t["name"] == "output.weight"), None)
        assert target_idx is not None, "output.weight not found"
        tgt = tensors[target_idx]
        assert tgt["type"] == GGML_TYPE_BF16, f"Expected output.weight BF16 (30), got {tgt['type']}"
        n_per_row, nrows = tgt["dims"][0], tgt["dims"][1]
        assert n_per_row % Q6_K_BLOCK_SIZE == 0, f"n_per_row {n_per_row} not divisible by 256"

        new_q6k_bytes = nrows * (n_per_row // Q6_K_BLOCK_SIZE) * Q6_K_TYPE_SIZE
        print(f"Shrinking output.weight [{n_per_row}, {nrows}]: BF16 {tgt['alloc_bytes']/(1024**2):.2f} MiB -> Q6_K {new_q6k_bytes/(1024**2):.2f} MiB")

        # Build new offsets in original physical order
        new_offsets = {}
        cur_off = 0
        for idx in by_off:
            new_offsets[idx] = cur_off
            sz = new_q6k_bytes if idx == target_idx else tensors[idx]["alloc_bytes"]
            cur_off = align_up(cur_off + sz, alignment)

        args.output_gguf.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_gguf, "wb") as fout:
            # 1. Copy header + all KV metadata verbatim
            fin.seek(0)
            remaining = kv_end
            while remaining > 0:
                chunk = fin.read(min(remaining, 8 * 1024 * 1024))
                fout.write(chunk)
                remaining -= len(chunk)

            # 2. Write updated tensor info table
            for idx, t in enumerate(tensors):
                nb = t["name"].encode("utf-8")
                fout.write(struct.pack("<Q", len(nb)))
                fout.write(nb)
                fout.write(struct.pack("<I", len(t["dims"])))
                fout.write(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
                out_type = GGML_TYPE_Q6_K if idx == target_idx else t["type"]
                fout.write(struct.pack("<IQ", out_type, new_offsets[idx]))

            # Pad to data_start
            cur_pos = fout.tell()
            pad = align_up(cur_pos, alignment) - cur_pos
            if pad:
                fout.write(b"\x00" * pad)
            out_data_start = fout.tell()
            assert out_data_start == data_start, "Header size unexpectedly changed"

            # 3. Stream tensor data
            slab_rows = 4096
            row_q6k_bytes = (n_per_row // Q6_K_BLOCK_SIZE) * Q6_K_TYPE_SIZE
            dst_buf = bytearray(slab_rows * row_q6k_bytes)
            dst_ptr = (ctypes.c_char * len(dst_buf)).from_buffer(dst_buf)

            for idx in by_off:
                t = tensors[idx]
                expected_pos = out_data_start + new_offsets[idx]
                if fout.tell() < expected_pos:
                    fout.write(b"\x00" * (expected_pos - fout.tell()))
                fin.seek(data_start + t["offset"])

                if idx == target_idx:
                    written = 0
                    for r0 in range(0, nrows, slab_rows):
                        rc = min(slab_rows, nrows - r0)
                        raw_bf16 = fin.read(rc * n_per_row * 2)
                        f32 = bf16_bytes_to_f32(raw_bf16)
                        qbytes = lib.ggml_quantize_chunk(
                            GGML_TYPE_Q6_K,
                            f32.ctypes.data,
                            ctypes.addressof(dst_ptr),
                            0,
                            rc,
                            n_per_row,
                            None,
                        )
                        assert qbytes == rc * row_q6k_bytes, f"Unexpected qbytes {qbytes}"
                        fout.write(memoryview(dst_buf)[:qbytes])
                        written += qbytes
                    assert written == new_q6k_bytes
                else:
                    rem = t["alloc_bytes"]
                    while rem > 0:
                        chunk = fin.read(min(rem, 16 * 1024 * 1024))
                        fout.write(chunk)
                        rem -= len(chunk)

    out_size = args.output_gguf.stat().st_size
    dt = time.time() - t0
    print(f"DONE in {dt:.1f}s! {in_size/(1024**3):.2f} GiB -> {out_size/(1024**3):.2f} GiB (saved {(in_size-out_size)/(1024**3):.2f} GiB)")


if __name__ == "__main__":
    main()
