"""Fingerprint of a set of per-item forecast files (names, sizes, modification times)."""
import hashlib
import os


def fingerprint(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()
