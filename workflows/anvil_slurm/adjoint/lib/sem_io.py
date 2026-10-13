"""Minimal reader/writer for SPECFEM3D 'unformatted' proc*.bin kernel files.

Each file is a single Fortran sequential-access record:
    [int32 marker = nbytes] [float32 x (nbytes/4)] [int32 marker = nbytes]
CUSTOM_REAL = 4 (single precision); little-endian on this build.

read_sem_bin returns the float32 array; the length is self-describing from the
record marker, so files with different NGLOB per proc are handled automatically.
write_sem_bin writes the same single-record layout. A read->write round-trip is
byte-identical to the original file (verified against ORI_KERNELS).
"""

import numpy as np

_MARK = np.dtype("<i4")
_REAL = np.dtype("<f4")


def read_sem_bin(path):
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype=_MARK, count=1)
        if head.size != 1:
            raise IOError(f"{path}: empty / truncated (no leading marker)")
        nbytes = int(head[0])
        data = np.fromfile(f, dtype=_REAL, count=nbytes // _REAL.itemsize)
        tail = np.fromfile(f, dtype=_MARK, count=1)
    if tail.size != 1 or int(tail[0]) != nbytes:
        raise IOError(f"{path}: record marker mismatch "
                      f"(lead={nbytes}, trail={tail.tolist()})")
    if data.size != nbytes // _REAL.itemsize:
        raise IOError(f"{path}: short read ({data.size} of "
                      f"{nbytes // _REAL.itemsize} floats)")
    return data


def write_sem_bin(path, data):
    data = np.ascontiguousarray(data, dtype=_REAL)
    mark = np.array([data.nbytes], dtype=_MARK)
    with open(path, "wb") as f:
        mark.tofile(f)
        data.tofile(f)
        mark.tofile(f)
