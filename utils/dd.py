import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np


_C_FILE = Path(__file__).with_name("dd_counter.c")


def _shared_object_path() -> Path:
    digest = hashlib.md5(str(_C_FILE.resolve()).encode("utf-8")).hexdigest()[:12]
    build_dir = Path(tempfile.gettempdir()) / "utils_dd_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir / f"dd_counter_{digest}.so"


def _compile_shared_object(so_file: Path) -> None:
    cflags = ["-O3", "-fopenmp"]
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").lower()
        if "avx2" in cpuinfo:
            cflags.extend(["-mavx2", "-mfma"])
    except OSError:
        pass

    extra_cflags = os.environ.get("WEIGHTED_DD_CFLAGS", "").strip()
    if extra_cflags:
        cflags.extend(extra_cflags.split())

    cmd = [
        "gcc",
        *cflags,
        "-shared",
        "-fPIC",
        "-o",
        str(so_file),
        str(_C_FILE),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"Failed to compile dd_counter.c: {stderr}") from exc


def _ensure_library() -> Path:
    so_file = _shared_object_path()
    if not so_file.exists() or so_file.stat().st_mtime < _C_FILE.stat().st_mtime:
        _compile_shared_object(so_file)
    return so_file


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_ensure_library()))
    fn = lib.dd_auto_no_weight
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_int
    return lib


def dd_auto(
    sample_xyz: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int = 1,
    approx_cell_size: Optional[float] = None,
    refine_factor: int = 2,
    max_cells_per_dim: int = 100,
    use_float32: bool = False,
) -> Dict[str, np.ndarray]:
    arr = np.ascontiguousarray(np.asarray(sample_xyz, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sample_xyz must have shape (N, 3)")

    x = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
    z = np.ascontiguousarray(arr[:, 2], dtype=np.float64)

    edges = np.ascontiguousarray(np.asarray(rbins, dtype=np.float64))
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("rbins must be a 1D array with at least 2 edges")

    nbins = edges.size - 1
    dd_counts = np.zeros(nbins, dtype=np.uint64)

    lib = _load_library()
    if approx_cell_size is None:
        approx_cell_size = float(boxsize) / 10.0

    rc = lib.dd_auto_no_weight(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(arr.shape[0]),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(nbins),
        ctypes.c_double(float(boxsize)),
        ctypes.c_int(int(nthreads)),
        ctypes.c_double(float(approx_cell_size)),
        ctypes.c_int(int(refine_factor)),
        ctypes.c_int(int(max_cells_per_dim)),
        ctypes.c_int(1 if use_float32 else 0),
        dd_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    if rc != 0:
        raise RuntimeError(f"dd_auto_no_weight failed with status code {rc}")

    return {"dd_counts": dd_counts.astype(np.float64)}
