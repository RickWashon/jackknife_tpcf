import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np


_C_FILE = Path(__file__).with_name("weighted_dd_counter.c")


def _shared_object_path() -> Path:
    digest = hashlib.md5(str(_C_FILE.resolve()).encode("utf-8")).hexdigest()[:12]
    build_dir = Path(tempfile.gettempdir()) / "utils_weighted_dd_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir / f"weighted_dd_counter_{digest}.so"


def _compile_shared_object(so_file: Path) -> None:
    cflags = ["-O3", "-fopenmp"]
    # Enable AVX2 path only when CPU flags indicate support.
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
        raise RuntimeError(f"Failed to compile weighted_dd_counter.c: {stderr}") from exc


def _ensure_library() -> Path:
    so_file = _shared_object_path()
    if not so_file.exists() or so_file.stat().st_mtime < _C_FILE.stat().st_mtime:
        _compile_shared_object(so_file)
    return so_file


def _load_library() -> ctypes.CDLL:
    return ctypes.CDLL(str(_ensure_library()))


def _setup_auto_signature(lib: ctypes.CDLL) -> None:
    fn = lib.weighted_dd_1h2h_auto
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
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
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_int


def _setup_cross_signature(lib: ctypes.CDLL) -> None:
    fn = lib.weighted_dd_1h2h_cross
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
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
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_int


def weighted_dd_1h2h_auto(
    sample_xyz: np.ndarray,
    host_halo_id: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int = 1,
    approx_cell_size: float = None,
    refine_factor: int = 2,
    max_cells_per_dim: int = 100,
    use_float32: bool = False,
) -> Dict[str, np.ndarray]:
    arr = np.ascontiguousarray(np.asarray(sample_xyz, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sample_xyz must have shape (N, 3)")
    # ctypes passes raw pointers; ensure each coordinate array is truly contiguous.
    x = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
    z = np.ascontiguousarray(arr[:, 2], dtype=np.float64)

    host = np.ascontiguousarray(np.asarray(host_halo_id, dtype=np.int64))
    if host.ndim != 1 or host.size != arr.shape[0]:
        raise ValueError("host_halo_id must be a 1D array with length N")

    edges = np.ascontiguousarray(np.asarray(rbins, dtype=np.float64))
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("rbins must be a 1D array with at least 2 edges")

    nbins = edges.size - 1
    dd_total = np.zeros(nbins, dtype=np.uint64)
    dd_1h = np.zeros(nbins, dtype=np.uint64)
    dd_2h = np.zeros(nbins, dtype=np.uint64)

    lib = _load_library()
    _setup_auto_signature(lib)
    if approx_cell_size is None:
        approx_cell_size = float(boxsize) / 10.0
    rc = lib.weighted_dd_1h2h_auto(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        host.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(arr.shape[0]),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(nbins),
        ctypes.c_double(float(boxsize)),
        ctypes.c_int(int(nthreads)),
        ctypes.c_double(float(approx_cell_size)),
        ctypes.c_int(int(refine_factor)),
        ctypes.c_int(int(max_cells_per_dim)),
        ctypes.c_int(1 if use_float32 else 0),
        dd_total.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        dd_1h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        dd_2h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    if rc != 0:
        raise RuntimeError(f"weighted_dd_1h2h_auto failed with status code {rc}")

    return {
        "dd_total": dd_total.astype(np.float64),
        "dd_1h": dd_1h.astype(np.float64),
        "dd_2h": dd_2h.astype(np.float64),
    }


def weighted_dd_1h2h_cross(
    sample1_xyz: np.ndarray,
    host1_halo_id: np.ndarray,
    sample2_xyz: np.ndarray,
    host2_halo_id: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int = 1,
    approx_cell_size: float = None,
    refine_factor: int = 2,
    max_cells_per_dim: int = 100,
    use_float32: bool = False,
) -> Dict[str, np.ndarray]:
    arr1 = np.ascontiguousarray(np.asarray(sample1_xyz, dtype=np.float64))
    arr2 = np.ascontiguousarray(np.asarray(sample2_xyz, dtype=np.float64))
    if arr1.ndim != 2 or arr1.shape[1] != 3:
        raise ValueError("sample1_xyz must have shape (N1, 3)")
    if arr2.ndim != 2 or arr2.shape[1] != 3:
        raise ValueError("sample2_xyz must have shape (N2, 3)")

    x1 = np.ascontiguousarray(arr1[:, 0], dtype=np.float64)
    y1 = np.ascontiguousarray(arr1[:, 1], dtype=np.float64)
    z1 = np.ascontiguousarray(arr1[:, 2], dtype=np.float64)
    x2 = np.ascontiguousarray(arr2[:, 0], dtype=np.float64)
    y2 = np.ascontiguousarray(arr2[:, 1], dtype=np.float64)
    z2 = np.ascontiguousarray(arr2[:, 2], dtype=np.float64)

    host1 = np.ascontiguousarray(np.asarray(host1_halo_id, dtype=np.int64))
    host2 = np.ascontiguousarray(np.asarray(host2_halo_id, dtype=np.int64))
    if host1.ndim != 1 or host1.size != arr1.shape[0]:
        raise ValueError("host1_halo_id must be a 1D array with length N1")
    if host2.ndim != 1 or host2.size != arr2.shape[0]:
        raise ValueError("host2_halo_id must be a 1D array with length N2")

    edges = np.ascontiguousarray(np.asarray(rbins, dtype=np.float64))
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("rbins must be a 1D array with at least 2 edges")

    nbins = edges.size - 1
    dd_total = np.zeros(nbins, dtype=np.uint64)
    dd_1h = np.zeros(nbins, dtype=np.uint64)
    dd_2h = np.zeros(nbins, dtype=np.uint64)

    lib = _load_library()
    _setup_cross_signature(lib)
    if approx_cell_size is None:
        approx_cell_size = float(boxsize) / 10.0

    rc = lib.weighted_dd_1h2h_cross(
        x1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        host1.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(arr1.shape[0]),
        x2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        host2.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(arr2.shape[0]),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(nbins),
        ctypes.c_double(float(boxsize)),
        ctypes.c_int(int(nthreads)),
        ctypes.c_double(float(approx_cell_size)),
        ctypes.c_int(int(refine_factor)),
        ctypes.c_int(int(max_cells_per_dim)),
        ctypes.c_int(1 if use_float32 else 0),
        dd_total.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        dd_1h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        dd_2h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    if rc != 0:
        raise RuntimeError(f"weighted_dd_1h2h_cross failed with status code {rc}")

    return {
        "dd_total": dd_total.astype(np.float64),
        "dd_1h": dd_1h.astype(np.float64),
        "dd_2h": dd_2h.astype(np.float64),
    }
