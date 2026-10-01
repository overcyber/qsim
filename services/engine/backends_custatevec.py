"""Backend cuStateVec (NVIDIA cuQuantum) — EXPERIMENTAL.

Substitui a aplicacao de portas do simulador pelos kernels otimizados da
biblioteca cuStateVec, mantendo o estado em VRAM via CuPy. O restante do
pipeline (amostragem, medicao, Bloch, entropia, observaveis) reutiliza os
caminhos CuPy da classe base, que operam sobre o mesmo array.

Requisitos:
  - GPU com compute capability >= 7.0 (Volta+): RTX 3090 (8.6) e
    RTX 4060 Ti (8.9) atendem; a Tesla P40 (Pascal 6.1) NAO — nela use
    backend "cupy";
  - pip install cupy-cuda12x cuquantum-python-cu12
    (testado com a API do cuquantum-python 23.x/24.x).

Uso: campo "backend": "custatevec" no circuito. Validacao no hardware:
PYTHONPATH=. python3 tests/test_gpu.py

Convencao de alvos: o cuStateVec ordena os bits do indice da matriz como
(targets[n-1], ..., targets[0]) — targets[0] e o bit MENOS significativo.
No QSim, targets[0] e o eixo MAIS significativo. Por isso a lista de alvos
e invertida antes da chamada. O teste diferencial custatevec vs cupy em
tests/test_gpu.py e o criterio de aceitacao dessa convencao.
"""
from __future__ import annotations

import numpy as np

from .simulator import StatevectorSimulator


class CuStateVecSimulator(StatevectorSimulator):
    """Simulador com aplicacao de portas via cuStateVec (estado em VRAM)."""

    def __init__(self, num_qubits: int, seed: int | None = None,
                 precision: str = "complex128"):
        super().__init__(num_qubits, seed=seed, precision=precision,
                         backend="cupy")
        self.backend_name = "custatevec"
        try:
            import cupy as cp
            from cuquantum import ComputeType, cudaDataType
            from cuquantum import custatevec as cusv
        except ImportError as err:
            raise RuntimeError(
                "Backend custatevec requer cupy e cuquantum-python "
                f"(pip install cupy-cuda12x cuquantum-python-cu12): {err}"
            )
        cc = int(cp.cuda.Device().compute_capability)
        if cc < 70:
            raise RuntimeError(
                f"cuStateVec requer compute capability >= 7.0; esta GPU tem "
                f"{cc / 10:.1f}. Na Tesla P40 use backend 'cupy'."
            )
        self._cp = cp
        self._cusv = cusv
        self._handle = cusv.create()
        if self.dtype == np.complex64:
            self._sv_dtype = cudaDataType.CUDA_C_32F
            self._mat_dtype = cudaDataType.CUDA_C_32F
            self._compute = ComputeType.COMPUTE_32F
        else:
            self._sv_dtype = cudaDataType.CUDA_C_64F
            self._mat_dtype = cudaDataType.CUDA_C_64F
            self._compute = ComputeType.COMPUTE_64F

    def __del__(self):  # pragma: no cover
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                self._cusv.destroy(handle)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def apply_matrix(self, mat, targets: list[int]) -> None:
        cusv = self._cusv
        k = len(targets)
        if getattr(mat, "shape", None) != (2**k, 2**k):
            raise ValueError("Dimensao da matriz incompativel com os alvos")
        if len(set(targets)) != k:
            raise ValueError("Qubits alvo repetidos")
        for t in targets:
            if not isinstance(t, int) or not 0 <= t < self.n:
                raise ValueError(f"Qubit alvo fora do registro: {t}")

        # Matriz na memoria do host, mesma precisao do estado, row-major.
        mat_h = np.ascontiguousarray(np.asarray(mat, dtype=self.dtype))
        # Inversao de convencao (ver docstring do modulo).
        cusv_targets = np.asarray(list(reversed(targets)), dtype=np.int32)
        empty = np.asarray([], dtype=np.int32)

        layout = cusv.MatrixLayout.ROW
        ws_size = cusv.apply_matrix_get_workspace_size(
            self._handle, self._sv_dtype, self.n,
            mat_h.ctypes.data, self._mat_dtype, layout, 0,
            k, 0, self._compute,
        )
        if ws_size > 0:
            workspace = self._cp.cuda.alloc(ws_size)
            ws_ptr = workspace.ptr
        else:
            workspace = None
            ws_ptr = 0

        cusv.apply_matrix(
            self._handle,
            self.state.data.ptr, self._sv_dtype, self.n,
            mat_h.ctypes.data, self._mat_dtype, layout, 0,
            cusv_targets.ctypes.data, k,
            empty.ctypes.data, 0, 0,
            self._compute, ws_ptr, ws_size,
        )
        del workspace
