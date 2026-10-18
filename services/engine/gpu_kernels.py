"""Kernels CUDA in-place para o simulador (P1.1 do plano de melhoria).

O caminho generico (moveaxis + matmul) aloca 2-3 buffers do tamanho do
estado por porta; acima de ~29 qubits isso excede a VRAM. Estes kernels
aplicam portas de 1-3 qubits e avaliam observaveis Pauli DIRETAMENTE no
buffer do estado, sem nenhuma alocacao proporcional — 31 qubits c64
(8 GiB de estado) passam a caber em uma P40 de 24 GB.

Requisitos: CuPy com toolkit (headers) disponivel para compilacao JIT
(cupy-cuda12x[ctk]); os binarios compilados sao cacheados em
CUPY_CACHE_DIR (volume persistente no deploy). Se a compilacao falhar,
o simulador volta ao caminho generico — kernels sao aceleracao, nao
dependencia.

Convencao de indices (identica ao nucleo NumPy): qubit t = bit t do
indice do vetor de estado (little-endian). Na matriz da porta, a linha j
tem targets[0] como bit MAIS significativo — mesma ordem do caminho
moveaxis+reshape(2^k, -1), garantida pelos testes diferenciais
(tests/test_inplace.py).
"""
from __future__ import annotations

import numpy as np

_SRC = r"""
extern "C" {

// ------------------------------------------------------------------
// Aplicacao de porta de k qubits (k <= 3): um grupo (2^k amplitudes) por
// thread. `base` e o indice com bits zero nas posicoes alvo; a linha j da
// matriz corresponde aos bits (targets[0]=MSB ... targets[k-1]=LSB).
// ------------------------------------------------------------------
__device__ __forceinline__ __T__ cmul(__T__ a, __T__ b) {
    __T__ r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

__global__ void qsim_apply_gate(const __T__* __restrict__ mat,
                                __T__* __restrict__ state,
                                long long ngroups, int n, int k,
                                const int* __restrict__ tmasks) {
    long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= ngroups) return;

    long long base;
    if (k == 1) {
        // caminho O(1): insere um bit 0 na posicao t deslocando os
        // superiores — equivale a iteracao de bits, sem o loop O(n)
        long long mask = (1LL << tmasks[0]) - 1;
        base = (g & mask) | ((g & ~mask) << 1);
    } else {
        // caminho generico (k = 2, 3): distribui os bits de g nas posicoes
        // que NAO sao alvo, cada bit posicionado no seu qubit real
        base = 0;
        long long rest = g;
        for (int b = n - 1; b >= 0; b--) {
            bool is_target = false;
            for (int i = 0; i < k; i++) if (tmasks[i] == b) { is_target = true; break; }
            if (!is_target) {
                base |= (rest & 1) << b;
                rest >>= 1;
            }
        }
    }

    const int dim = 1 << k;
    __T__ v[8];
    long long idx[8];
    for (int j = 0; j < dim; j++) {
        long long off = 0;
        for (int i = 0; i < k; i++) {
            if ((j >> (k - 1 - i)) & 1) off |= 1LL << tmasks[i];
        }
        idx[j] = base | off;
        v[j] = state[idx[j]];
    }
    for (int j = 0; j < dim; j++) {
        __T__ acc; acc.x = 0; acc.y = 0;
        for (int i = 0; i < dim; i++) {
            acc = cadd(acc, cmul(mat[(long long)j * dim + i], v[i]));
        }
        state[idx[j]] = acc;
    }
}

__device__ __forceinline__ __T__ cadd(__T__ a, __T__ b) {
    __T__ r; r.x = a.x + b.x; r.y = a.y + b.y; return r;
}

// ------------------------------------------------------------------
// Porta diagonal (rz, cz, p, cp): fase por amplitude, sem gather.
// ------------------------------------------------------------------
__global__ void qsim_apply_diag(const __T__* __restrict__ diag,
                                __T__* __restrict__ state,
                                long long numel, int k,
                                const int* __restrict__ tmasks) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numel) return;
    int j = 0;
    for (int t = 0; t < k; t++) {
        if ((i >> tmasks[t]) & 1) j |= 1 << (k - 1 - t);
    }
    state[i] = cmul(diag[j], state[i]);
}

// ------------------------------------------------------------------
// Observaveis Pauli (P1.3): uma passada, sem copia do estado.
//   qsim_pauli_z:  <P> para P com apenas I/Z   -> soma de |a_i|^2 com sinal
//   qsim_pauli_xy: <P> com X/Y no qubit q0 (menor indice) -> pares
//                  (i0, i0 ^ xymask) com fases por amplitude.
// Acumulo por atomicAdd (double suportado a partir de sm_60).
// ------------------------------------------------------------------
__global__ void qsim_pauli_z(unsigned long long zmask,
                             const __T__* __restrict__ state,
                             long long numel, __BASE__* out) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numel) return;
    unsigned long long bits = (unsigned long long)i & zmask;
    int pop = 0;
    while (bits) { pop += (int)(bits & 1ULL); bits >>= 1; }
    __BASE__ s = (pop & 1) ? (__BASE__)-1.0 : (__BASE__)1.0;
    __BASE__ p = state[i].x * state[i].x + state[i].y * state[i].y;
    atomicAdd(out, s * p);
}

__global__ void qsim_pauli_xy(int q0, unsigned long long xmask,
                              unsigned long long ymask,
                              unsigned long long zmask,
                              const __T__* __restrict__ state,
                              long long npairs, int n, __BASE__* out) {
    long long p = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= npairs) return;

    // i0: par com bit q0 = 0, calculo O(1) (mesma insercao de bit do kernel
    // de portas k=1)
    long long mask = (1LL << q0) - 1;
    long long i0 = (p & mask) | ((p & ~mask) << 1);
    long long j = i0 ^ (long long)(xmask | ymask);  // inverte TODOS os X/Y

    __T__ a0 = state[i0], a1 = state[j];

    // fase de (P psi)_i para um indice i: Y_q -> +-i conforme o bit;
    // Z_q -> -1 se o bit esta setado.
    __T__ ph0; ph0.x = 1; ph0.y = 0;
    __T__ ph1; ph1.x = 1; ph1.y = 0;
    for (int b = 0; b < n; b++) {
        unsigned long long bit = 1ULL << b;
        if (ymask & bit) {
            __T__ f0, f1;  // +i se bit setado, -i caso contrario
            f0.x = 0; f0.y = (i0 & bit) ? 1 : -1;
            f1.x = 0; f1.y = (j & bit) ? 1 : -1;
            ph0 = cmul(ph0, f0);
            ph1 = cmul(ph1, f1);
        } else if (zmask & bit) {
            if (i0 & bit) { ph0.x = -ph0.x; ph0.y = -ph0.y; }
            if (j & bit)  { ph1.x = -ph1.x; ph1.y = -ph1.y; }
        }
    }
    // v = conj(a0)*ph0*a1 + conj(a1)*ph1*a0  (real por construcao)
    __T__ c0; c0.x = a0.x; c0.y = -a0.y;
    __T__ c1; c1.x = a1.x; c1.y = -a1.y;
    __T__ t0 = cmul(cmul(c0, ph0), a1);
    __T__ t1 = cmul(cmul(c1, ph1), a0);
    atomicAdd(out, (__BASE__)(t0.x + t1.x));
}

}  // extern "C"
"""


def _materialize(src: str, cpt: str, base: str) -> str:
    out = src
    out = out.replace("__T__", cpt).replace("__BASE__", base)
    # cadd e usado antes de sua definicao no fonte; em CUDA isso e valido
    # apenas com prototipo — inserir prototipo no topo.
    proto = f"__device__ __forceinline__ {cpt} cadd({cpt} a, {cpt} b);\n"
    return out.replace("extern \"C\" {\n", "extern \"C\" {\n" + proto, 1)


class GpuKernels:
    """Compila (uma vez por dtype, cacheado em disco) e expoe os kernels."""

    def __init__(self, cp, dtype):
        self.cp = cp
        self.dtype = dtype
        if dtype == np.complex64:
            cpt, base, np_base = "float2", "float", np.float32
        else:
            cpt, base, np_base = "double2", "double", np.float64
        src = _materialize(_SRC, cpt, base)
        module = cp.RawModule(code=src, options=("-std=c++11",),
                              name_expressions=[])
        self._gate = module.get_function("qsim_apply_gate")
        self._diag = module.get_function("qsim_apply_diag")
        self._pz = module.get_function("qsim_pauli_z")
        self._pxy = module.get_function("qsim_pauli_xy")
        self._np_base = np_base
        self._blocks = 256

    # ------------------------------------------------------------------
    def apply(self, state, mat, targets: list[int], n: int) -> None:
        """Aplica `mat` (2^k x 2^k) in-place nos qubits `targets`."""
        cp = self.cp
        k = len(targets)
        mat_d = cp.asarray(mat, dtype=self.dtype)
        # indices dos qubits alvo (NAO mascaras: o kernel faz 1LL << tmasks[i])
        tmasks = cp.asarray(targets, dtype=cp.int32)

        # caminho diagonal: fase por amplitude, sem gather
        diag = cp.diag(mat_d)
        dense = mat_d - cp.diag(diag)
        if int(cp.count_nonzero(dense)) == 0:
            numel = state.size
            grid = (int((numel + self._blocks - 1) // self._blocks),)
            self._diag(grid, (self._blocks,),
                       (diag, state, np.int64(numel), np.int32(k), tmasks))
            return

        ngroups = state.size >> k
        grid = (int((ngroups + self._blocks - 1) // self._blocks),)
        self._gate(grid, (self._blocks,),
                   (mat_d, state, np.int64(ngroups), np.int32(n),
                    np.int32(k), tmasks))

    # ------------------------------------------------------------------
    def pauli_expectation(self, state, pauli: str, n: int) -> float:
        """<psi|P|psi> em uma passada sobre o estado, sem copia."""
        cp = self.cp
        xmask = zmask = ymask = 0
        for q, ch in enumerate(pauli):
            if ch == "X":
                xmask |= 1 << q
            elif ch == "Y":
                ymask |= 1 << q
            elif ch == "Z":
                zmask |= 1 << q
        out = cp.zeros(1, dtype=self._np_base)
        blocks = self._blocks
        if xmask == 0 and ymask == 0:
            if zmask == 0:
                return 1.0
            grid = (int((state.size + blocks - 1) // blocks),)
            self._pz(grid, (blocks,),
                     (np.uint64(zmask), state, np.int64(state.size), out))
            return float(out[0])
        q0 = (xmask | ymask) & -(xmask | ymask)  # menor bit setado
        q0 = q0.bit_length() - 1
        npairs = state.size >> 1
        grid = (int((npairs + blocks - 1) // blocks),)
        self._pxy(grid, (blocks,),
                  (np.int32(q0), np.uint64(xmask), np.uint64(ymask),
                   np.uint64(zmask), state, np.int64(npairs), np.int32(n),
                   out))
        return float(out[0])
