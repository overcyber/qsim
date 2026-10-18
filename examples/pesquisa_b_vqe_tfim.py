"""Pesquisa B — Estado fundamental variacional (estilo VQE) no Ising
transversal (TFIM), otimizador clássico sobre medições do QSim.

H = -sum_i Z_i Z_{i+1} (anel) - h sum_i X_i, h=1, n=6.
Ansatz hardware-efficient: camadas de ry + cadeia cx.
Oráculo independente: diagonalização densa em numpy (H completa).

Uso: docker run --rm --network host -v $PWD:/src:ro -w /src \
        -e PYTHONPATH=/src qsim-engine python3 examples/pesquisa_b_vqe_tfim.py
"""
import numpy as np
from scipy.optimize import minimize

from sdk.qsim_client import run_sync

URL = "nats://127.0.0.1:4222"
N = 6
H_FIELD = 1.0
LAYERS = 3


def tfim_exact_ground():
    dim = 2 ** N
    mat = np.zeros((dim, dim))
    for d in range(dim):
        for i in range(N):
            si = 1 - 2 * ((d >> i) & 1)
            sj = 1 - 2 * ((d >> ((i + 1) % N)) & 1)
            mat[d, d] -= si * sj                     # -Z_i Z_{i+1}
            mat[d ^ (1 << i), d] -= H_FIELD          # -h X_i
    return float(np.linalg.eigvalsh(mat).min())


def ansatz_ops(params):
    ops = []
    p = list(params)
    for _ in range(LAYERS):
        for q in range(N):
            ops.append({"gate": "ry", "targets": [q], "params": [p.pop(0)]})
        for q in range(N - 1):
            ops.append({"gate": "cx", "targets": [q, q + 1]})
    return ops


ZZ = ["".join("Z" if i in (k, (k + 1) % N) else "I" for i in range(N))
      for k in range(N)]
XX = ["".join("X" if i == k else "I" for i in range(N)) for k in range(N)]
OBS = ZZ + XX


def energy(expect):
    zz = sum(expect[k] for k in ZZ)
    xpart = sum(expect[k] for k in XX)
    return -zz - H_FIELD * xpart


def main():
    e0 = tfim_exact_ground()
    print(f"oraculo exato (numpy, H completa): E0 = {e0:.6f}")
    rng = np.random.default_rng(2)
    x0 = rng.uniform(-1, 1, N * LAYERS)
    calls = {"n": 0, "best": 9e9}

    def objective(theta):
        payload = run_sync({"num_qubits": N, "shots": 1, "seed": 0,
                            "observables": OBS,
                            "operations": ansatz_ops(theta)},
                           url=URL, timeout=60.0)
        assert payload["status"] == "done", payload
        e = energy(payload["result"]["expectation"])
        calls["n"] += 1
        calls["best"] = min(calls["best"], e)
        return e

    minimize(objective, x0, method="COBYLA",
             options={"maxiter": 400, "rhobeg": 0.8})
    gap = calls["best"] - e0
    print(f"variacional via QSim: E = {calls['best']:.6f} "
          f"apos {calls['n']} avaliacoes")
    print(f"principio variacional (E - E0) = {gap:+.6f} (deve ser >= 0)")
    print(f"precisao relativa: {gap / abs(e0) * 100:.2f}%")
    print("\nExtensões: aumentar LAYERS/iterações para fechar o gap; varrer"
          " h e mapear a transição de fase; trocar o ansatz (ring de cx,"
          " rx/ry/rz); barulhar as medições (shots>>1) e ver o custo"
          " estatístico.")


if __name__ == "__main__":
    main()
