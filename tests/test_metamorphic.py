"""Suite metamorfica: identidades algebricas que devem valer para QUALQUER
circuito, sem precisar conhecer o resultado analitico.

Complementa o oraculo fisico: cobre combinacoes que ninguem calculou a mao.
"""
import numpy as np

from services.engine.simulator import StatevectorSimulator


def _random_state(n: int, seed: int) -> StatevectorSimulator:
    """Prepara um estado pseudo-aleatorio nao trivial."""
    rng = np.random.default_rng(seed)
    sim = StatevectorSimulator(n, seed=seed)
    for _ in range(30):
        sim.apply_gate(str(rng.choice(["h", "t", "sx"])),
                       [int(rng.integers(0, n))])
        a, b = rng.choice(n, size=2, replace=False)
        sim.apply_gate("cx", [int(a), int(b)])
    return sim


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def test_inverse_circuit_returns_identity():
    """U seguido de U-adaga devolve o estado inicial."""
    rng = np.random.default_rng(1)
    inverses = {"h": ("h", None), "x": ("x", None), "y": ("y", None),
                "z": ("z", None), "s": ("sdg", None), "t": ("tdg", None),
                "cx": ("cx", None), "cz": ("cz", None), "swap": ("swap", None)}
    n = 7
    sim = StatevectorSimulator(n, seed=0)
    initial = sim.state.copy()
    applied = []
    for _ in range(150):
        name = str(rng.choice(list(inverses.keys())))
        nq = 2 if name in ("cx", "cz", "swap") else 1
        targets = [int(q) for q in rng.choice(n, size=nq, replace=False)]
        sim.apply_gate(name, targets)
        applied.append((name, targets))
    for name, targets in reversed(applied):
        inv, _ = inverses[name]
        sim.apply_gate(inv, targets)
    fid = _fidelity(initial, sim.state)
    assert fid > 1 - 1e-10, f"fidelidade U Udg = {fid}"
    print(f"Inversao OK: fidelidade {fid:.14f} apos 300 portas")


def test_hzh_equals_x():
    for q in range(5):
        s1 = _random_state(5, seed=q)
        s2 = StatevectorSimulator(5, seed=q)
        s2.state = s1.state.copy()
        s1.apply_gate("h", [q]); s1.apply_gate("z", [q]); s1.apply_gate("h", [q])
        s2.apply_gate("x", [q])
        assert _fidelity(s1.state, s2.state) > 1 - 1e-12
    print("Identidade HZH = X OK em todos os qubits")


def test_ss_equals_z_and_tt_equals_s():
    s1 = _random_state(4, seed=3)
    s2 = StatevectorSimulator(4, seed=3)
    s2.state = s1.state.copy()
    s1.apply_gate("s", [2]); s1.apply_gate("s", [2])
    s2.apply_gate("z", [2])
    assert _fidelity(s1.state, s2.state) > 1 - 1e-12
    s1.apply_gate("t", [1]); s1.apply_gate("t", [1])
    s2.apply_gate("s", [1])
    assert _fidelity(s1.state, s2.state) > 1 - 1e-12
    print("Identidades SS = Z e TT = S OK")


def test_rotation_composition():
    """RZ(a) RZ(b) = RZ(a+b) em estado arbitrario."""
    a, b = 0.7, 1.9
    s1 = _random_state(4, seed=5)
    s2 = StatevectorSimulator(4, seed=5)
    s2.state = s1.state.copy()
    s1.apply_gate("rz", [0], [a]); s1.apply_gate("rz", [0], [b])
    s2.apply_gate("rz", [0], [a + b])
    assert _fidelity(s1.state, s2.state) > 1 - 1e-12
    print("Composicao RZ(a) RZ(b) = RZ(a+b) OK")


def test_swap_equals_three_cnots():
    s1 = _random_state(3, seed=8)
    s2 = StatevectorSimulator(3, seed=8)
    s2.state = s1.state.copy()
    s1.apply_gate("swap", [0, 2])
    s2.apply_gate("cx", [0, 2]); s2.apply_gate("cx", [2, 0]); s2.apply_gate("cx", [0, 2])
    assert _fidelity(s1.state, s2.state) > 1 - 1e-12
    print("SWAP = CX CX CX OK")


def test_global_phase_invariance_of_probabilities():
    """Z X Z X = -I: fase global nao altera probabilidades."""
    s1 = _random_state(4, seed=13)
    probs_before = np.abs(s1.state) ** 2
    for _ in range(1):
        s1.apply_gate("z", [1]); s1.apply_gate("x", [1])
        s1.apply_gate("z", [1]); s1.apply_gate("x", [1])
    probs_after = np.abs(s1.state) ** 2
    assert np.allclose(probs_before, probs_after, atol=1e-12)
    print("Invariancia de fase global OK")


def test_toffoli_from_controls():
    """CCX so age quando ambos os controles estao em |1>."""
    for c0, c1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sim = StatevectorSimulator(3, seed=0)
        if c0:
            sim.apply_gate("x", [0])
        if c1:
            sim.apply_gate("x", [1])
        sim.apply_gate("ccx", [0, 1, 2])
        expected_t = 1 if (c0 and c1) else 0
        idx = c0 | (c1 << 1) | (expected_t << 2)
        assert abs(abs(sim.state[idx]) - 1.0) < 1e-12
    print("Tabela-verdade do Toffoli OK")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nSuite metamorfica: todos os testes passaram.")
