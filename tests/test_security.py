"""Suite de seguranca: entradas hostis devem ser rejeitadas com ValueError
controlado — nunca travar, estourar memoria ou executar parcialmente.

Cobre a camada compartilhada (shared.validation, usada pelo gateway E pelo
worker) e as barreiras internas do engine.
"""
from services.engine.simulator import run_circuit
from shared.validation import validate_circuit

BASE = {
    "num_qubits": 2,
    "shots": 100,
    "operations": [{"gate": "h", "targets": [0]}],
}


def _rejected(circuit: dict, label: str) -> None:
    try:
        validate_circuit(circuit)
    except ValueError:
        print(f"  rejeitado como esperado: {label}")
        return
    raise AssertionError(f"validacao ACEITOU entrada hostil: {label}")


def test_resource_exhaustion_vectors():
    """Vetores de negacao de servico por consumo de recursos."""
    _rejected({**BASE, "num_qubits": 64}, "64 qubits (2^64 amplitudes)")
    _rejected({**BASE, "num_qubits": 0}, "0 qubits")
    _rejected({**BASE, "num_qubits": -3}, "qubits negativos")
    _rejected({**BASE, "num_qubits": 2**40}, "qubits absurdos")
    _rejected({**BASE, "shots": 10**9}, "1 bilhao de shots")
    _rejected({**BASE, "shots": 0}, "0 shots")
    _rejected(
        {**BASE, "operations": [{"gate": "h", "targets": [0]}] * 50_000},
        "50 mil operacoes",
    )


def test_type_confusion_vectors():
    """Confusao de tipos: strings, bools e floats onde se espera int."""
    _rejected({**BASE, "num_qubits": "20"}, "num_qubits string")
    _rejected({**BASE, "num_qubits": True}, "num_qubits booleano")
    _rejected({**BASE, "num_qubits": 2.5}, "num_qubits float")
    _rejected({**BASE, "shots": "muitos"}, "shots string")
    _rejected({**BASE, "seed": "abc"}, "seed string")
    _rejected({**BASE, "operations": "h(0)"}, "operations string")
    _rejected("nao sou um dict", "circuito string")
    _rejected({**BASE, "noise": {"depolarizing": "sim"}}, "ruido string")


def test_injection_and_structure_vectors():
    """Campos e estruturas fora do contrato sao rejeitados (nega por padrao)."""
    _rejected({**BASE, "__class__": "x"}, "chave dunder extra")
    _rejected({**BASE, "cmd": "rm -rf /"}, "campo desconhecido")
    _rejected(
        {**BASE, "operations": [{"gate": "h; drop table", "targets": [0]}]},
        "nome de porta com payload",
    )
    _rejected(
        {**BASE, "operations": [{"op": "shell", "targets": [0]}]},
        "tipo de operacao desconhecido",
    )
    _rejected(
        {**BASE, "operations": [{"gate": "h" * 100, "targets": [0]}]},
        "nome de porta gigante",
    )
    _rejected(
        {**BASE, "observables": ["z0; select"]},
        "observavel malformado",
    )


def test_bounds_vectors():
    """Alvos e parametros fora de faixa."""
    _rejected(
        {**BASE, "operations": [{"gate": "h", "targets": [5]}]},
        "alvo fora do registro",
    )
    _rejected(
        {**BASE, "operations": [{"gate": "h", "targets": [-1]}]},
        "alvo negativo",
    )
    _rejected(
        {**BASE, "operations": [{"gate": "cx", "targets": [0, 0]}]},
        "alvos repetidos",
    )
    _rejected(
        {**BASE, "operations": [{"gate": "rz", "targets": [0],
                                 "params": [float("nan")]}]},
        "parametro NaN",
    )
    _rejected(
        {**BASE, "operations": [{"gate": "rz", "targets": [0],
                                 "params": [float("inf")]}]},
        "parametro infinito",
    )
    _rejected(
        {**BASE, "operations": [{"gate": "rz", "targets": [0],
                                 "params": [1e12]}]},
        "parametro gigantesco",
    )
    _rejected({**BASE, "noise": {"depolarizing": 1.5}}, "probabilidade > 1")
    _rejected({**BASE, "noise": {"depolarizing": -0.1}}, "probabilidade < 0")


def test_engine_internal_barriers():
    """Mesmo pulando a validacao, o engine mantem barreiras proprias."""
    for bad in [
        {"num_qubits": 40, "operations": [{"gate": "h", "targets": [0]}]},
        {"num_qubits": 2, "shots": 10**8,
         "operations": [{"gate": "h", "targets": [0]}]},
        {"num_qubits": 2,
         "operations": [{"gate": "hack", "targets": [0]}]},
        {"num_qubits": 2,
         "operations": [{"gate": "cx", "targets": [0, 0]}]},
        {"num_qubits": 2, "operations": []},
    ]:
        try:
            run_circuit(bad)
        except (ValueError, KeyError):
            continue
        raise AssertionError(f"engine executou entrada hostil: {bad}")
    print("  barreiras internas do engine OK")


def test_statevector_payload_cap():
    """return_statevector nao pode virar canal de exfiltracao de memoria:
    acima do limite, o vetor simplesmente nao e serializado."""
    ops = [{"gate": "h", "targets": [i]} for i in range(16)]
    result = run_circuit({"num_qubits": 16, "shots": 1, "seed": 0,
                          "return_statevector": True, "operations": ops})
    assert result.statevector is None
    assert len(result.probabilities) <= 256
    print("  teto de payload do statevector e probabilities OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"{name}:")
            fn()
    print("\nSuite de seguranca: todos os testes passaram.")
