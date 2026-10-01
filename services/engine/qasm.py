"""Parser OpenQASM do QSim (subconjunto das versoes 2.0 e 3.0).

Suportado:
  - Cabecalhos: OPENQASM 2.0/3; include "...";  (ignorados)
  - Registros: qubit[n] q;  |  qreg q[n];  (um unico registro quantico)
  -            bit[n] c;    |  creg c[n];  (ignorado; medicao usa qubits)
  - Portas:    h q[0];  cx q[0], q[1];  rz(pi/4) q[2];  ccx q[0],q[1],q[2];
  - Broadcast: h q;  (porta de 1 qubit aplicada a todo o registro)
  - Medicao:   measure q[0] -> c[0];  |  c[0] = measure q[0];  |  measure q;
  - Outros:    barrier ...;  reset q[0];

Parametros aceitam expressoes com pi (ex.: pi/2, 3*pi/4, -pi), avaliadas
com whitelist estrita de caracteres — nunca eval de entrada livre.
"""
from __future__ import annotations

import math
import re

MAX_SOURCE_LEN = 200_000
_PARAM_SAFE = re.compile(r"^[0-9pi+\-*/(). eE]+$")
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


class QasmError(ValueError):
    pass


def _eval_param(expr: str) -> float:
    expr = expr.strip()
    if not expr or len(expr) > 64 or not _PARAM_SAFE.match(expr):
        raise QasmError(f"Parametro invalido: '{expr}'")
    if "**" in expr:
        raise QasmError(f"Parametro invalido (exponenciacao): '{expr}'")
    # 'p'/'i'/'e'/'E' apenas como 'pi' ou expoente numerico
    if re.search(r"[pi]", expr) and re.sub(r"pi", "", expr).strip("0123456789+-*/(). eE"):
        raise QasmError(f"Parametro invalido: '{expr}'")
    try:
        value = eval(  # noqa: S307 - entrada ja filtrada por whitelist
            expr, {"__builtins__": {}}, {"pi": math.pi}
        )
    except Exception as err:
        raise QasmError(f"Parametro invalido: '{expr}' ({err})")
    f = float(value)
    if f != f or abs(f) > 1e6:
        raise QasmError(f"Parametro fora de faixa: '{expr}'")
    return f


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    source = re.sub(r"//[^\n]*", " ", source)
    return source


def parse(source: str) -> dict:
    """Converte OpenQASM para o formato JSON do QSim.

    Retorna {"num_qubits": n, "operations": [...]}.
    """
    if not isinstance(source, str) or not source.strip():
        raise QasmError("Fonte QASM vazia")
    if len(source) > MAX_SOURCE_LEN:
        raise QasmError("Fonte QASM excede o tamanho maximo")

    reg_name: str | None = None
    num_qubits = 0
    operations: list[dict] = []

    statements = [s.strip() for s in _strip_comments(source).split(";")]
    for stmt in statements:
        if not stmt:
            continue

        if re.match(r"^OPENQASM\s+[\d.]+$", stmt, re.I):
            continue
        if re.match(r"^include\s+", stmt, re.I):
            continue
        if re.match(rf"^(bit\s*\[\s*\d+\s*\]\s*{_IDENT}|creg\s+{_IDENT}\s*\[\s*\d+\s*\])$", stmt):
            continue

        m = re.match(rf"^qubit\s*\[\s*(\d+)\s*\]\s*({_IDENT})$", stmt) or \
            re.match(rf"^qreg\s+({_IDENT})\s*\[\s*(\d+)\s*\]$", stmt)
        if m:
            if reg_name is not None:
                raise QasmError("Apenas um registro quantico e suportado")
            g1, g2 = m.group(1), m.group(2)
            if g1.isdigit():
                num_qubits, reg_name = int(g1), g2
            else:
                reg_name, num_qubits = g1, int(g2)
            if not 1 <= num_qubits <= 31:
                raise QasmError("Registro deve ter entre 1 e 31 qubits")
            continue

        if reg_name is None:
            raise QasmError(f"Declaracao antes do registro: '{stmt[:40]}'")

        if re.match(r"^barrier\b", stmt):
            operations.append({"op": "barrier"})
            continue

        m = re.match(rf"^reset\s+{reg_name}\s*\[\s*(\d+)\s*\]$", stmt)
        if m:
            operations.append({"op": "reset",
                               "targets": [_idx(m.group(1), num_qubits)]})
            continue

        # measure q[i] -> c[j];  |  c[j] = measure q[i];  |  measure q;
        m = re.match(rf"^measure\s+{reg_name}\s*\[\s*(\d+)\s*\]"
                     rf"(\s*->\s*{_IDENT}\s*\[\s*\d+\s*\])?$", stmt)
        if m:
            operations.append({"op": "measure",
                               "targets": [_idx(m.group(1), num_qubits)]})
            continue
        m = re.match(rf"^{_IDENT}\s*\[\s*\d+\s*\]\s*=\s*measure\s+"
                     rf"{reg_name}\s*\[\s*(\d+)\s*\]$", stmt)
        if m:
            operations.append({"op": "measure",
                               "targets": [_idx(m.group(1), num_qubits)]})
            continue
        m = re.match(rf"^measure\s+{reg_name}(\s*->\s*{_IDENT})?$", stmt)
        if m:
            operations.append({"op": "measure",
                               "targets": list(range(num_qubits))})
            continue

        # porta: nome(params)? alvo(, alvo)*
        m = re.match(
            rf"^({_IDENT})\s*(\(([^)]*)\))?\s+(.+)$", stmt
        )
        if not m:
            raise QasmError(f"Instrucao nao reconhecida: '{stmt[:40]}'")
        gate = m.group(1).lower()
        params = None
        if m.group(3) is not None:
            params = [_eval_param(p) for p in m.group(3).split(",") if p.strip()]

        targets_src = [t.strip() for t in m.group(4).split(",")]
        if targets_src == [reg_name]:
            # broadcast de porta de 1 qubit no registro inteiro
            for q in range(num_qubits):
                operations.append(_gate_op(gate, [q], params))
            continue
        targets = []
        for t in targets_src:
            tm = re.match(rf"^{reg_name}\s*\[\s*(\d+)\s*\]$", t)
            if not tm:
                raise QasmError(f"Alvo invalido: '{t}'")
            targets.append(_idx(tm.group(1), num_qubits))
        operations.append(_gate_op(gate, targets, params))

    if reg_name is None:
        raise QasmError("Nenhum registro quantico declarado")
    if not operations:
        raise QasmError("Nenhuma operacao no circuito")
    return {"num_qubits": num_qubits, "operations": operations}


def _idx(raw: str, n: int) -> int:
    q = int(raw)
    if not 0 <= q < n:
        raise QasmError(f"Indice de qubit fora do registro: {q}")
    return q


def _gate_op(gate: str, targets: list[int],
             params: list[float] | None) -> dict:
    op: dict = {"gate": gate, "targets": targets}
    if params:
        op["params"] = params
    return op
