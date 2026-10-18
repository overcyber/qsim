# QSim — Guia de Testes Práticos

Receitas prontas para experimentar o simulador, **todas executadas e
validadas** contra a stack viva (P40 para circuitos exatos; worker CPU para
ruído; números desta seção são saídas reais medidos em 2026-10-18).

Pré-requisitos:
- Stack no ar: `cd qsim && docker compose -f docker-compose.yml -f docker-compose.test-override.yml --profile gpu up -d nats scheduler engine engine-gpu2 gateway prometheus grafana dcgm-exporter nats-exporter`
- Token de acesso: `grep '^QSIM_ADMIN_TOKEN=' .env` (campo no topo do console)
- Limite atual de qubits: `QSIM_MAX_QUBITS` no `.env` (hoje: 20; teto do motor 31)

Índice: [console](#console) · [curl](#curl) · [SDK/varreduras](#sdk) ·
[o que esperar](#esperar) · [erros comuns](#erros)

---

## <a name="console"></a>1. Pelo console web (http://localhost:10000)

1. **Ctrl+F5** (limpa cache antigo), cole o token no campo do topo, aguarde
   "WS: conectado".
2. **Bell com verificação**: aba *Editor* → clique em `H` e depois na célula
   q0/coluna 1; clique em `CX` e depois em q0 e q1 **na mesma coluna** →
   "Executar simulação". No resultado: barras 50/50 em |00⟩/|11⟩, linha
   "verificação complex128 vs complex64: APROVADA", esferas de Bloch com
   r=0 (estado misto reduzido) e entropia 1 bit/qubit.
3. **QASM**: aba *QASM* → cole `OPENQASM 3; qubit[3] q; h q[0]; cx q[0],q[1]; cx q[1],q[2];` → executar → histograma 000/111.
4. **Observáveis**: campo "Observaveis" = `ZZ, XX` para 2 qubits (ou string
   Pauli do tamanho do registro, ex. `ZIXY` para 4).
5. **Ruído**: "Ruido depol." = 0.05 → o histograma ganha estados impuros
   (01/10) — a classe `noisy` roda no worker CPU com 14 workers.
6. **Motor**: `auto` usa o melhor caminho do worker (cupy/kernels na P40;
   cuStateVec em placas Volta+; tableau Clifford automático >20q Clifford).
7. **Comparador A vs B**: execute o mesmo circuito com e sem ruído e
   compare no painel de baixo (mostra TVD).
8. **Exportações**: botões `exportar json/csv/latex` no resultado (o LaTeX
   sai pronto para artigos, com `\usepackage{braket}`).

## <a name="curl"></a>2. Pela API REST (curl)

    TOKEN=$(grep '^QSIM_ADMIN_TOKEN=' .env | cut -d= -f2)
    curl -s -X POST http://localhost:10000/api/jobs \
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d '{"num_qubits": 2, "shots": 4096, "seed": 7, "verify": true,
           "return_entanglement": true, "observables": ["ZZ","XX"],
           "operations": [{"gate":"h","targets":[0]},{"gate":"cx","targets":[0,1]}]}'
    # -> {"job_id":"...","status":"queued"}
    curl -s -H "Authorization: Bearer $TOKEN" http://localhost:10000/api/jobs/<ID>

Saída real medida (Bell, 4096 shots): `counts {00: 2066, 11: 2030}`,
`<ZZ>=<XX>=1.0`, entropia 1.0 bit/qubit, verificação APROVADA, **1,5 s**.

**GHZ-20 na GPU (P40, kernels in-place)** — saída real: counts
{0²⁰: 2061, 1²⁰: 2035} em **22 ms**:

    curl -s -X POST .../api/jobs -H ... -d '{"num_qubits":20,"shots":4096,
      "seed":1,"precision":"complex64",
      "operations":[{"gate":"h","targets":[0]}]
      + [{"gate":"cx","targets":[0,1]},{"gate":"cx","targets":[1,2]}, ...]}'

**Observáveis Pauli em 20 qubits** (kernel sem cópia na GPU; para o GHZ
simétrico **⟨Z⊗…⊗Z⟩ = 1 e ⟨X⊗…⊗X⟩ = 1** — ambos são estabilizadores do
GHZ+; o sinal de X^n distingue GHZ+ de GHZ−):

    "observables": ["ZZZZZZZZZZZZZZZZZZZZ", "XXXXXXXXXXXXXXXXXXXX"]

**Clifford 500 qubits** (tableau O(n²); impossível no statevector) — saída
real: counts {0⁵⁰⁰: 33, 1⁵⁰⁰: 31} em ~27 s (64 shots):

    {"num_qubits": 500, "shots": 64, "backend": "clifford",
     "operations": [h(0)] + [cx(i,i+1) para i em 0..498]}

**Curva de decoerência** (Bell + depolarizante; 4000 shots por ponto) —
valores medidos via stack (worker CPU, 14 workers; 20k shots com dep 0.05
completa em **1,2 s** com o reuso de trajetórias P1.4):

| dep | fração em \|00⟩/\|11⟩ |
|---|---|
| 0,00 | 1,000 |
| 0,02 | 0,976 |
| 0,05 | 0,933–0,936 |
| 0,10 | 0,873 |
| 0,20 | 0,767 |

**Exportação e cancelamento**:

    curl -s -H "Authorization: Bearer $TOKEN" ".../api/jobs/<ID>/export?fmt=csv"
    # state,counts,probability
    # 00,2066,0.504394...
    curl -s -X POST .../api/jobs/<ID>/cancel -H ...   # job pesado → status "cancelled"

## <a name="sdk"></a>3. Pelo SDK (varreduras — o fluxo de pesquisa)

`examples` abaixo rodam em qualquer ambiente com `nats-py` + `msgpack`
(p.ex. `docker run --rm --network host -v $PWD:/src -w /src qsim-engine
python3 script.py`). Saída real da varredura RX (2048 shots/ponto):

    theta      <Z> medido   cos(theta)      |erro|
     0.00      +1.000000    +1.000000       0.00e+00
     0.79      +0.707107    +0.707107       1.11e-16
     1.57      +0.000000    +0.000000       1.61e-16
     3.14      -1.000000    -1.000000       0.00e+00

```python
import asyncio, math
from sdk.qsim_client import QSimClient

async def main():
    async with QSimClient("nats://localhost:4222") as client:
        thetas = [i * math.pi / 12 for i in range(13)]
        results = await client.sweep(                      # ate 4 simultaneos
            lambda th: {"num_qubits": 1, "shots": 2048, "seed": 0,
                        "observables": ["z0"],
                        "operations": [{"gate": "rx", "targets": [0],
                                        "params": [th]}]},
            thetas, concurrency=4)
        for r, th in zip(results, thetas):
            print(th, r["result"]["expectation"]["z0"])

asyncio.run(main())
```

## <a name="esperar"></a>4. Como interpretar o que sai

- **counts**: chaves little-endian — o caractere da DIREITA é o qubit 0.
- **verificação**: fidelidade c64 vs c128 do mesmo circuito; "APROVADA"
  exige > 1−10⁻⁶. Em c128 o compare é contra c64 (e vice-versa).
- **entanglement**: entropia de von Neumann (bits) do qubit vs resto;
  1 = maximamente emaranhado (Bell/GHZ), 0 = separável.
- **Bloch**: vetor r por qubit; r=0 no Bell (reduzido misto), r=1 em
  estados puros de 1 qubit.
- **metadados do job** dizem o caminho real: `backend: cupy` (kernels P40),
  `custatevec` (Volta+), `mode: clifford` (tableau), `mode: trajectories`
  (ruído, com `parallel_workers`).
- **tempos de referência nesta máquina**: Bell c/ verificação ~1,5 s;
  GHZ-20 GPU 22 ms; Bell+ruído 20k shots 1,2 s; Clifford-500 (64 shots)
  ~27 s; GHZ-31 c64 direto na P40 ~53 s (fora do REST enquanto o limite
  for 20).

## <a name="erros"></a>5. Erros comuns e o que significam

| Mensagem | Causa | O que fazer |
|---|---|---|
| 401 / "Token inválido" | token ausente/errado no campo do topo | copiar de `.env` sem quebra de linha; Ctrl+F5 |
| 422 "num_qubits deve ser inteiro entre 1 e 20" | limite operacional atual | editar `QSIM_MAX_QUBITS` no `.env` e `docker compose ... up -d` (máx. físico c64 na P40: 29 via REST/30–31 direto; ver EXECUCAO_PLANO) |
| 422 "backend deve ser..." | typo no backend | numpy/cupy/custatevec/aer/aer-gpu/clifford/auto |
| `custatevec requer CC >= 7.0` | pedido custatevec na P40 (Pascal) | usar `cupy`/`auto` na P40 |
| `VRAM insuficiente para N qubits` | guarda P0.2 | reduzir precisão (c64) ou qubits |
| job `timeout` | excedeu `QSIM_JOB_TIMEOUT_S` (300 s) | menos shots/portas, ou subir o limite |
| 429 | rate limit por IP (240/min) | esperar ~1 min |

**Suítes completas** (física, metamórfica, diferencial, pesquisa, segurança,
Aer, Clifford, aceleradores, trajetórias, fuzzer):
`PYTHONPATH=. python3 tests/<suite>.py` — ou tudo de uma vez no CI do GitHub.
