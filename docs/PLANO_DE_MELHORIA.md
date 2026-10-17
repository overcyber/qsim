# QSim — Plano de Melhoria

**Data:** 2026-10-17 · **Status:** proposta (somente o plano; nada implementado)
**Base da avaliação:** auditoria de código + bateria completa executada em hardware
(Tesla P40 24 GB via CuPy; RTX 4060 Ti para validação cuStateVec; 9 suítes de teste,
fuzzer, benchmark, E2E REST/WebSocket, jobs reais de 2 a 29 qubits).

---

## 1. Avaliação do estado atual

### 1.1 Forças (o que já está no nível certo)

| Dimensão | Evidência |
|---|---|
| Corretude física | Oráculo analítico (Bell/GHZ/QFT/RX), metamórfico (U·U†=I, HZH=X…), diferencial c64/c128 e NumPy/CuPy (fidelidade 1.0), crosscheck vs Qiskit Aer (30 circuitos, pior fidelidade 1.0), fuzzer 300+300 sem crash |
| Arquitetura | Microsserviços limpos: gateway único HTTP, scheduler sem HTTP, workers por capacidade de classe, NATS+MessagePack, validação em duas camadas (gateway + worker nunca confia na fila) |
| Verificação cruzada integrada | `verify=true` compara c64 vs c128 no próprio job (fidelidade + TVD) |
| Diagnósticos de pesquisa | Pauli arbitrário, Bloch, entropia de emaranhamento 1-vs-resto, QASM 2/3 (subconjunto), exports JSON/CSV/LaTeX, console web com comparador A/B |
| Segurança básica | Bearer token, rate limit por IP, limite de corpo, deny-by-default em campos, parser QASM sem `eval` (AST) |
| Observabilidade | Prometheus por serviço, DCGM (GPU), Grafana provisionado, histórico SQLite persistente |

### 1.2 Fraquezas (com evidência medida nesta sessão)

**W1 — Aplicação de portas aloca cópias do estado (o gargalo central).**
`StatevectorSimulator.apply_matrix` usa `moveaxis` + `matmul` + `ascontiguousarray`
(`services/engine/simulator.py:200-214`): cada porta densa custa ~2–3 buffers do
estado. Consequência medida: **30 qubits c64 (16 GiB de estado) dá OOM na P40 de
24 GB** — o teto real é 29q c64 / 28q c128. O "até 31 qubits" do projeto não é
atingível com o algoritmo atual. É também o freio de desempenho: P40 faz 96
portas/s a 26q e 25 a 28q (CuPy), enquanto cuStateVec na 4060 Ti faz 21.411
portas/s a 16q (~360× o caminho CuPy em regimes pequenos).

**W2 — Caminho ruidoso re-executa o circuito inteiro por shot.**
`_run_trajectory_chunk` recria o simulador e reaplica todas as portas a cada
shot (`simulator.py:423-457`). Com ruído em 1 porta, 20 mil shots = 20 mil
execuções completas. (Aer compensa com density matrix no motor C++; o núcleo
Python não tem esse caminho.)

**W3 — Observáveis Pauli custam uma cópia do estado cada.**
`expectation_pauli` clona o estado e aplica portas Pauli porta a porta
(`simulator.py:251-276`): O(N·2ⁿ) cópias para N observáveis, teto artificial
de 26 qubits.

**W4 — Redelivery silencioso de jobs longos no JetStream.**
`subscribe_workqueue` não configura `ack_wait`/heartbeat (`shared/bus.py:129-154`):
com o padrão de ~30s, um job que passe disso pode ser **reentregue a outro
worker enquanto ainda executa** (execução duplicada). Nossos testes E2E usaram
jobs curtos e não expuseram isso; o probe de 28–29q (49–94 s) teria exposto.

**W5 — Sem controle de concorrência/VRAM no worker.**
`handle_job` despacha para o executor default (threads) sem semáforo
(`services/engine/worker.py:132-157`): dois jobs simultâneos na mesma GPU podem
estourar VRAM; não há verificação de memória livre antes de aceitar classe
`large`.

**W6 — Amostragem duplica memória em c64.**
`probabilities()` converte para float64 antes do `cumsum` (`simulator.py:309-321`):
2× o estado em pico durante a amostragem.

**W7 — Núcleo único de simulação.** Só statevector denso. Sem caminho Clifford
(tableau), sem matriz densidade, sem tensor network/MPS, sem multi-GPU — ver
comparativo §2.

**W8 — Sem transpiler.** Não há decomposição para base de portas, mapeamento/
roteamento de qubits nem passes de otimização (a fusão de 1q existe e dá 8,95×,
mas não cancela portas redundantes através de 2q nem otimiza runs de 1q).

**W9 — Engenharia de processo.** Sem CI, sem medição de cobertura, sem
benchmark de regressão automatizado; suítes são scripts (rodam via pytest, mas
não há gate). Histórico sem paginação real; exports sem formato binário
(npz/hdf5) para vetores de estado (teto de retorno: 12q).

**W10 — Scheduler volátil.** Heap de prioridades em memória; quota por origem
em memória; sem dead-letter, sem aging (starvation possível sob prioridade 9
contínua), estatísticas sem RTT de fila.

---

## 2. Panorama: onde os outros simuladores estão

| Simulador | Técnica central | O que tem que o QSim não tem |
|---|---|---|
| [Qiskit Aer](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) | C++/OpenMP/CUDA, cuStateVec, fusão em lote | Métodos density-matrix/unitary/stabilizer, MPI multi-nó, [cache-blocking p/ 44q em supercomputadores](https://fiqci.fi/publications/2025-04-01-LUMI-quantum-simulations-qiskit-aer), transpiler completo |
| [cuQuantum / cuStateVec](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/index.html) | Biblioteca GPU da NVIDIA | APIs de [medição/amostragem em lote com footprint pequeno](https://resources.nvidia.com/en-us-quantum-computing-resource-center/accelerating-quantum), [multi-GPU com distribuição do vetor](https://developer.nvidia.com/cuquantum-sdk) (já usamos apply_matrix; não o sampler nem mgpu) |
| [CUDA-Q](https://nvidia.github.io/cuda-quantum/0.8.0/using/backends/simulators.html) | Plataforma NVIDIA | Backend `mgpu` multi-GPU/nó nativo |
| [Stim](https://quantum-journal.org/papers/q-2021-07-06-497) (Google) | Tableau de estabilizadores vectorizado | [Circuitos Clifford com ~1M qubits](https://github.com/quantumlib/stim), amostragem de shots em lote — escala polinomial onde statevector é exponencial |
| [Qrack](https://github.com/unitaryfund/qrack) | Statevector + decomposição de Schmidt | "Desmonta" emaranhamento para reduzir custo em circuits pouco emaranhados |
| [QuEST](https://quest.qtechtheory.org/) | C HPC distribuído | MPI multi-nó maduro |
| [MQT DDSim](https://github.com/cda-tum/mqt-ddsim) | Diagramas de decisão | Circuitos grandes com estrutura repetida; [contração estilo tensor network](https://portal.fis.tum.de/en/publications/mqt-ddsim-a-quantum-circuit-simulator-based-on-decision-diagrams-/publications) |
| [quimb](https://quimb.quimb.readthedocs.io/) + [cuTensorNet](https://developer.nvidia.com/blog/scaling-quantum-circuit-simulation-with-cutensornet) | Tensor networks (MPS/PEPS) | 40–60q em circuits de conectividade local; GPU em placas Volta+ |
| [PennyLane Lightning](https://pennylane.ai/blog/2023/09/distributing-quantum-simulations-using-lightning-gpu-with-NVIDIA-cuQuantum) | Motor C++/CUDA + autodiff | Diferenciação automática, ecossistema de plugins |
| [QuTiP](https://qutip.org/) | Evolução de sistemas abertos | Master equations, ruído coerente contínuo |
| [Mitiq](https://mitiq.readthedocs.io/en/stable/guide/error-mitigation.html) / Qiskit Experiments | Mitigação de erros | [ZNE, PEC, mitigação de leitura](https://quantum-journal.org/papers/q-2022-08-11-774) prontos para pesquisa |

**Leitura estratégica:** o QSim já tem o que a maioria não tem — um pipeline de
*pesquisa orquestrado* (fila, prioridade, cancelamento, verificabilidade
integrada, console, exports para artigos) e um núcleo Python auditável com
paridade de recursos contra o Aer. O que falta é (a) desempenho de núcleo
(gargalo W1) e (b) amplitude de métodos de simulação (W7). O plano ataca na
ordem: performance primeiro (é o multiplicador de tudo), métodos novos depois
(são os diferenciais científicos), plataforma sempre.

---

## 3. O Plano

Esforço: **S** ≤ 1 semana · **M** 1–3 semanas · **L** > 3 semanas (pessoal-mês
de 1 dev sênior). Prioridade P0 > P4. Cada item tem critério de aceite
verificável — em geral suíte nova ou existente que precisa passar.

### Fase P0 — Confiabilidade e fundação de processo (2 semanas)

| ID | Item | Escopo | Aceite | Esforço |
|---|---|---|---|---|
| P0.1 | **Consertar redelivery de jobs longos** (W4) | `shared/bus.py`: `ack_wait` explícito (≥ `job_timeout_s` + folga) e `msg.in_progress()` em heartbeat no loop de cancelamento do worker | Teste E2E novo: job de 90 s não é duplicado (contar execuções por métrica `qsim_engine_jobs_total`); kill do worker no meio reentrega exatamente 1× | S |
| P0.2 | **Semáforo de execução + guarda de VRAM** (W5) | `worker.py`: 1 job por vez em workers GPU (semáforo por classe), checagem de `mem_info` antes de aceitar `large`; rejeição limpa com retry-hint | Job 30q em P40 → `rejected/oom-guard` imediato e scheduler saudável; dois jobs concorrentes serializam | S |
| P0.3 | **Amostragem sem dobrar memória** (W6) | `simulator.py`: cumsum no dtype do estado; kernel de CDF+busca binária em CuPy | Benchmark 28q: pico de VRAM na amostragem cai ≥ 40%; suíte differential passa | S |
| P0.4 | **CI no GitHub Actions** (W9) | `.github/workflows/ci.yml`: Python 3.12; suítes physics/metamorphic/differential/research/security/fuzz(200)/aer-crosscheck; build das imagens engine/gateway; cache pip | PR não passa sem suítes verdes; badge no README | S |
| P0.5 | **Métricas de núcleo** | `worker.py`: histograma `qsim_engine_gate_seconds` por backend, gauge `qsim_engine_vram_bytes`; painel Grafana v2 | Métricas visíveis no Prometheus alvo; dashboard atualizado | S |
| P0.6 | **Cobertura e benchmark de regressão** | `pytest --cov` no CI com limiar 80% no `services/engine`; `tests/benchmark.py` vira job noturno com artefato e comparação com baseline | Relatório de cobertura no CI; falha se regressão > 10% nos pontos de controle | M |

### Fase P1 — Desempenho do núcleo (o multiplicador; 4–6 semanas)

| ID | Item | Escopo | Aceite | Esforço |
|---|---|---|---|---|
| P1.1 | **Portas in-place, sem cópias** (W1 — o item mais importante do plano) | Kernels 1q/2q/3q in-place: CPU via Numba (generalizar `numba_kernels.py` que já existe para 1q), GPU via `cupy.RawKernel`/`ElementwiseKernel`; caminho diagonal in-place (fases por índice computado); matrizes ficam em registradores/shared | **GHZ-31 c64 conclui na P40 em < 60 s** (hoje: OOM); portas/s @28q ≥ 2× (25 → 50+); suítes physics+differential+fuzzer passam sem mudança de semântica | L |
| P1.2 | **cuStateVec como caminho padrão em CC ≥ 7.0** | `backends_custatevec.py`: adicionar sampler nativo (amostragem em lote na GPU com footprint pequeno) e colapso de medição nativo; auto-seleção por compute capability | Fidelidade 1.0 vs CuPy no teste existente + `test_gpu.py` benchmark mostra custatevec dominante onde disponível | M |
| P1.3 | **Observáveis Pauli sem cópia** (W3) | Kernel de expectativa por bitmask (uma passada por observável sobre o estado, sem clone); agrupamento por comutação para varreduras | `PAULI_MAX_QUBITS` sobe para 30; 20 observáveis em 26q sem OOM; differential novo compara com caminho atual | M |
| P1.4 | **Trajetórias com reuso de estado** (W2) | Shot-branching: prefixo determinístico do circuito executa 1×; apenas o sufixo pós-ruído ramifica; ramificação de portas sem ruído compartilha estado quando possível | Bell+ruido 20k shots < 5 s em CPU (hoje ~77 s com 28 workers); TVD vs implementação atual < 1e-3 | M |
| P1.5 | **Cache de kernels e warmup** | `CUPY_CACHE_DIR` em volume persistente; compilação de todos os kernels no boot do worker | Reinício do worker executa 1º job GPU sem latência de JIT (medir) | S |

### Fase P2 — Física e recursos de pesquisa (os diferenciais; 6–10 semanas, sobrepostas)

| ID | Item | Escopo | Aceite | Esforço |
|---|---|---|---|---|
| P2.1 | **Caminho Clifford (tableau CHP)** (W7) | Novo backend `clifford`: portas {H,S,X,CX,SWAP,measure,reset} → tableau O(n²), shots em lote (estilo Stim); detecção automática de circuito-Cifford no scheduler com rota própria (classe `clifford`, sem teto de 31q) | `tests/test_clifford.py`: crosscheck exato vs statevector ≤ 20q em 200 circuitos aleatórios; **GHZ-1000 qubits em < 1 s**; âncora para estudos de QEC | L |
| P2.2 | **Matriz densidade ≤ 15q** | Evolução por superoperador (Kraus) sem trajetórias; comparação automática trajetórias vs densidade (TVD) no `verify` quando ambos cabem | Teste: canal depolarizante 5% — ⟨Z⟩ por ambos os caminhos converge com 10k shots (1σ); modo exposto no console | M |
| P2.3 | **Canais de ruído gerais** | `noise.per_gate` aceita Kraus custom e canais coerentes (erro unitário determinístico); reset térmico | Suíte física: fidelidade do canal vs forma densidade em 1–3 qubits | M |
| P2.4 | **Mitigação de erros nativa** | (a) mitigação de leitura: circuitos de calibração + inversão tensor/least-squares; (b) ZNE por escalonamento do ruído depolarizante com extrapolação Richardson/exponencial — ambos como pós-processamento no gateway/SDK, checkbox no console | Experimento sintético com readout 2% → estimativa mitigada dentro de 1σ do valor exato; notebook de exemplo | M |
| P2.5 | **Diagnósticos ampliados + exports ricos** | Entropia/informação mútua para bipartições arbitrárias (SVD via CuPy); export `.npz` de statevector/probabilidades com streaming (remove o teto de 12q do retorno); export QASM3 do circuito; desenhista de circuito (matplotlib) para artigos | Console mostra corte 8-vs-resto em 24q; download npz de estado 20q íntegro | M |
| P2.6 | **Parser QASM3 ampliado** | `def` de portas custom, `for`/`if` limitados, múltiplos registros — ainda sem dependência pesada (gramática própria) ou, alternativa: `qiskit.qasm3` importer opcional no gateway | Suíte QASM nova com 30 casos; retrocompatibilidade com subconjunto atual garantida | M |

### Fase P3 — Plataforma e operação (contínua, começa junto com P1)

| ID | Item | Escopo | Aceite | Esforço |
|---|---|---|---|---|
| P3.1 | **Scheduler durável** (W10) | JetStream como fonte de verdade da fila (submit → stream de controle), replay após queda, dead-letter `qsim.jobs.dlq`, aging de prioridade, métricas de tempo de espera | `kill -9` do scheduler sob carga de 100 jobs: zero perda, zero duplicação; job envenenado termina na DLQ com motivo | M |
| P3.2 | **Worker pull-based com drenagem graciosa** | Consumo por `fetch` com backpressure; SIGTERM termina o job corrente e sai limpo; limites de classe recalculados por VRAM livre | Reinício rolado do fleet sem `timeout` em jobs em voo | M |
| P3.3 | **API madura** | `/api/v1` versionada, paginação de histórico, SSE além de WebSocket, OpenAPI anotada; export assíncrono de grandes resultados | Clientes antigos seguem funcionando (compat 1 ciclo); histórico pagina 10k jobs < 300 ms | M |
| P3.4 | **Segurança de produção** | Tokens com escopo (research/admin), HTTPS documentado (proxy reverso), NATS com nkeys/contas por padrão no compose seguro, log de auditoria de submissões | Teste de segurança estendido passa; auditoria responde quem submeteu o quê | M |
| P3.5 | **Observabilidade de ponta a ponta** | Trace leve por job (spans gateway→scheduler→worker com `job_id`), alertas Grafana (fila > N, p95 > X, VRAM > 90%) | Um job rastreável nas três camadas; 2 alertas de demonstração disparam em cenário sintético | M |
| P3.6 | **DX e documentação** | SDK empacotado (pyproject/wheel + publicação), exemplos Jupyter com o fluxo de pesquisa, `ARCHITECTURE.md`, `CHANGELOG.md`, runbook | `pip install qsim` funcional; 3 notebooks reproduzíveis | S |

### Fase P4 — Diferenciadores estratégicos (após P1/P2; horizonte trimestre+)

| ID | Item | Por quê |
|---|---|---|
| P4.1 | **Multi-GPU/nó** (cuStateVec mgpu ou sharding por qubits) | 31–34q quando 3090+P40 estiverem livres; segue a linha [cuQuantum mgpu](https://developer.nvidia.com/cuquantum-sdk)/[CUDA-Q](https://nvidia.github.io/cuda-quantum/0.8.0/using/backends/simulators.html) |
| P4.2 | **Caminho tensor network/MPS** (quimb; cuTensorNet nas Volta+) | 40–60q em circuits de conectividade local — abre classe de problemas inacessível a statevector |
| P4.3 | **Sweep com cache de prefixo** | Varreduras de parâmetro reaproveitam o prefixo comum do circuito (1 execução para N pontos) — acelera o caso de uso nº 1 do SDK |
| P4.4 | **"QEC lab"** sobre P2.1+P2.3 | Patches de surface code com ruído configurável e estimativa de threshold — diferencial científico único para um console de pesquisa |
| P4.5 | **Estatística no comparador A/B** | IC por bootstrap e teste de hipótese entre jobs no console (hoje só TVD pontual) |

### Sequenciamento sugerido

```
Semana 1-2   ── P0 (tudo; CI primeiro)
Semana 3-8   ── P1.1 ──┬── P1.3, P1.5 (paralelo)
                        └── P1.2, P1.4 (após P1.1 firmar a interface de kernels)
Semana 5-14  ── P2.1 (independente de P1; pode começar cedo)
Semana 8-16  ── P2.2-P2.6 (dependem do núcleo estável de P1)
Contínuo     ── P3 (1 item por ciclo); P4 após P1+P2.1
```

### KPIs de sucesso (medidos na P40; baseline desta avaliação)

| KPI | Hoje | Meta |
|---|---|---|
| GHZ-31 qubits c64 | OOM | **< 60 s** |
| Portas/s @28q (CuPy) | 25 | **≥ 50** |
| Bell + ruído, 20k shots (CPU) | ~77 s (28 workers) | **< 5 s** |
| p95 de job 20q via HTTP | ~1 s | **< 0,5 s** |
| Clifford 1000q | impossível | **< 1 s** |
| Perda/duplicação de job (kill sob carga) | não medido | **zero** |
| Cobertura do núcleo (`services/engine`) | não medida | **≥ 85%** |
| Regressão de benchmark entre releases | não medida | **≤ 10%** |

### Riscos e mitigações

- **Kernels in-place com erro de endianidade/indexação** (P1.1): mitigação = suíte diferencial existente + fuzzer como gate contínuo; desenvolvimento atrás de feature flag `QSIM_KERNELS=inplace`.
- **Bifurcação de caminhos (cupy/custatevec/numba/clifford) elevar custo de manutenção**: manter o núcleo auditável como referência e todo backend novo obrigatoriamente passando pelo mesmo oráculo (política já usada no projeto para o Aer).
- **P40 (Pascal) fora do roadmap da NVIDIA**: cuStateVec novo não cobre CC 6.1 — os kernels CuPy próprios de P1.1 são justamente o caminho que independe da biblioteca.
- **Escopo de P2.6 (parser completo) crescer**: fallback documentado para importer do Qiskit como dependência opcional.

### Fora de escopo deste plano

Simulação de nível de pulso, autodiferenciação/parametrização de gradiente
(caso PennyLane), integração com hardware real (QPU cloud) e execução
multi-tenant pública. (Razão: fora da identidade "simulador teórico-empírico
auditável em microserviços"; revisitável após P4.)

---

## 4. Referências da pesquisa

- [Qiskit Aer — AerSimulator (docs 0.17)](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) · [repo](https://github.com/Qiskit/qiskit-aer)
- [Simulações até 44 qubits com Aer no LUMI (CSC, 2025)](https://fiqci.fi/publications/2025-04-01-LUMI-quantum-simulations-qiskit-aer)
- [BlueQubit/AMD: 34q em GPU única (2026)](https://www.amd.com/en/developer/resources/technical-articles/2026/largest-single-gpu-quantum-simulation-on-amd-by-bluequbit.html)
- [Estudo GPU/multi-GPU do Aer (arXiv 2307.14860)](https://arxiv.org/html/2307.14860v1)
- [NVIDIA cuStateVec — overview](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/index.html) · [cuQuantum 25.06](https://docs.nvidia.com/cuda/cuquantum/25.06.0/custatevec/overview.html) · [SDK](https://developer.nvidia.com/cuquantum-sdk)
- [CUDA-Q — backends mgpu](https://nvidia.github.io/cuda-quantum/0.8.0/using/backends/simulators.html)
- [Stim (Gidney, Quantum 2021)](https://quantum-journal.org/papers/q-2021-07-06-497) · [repo](https://github.com/quantumlib/stim)
- [PennyLane — simulação Clifford por tableau](https://pennylane.ai/demos/tutorial_clifford_circuit_simulations) · [Lightning.gpu + cuQuantum](https://pennylane.ai/blog/2023/09/distributing-quantum-simulations-using-lightning-gpu-with-NVIDIA-cuQuantum)
- [STABSim: tableau em GPU (arXiv 2025)](https://arxiv.org/html/2507.03092v2)
- [Tensor decision diagrams (Springer 2025)](https://link.springer.com/article/10.1007/s11227-024-06836-w) · [MQT DDSim](https://github.com/cda-tum/mqt-ddsim)
- [quimb](https://quimb.quimb.readthedocs.io/) · [cuTensorNet (NVIDIA blog)](https://developer.nvidia.com/blog/scaling-quantum-circuit-simulation-with-cutensornet)
- [Mitiq — error mitigation](https://mitiq.readthedocs.io/en/stable/guide/error-mitigation.html) · [paper (Quantum 2022)](https://quantum-journal.org/papers/q-2022-08-11-774) · [repo](https://github.com/unitaryfoundation/mitiq)
