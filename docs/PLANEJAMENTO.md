# QSim — Planejamento e Estado do Projeto (v4)

Simulador teórico-empírico de computador quântico em microserviços, para
testes de programação quântica e pesquisa. Comunicação interna exclusivamente
por NATS + MessagePack (zero HTTP entre serviços); HTTP existe apenas entre o
navegador e o gateway de administração (porta 10000).

## Arquitetura

    navegador ── HTTP/WS ──> gateway (FastAPI :10000, SQLite, exports)
                                │ NATS (MessagePack)
                                v
                       qsim.jobs.submit
                                │
                           scheduler ── prioridades, quotas, cancelamento
                                │ classifica e roteia
              ┌─────────────┬───┴────────┬─────────────┐
        exec.small    exec.medium   exec.large    exec.noisy
              │             │            │              │
         worker CPU    worker GPU   workers 3090/P40  todos
        (Xeon 2x28c)  (4060 Ti 16G)  (24 GB, 31 qb)  (trajetórias)
              └─────────────┴───┬────────┴─────────────┘
                        qsim.jobs.result.<id>
                                │
               gateway ─ WebSocket ─> console  +  SQLite (histórico)

Filas de classe em JetStream (work queue durável: job de worker morto é
reentregue) com fallback automático para queue groups core NATS.
Observabilidade: Prometheus (engine 9101, gateway 9102, scheduler 9103,
nats-exporter, dcgm-exporter para GPUs) + Grafana com dashboard versionado.

## Decisões registradas

NATS em vez de RabbitMQ (mais leve, request-reply nativo, JetStream quando
precisa de durabilidade); MessagePack em vez de JSON interno; complex64 como
precisão de produção em GPU com verificação cruzada c128 integrada; teto de
31 qubits (16 GiB em c64, cabe nas placas de 24 GB); NPU descartada
estruturalmente para simulação de vetor de estado; Tesla P40 (Pascal, CC 6.1)
atendida via CuPy com CUDA 12.x — sem cuQuantum, que exige Volta+.

## Fase 1 — Núcleo e esqueleto (ENTREGUE)

Simulador de vetor de estado NumPy/CuPy com portas de 1-3 qubits fixas e
paramétricas, fusão de portas 1q (8,5x medido), caminho otimizado para
matrizes diagonais, amostragem vetorizada por CDF (0,73 M shots/s), medição
sem alocação intermediária, worker NATS, gateway FastAPI com console web,
validação em duas camadas (gateway Pydantic + revalidação estrutural no
worker), Dockerfiles, compose com Prometheus/Grafana, suítes física,
metamórfica, diferencial e de segurança, fuzzer e benchmark.

## Fase 2 — Distribuição e operação (ENTREGUE)

Scheduler dedicado com fila de prioridades (0-9), classificação por classe
(small <= 20 qubits, medium 21-29, large 30-31, noisy = com ruído), quotas
por origem (8 jobs ativos, resposta `rejected` acima), cancelamento de jobs
na fila e em execução (cooperativo, via `qsim.control.cancel.<id>`);
JetStream work queue com fallback; histórico persistente em SQLite no
gateway; workers heterogêneos por GPU no compose (perfil `gpu`, um serviço
por placa via `CUDA_VISIBLE_DEVICES` e classes por capacidade);
Dockerfile.cuda (CUDA 12.4 + cupy-cuda12x); NATS com contas e permissões de
menor privilégio por serviço (`deploy/nats-secure.conf` +
`docker-compose.secure.yml`); dcgm-exporter e dashboard Grafana versionado;
teste de carga k6.

## Fase 3 — Capacidades de simulação (ENTREGUE)

Parser OpenQASM 2.0/3.0 (subconjunto; parâmetros com `pi` sob whitelist
estrita, broadcast de registro, medições nas duas sintaxes); observáveis
Pauli arbitrários (strings IXYZ, valores esperados exatos até 26 qubits);
modelo de ruído em três níveis (global, por porta, por qubit — depolarizante,
amplitude damping, erro de leitura); trajetórias estocásticas paralelizadas
por processos nos núcleos da CPU com resultados agregados; cancelamento
cooperativo dentro do laço de execução; validação cruzada externa contra o
Qiskit Aer (20 circuitos aleatórios, pior fidelidade 1,000000000000).

## Fase 4 — Pesquisa e experiência (ENTREGUE)

Console web com editor visual de circuitos em grade (paleta, portas
multi-qubit por cliques, parâmetros em radianos com `pi`), abas JSON e QASM;
esferas de Bloch por qubit (SVG) e barras de entropia de emaranhamento;
comparador A vs B com sobreposição de histogramas e TVD (teórico vs
empírico); botão de cancelamento; autenticação por token no console;
exportação JSON/CSV/LaTeX (tabela booktabs com kets); SDK Python assíncrono e
síncrono direto no NATS com varreduras de parâmetros (`sweep`), prioridade e
cancelamento; documento `docs/PESQUISA.md` com receitas por linha de
pesquisa; teste de integração ponta a ponta com NATS real cobrindo todo o
fluxo.

## Motor de execução em C++ (Qiskit Aer) — ENTREGUE

Após revisão, o hot loop de produção passou a ser o simulador do Qiskit Aer
(C++, AVX, OpenMP; GPU opcional via qiskit-aer-gpu), selecionado por job com
"backend": "aer" ou "aer-gpu". O núcleo Python permanece no sistema com papel
definido: implementação de referência auditável e oráculo de verificação,
validada contra o Aer com fidelidade 1,0. A paridade de recursos é total no
backend aer: verify c64/c128, seed reprodutível, observáveis Pauli, Bloch,
entropia, vetor de estado, medição intermediária e ruído em três níveis. O
canal depolarizante é convertido pelo fator 4p/3 para que os dois motores
implementem o mesmo canal físico (validado com TVD 0,005 em 20 mil shots
entre motores). Limitação documentada: noise.per_qubit combinado com portas
de 3 qubits não é mapeável no Aer e é rejeitado com orientação de usar as
trajetórias do núcleo. Suíte: tests/test_backend_aer.py (6 blocos, todos
aprovados; 2,5x sobre o núcleo já em 1 núcleo de CPU, ganho muito maior com
OpenMP nos Xeon). Workers CPU com Aer instalado passam a atender também a
classe medium (21 a 29 qubits).

## Verificação (estado atual)

Nove suítes, todas passando nesta versão: backend Aer (paridade de recursos), física (oráculo), metamórfica,
diferencial (c64/c128, numpy/cupy, reprodutibilidade), pesquisa (QASM, Pauli,
Bloch, entropia, ruído estendido, paralelismo, cancelamento), segurança (30
vetores hostis + endurecimento do avaliador de parâmetros QASM contra DoS por
exponenciação), fuzzer (300 válidos + 300 mutações hostis) e integração ponta
a ponta (auth, JSON, QASM, exports, ruído, histórico, cancelamento,
scheduler, SDK). Crosscheck externo contra Qiskit Aer aprovado.

## Fase 5 — Aceleradores experimentais (CÓDIGO ENTREGUE, EM VALIDAÇÃO NO HARDWARE)

Backend cuStateVec (`services/engine/backends_custatevec.py`): aplicação de
portas pelos kernels do cuQuantum com estado em VRAM via CuPy; requer CC >=
7.0 (3090 e 4060 Ti; a P40 permanece em CuPy por ser Pascal 6.1); selecionado
por job com `"backend": "custatevec"`. Kernels Numba
(`services/engine/numba_kernels.py`): portas 1q in-place paralelas nos
núcleos da CPU, ativadas com `QSIM_USE_NUMBA=1` (limiar
`QSIM_NUMBA_MIN_QUBITS`, padrão 18); warmup no boot do worker. Dependências
em `requirements-gpu.txt`; Dockerfile.cuda já instala ambos.

Critério de aceitação: `PYTHONPATH=. python3 tests/test_gpu.py` na máquina
com as placas — diferencial cupy vs numpy, custatevec vs cupy (inclui
confirmação da convenção de alvos em CX/CCX), numba vs numpy (fidelidade >
1-1e-10 exigida), smoke opcional de GHZ-31 (`QSIM_GPU_SMOKE_31=1`) e
benchmark de portas/s por backend. Os kernels Numba já foram validados em
CPU (pior fidelidade 1,000000000000 em 6 circuitos); cupy e custatevec
aguardam execução no chassi com as GPUs. Itens que permanecem futuros:
fusão de blocos 2q, fatiamento de trajetórias entre workers, redes de
tensores acima de 31 qubits, editor arrastar-e-soltar e contas de usuário.

## Operação

    docker compose up --build                       # CPU
    docker compose --profile gpu up --build          # 3 GPUs
    docker compose -f docker-compose.yml -f docker-compose.secure.yml up
    QSIM_ADMIN_TOKEN=... (obrigatório fora de dev)

Console: http://localhost:10000 — Grafana: http://localhost:3000 —
Prometheus: http://localhost:9090. Guia de pesquisa: docs/PESQUISA.md.
