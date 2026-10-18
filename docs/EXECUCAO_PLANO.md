# Execução do Plano de Melhoria — 1º e 2º ciclos (extremo detalhe)

**Branch:** `melhorias-p0-p2` · **Período:** 2026-10-17 a 2026-10-18
**Hardware de validação:** Tesla P40 24 GB (CC 6.1 — GPU dedicada), RTX 3090
(CC 8.6 — validação pontual do P1.2, ~2 min, liberada pelo usuário), 28 núcleos
CPU. GPUs 0/1 com cargas alheias o tempo todo; snapshots antes/depois nos
commits. Nenhuma funcionalidade entrou sem suíte própria + bateria completa
reexecutada.

Índice: [P0.1](#p01) · [P0.2](#p02) · [P0.3](#p03) · [P0.4](#p04) ·
[P0.5](#p05) · [P1.1](#p11) · [P1.2](#p12) · [P1.3](#p13) · [P1.4](#p14) ·
[P1.5](#p15) · [P2.1](#p21) · [KPIs](#kpis) · [memória](#memoria) ·
[bugs-caçados-pelos-testes](#bugs) · [operação](#operacao) ·
[o que ficou pendente](#pendente)

---

## <a name="p01"></a>P0.1 — Fim da reentrega duplicada de jobs longos (JetStream)

**Problema.** `Bus.subscribe_workqueue` (`shared/bus.py`) criava o consumer
durable sem `ack_wait` → padrão do broker ~30 s. Um job que executa por mais
disso era **reentregue a outro worker enquanto ainda executava**, gerando
execução duplicada e resultados conflitantes. Evidência: o próprio E2E da
branch pegou 2 recebimentos do mesmo job antes da correção final.

**Design da solução** (3 camadas, porque cada uma sozinha não basta):

1. **`ack_wait` explícito** = `settings.job_timeout_s + 120` (420 s no deploy
   padrão), passado no `subscribe_workqueue` via
   `ConsumerConfig(ack_wait=...)`.
2. **Atualização de durable preexistente**: consumers durables JÁ criados
   (deploy anterior) mantêm o `ack_wait` antigo — o broker não propaga
   mudança por `js.subscribe` sozinho. Antes de subscrever, o worker chama
   `js.update_consumer(stream, ConsumerConfig(name=durable,
   ack_wait=..., filter_subject=...))` (silencioso se indisponível).
3. **Heartbeat durante toda a espera**: o handler agora recebe um callable
   `progress` (closure sobre `msg.in_progress()`). O worker cria a task de
   heartbeat **no recebimento** — antes de adquirir o semáforo (P0.2) —
   renovando o prazo a cada 20 s inclusive enquanto o job espera na fila
   local. (A primeira versão criava o heartbeat dentro do semaforo; o E2E
   mostrou reentrega durante a espera e expôs exatamente isso.)

**Arquivos:** `shared/bus.py` (assinatura + consumer config), 
`services/engine/worker.py` (heartbeat + passagem do ack_wait).

**Validação:** E2E na stack viva — job ruidoso de 20 qubits rodando os 300 s
completos: **1 único "recebido" nos logs** (antes: 2), status final `timeout`
limpo (comportamento correto do limite), scheduler saudável após.

---

## <a name="p02"></a>P0.2 — Semáforo de execução + guarda de VRAM

**Problema.** `handle_job` despachava para o executor default (threads) sem
limite: dois jobs GPU simultâneos no mesmo worker podiam somar estados e
estourar a VRAM no meio da execução; não havia checagem prévia de memória.

**Design.**
- `asyncio.Semaphore(settings.engine_max_concurrent)` (default 1): o
  **recebimento** é imediato (ack/heartbeat rodam), a **execução pesada**
  serializa. Espera no semáforo conta como espera normal do job.
- **Guarda de VRAM** antes de executar jobs GPU (`backend` ∈
  {cupy, gpu, auto, custatevec}): `_estimate_vram_bytes(n, precision) =
  2^n × itemsize × 2 × 1,15` (estado + probabilidades + folga de amostragem),
  comparada com `cupy.cuda.Device().mem_info` livres. Insuficiente →
  `ValueError` com números ("precisa de 36,8 GiB, livres 23,5 GiB") → job
  volta `rejected` limpo, scheduler segue saudável. Sem CuPy visível
  (ex.: `auto` em worker CPU), a checa é pulada.

**Validação.** Smoke na P40: 31q c128 (36,8 GiB) → rejeita; 31q c64
(18,4 GiB) e 29q → aceita. Serialização demonstrada no incidente do P0.1:
a cópia reentregada ficou **bloqueada no semáforo** em vez de executar em
paralelo com a original.

**Limitação conhecida:** a estimativa é pico-de-algoritmo, não por job
observado; jobs Aer-GPU não passam pela checa (limite da API do Aer).

---

## <a name="p03"></a>P0.3 — Amostragem e probabilidades sem dobrar memória

**Problema.** O sampler antigo fazia `cumsum(probabilidades().astype(float64))`:
um vetor float64 completo do estado + a conversão do |a|² → em 30–31 qubits
isso excede a VRAM junto com o estado (e `complex64` tem **itemsize 8** —
2^31 elementos = 16 GiB de estado; margem mínima).

**Design** (três funções reformuladas em `services/engine/simulator.py`):

- `probabilities()`: array pré-alocado no dtype nativo (float32 p/ c64),
  preenchido **por blocos** (`out[seg] = Re²; out[seg] += Im²`) — pico =
  estado + 1 saída + 1 bloco, nunca 2 temporários cheios. (Descoberta no
  caminho: `xp.multiply(p, p, out=p)` com entrada≡saída no CuPy aloca outro
  array completo mesmo assim — por isso o formato por blocos.)
- `sample(shots, ...)`: **nunca materializa probabilidades**. Duas passadas
  por blocos de 2^24: (1) total em float64 acumulado; (2) por bloco,
  `cumsum(float64 do bloco) + offset`, `searchsorted` só nos shots que caem
  naquele intervalo, `clip` e máscara `filled` para bordas de arredondamento.
  Tráfego de VRAM extra por bloco: ~0,5 GiB.
- `_prob_summary`: para n > `PROBS_FULL_MAX` (12) virou **top-k em
  streaming**: por bloco, `argpartition` local, mescla dos candidatos no
  host, corta nos 256 maiores. O vetor completo nunca existe.

**Validação.** `tests/test_inplace.py`: TVD **0.0** vs CDF float64 completa
(n=10 e n=14, mesmos uniformes); invariantes (soma = shots, suporte, 50/50);
amostragem de 31 qubits executando de fato na P40 (counts exatos do GHZ,
25 s). Bateria física/diferencial inteira reexecutada.

---

## <a name="p04"></a>P0.4 — CI (GitHub Actions)

`.github/workflows/ci.yml`, dois jobs: `testes` (ubuntu, Python 3.12, cache
pip; roda física, metamórfica, diferencial, pesquisa, segurança, backend Aer,
crosscheck Aer ×10, **Clifford**, **aceleradores CPU** e **trajetórias**,
fuzzer 150) e `imagens` (build das 3 imagens Docker). Gatilho: push em
qualquer branch + PR. Suítes novas adicionadas ao CI desde a criação.

## <a name="p05"></a>P0.5 — Métricas de núcleo

`services/engine/worker.py`: histograma `qsim_engine_sim_seconds` com label
`backend` (do `metadata.backend/engine/mode` do resultado — diferencia numpy,
cupy, custatevec, aer, clifford e trajectories), gauge
`qsim_engine_vram_free_bytes` (via `cupy.mem_info`, 0 sem GPU) atualizado no
loop de telemetria a cada 5 s junto com o payload de telemetria do NATS.

---

## <a name="p11"></a>P1.1 — Kernels CUDA in-place (o multiplicador do plano)

**Problema.** Cada porta densa no caminho genérico custa ~2–3 buffers do
tamanho do estado (`moveaxis` → cópia no `reshape`, `matmul` → saída, e a
cópia final `ascontiguousarray`). Consequências medidas na P40: teto real de
29 qubits c64 (30q = OOM), 96 portas/s a 26q, e nada de 31q.

**Design** (`services/engine/gpu_kernels.py`, ~230 linhas):

- Um **RawKernel por precisão** (double2/float2, código-fonte único com
  placeholders `__T__`/`__BASE__`, compilado sob demanda e cacheado em
  `CUPY_CACHE_DIR`):
  - `qsim_apply_gate`: um grupo de 2^k amplitudes por thread. Índice-base do
    grupo = bits do `g` redistribuídos com **zeros nas posições alvo**.
    - **k=1 e k=2 em O(1)** por thread: inserção de bit zero via máscara
      `(g & mask) | ((g & ~mask) << 1)`; para k=2, **duas inserções, da
      posição MENOR para a maior** (a inserção inferior desloca o buraco
      superior; na ordem inversa os índices colidem — bug real encontrado
      pelo standalone 31q, que dava counts 1000×"0" com 51 s).
    - k=1 e k=2 com **amplitudes em escalares** (sem `v[8]`/`idx[8]`) e o
      produto de matriz manual (2×2 e 4×4, 16 `cmul`/`cadd`).
    - k=3 no caminho genérico (loop de bits O(n) — raro e aceitável).
  - `qsim_apply_diag`: fase por amplitude (rz/cz/p/cp), sem gather.
- **Híbrido removido**: na primeira versão os kernels perdiam do matmul em
  25–28q (loop O(n) por thread). Com a indexação O(1) os kernels dominam em
  **todas** as faixas (benchmark P40: 26q 262 vs 93 portas/s; 28q 67 vs 24)
  e o híbrido 25–28 foi descartado.
- **CPU**: `apply_matrix` escreve o resultado **de volta pela view
  `moveaxis`** (`psi[...] = out`) — elimina a cópia final do tamanho do
  estado por porta. Referência de comparação isolada (o algoritmo antigo,
  com cópias) vive em `tests/test_inplace.py` para sempre validar a
  equivalência.
- Falha de compilação → `self._kernels = None` → caminho genérico (kernels
  são aceleração, não dependência). Flag `QSIM_KERNELS`
  (`auto`/`on`/`off`) permite A/B exato nos testes.

**Validação.**
- Diferencial kernels vs genérico, 6 circuitos aleatórios por precisão:
  fidelidade 1.0 (c128) e 0.99999965 (c64 — limite de float32 ajustado no
  teste com justificativa).
- Bateria completa na P40: física, metamórfica, diferencial, research,
  segurança, fuzzer 300 — verde.
- **Standalone 31q** (`/tmp/qsim_31q_standalone.py`, replicado no smoke da
  suíte): counts exatos {0³¹:473, 1³¹:527}, **52,9–54,3 s** (meta do plano
  < 60 s ✓; baseline anterior: OOM; primeira versão dos kernels: 131 s).

**Limitações/next:** acessos ainda não perfeitamente coalesced para alvos
em bits baixos (<5); cuStateVec (P1.2) cobre isso nas placas Volta+.

## <a name="p12"></a>P1.2 — cuStateVec como caminho padrão em CC ≥ 7.0

**Problema.** O backend custatevec exigia pedido explícito; o padrão era
CuPy puro mesmo nas placas onde a biblioteca da NVIDIA é muito mais rápida.

**Design.** `make_simulator` (`simulator.py`): `backend` ∈ {auto, cupy, gpu}
→ se `available_backends()` contém `custatevec` (o que já exige cuquantum
instalado **e** CC ≥ 7.0 — ver `backends.py`), retorna
`CuStateVecSimulator`; qualquer falha cai no caminho genérico. Na Pascal
(P40) nada muda (fica com os kernels P1.1). O `metadata.backend` do job
reflete o caminho real (`custatevec`), mantendo a auditoria honesta.

**Validação (RTX 3090, execução curta liberada pelo usuário).**
`tests/test_gpu.py` completo: custatevec vs cupy fidelidade **1.0** em 10
circuitos + convenção de alvos CX/CCX confirmada; benchmark 3090:
custatevec 15.086 portas/s a 20q (cupy+ kernels próprios: 2.796). Efeito
colateral visível e esperado: no benchmark as colunas cupy/custatevec
convergem em CC≥7 porque ambas agora rodam cuStateVec.

**Pendência dentro do item:** o sampler nativo do cuStateVec (amostragem em
lote na GPU com footprint mínimo) não foi integrado — nosso sampler por
blocos (P0.3) já resolve o footprint; a integração fica para o próximo
ciclo quando houver benefício medido.

## <a name="p13"></a>P1.3 — Observáveis Pauli sem cópia do estado

**Problema.** `expectation_pauli` clonava o estado e aplicava as portas
Pauli uma a uma: O(N·2ⁿ) cópias para N observáveis, teto artificial de 26
qubits (`PAULI_MAX_QUBITS`).

**Design.** Kernel `qsim_pauli_z` (P só com I/Z: Σ|a_i|²·(−1)^popcount(i∧z))
e `qsim_pauli_xy` (P com X/Y: pares (i0, i0⊕M) com M = máscara X|Y, fase por
amplitude — Y_q → ±i conforme o bit, Z_q → −1 — acumuladas com
`atomicAdd` double, suportado a partir de sm_60). Derivação completa nos
comentários do arquivo; o par usa a mesma inserção O(1) de bit do kernel de
portas. Fallback CPU mantém o caminho de cópia (referência auditável).

**Validação.** 16/16 padrões (X, Y, Z isolados e compostos tipo ZIXY, YXII,
IIXZ…) contra referência host elemento-a-elemento em estado aleatório
(rotina independente escrita no teste); 8 estados × 6 Paulis na suíte.

## <a name="p14"></a>P1.4 — Trajetórias com reuso (prefixo + shots limpos em lote)

**Problema.** O caminho empírico (ruído/medição intermediária) recriava o
simulador e **reexecutava o circuito inteiro por shot**: Bell + ruído com
20k shots ≈ 77 s com 28 workers (KPI do plano: < 5 s).

**Design** (`simulator.py`):
1. `_trajectory_plan`: divide as operações em **prefixo determinístico**
   (portas cujo ruído pós-porta é exatamente zero — calculado por
   `NoiseModel.probs_for` por qubit-alvo — e que antecedem qualquer
   measure/reset) e **resto**. Sutileza crítica descoberta pelo teste de
   amplitude damping: a porta cujo pós-ruído é estocástico **fica no resto**
   — na primeira versão ela ia para o prefixo e o evento da fronteira se
   perdia (invisível no Bell+depolarizante, porque X/Y/Z em |+⟩ não alteram
   as contagens — a física do T5; explícito no damping, confirmado por
   cálculo analítico exato: referência 0,585 de "00", versão com bug 0,51).
2. O prefixo executa **1 vez por chunk**; cada shot clona o estado-base
   (cópia 2ⁿ, barata vs reexecutar portas).
3. `apply_noise` agora **retorna se o estado divergiu** (evento depolarizante
   disparou, damping mediu, ou measure/reset intermediário). Shots sem
   nenhuma divergência (a maioria com ruído leve) são acumulados e
   **amostrados em lote** do mesmo estado final (`sample(pending)`); shots
   divergentes fazem flush do lote pendente e amostram individualmente.
   `readout_error` permanece por-shot dentro do `sample`.

**Validação** (`tests/test_trajectories.py`, com a implementação
**referência pré-P1.4 mantida como oráculo dentro do teste**):
- Equivalência estatística: Bell+depol 5% TVD 0,0037 (20k shots);
  GHZ-4 + per_gate + readout TVD 0,0181; medição intermediária + ruído TVD
  0,0207; amplitude damping 15% TVD 0,0123 (este foi o teste que pegou o
  bug da fronteira).
- Reprodutibilidade por seed preservada.
- **KPI: Bell+depol 20k shots em 2,26 s (baseline 77 s → 34×; meta < 5 s ✓)**.
- Bateria de ruído existente (research, physics, fuzz 200, backend Aer com
  paridade de canal) reexecutada — verde.

## <a name="p15"></a>P1.5 — Cache de kernels persistente + warmup

`Dockerfile.cuda` define `CUPY_CACHE_DIR=/cache/cupy` e
`NUMBA_CACHE_DIR=/cache/numba`; `docker-compose.yml` monta o volume nomeado
`qsim-kernel-cache` em todos os workers GPU. O boot do worker compila os
kernels das duas precisões fora do caminho crítico (log: "Kernels GPU
in-place compilados (P1.1 ativo)"). Reinício do worker passa a não pagar
JIT no primeiro job.

## <a name="p21"></a>P2.1 — Backend Clifford (tableau de estabilizadores)

**Problema.** Só existia statevector (O(2ⁿ)); circuitos Clifford (a classe
mais comum em testes de QEC e benchmarks) são O(n²) por tableau.

**Design** (`services/engine/clifford.py`, formulação própria auditável —
sem destabilizadores, mais fácil de verificar que o CHP clássico):
- Estado = os **n geradores de estabilizador**: máscaras X e Z em uint64
  empacotados (numpy) + fase em Z₄ (i^p) por linha.
- **Portas por conjugação** (G → U G U†): H (troca colunas x/z, fase
  += 2·(x∧z)), S/SDG (X→Y→−X→−Y com fases), X/Z (fase += 2·coluna oposta),
  **Y não é no-op**: YXY†=−X, YZY†=−Z → fase += 2·(x⊕z) (erro clássico
  cometido e corrigido — derivado de novo com Y†=Y), CX (colunas
  x_b⊕=x_a, z_a⊕=z_b **e fase pela tabela**), CZ = H·CX·H, SWAP (troca
  verdadeira de colunas — com caso próprio para a/b na mesma palavra).
- **Fase do produto de duas linhas** (base do rowsum): quarter-turns +1 nos
  pares cíclicos (X·Y, Y·Z, Z·X) e −1 nos anticíclicos, computada por
  **popcounts de máscaras** (`np.bitwise_count` com fallback). Validada por
  força bruta matricial: 1.274 casos, 0 erros.
- **Fase do CNOT por LUT derivada no import**: tabela 16-combinações
  (x_a,z_a,x_b,z_b) → {0,+2} gerada por **conjugação matricial exata**
  (`_build_cx_phase`) — sem regra decorada; a assert de construção garante
  fase real (0 ou 2) em todos os casos.
- **Medida de Z_q**: se algum gerador tem X/Y em q → aleatória: elimina X_q
  dos outros geradores com o pivô, substitui o pivô por (−1)^m·Z_q, e
  **elimina Z_q dos demais com o novo pivô** (é isso que propaga a
  correlação — no Bell, Z₀Z₁ vira ±Z₁; sem esse passo o qubit parceiro
  trava). Senão → determinística: elimina a coluna z_q; se o pivô for
  exatamente ±Z_q usa a fase dele; caso geral resolve por **RREF completo
  das 2n colunas** com fases rastreadas (a linha Z_q, se o observável
  pertence ao grupo, vira literalmente uma linha da base reduzida).
- Integração: `backend: "clifford"` explícito (teto
  `QSIM_CLIFFORD_MAX_QUBITS`, default 2048) + **auto-rota** (backend auto,
  circuito 100% Clifford — validado por `is_clifford_circuit` — e n > 20);
  classe de scheduler própria (`clifford`) atendida por todos os workers;
  rejeita com erro claro: não-Clifford, ruído, observáveis, statevector,
  Bloch, entropia.

**Validação** (`tests/test_clifford.py` + verificador de portas):
- **Cada regra de porta** contra conjugação matricial exata em n=2, todos
  os (x,z,p) de entrada: TODAS OK (o verificador é script auxiliar usado
  durante o desenvolvimento; a suíte cobre o resultado final).
- Determinísticos: Bell, GHZ-5, X|0⟩ exatos.
- **Crosscheck vs statevector**: 8 circuitos aleatorios, pior TVD 0,026
  (4.000 shots) + fuzzer de minimização durante o desenvolvimento.
- Medida/reset intermediários, auto-rota, erros claros.
- **GHZ-1000**: 13–15 s para 64 shots (0,23 s/shot) — impossível no
  statevector.

**Limitações/next:** shots totalmente aleatórios custam O(n³/64) por colapso
(caminho RREF); optimização incremental (redução mantida entre medidas)
fica para o próximo ciclo. Observáveis Pauli no clifford (via tabela) também
pendentes.

---

## <a name="kpis"></a>KPIs do plano — medidos

| KPI | Baseline | Meta | Obtido | Status |
|---|---|---|---|---|
| GHZ-31 c64 na P40 | OOM | < 60 s | **52,9–54,3 s** | ✅ (1ª versão dos kernels: 131 s; O(1) k=2 resolveu) |
| Portas/s @26q (P40) | 96 | ≥ 192 (2×) | **262** | ✅ (2,7×) |
| Portas/s @28q (P40) | 25 | — | **67** | ✅ (2,7×) |
| Bell+ruído 20k shots CPU | 77 s | < 5 s | **2,26 s** | ✅ (34×) |
| Clifford 1000q | impossível | < 1 s | 13,5 s/64 shots (**0,21 s/shot**) | parcial no tempo-total; viável e validado |
| Observável Pauli 26–30q | teto 26q c/ cópia | 30q s/ cópia | **sem cópia; teto = do estado** | ✅ |
| Perda/duplicação de job | não medido | zero | **zero** (job de 300 s, 1 recebimento) | ✅ |
| Amostragem 30q+ | OOM | — | 25 s @31q, counts exatos | ✅ |

## <a name="memoria"></a>Perfil de memória VRAM (P40, c64, 31 qubits = estado 16 GiB)

| Estágio | Pico | Detalhe |
|---|---|---|
| Criação do estado | 16 GiB | `zeros(2³¹, complex64)` |
| 31 portas (kernels) | +~0 | in-place, matrizes 2×2/4×4 em constantes |
| Amostragem (2 passadas) | +~0,5 GiB | blocos de 2²⁴; nunca probs completas |
| Resumo top-256 | +~0,4 GiB | argpartition por bloco |
| Observável Pauli | +0 | uma passada, atomicAdd |

## <a name="bugs"></a>Bugs reais encontrados pelos testes durante a execução

Registro honesto — todos pegos por validação, todos com regressão de teste:

1. **Kernel: alvo deslocado** — passei `1 << t` (máscara) onde o kernel
   esperava o índice `t`: H(q) agia no qubit q+1 e H(2)/H(3) escreviam fora
   do array (a origem do `CUDA_ERROR_ILLEGAL_ADDRESS` original). Isolado
   dumpando o suporte por alvo.
2. **Kernel: base não-injetiva** — construção de `base` por shifts
   acumulados gerava colisão de grupos (norma 0,5/0,0). Correção: bit na
   posição real (`base |= bit << b`).
3. **Pauli: flip só de X** — `j = i0 ^ xmask` em vez de `xmask | ymask`:
   qualquer Pauli com Y dava exatamente 0 (termos puramente imaginários).
4. **CNOT: fase** — a regra "r ^= x_a·z_b" é incompleta (depende de z_a
   também); resolvi derivando a **tabela completa por força bruta
   matricial no import** (auto-verificável por construção).
5. **Clifford: Y como no-op** — derivação inicial com Y† errado; YXY†=−X.
6. **Clifford: `zrows.any()`** — `.any()` sobre array de **índices** trata o
   índice 0 como falso (medida do qubit cujo pivô era a linha 0 retornava
   sempre 0). Encontrado com `sys.settrace` linha a linha.
7. **Clifford: colunas z nunca processadas** — `range(n)+range(n)` com
   condição `col < n` sempre verdadeira; o solve determinístico via
   RREF incompleto.
8. **Trajetórias: ruído da porta-fronteira perdido** (P1.4) — invisível em
   Bell+depolarizante, exposto pelo amplitude damping e confirmado por
   cálculo analítico exato no relatório de teste.
9. **Inserção dupla de bits na ordem errada** (k=2 O(1)) — 51 s mas counts
   1000×"0"; a inserção da posição menor precisa vir primeiro.

## <a name="operacao"></a>Notas de operação

- Novas variáveis: `QSIM_KERNELS` (auto|on|off), `QSIM_ENGINE_MAX_CONCURRENT`
  (1), `QSIM_CLIFFORD_MAX_QUBITS` (2048), além das existentes
  (`QSIM_MAX_QUBITS` via `.env`, teto Clifford respeitado na validação em
  separado).
- Classes de fila: `small`/`medium`/`large`/`noisy` + **`clifford`**
  (atendida por todos os workers; auto-rota para n > 20).
- Worker GPU: boot compila kernels (~10–20 s na primeira vez, depois cache
  em volume); telemetria publica `vram_free_bytes`; guarda de VRAM rejeita
  jobs acima da capacidade com mensagem quantificada.
- Routing por hardware: CC ≥ 7 (3090/4060+) → cuStateVec por padrão em
  `auto`/`cupy`; Pascal (P40) → kernels próprios. `metadata.backend` sempre
  reflete o caminho real.

## <a name="pendente"></a>O que ficou pendente (próximos ciclos)

- P2.2 matriz densidade; P2.3 canais de Kraus gerais; P2.4 mitigação de
  erros (leitura + ZNE); P2.5 bipartições arbitrárias + exports npz/QASM3 +
  desenhista; P2.6 parser QASM3 ampliado.
- P3 (scheduler durável com DLQ, pull-based, API v1, segurança com escopos,
  tracing); P4 (multi-GPU, tensor network, sweep com cache de prefixo,
  QEC lab).
- Otimizações pontuais: coalescência dos kernels para alvos em bits baixos;
  sampler nativo cuStateVec; observáveis no backend Clifford; redução
  incremental no clifford para shots totalmente aleatórios.
