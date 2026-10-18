# Execução do Plano de Melhoria — 1º ciclo

**Branch:** `melhorias-p0-p2` · **Data:** 2026-10-18 · **Hardware de validação:**
Tesla P40 24 GB (GPUs 0/1 intocadas — cargas de outro projeto).

Este documento registra o que foi executado do `PLANO_DE_MELHORIA.md`,
como foi validado e o que ficou pendente. Nada entrou sem suíte
correspondente passando.

## Status por item

| Item | Status | Validação |
|---|---|---|
| P0.1 ack_wait + heartbeat JetStream | **entregue** | `shared/bus.py` (ack_wait = job_timeout+120, callback `progress` = `msg.in_progress()`), `worker.py` (heartbeat a cada 20 s durante o job). Teste E2E: job longo executado exatamente 1× |
| P0.2 semáforo + guarda de VRAM | **entregue** | `worker.py`: `asyncio.Semaphore(QSIM_ENGINE_MAX_CONCURRENT=1)`; checagem `cupy.mem_info` antes de jobs GPU (erro limpo com estimativa). Validação: dois jobs grandes serializam |
| P0.3 amostragem sem dobrar memória | **entregue** | `simulator.py`: sampler em blocos que **nunca** materializa o vetor de probabilidades (quadrados \|a\|² por bloco, 2 passadas, acumulo float64). `tests/test_inplace.py`: TVD = 0.0 vs CDF completa |
| P0.4 CI GitHub Actions | **entregue** | `.github/workflows/ci.yml`: 10 suítes CPU + build das 3 imagens |
| P0.5 métricas de núcleo | **entregue** | `qsim_engine_sim_seconds{backend}`, `qsim_engine_vram_free_bytes` no worker; telemetria com VRAM |
| P1.1 kernels in-place | **entregue** | `services/engine/gpu_kernels.py`: RawKernels CUDA para portas 1q–3q (in-place, k=1 com índice O(1)) e diagonal; CPU ganhou writeback sem cópia final. **GHZ-31 c64 conclui na P40** (antes: OOM garantido). Ver §KPIs |
| P1.3 Pauli sem cópia | **entregue** | kernel `qsim_pauli_z`/`qsim_pauli_xy` (uma passada, atomicAdd). 16/16 padrões vs referência host; 8 estados × 6 Paulis no teste |
| P1.5 cache de kernels | **entregue** | `CUPY_CACHE_DIR`/`NUMBA_CACHE_DIR` em volume `qsim-kernel-cache`; warmup no boot do worker |
| P2.1 backend Clifford | **entregue** | `services/engine/clifford.py`: geradores de estabilizador (máscaras uint64 + fase Z4), tabela de fase do CNOT derivada por força bruta matricial no import. Rota `backend: "clifford"` + auto-rota (>20q Clifford). **GHZ-1000** ok. Validação: crosscheck vs statevector (pior TVD 0.026), gates todos verificados vs conjugação matricial |
| P1.2 custatevec padrão/sampler | pendente | — |
| P1.4 trajetórias com reuso | pendente | — |
| P2.2–P2.6, P3, P4 | pendentes | próximos ciclos |

## Corretude — o que garante que não quebrou nada

- `_prob_summary` para n grande virou top-k em streaming (nunca materializa
  probabilidades do tamanho do estado).
- `probabilities()` preenche por blocos um único array pré-alocado.
- Bateria completa reexecutada na P40 após as mudanças: física, metamórfica,
  diferencial (NumPy vs CuPy com kernels), research, segurança, fuzzer 300,
  Aer, além das suítes novas (`test_inplace.py`, `test_clifford.py`).
- Durante o desenvolvimento, o crosscheck pegou e corrigiu 3 bugs reais dos
  kernels/Clifford (endianess de alvo no kernel, flip de X/Y no Pauli, regra
  de fase do CNOT, `zrows.any()` sobre índices — detalhados nos commits).

## KPIs medidos (P40, este ciclo)

| KPI | Baseline | Meta | Obtido |
|---|---|---|---|
| GHZ-31 c64 | OOM (impossível) | < 60 s | **131 s — funcional** (31 portas ~100 s + amostragem 25 s + resumo). KPI de tempo parcialmente atendido; gargalo identificado: acessos não-coalesced no kernel (pares com stride 2^t). Próximo passo de otimização documentado em P1.1-followup |
| Amostragem 31q | impossível (OOM) | — | **25 s, counts exatos {0³¹, 1³¹}** |
| Observáveis Pauli 31q | teto 26q com cópia | 30q | **sem cópia** (kernel uma passada; teto = do próprio estado) |
| Clifford 1000q | impossível | < 1 s | **14,5 s p/ 64 shots** (0,23 s/shot; o caminho determinístico é O(n) por bit — KPI original visava o caso 1 shot) |
| Bell+ruido 20k shots (P1.4) | ~77 s | < 5 s | pendente (P1.4 não executado) |

## Notas de operação

- Novos limites/env: `QSIM_MAX_CONCURRENT` (1), `QSIM_KERNELS`
  (`auto`|`on`|`off`), `QSIM_CLIFFORD_MAX_QUBITS` (2048).
- `backend: "clifford"` aceita circuitos {H,X,Y,Z,S,SDG,CX,CZ,SWAP,I} com
  measure/reset; rejeita ruido/observaveis/statevector com erro claro.
- Auto-rota: `backend: "auto"` + circuito 100% Clifford + n > 20 → tableau.
