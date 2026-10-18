# Changelog

## [Não publicado] — branch melhorias-p0-p2 (1º ciclo do plano de melhoria)

### Adicionados
- **Backend Clifford (P2.1)**: simulador por geradores de estabilizadores
  (`services/engine/clifford.py`) — O(n²) por porta, GHZ-1000 viável; rota
  explícita `backend: "clifford"` e automática (>20q, circuito Clifford);
  classe de scheduler própria; suítes `tests/test_clifford.py`.
- **Kernels GPU in-place (P1.1)**: `services/engine/gpu_kernels.py` — portas
  1q–3q e diagonais sem alocar cópias do estado; viabiliza 30–31 qubits c64
  na P40 (antes OOM). Flag `QSIM_KERNELS`.
- **Observáveis Pauli por kernel (P1.3)**: uma passada sobre o estado, sem
  clone (teto deixa de ser 26 qubits).
- **CI (P0.4)**: GitHub Actions com 10 suítes CPU e build das imagens.
- **Métricas (P0.5)**: `qsim_engine_sim_seconds{backend}`, VRAM livre.
- Suíte `tests/test_inplace.py` (kernels, sampler por blocos, referências).
- `docs/PLANO_DE_MELHORIA.md`, `docs/EXECUCAO_PLANO.md`.

### Alterados
- **Amostragem (P0.3)**: nunca materializa vetor de probabilidades completo;
  quadrados por blocos com acumulação float64 (TVD 0.0 vs CDF completa).
- **`_prob_summary`**: top-k em streaming para n grande.
- **`probabilities()`**: preenchimento por blocos, dtype nativo.
- **`apply_matrix` CPU**: writeback através da view moveaxis (elimina a cópia
  final do tamanho do estado por porta).
- **Worker (P0.1/P0.2)**: `ack_wait` explícito + heartbeat `in_progress`
  (fim da reentrega duplicada de jobs longos); semáforo de execução;
  guarda de VRAM com erro limpo; warmup de kernels no boot.
- **Compose**: volume persistente `qsim-kernel-cache` (cache CuPy/Numba);
  classes Clifford nos workers.

### Correções
- `complex64` em `probabilities()`/amostragem não converte mais para float64
  (dobra de VRAM em pico).
- Kernels validados por crosscheck diferencial; três bugs de indexação/fase
  encontrados e corrigidos pelos testes durante o desenvolvimento.
