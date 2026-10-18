# Changelog

## [Não publicado] — branch melhorias-p0-p2 (2º lote: P1.2, P1.4 e follow-up P1.1)

### Alterados
- **Kernels O(1) para portas de 2 qubits** (P1.1 follow-up): inserção dupla
  de bit zero (posição menor primeiro) elimina o loop O(n) por thread —
  GHZ-31 c64 na P40 de 131 s para **52,9 s (meta do plano < 60 s)**;
  benchmark 26q 262 portas/s (baseline 96), 28q 67 (baseline 25). Híbrido
  25–28q removido (kernels agora dominam em todas as faixas). k=1/k=2 com
  amplitudes em escalares (sem arrays locais).
- **cuStateVec como padrão em CC ≥ 7.0** (P1.2): `make_simulator` roteia
  auto/cupy/gpu para `CuStateVecSimulator` quando disponível; Pascal segue
  com os kernels próprios. Validado na RTX 3090 (fidelidade 1.0; 15.086
  portas/s a 20q).
- **Trajetórias com reuso** (P1.4): prefixo determinístico executa 1× por
  chunk; shots sem evento estocástico são amostrados em lote do mesmo
  estado; `apply_noise` reporta divergência. Bell+depol 20k shots:
  **2,26 s** (baseline 77 s; meta < 5 s). A porta da fronteira estocástica
  fica no resto (bug da fronteira pego pelo teste de amplitude damping).

### Adicionados
- `tests/test_trajectories.py`: equivalência estatística com a implementação
  de referência (pré-P1.4, mantida como oráculo no teste) em 4 cenários de
  ruído, reprodutibilidade por seed e o KPI de tempo.

## [Não publicado] — branch melhorias-p0-p2 (1º lote: P0 completo, P1.1, P1.3, P1.5, P2.1)

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
