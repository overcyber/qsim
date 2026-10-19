# Changelog

## [Não publicado] - branch exp/tema1-canario

### Adicionados
- **Phase damping** (`phase_damping: λ`) nos três níveis do modelo de
  ruído, no núcleo e no Aer.
- **Relaxamento térmico por T1, T2 e duração de porta**: `t1`/`t2` no
  nível global ou em `per_qubit`, `gate_time` no global ou em `per_gate`.
  Convertido em amplitude damping seguido de phase damping, igual ao
  `thermal_relaxation_error` do Qiskit (1e-12). Validação de T2 <= 2 T1 e
  dos campos no nível certo.
- `tests/test_noise_channels.py`: phase damping contra a fórmula, canal
  térmico contra o do Qiskit, curvas de Ramsey e de decaimento de T1 com
  T1/T2 diferentes por qubit nos dois motores, validação.
- Experimento do Tema 1 (`experiments/tema1_canario/`): piloto binomial
  versionado, benchmark sequencial, critérios de viabilidade e pipeline de
  calibração real da IBM.
- Tema 1: detector proposto `cusum_multicomprimento` (segunda diferença
  da log-coerência em canários de 0, 30 e 60 janelas, invariante a erro
  de leitura e decoerência), cenários de deriva de leitura, atacante
  adaptativo e deriva de frequência, e controle com canário longo na
  linha de base.

### Alterados
- Backend Clifford e rota automática decidem "há ruído" pelo próprio
  `NoiseModel`, em vez de listas de chaves que esqueciam canais novos.

## [Não publicado] - branch fix/fisica-ruido-seeds-aer

### Corrigidos
- **Amplitude damping no núcleo Python era um canal de reset.** Com
  probabilidade gamma o qubit era medido e levado a |0>: as populações
  coincidiam, mas as coerências decaíam por (1-gamma) em vez de
  sqrt(1-gamma). Medido antes da correção: |+> com gamma = 0,5 dava
  P(0) = 0,751 contra 0,854 analítico (40 sigma). Agora é um desdobramento
  em saltos quânticos dos operadores de Kraus, compatível com o reuso de
  trajetórias do P1.4. Resultados com amplitude damping em numpy, cupy ou
  auto obtidos antes desta versão devem ser refeitos.
- **Aer ignorava ruído em `rzz`, `rxx` e `id`** e falhava com os apelidos
  `cnot`, `toffoli`, `fredkin`, `phase`, `i` e com `iswap`. O ruído agora é
  registrado por instância (porta canônica, qargs) presente no circuito;
  portas sem instrução nativa no método do Aer vão como unitary rotulado
  (ex.: `cswap` no método density_matrix). `per_qubit` com portas de 3
  qubits passou a funcionar no Aer.
- **A mesma seed dava contagens diferentes conforme o número de núcleos**
  e entre os modos paralelo e sequencial. Cada bloco de 256 shots recebe
  agora um filho de `SeedSequence(seed)`; a distribuição dos blocos entre
  processos não altera o resultado.
- **`per_gate` com porta desconhecida era ignorado em silêncio**; agora é
  rejeitado. Apelidos e nome canônico são equivalentes no `per_gate`.
- **Roteamento**: job com ruído em registro grande caía na classe `noisy`
  (CPU). Agora só registros pequenos vão para `noisy`; erro de leitura
  sozinho não conta como ruído.
- **Documentação**: o estado W tem entropia 0,918 por qubit, não 1, e a
  porta `ch` sugerida não existe; receitas de W e de teleporte corrigidas.

### Alterados
- **`backend: "auto"`** resolve para o melhor motor instalado e registra
  `backend_requested`/`backend_resolved` nos metadados: ruído ou medição
  intermediária vão para aer-gpu, aer ou numpy; circuitos exatos para
  aer-gpu, cupy, aer ou numpy. Clifford puro acima de 20 qubits continua
  indo para o tableau.
- **Aer com ruído usa matriz densidade até 12 qubits** (uma evolução do
  canal; medido 14x a 28x mais rápido que trajetórias de 2 a 10 qubits) e
  shot branching acima disso.
- **Erro de leitura sozinho roda no caminho exato** (antes forçava
  trajetórias).
- `tests/test_trajectories.py`: a implementação de referência passa a usar
  o canal corrigido.

### Adicionados
- `tests/test_noise_channels.py`: cada canal contra a fórmula analítica nos
  dois motores, todos os 29 nomes de porta com ruído nos dois motores,
  `per_qubit` com porta de 3 qubits, seed independente de workers,
  resolução do `auto` e roteamento do scheduler. Incluída no CI.
- `examples/pesquisa_d_bb84_interceptacao.py` e
  `examples/pesquisa_e_teletransporte.py`, com opção `--local`.
- `docs/LINHAS_DE_PESQUISA.md` reestruturado: como transformar uma pergunta
  em experimento e mapa de linhas relacionadas, correlatas e distintas.

### Limitação conhecida
- Jobs executados no Aer só verificam cancelamento antes e depois da
  execução; com `auto`, o job pesado do teste ponta a ponta termina como
  `done` em vez de `cancelled`.

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
