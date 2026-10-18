# QSim — Guia de Linhas de Pesquisa Científica

Como usar o simulador para **pesquisa de verdade**: cada linha abaixo tem
pergunta científica, receita executável, oráculo independente para validar
o que sai, e extensões. Os três experimentos-bandeira (A, B, C) foram
**executados na íntegra** — as saídas mostradas são reais (2026-10-18),
e os scripts completos vivem em `examples/`.

Mapa rápido — o que o QSim tem hoje e o que isso habilita:

| Capacidade | Habilita pesquisar |
|---|---|
| Statevector exato ≤ 20q (REST; 29–31 direto na P40) | física de circuitos, emaranhamento, algoritmos variacionais |
| Ruído depolarizante/amplitude-damping/leitura, por porta e por qubit, com trajetórias rápidas | sistemas abertos, decoerência, QEC sob ruído realista |
| Observáveis Pauli arbitrários + ⟨Z⟩ por qubit | energia de Hamiltonianos Pauli (VQE), magnetização |
| Bloch por qubit + entropia de emaranhamento (1 vs resto) | estrutura de emaranhamento, localidade |
| Medição/reset intermediários | protocolos com clássico-quântico (teleporte) |
| Backend Clifford (tableau) até 2048q | QEC de estabilizadores em escala |
| verify c64/c128 integrado + comparador A/B | metodologia/precisão numérica |
| SDK com sweep + export LaTeX/CSV | curvas para artigos |

---

## Bandeira A — Decoerência do emaranhamento GHZ (sistemas abertos)

**Pergunta.** Quão rápido um estado GHZ-n macroscopicamente emaranhado
degrada sob ruído local, comparado a estados produto?

**Método.** GHZ-n (h + cadeia de cx), canal depolarizante p=2% por
porta-qubit; métrica = fração de shots nos estados |0…0⟩/|1…1⟩.
**Oráculo**: P ≈ (1−2p/3)^(2(n−1)) (só X/Y flipam bits).

**Saída real** (`examples/pesquisa_a_ghz_decoerencia.py`):

```
  n   P(GHZ) medido   teorico
  2      0.974        0.974
  4      0.923        0.923
  6      0.872        0.874
  8      0.821        0.829
 10      0.778        0.785
 12      0.740        0.744
```

**Interpretação.** Decaimento exponencial em n (o "custo do tamanho" do
emaranhamento global) — o argumento físico por trás da fragilidade de
estados de gato em hardware ruidoso. **Extensões**: ajustar a exponencial
por mínimos quadrados e extrair o comprimento de coerência 1/λ(p);
repetir com amplitude damping (canal assimétrico); comparar contra |+⟩^n
(estado produto: decai linearmente) — a diferença de escalas é o artigo.

## Bandeira B — Estado fundamental variacional (VQE) no Ising transversal

**Pergunta.** Um ansatz hardware-efficient + otimizador clássico sobre
medições do QSim recupera o ground state do TFIM? O princípio variacional
(E ≥ E0) é respeitado?

**Método.** H = −Σ Z_iZ_{i+1} (anel) − h Σ X_i (h=1, n=6); ansatz = 3
camadas ry + cadeia cx (18 parâmetros); energia = combinação de 12
observáveis Pauli medidos por job; COBYLA 400 avaliações. **Oráculo**:
diagonalização densa em numpy (H completa, independente do QSim).

**Saída real** (`examples/pesquisa_b_vqe_tfim.py`):

```
oraculo exato (numpy, H completa): E0 = -7.727407
variacional via QSim: E = -7.304143 apos 400 avaliacoes
principio variacional (E - E0) = +0.423264 (deve ser >= 0)
precisao relativa: 5.48%
```

**Interpretação.** O otimizador encontra um estado com energia dentro de
5,5% do fundamental **nunca violando E ≥ E0** — a assinatura de que o
pipeline de medição é consistente. O gap é falta de expressividade do
ansatz (não do estimador): mais camadas fecha. **Extensões**: varrer h
de 0 a 2 e localizar a transição de fase (E0(h) suave, ⟨Z⟩ espontâneo
aparece); medir com shots>1 e quantificar o ruído do estimador (∝1/√N);
trocar ansatz (anel de cx; rx/ry/rz) — estudo de barren plateaus fica
natural aqui.

## Bandeira C — Correção de erros: código de repetição (QEC)

**Pergunta.** Codificar 1 qubit em d físicos suprime o erro lógico abaixo
do físico, como a teoria de códigos prevê?

**Método.** Repetição contra bit-flip (|0_L⟩=|0…0⟩), canal depolarizante p
numa camada de identidades, decodificação por maioria nos counts.
**Oráculo**: erro binomial Bin(d, q), q=2p/3.

**Saída real** (`examples/pesquisa_c_codigo_repeticao.py`):

```
  p     d=1     d=3(m/t)    d=5(m/t)    d=7(m/t)
 0.05   0.033   0.005/0.003   0.001/0.000   0.000/0.000
 0.12   0.080   0.018/0.018   0.006/0.005   0.002/0.001
 0.25   0.167   0.074/0.074   0.035/0.035   0.019/0.018
```

**Interpretação.** A distância suprime o erro exponencialmente (p=0,25:
de 16,7% para 1,9% com d=7 — ~9×) e a curva medida cola na binomial.
**Extensões**: decoder por síndrome com âncilas e medição intermediária
(vs. maioria pós-hoc — compara "medir e corrigir" com "extrair síndrome");
código [[4,2,2]]; amplitude damping; e para escala grande (centenas de
qubits Clifford), usar `backend: "clifford"` para lógica de estabilizadores.

---

## Catálogo de linhas correlatas e distintas (receitas curtas)

**1. Estrutura de emaranhamento: GHZ vs W.** Estado W (1 excitação
simétrica): preparar com ry(2·acos(1/√n)) + cx/ccx em cadeia; comparar a
**entropia por qubit** (`return_entanglement`) e a resposta dos ⟨Z_i⟩ ao
ruído: W perde 1 bit de coerência global mas mantém emaranhamento
bipartido — GHZ perde tudo de uma vez. Pergunta: robustez relativa de
classes de emaranhamento.

**2. Teleporte quântico com canal clássico.** Bell em q0-q1, estado |ψ⟩=ry(θ)
em q2, measure q0 (medição intermediária), cx q0→q1 ANTES do measure...
sequência clássica: cx(0,1), h(0), measure(0), correções condicionais via
cx/cz controlados por resultado — como o measure colapsa, rodar as 4
correções em jobs com seeds diferentes e verificar fidelidade ≈ 1 do qubit
teleportado (⟨X⟩,⟨Y⟩,⟨Z⟩ vs cos/sin(θ)). Protocolo clássico-quântico.

**3. Circuitos aleatórios e caos quântico (Porter–Thomas).** Circuitos
aleatórios profundos (h/t/rx + cx); a distribuição de saída converge para
Porter–Thomas P(k) = 2^n·p_k ~ exponencial: usar `probabilities` (top-256)
e testar colapsão do XEB linear vs profundidade. Métrica de caos/ scrambling.

**4. Precisão numérica c64 vs c128.** `verify: true` reporta fidelidade e
TVD entre precisões; varrer profundidade (10…1000 portas) e n — quando
float64 deixa de ser seguro? Receita: jobs com seed fixo, plotar
fidelidade vs profundidade (`export?fmt=csv`). Metaciência de simulação.

**5. Mitigação de erro de leitura (pós-processo clássico).** Calibrar a
matriz de confusão A: para cada estado base |i⟩ (job com x-gates), medir
p(j|i); estimativa mitigada = A⁻¹·counts (numpy no seu script). Validar:
Bell com readout 2% → TVD mitigado ≪ TVD cru. (A mitigação nativa é o
item P2.4 do plano.)

**6. Mapas de sensibilidade a ruído heterogêneo.** `per_qubit`: ruído só
em um qubit ou `per_gate`: só em cx; medir qual posição/topa derruba mais
a fidelidade do GHZ — mapa de hotspots → análogo a layouts de hardware.

**7. Comparação de motores (metodologia).** Mesmo circuito+seed em
numpy/cupy/aer; TVD entre contagens e tempos (`metadata.elapsed_ms`) —
estudo sistemático de trade-offs de engine (o repo já faz isso como
teste; virar experimento com sweep de n).

**8. Escala Clifford para QEC.** `backend: "clifford"` com 500–2048
qubits: medir como o tempo escala com n (O(n²) por porta) e shots;
preparar estabilizadores de códigos de superfície pequenos via portas
Clifford e estudar ciclos de medição (sem ruído no backend clifford —
o ruído roda no statevector; combinar os dois mundos é a fronteira).

## Boas práticas de metodologia

- **Seeds fixos** para reprodutibilidade; variar seed para barras de erro.
- **Barras de erro √N**: com S shots, incerteza de uma probabilidade ~
  √(p(1−p)/S) — 4000 shots ⇒ ~0,8%. Compare sempre com isso antes de
  declarar desvio.
- **verify: true** em todo resultado publicado (custa 2×) e
  **comparador A/B** no console para ruído vs ideal.
- **Exports**: `?fmt=latex` gera a tabela pronta (`\usepackage{braket}`);
  `?fmt=csv` para análise.
- **Oráculo independente** sempre que existir (analítico, numpy denso, ou
  segundo backend) — os exemplos A/B/C mostram o padrão.
- Cite o motor usado (`metadata.backend`) — resultados c64 vs c128 vs
  custatevec têm significâncias numéricas diferentes.

## Limitações atuais (o que ainda não simula)

Canais de Kraus gerais e matriz densidade (P2.2/P2.3), mitigação nativa
(P2.4), bipartições arbitrárias de emaranhamento (P2.5) — todos no plano
de melhoria. Ruído no backend Clifford não é suportado (use statevector
para ruído; Clifford para escala).
