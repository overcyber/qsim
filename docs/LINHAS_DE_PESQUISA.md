# QSim: guia de linhas de pesquisa científica

O QSim não é a pesquisa; é o instrumento. Ele substitui o hardware quântico
por uma simulação exata (ou com ruído controlado), e isso só vira ciência
quando há uma pergunta mensurável, um oráculo independente e um registro
que permita reproduzir o resultado. Este guia mostra como fazer essa ponte
e traz cinco experimentos completos (A a E), executados com saídas reais e
scripts em `examples/`.

## 1. Como usar o QSim numa pesquisa (comece aqui)

Todo experimento deste guia segue os mesmos cinco passos.

1. **Transforme a pergunta numa quantidade.** Probabilidade de um
   resultado, valor esperado de um observável, energia, fidelidade, taxa de
   erro. "O emaranhamento é frágil?" vira "como a fração de shots em
   |0…0⟩ e |1…1⟩ de um GHZ-n cai com n sob ruído p?".
2. **Escreva o circuito e escolha o regime.**
   - *Exato* (sem ruído quântico): probabilidades, observáveis Pauli,
     Bloch e entropia saem do vetor de estado, sem erro estatístico.
   - *Empírico* (ruído ou medição intermediária): só contagens, com erro
     de amostragem √(p(1−p)/N).
   Editor visual, aba QASM ou JSON no console; ou o SDK em Python.
3. **Defina o oráculo antes de rodar.** Fórmula analítica, diagonalização
   densa em numpy, força bruta clássica, um segundo motor do QSim ou um
   limite conhecido da literatura. Sem oráculo, um defeito do simulador
   vira "resultado".
4. **Varra o parâmetro de interesse** (ruído, ângulo, tamanho, profundidade)
   com `sweep` do SDK ou com um script como os de `examples/`. Registre
   seed, shots, `metadata.backend_resolved` e o commit do QSim.
5. **Compare medido contra oráculo com barra de erro** e só então
   interprete. Exporte com `?fmt=csv` para análise ou `?fmt=latex` para o
   artigo.

Os scripts de `examples/` rodam de dois jeitos: com a stack no ar (via
NATS, como um usuário do SDK) ou com `--local`, que executa o motor no
próprio processo, sem Docker.

    PYTHONPATH=. python3 examples/pesquisa_d_bb84_interceptacao.py --local

## 2. Que tipo de pesquisa cabe no QSim

**Relacionadas**: o objeto de estudo é a própria computação quântica.

| Tema | Experimento |
|---|---|
| Decoerência de estados emaranhados | Bandeira A |
| Algoritmos variacionais e física de muitos corpos | Bandeira B |
| Correção de erros (QEC) | Bandeira C, catálogo 8 |
| Protocolos e redes quânticas | Bandeira E |
| Classes de emaranhamento (GHZ e W) | Catálogo 1 |
| Circuitos aleatórios e caos quântico | Catálogo 3 |

**Correlatas**: a simulação quântica é ferramenta de outra área.

| Área | Pergunta típica | Como mapear no QSim | Oráculo |
|---|---|---|---|
| Segurança e criptografia | Um espião é detectável sob o ruído do canal? | Bandeira D (BB84 com interceptação) | QBER = e + n − 2en |
| Criptoanálise pós-quântica (escala didática) | Quantas iterações de Grover encontram a chave de uma cifra de brinquedo de 4 a 8 bits, e quanto ruído destrói o ganho? | Oráculo da cifra com x/cx/ccx, difusor padrão, sweep de iterações t e de ruído | sin²((2t+1)·arcsin(1/√N)) sem ruído |
| Física da matéria condensada | Onde está a transição de fase do Ising transversal? | Bandeira B com sweep de h; magnetização por observáveis | Diagonalização densa |
| Química quântica | O VQE recupera a energia de ligação do H₂? | Hamiltoniano qubit do H₂ (STO-3G) gerado no OpenFermion ou Qiskit Nature e colado como soma de Paulis | Diagonalização da mesma soma |
| Otimização combinatória | Qual a razão de aproximação do QAOA no MaxCut com p camadas? | `rzz` para o custo, `rx` para o misturador, ⟨Z_iZ_j⟩ por observáveis | MaxCut por força bruta (n ≤ 20) |
| Aprendizado de máquina | Um classificador variacional resiste a ruído melhor ou pior que a linha de base clássica? | Codificação por ângulo `ry(x_i)`, ⟨Z₀⟩ como escore, gradiente por parameter-shift (2 jobs por parâmetro via `sweep`) | Regressão logística nos mesmos dados |

Nessas áreas, a contribuição honesta em escala de simulação é sobre
comportamento sob ruído, escalonamento e metodologia, não sobre vantagem
quântica.

**Distintas**: o objeto de estudo é a simulação em si.

| Tema | Experimento |
|---|---|
| Precisão numérica (complex64 contra complex128) | Catálogo 4 |
| Verificação diferencial de simuladores | Catálogo 7 |
| Desempenho em CPU e GPU (3090, 4060 Ti, P40) | `tests/benchmark.py`, `tests/test_gpu.py` |
| Ensino | Editor visual, Bloch e comparador do console |

## 3. Capacidades e o que cada uma habilita

| Capacidade | Habilita |
|---|---|
| Vetor de estado exato até 20 qubits via REST (29 a 31 direto na P40) | física de circuitos, emaranhamento, algoritmos variacionais |
| Ruído depolarizante, amplitude damping e leitura, global, por porta e por qubit; porta `id` como canal | sistemas abertos, decoerência, QEC, canais de comunicação |
| Observáveis Pauli arbitrários e ⟨Z⟩ por qubit | energias de Hamiltonianos (VQE), correlações, tomografia |
| Bloch por qubit e entropia 1 contra o resto | estrutura de emaranhamento |
| Medição e reset intermediários; medição adiada para correções clássicas | protocolos (teleporte, QKD) |
| Backend Clifford (tableau) até 2048 qubits | QEC de estabilizadores em escala |
| `verify` c64/c128 e comparador A/B | metodologia e precisão numérica |
| SDK com `sweep`, exports CSV e LaTeX | curvas para artigos |

---

## Bandeira A: decoerência do emaranhamento GHZ (sistemas abertos)

**Pergunta.** Quão rápido um estado GHZ-n degrada sob ruído local?

**Método.** GHZ-n (h e cadeia de cx), depolarizante p = 2% por porta e
qubit; métrica é a fração de shots em |0…0⟩ e |1…1⟩.
**Oráculo**: P ≈ (1−2p/3)^(2(n−1)) (só X e Y invertem bits).

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

**Interpretação.** Decaimento exponencial em n, o custo do tamanho do
emaranhamento global e o argumento físico por trás da fragilidade de estados
de gato em hardware ruidoso. **Extensões**: ajustar a exponencial e extrair
o comprimento de coerência 1/λ(p); repetir com amplitude damping (canal
assimétrico, corrigido nesta versão: veja `docs/PESQUISA.md` §2.2);
comparar contra |+⟩^n.

## Bandeira B: estado fundamental variacional (VQE) no Ising transversal

**Pergunta.** Um ansatz hardware-efficient com otimizador clássico sobre
medições do QSim recupera o estado fundamental do TFIM, respeitando E ≥ E0?

**Método.** H = −Σ Z_iZ_{i+1} (anel) − h Σ X_i (h = 1, n = 6); ansatz de 3
camadas ry e cadeia cx (18 parâmetros); energia por 12 observáveis Pauli
por job; COBYLA com 400 avaliações. **Oráculo**: diagonalização densa em
numpy, independente do QSim.

**Saída real** (`examples/pesquisa_b_vqe_tfim.py`):

```
oraculo exato (numpy, H completa): E0 = -7.727407
variacional via QSim: E = -7.304143 apos 400 avaliacoes
principio variacional (E - E0) = +0.423264 (deve ser >= 0)
precisao relativa: 5.48%
```

**Interpretação.** Energia a 5,5% do fundamental sem nunca violar E ≥ E0,
sinal de que o pipeline de medição é consistente. A diferença vem da
expressividade do ansatz, não do estimador. **Extensões**: varrer h de 0 a 2
e localizar a transição de fase; medir com shots e quantificar o ruído do
estimador (∝1/√N); trocar o ansatz e estudar barren plateaus.

## Bandeira C: código de repetição (QEC)

**Pergunta.** Codificar 1 qubit em d físicos suprime o erro lógico abaixo
do físico?

**Método.** Repetição contra bit-flip (|0_L⟩ = |0…0⟩), depolarizante p numa
camada de identidades, decodificação por maioria. **Oráculo**: erro
binomial Bin(d, q), q = 2p/3.

**Saída real** (`examples/pesquisa_c_codigo_repeticao.py`):

```
  p     d=1     d=3(m/t)    d=5(m/t)    d=7(m/t)
 0.05   0.033   0.005/0.003   0.001/0.000   0.000/0.000
 0.12   0.080   0.018/0.018   0.006/0.005   0.002/0.001
 0.25   0.167   0.074/0.074   0.035/0.035   0.019/0.018
```

**Interpretação.** A distância suprime o erro (p = 0,25: de 16,7% para 1,9%
com d = 7) e a curva medida acompanha a binomial. **Extensões**: decodificador
por síndrome com âncilas; código [[4,2,2]]; amplitude damping; escala com
`backend: "clifford"`.

## Bandeira D: BB84 com interceptação e reenvio (segurança)

**Pergunta.** Qual QBER um espião que intercepta uma fração f dos qubits
provoca, e a partir de que f ele deixa de se esconder no ruído do canal?

**Método.** Sete qubits: sinal, bit e base da Alice, base da Eve, base do
Bob, decisão de interceptar (ry com P = f) e registro da Eve. Todos os
sorteios são feitos por qubits auxiliares em superposição (medição adiada) e
a medição da Eve é um CNOT controlado pela decisão. Hadamard controlado é
montado com `ry(π/4)`, `cx`, `ry(−π/4)`. O canal Eve→Bob é uma porta `id`
com depolarizante. **Oráculo**: QBER = f/4 sem ruído; com o canal invertendo
bits com probabilidade n = 2p/3, QBER = e + n − 2en, e = f/4. Limite de
Shor-Preskill: QBER acima de ~11% aborta a chave.

**Saída real** (`examples/pesquisa_d_bb84_interceptacao.py --local`, 1,4 s):

```
1) Exato (sem ruido): QBER x fracao interceptada f
    f     QBER     oraculo f/4   peneirada
  0.00   0.0000     0.0000      0.500
  0.25   0.0625     0.0625      0.500
  0.50   0.1250     0.1250      0.500
  0.75   0.1875     0.1875      0.500
  1.00   0.2500     0.2500      0.500

2) Canal Eve->Bob com depolarizante p=0.03 (n = 2p/3 = 0.020), 40000 shots
    f     QBER     oraculo   sigma    decisao
  0.00   0.0197   0.0200   0.0010   aceita
  0.20   0.0686   0.0680   0.0018   aceita
  0.30   0.0926   0.0920   0.0020   aceita
  0.40   0.1140   0.1160   0.0023   aborta (Eve detectada)
  0.50   0.1398   0.1400   0.0025   aborta (Eve detectada)
  fronteira teorica: f* = 4(0.11 - n)/(1 - 2n) = 0.375
```

**Interpretação.** Sem ruído a QBER sai exata das probabilidades (circuito
unitário). Com ruído, todos os pontos ficam dentro de 1σ do oráculo, e um
espião que intercepta até 37,5% dos qubits passa abaixo do limite de
aborto. **Extensões**: mapa de detecção no plano (p, f); amplitude damping
no canal, que afeta as bases de forma diferente; ataque em base
intermediária (Breidbart); taxa de chave após correção e amplificação de
privacidade.

## Bandeira E: teleporte com memória ruidosa (protocolos)

**Pergunta.** Até que ruído no par emaranhado o teleporte supera o melhor
protocolo clássico (medir e reenviar, fidelidade média 2/3)?

**Método.** O QSim não tem controle clássico por resultado de medição, e
não precisa: pelo princípio da medição adiada, as correções X e Z da Alice
viram CX(q1→q2) e CZ(q0→q2) antes de qualquer medição. Ruído: porta `id`
com depolarizante p em q1 e q2 após criar o par. Fidelidade por tomografia
do qubit 2 em três bases, média sobre seis estados de entrada. **Oráculo**:
cada canal encolhe o vetor de Bloch por (1 − 4p/3), logo
F = (1 + (1 − 4p/3)²)/2, que cruza 2/3 em p* = 0,317.

**Saída real** (`examples/pesquisa_e_teletransporte.py --local`, 3,2 s):

```
1) Sem ruido, valores esperados exatos no qubit de destino
  theta=0.000 phi=0.000  fidelidade = 1.000000000000
  theta=1.571 phi=0.000  fidelidade = 1.000000000000
  theta=1.571 phi=1.571  fidelidade = 1.000000000000
  theta=2.000 phi=1.000  fidelidade = 1.000000000000
  theta=1.000 phi=4.000  fidelidade = 1.000000000000
  theta=3.142 phi=0.000  fidelidade = 1.000000000000

2) Memoria ruidosa: fidelidade media sobre 6 estados (20000 shots por base)
    p      F medida   oraculo   supera 2/3?
  0.00     0.9985    1.0000   sim
  0.10     0.8745    0.8756   sim
  0.20     0.7670    0.7689   sim
  0.30     0.6787    0.6800   sim
  0.35     0.6400    0.6422   nao
  fronteira teorica: p* = 0.317
```

**Interpretação.** Teleporte exato sem ruído e transição para abaixo do
limite clássico entre p = 0,30 e 0,35, como prevê a fronteira de 0,317.
**Extensões**: amplitude damping na memória (canal não unital, a fidelidade
passa a depender do estado de entrada); purificação de emaranhamento antes
do teleporte; troca de emaranhamento em cadeia, base de repetidores.

---

## Catálogo de linhas (receitas curtas)

**1. Classes de emaranhamento: GHZ e W.** Prepare o W com a receita em QASM
de `docs/PESQUISA.md` §2.4 (a versão anterior deste catálogo estava
incorreta). Valores exatos: entropia por qubit 1 no GHZ e 0,918 no W;
descartando o qubit 0, ⟨X₁X₂⟩ = 0 no GHZ e 2/3 no W. Pergunta: como cada
classe responde a ruído localizado num qubit (`per_qubit`)?

**2. Teleporte, superdense coding e troca de emaranhamento.** Use medição
adiada como na Bandeira E: toda correção condicionada a uma medição vira
uma porta controlada pelo qubit que seria medido. Isso mantém o circuito
unitário e os observáveis exatos.

**3. Circuitos aleatórios e caos quântico (Porter-Thomas).** Circuitos
aleatórios profundos (h, t, rx e cx); a distribuição de N·p_k converge para
exponencial. Use n ≤ 12, onde `probabilities` traz a distribuição completa
(acima disso vêm só as 256 maiores, o que enviesa o histograma), e meça a
convergência do XEB linear com a profundidade.

**4. Precisão numérica c64 contra c128.** `verify: true` reporta fidelidade
e TVD entre precisões; varra profundidade (10 a 1000 portas) e n com seed
fixa e exporte em CSV. Pergunta: quando complex64 deixa de ser seguro?

**5. Mitigação de erro de leitura.** Calibre a matriz de confusão A com
jobs que preparam cada estado da base; estimativa mitigada = A⁻¹·contagens.
Validação: Bell com leitura de 2%, TVD mitigada contra a crua. Erro de
leitura sozinho roda no caminho exato, então a calibração é rápida.

**6. Mapas de sensibilidade a ruído heterogêneo.** Ruído só num qubit
(`per_qubit`) ou só numa porta (`per_gate`); meça qual posição mais derruba
a fidelidade do GHZ. É o análogo de escolher o layout num chip real.

**7. Comparação de motores.** Mesmo circuito e seed em numpy, cupy, aer e
clifford; TVD entre contagens e tempos (`metadata.elapsed_ms`). A suíte
`tests/test_noise_channels.py` é um exemplo: foi esse tipo de comparação que
encontrou o amplitude damping incorreto e o ruído ignorado em `rzz`, `rxx` e
`id` no Aer.

**8. Escala Clifford para QEC.** `backend: "clifford"` com 500 a 2048
qubits: escalonamento do tempo com n e com shots; estabilizadores de códigos
de superfície pequenos. O backend Clifford não tem ruído; ruído roda no
vetor de estado.

## Boas práticas de metodologia

- **Seed fixa** para reproduzir; varie a seed para barras de erro. A mesma
  seed reproduz dentro de um motor, em qualquer número de núcleos. Entre
  motores, os geradores diferem.
- **Barra de erro**: com S shots, uma probabilidade tem incerteza
  √(p(1−p)/S); 4000 shots dão cerca de 0,8%. Compare com isso antes de
  declarar desvio.
- **Oráculo independente** sempre que existir (analítico, numpy denso,
  força bruta ou outro motor). Os exemplos A a E mostram o padrão.
- **Registre o motor** (`metadata.backend_resolved` quando usar `auto`),
  a precisão, a seed e o commit do QSim no texto de métodos.
- **Declare a semântica do ruído**: depolarizante como probabilidade p de
  erro de Pauli, canais de 1 qubit independentes em cada alvo de portas
  multiqubit (`docs/PESQUISA.md` §2.2).
- **verify: true** em resultados publicados no regime exato (custa 2×).
- **Exports**: `?fmt=latex` gera a tabela (`\usepackage{braket}`);
  `?fmt=csv` para análise.

## Limitações atuais

- Canais de Kraus arbitrários, ruído coerente e matriz densidade como saída
  ainda não existem (P2.2 e P2.3 do plano); o Aer usa matriz densidade por
  dentro até 12 qubits, mas devolve contagens.
- Observáveis, Bloch e entropia só no regime exato; com ruído, estime pela
  tomografia por contagens (como na Bandeira E).
- Não há controle clássico por resultado de medição; use medição adiada.
- Entropia só para a bipartição 1 qubit contra o resto (P2.5).
- O backend Clifford não aceita ruído.
- Jobs executados no Aer só verificam cancelamento antes e depois da
  execução.
