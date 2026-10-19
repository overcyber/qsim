# Tema 1: canários contra injeção por crosstalk

Infraestrutura experimental do artigo principal: detecção de intrusão em
computadores quânticos multiusuário, tratada como detecção sequencial de
anomalias sobre circuitos-canário.

**Pergunta.** Qual a menor injeção coerente por crosstalk detectável com
falso alarme fixo, quantos shots de canário isso custa, e quais detectores
resistem a deriva de calibração e a um atacante que dosa o ataque?

## Modelo físico

Dois qubits: o atacante (q1, em |1⟩) e o canário (q0, em |+⟩). Em cada uma
das k janelas ociosas do canário, uma porta `id` com depolarizante p
representa o ruído de fundo e uma `rzz(θ)` representa o acoplamento ZZ
induzido pelo atacante. Depois, H e medição. A probabilidade de medir 1 é

    P(1) = (1 − r·cos(kθ)) / 2,   r = (1 − 4p/3)^k

Essa forma fechada é um oráculo: ela concorda com a matriz densidade
calculada pela tradução de ruído do próprio QSim até 1e-16, e com a
amostragem pela API pública dentro do erro estatístico
(`tests/test_tema1_canario.py`).

Hipóteses de modelagem, a revisar com dados de hardware: crosstalk apenas
ZZ coerente; ruído de fundo depolarizante isotrópico; a porta do atacante
sem ruído próprio; shots independentes entre si.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `modelo.py` | circuito do canário e P(1) por três fontes independentes |
| `piloto_binomial.py` | reprodução versionada do piloto de 18/10/2026 |
| `detectores.py` | Shewhart, CUSUM, CUSUM com referência móvel, CUSUM diferencial |
| `cenarios.py` | deriva de calibração, descasamento de deriva e perfis de atacante |
| `benchmark_sequencial.py` | calibração, avaliação e métricas dos detectores |
| `manifesto.py` | proveniência anexada a todo resultado |
| `resultados/` | referências gravadas (JSON e CSV) |

## Como reproduzir

    PYTHONPATH=. python3 tests/test_tema1_canario.py
    PYTHONPATH=. python3 experiments/tema1_canario/piloto_binomial.py --verificar
    PYTHONPATH=. python3 experiments/tema1_canario/benchmark_sequencial.py --verificar

Sem `--verificar`, os scripts regravam a referência. As três linhas acima
rodam no CI.

**Protocolo de proveniência.** O código é commitado primeiro; as
referências são geradas com a árvore limpa e commitadas em seguida. Cada
JSON traz um manifesto com commit, branch, estado da árvore, versões de
Python, numpy, qiskit e qiskit-aer, plataforma e SHA-256 dos arquivos do
experimento e do motor. As referências atuais foram geradas no commit
`b0ddca4`, com qiskit-aer 0.17.2 e numpy 2.5.4.

**Critério de verificação.** O piloto reproduz bit a bit com o mesmo
motor e a mesma versão do qiskit-aer; em outro ambiente, exige amostras a
menos de 5σ da fórmula fechada e a teoria dentro do IC de Wilson de 99,9%.
O benchmark reproduz bit a bit com a mesma versão do numpy.

## Resultados

### Piloto binomial

Canário com 10 janelas, fundo de 0,2%, injeção θ = 0,02 rad por janela,
teste unilateral com falso alarme nominal de 1%.

| Shots | Poder, normal (piloto original) | Poder, binomial exato | Poder empírico, 1000 jobs do QSim [IC 95%] |
|---|---|---|---|
| 500 | 0,364 | 0,259 | 0,261 [0,235; 0,289] |
| 2000 | 0,862 | 0,825 | 0,849 [0,825; 0,870] |

A aproximação normal do piloto original superestimava o poder com 500
shots; o valor empírico confirma o teste exato e exclui 0,364. Por
discretização, o falso alarme real do teste exato é 0,75%, não 1% (empírico:
0,80% e 0,90%).

### Benchmark sequencial

500 shots por janela, 300 janelas, ataque a partir da janela 100, base
calibrada nas janelas 0 a 49, limiares calibrados para 5% de falso alarme
no horizonte em 4000 réplicas independentes das 4000 de avaliação.

Falso alarme sem ataque:

| Cenário | Shewhart | CUSUM | CUSUM móvel | CUSUM diferencial |
|---|---|---|---|---|
| S0 estacionário | 0,049 | 0,043 | 0,051 | 0,045 |
| S1 deriva comum de 50% | 0,476 | 0,998 | 0,255 | 0,059 |
| S2 deriva descasada (κ = 0,5) | 0,481 | 0,997 | 0,255 | 0,883 |

Detecção com ataque (atraso mediano em janelas):

| Cenário | Shewhart | CUSUM | CUSUM móvel | CUSUM diferencial |
|---|---|---|---|---|
| S3 constante θ = 0,01 | 0,218 (91) | 0,825 (56) | 0,312 (24) | 0,957 (63) |
| S4 constante θ = 0,01 com deriva | 0,817 (100) | 0,951 (23) | 0,598 (23) | 0,872 (68) |
| S5 rampa até θ = 0,03 com deriva | 0,979 (99) | 0,954 (52) | 0,966 (89) | 0,997 (97) |
| S6 intermitente θ = 0,03 | 0,991 (1) | 0,997 (3) | 0,995 (2) | 0,995 (4) |
| S7 rampa lenta até θ = 0,015 | 0,249 (165) | 0,928 (158) | 0,290 (152) | 0,956 (157) |

Sob deriva, a detecção bruta se confunde com o falso alarme da própria
deriva. A detecção atribuível ao ataque em S4 (S4 menos o alarme pós-τ de
S1) é 0,37 no Shewhart, 0,01 no CUSUM, 0,38 no CUSUM móvel e 0,82 no CUSUM
diferencial.

### Leitura

1. Detectores com base fixa não servem sob deriva de calibração: o CUSUM
   dispara em 99,8% das réplicas sem nenhum ataque, e sua detecção
   atribuível em S4 é praticamente nula.
2. A referência móvel tolera parte da deriva, mas um atacante que cresce
   devagar a treina: em S7 ela detecta 29%, contra 96% do diferencial.
3. O canário diferencial resolve deriva comum e atacante lento, mas cai
   quando os qubits derivam de forma diferente (88% de falso alarme em S2).
   Medir o descasamento real de deriva entre qubits vizinhos e distantes, e
   construir um detector robusto a ele, é o problema central do artigo.

## Limitações e próximos passos

- Estimar deriva e descasamento (κ) a partir de séries de calibração
  publicadas pela IBM, em vez de fixá-los.
- Acrescentar dephasing e relaxação térmica T1/T2 ao QSim e repetir o
  benchmark com o ruído de fundo justificado fisicamente.
- Atacante adaptativo contra o detector diferencial (atingir também a
  referência) e detectores com múltiplas referências.
- Validar o canário e o modelo de injeção em hardware real.
