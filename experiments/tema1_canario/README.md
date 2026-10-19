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
| `detectores.py` | Shewhart, CUSUM, CUSUM com referência móvel, CUSUM diferencial e diferencial calibrado |
| `cenarios.py` | deriva de calibração, descasamento de deriva e perfis de atacante |
| `benchmark_sequencial.py` | calibração, avaliação e métricas dos detectores |
| `manifesto.py` | proveniência anexada a todo resultado |
| `viabilidade.py` | critérios C1 a C5 de viabilidade do artigo |
| `calibracao/` | coleta e análise do histórico de calibração da IBM e benchmark empírico |
| `resultados/` | referências gravadas (JSON e CSV) |

## Como reproduzir

    PYTHONPATH=. python3 tests/test_tema1_canario.py
    PYTHONPATH=. python3 tests/test_tema1_calibracao.py
    PYTHONPATH=. python3 experiments/tema1_canario/piloto_binomial.py --verificar
    PYTHONPATH=. python3 experiments/tema1_canario/benchmark_sequencial.py --verificar

Sem `--verificar`, os scripts regravam a referência. As quatro linhas acima
rodam no CI. O veredito de viabilidade sai de
`PYTHONPATH=. python3 experiments/tema1_canario/viabilidade.py`.

**Protocolo de proveniência.** O código é commitado primeiro; as
referências são geradas com a árvore limpa e commitadas em seguida. Cada
JSON traz um manifesto com commit, branch, estado da árvore, versões de
Python, numpy, qiskit e qiskit-aer, plataforma e SHA-256 dos arquivos do
experimento e do motor. As referências atuais foram geradas no commit
`259a4b5`, com qiskit-aer 0.17.2 e numpy 2.5.4.

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

| Cenário | Shewhart | CUSUM | CUSUM móvel | Diferencial | Diferencial calibrado |
|---|---|---|---|---|---|
| S0 estacionário | 0,049 | 0,043 | 0,051 | 0,045 | 0,047 |
| S1 deriva comum de 50% | 0,476 | 0,998 | 0,255 | 0,059 | 0,063 |
| S2 deriva descasada (κ = 0,5) | 0,481 | 0,997 | 0,255 | 0,883 | 0,396 |

Detecção com ataque (atraso mediano em janelas):

| Cenário | Shewhart | CUSUM | CUSUM móvel | Diferencial | Diferencial calibrado |
|---|---|---|---|---|---|
| S3 constante θ = 0,01 | 0,218 (91) | 0,825 (56) | 0,312 (24) | 0,957 (63) | 0,586 (89) |
| S4 constante θ = 0,01 com deriva | 0,817 (100) | 0,951 (23) | 0,598 (23) | 0,872 (68) | 0,469 (85) |
| S5 rampa até θ = 0,03 com deriva | 0,979 (99) | 0,954 (52) | 0,966 (89) | 0,997 (97) | 1,000 (117) |
| S6 intermitente θ = 0,03 | 0,991 (1) | 0,997 (3) | 0,995 (2) | 0,995 (4) | 0,994 (24) |
| S7 rampa lenta até θ = 0,015 | 0,249 (165) | 0,928 (158) | 0,290 (152) | 0,956 (157) | 0,592 (170) |

Sob deriva, a detecção bruta se confunde com o falso alarme da própria
deriva. A detecção atribuível ao ataque em S4 (S4 menos o alarme pós-τ de
S1) é:
Shewhart 0,372, CUSUM 0,007, CUSUM móvel 0,379, diferencial 0,817, diferencial calibrado 0,407.

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
4. O diferencial original supõe que os dois canários têm a mesma base de
   ruído, o que não vale entre qubits reais. A versão calibrada estima a
   razão de base e reduz o falso alarme sob deriva descasada (40% contra
   88%), ao custo de poder contra a rampa lenta (59% contra 96%).

## Calibração real da IBM

O pipeline em `calibracao/` substitui a deriva e o κ hipotéticos por dados
de calibração de um processador real.

1. Coleta (com a sua conta na IBM Quantum Platform; o plano aberto basta).
   Retomável: pode ser interrompida e rodada de novo.

        pip install qiskit-ibm-runtime
        PYTHONPATH=. python3 experiments/tema1_canario/calibracao/coletar_ibm.py \
            --backend ibm_torino --dias 90

2. Análise: deriva entre calibrações, fração de modo comum e κ por
   distância no chip, com IC por bootstrap.

        PYTHONPATH=. python3 experiments/tema1_canario/calibracao/analisar_deriva.py \
            --backend ibm_torino

3. Benchmark empírico: cada réplica sorteia um par real de qubits e uma
   transição real entre calibrações.

        PYTHONPATH=. python3 experiments/tema1_canario/calibracao/benchmark_empirico.py \
            --backend ibm_torino

A conversão de T1/T2 em ruído do canário iguala a fidelidade média do
relaxamento à do depolarizante (conferida contra o canal de relaxação
térmica do Qiskit a 1e-12). A análise recupera o κ verdadeiro de um
histórico sintético em todas as distâncias
(`tests/test_tema1_calibracao.py`). Pelo que se encontrou na literatura,
as flutuações de T1 em processadores IBM são da ordem de 20% da média e
atribuídas a defeitos locais, mas a correlação entre qubits não foi
medida ([Carroll et al., npj QI 2022](https://link.springer.com/article/10.1038/s41534-022-00643-y)).

Limitação: a calibração da IBM é diária, então o pipeline mede deriva na
escala de dias. Deriva dentro do dia exige rodar canários no hardware.

## Limitações e próximos passos

- Rodar a coleta em pelo menos um processador de 127 ou 133 qubits por 90
  dias e repetir o benchmark com os parâmetros medidos.
- Acrescentar dephasing e relaxação térmica T1/T2 ao QSim e repetir o
  benchmark com o ruído de fundo justificado fisicamente.
- Atacante adaptativo contra o detector diferencial (atingir também a
  referência) e detectores com múltiplas referências.
- Validar o canário e o modelo de injeção em hardware real.
