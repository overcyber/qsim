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
| `detectores.py` | linha de base (Shewhart, CUSUM, CUSUM móvel, diferenciais) e o detector proposto, multicomprimento |
| `calibracao/modelo_ruido.py` | conversão de T1/T2 em ruído do canário |
| `cenarios.py` | deriva de ruído, de leitura e de frequência, descasamento e perfis de atacante |
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
`25034be`, com qiskit-aer 0.17.2 e numpy 2.5.4.

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
no horizonte em 4000 réplicas independentes das 4000 de avaliação. Erro de
leitura de 1% em todos os canários. Detectores de base com canário de 10
janelas; o multicomprimento usa 0, 30 e 60 janelas com
56, 262 e 182 shots (o mesmo total de 500).

Falso alarme sem ataque:

| Cenário | Shewhart | CUSUM | CUSUM móvel | Diferencial | Dif. calibrado | Multicomprimento |
|---|---|---|---|---|---|---|
| S0 estacionário | 0,051 | 0,044 | 0,042 | 0,050 | 0,044 | 0,051 |
| S1 deriva comum de 50% | 0,309 | 0,942 | 0,149 | 0,056 | 0,049 | 0,042 |
| S2 deriva descasada (κ = 0,5) | 0,309 | 0,948 | 0,143 | 0,593 | 0,286 | 0,048 |
| S8 leitura do próximo de 1% a 1,5% | 0,207 | 0,790 | 0,110 | 0,956 | 0,540 | 0,048 |
| S9 ruído e leitura do próximo +100% | 1,000 | 1,000 | 0,645 | 1,000 | 1,000 | 0,053 |
| S11 deriva de frequência até 0,01 rad | 0,091 | 0,260 | 0,072 | 0,292 | 0,170 | 0,902 |

Detecção com ataque (atraso mediano em janelas):

| Cenário | Shewhart | CUSUM | CUSUM móvel | Diferencial | Dif. calibrado | Multicomprimento |
|---|---|---|---|---|---|---|
| S3 constante θ = 0,01 | 0,150 (98) | 0,569 (64) | 0,147 (24) | 0,689 (84) | 0,390 (85) | 0,991 (9) |
| S4 constante com deriva | 0,586 (116) | 0,956 (34) | 0,297 (26) | 0,595 (85) | 0,324 (83) | 0,988 (10) |
| S5 rampa até 0,03 com deriva | 0,982 (119) | 0,961 (65) | 0,973 (119) | 0,994 (108) | 0,997 (125) | 0,991 (69) |
| S6 intermitente θ = 0,03 | 0,992 (2) | 0,995 (3) | 0,994 (3) | 0,994 (6) | 0,944 (31) | 0,991 (0) |
| S7 rampa lenta até 0,015 | 0,158 (157) | 0,716 (167) | 0,145 (142) | 0,720 (169) | 0,415 (172) | 0,992 (122) |
| S10 adaptativo (atinge a referência) | 0,145 (101) | 0,562 (64) | 0,158 (25) | 0,050 (93) | 0,047 (91) | 0,990 (8) |

**Controle de comprimento.** O canário multicomprimento usa circuitos
mais longos, e o sinal coerente cresce com o quadrado do comprimento.
Repetindo a linha de base com canário de 30 janelas (`--janelas 30`; a
coluna do multicomprimento muda porque o ponto de projeto acompanha o
comprimento de referência):

| Cenário | Shewhart | CUSUM | CUSUM móvel | Diferencial | Dif. calibrado | Multicomprimento |
|---|---|---|---|---|---|---|
| S1 deriva comum de 50% | 0,821 | 1,000 | 0,227 | 0,051 | 0,048 | 0,046 |
| S2 deriva descasada (κ = 0,5) | 0,816 | 1,000 | 0,214 | 0,951 | 0,608 | 0,042 |
| S8 leitura do próximo de 1% a 1,5% | 0,117 | 0,372 | 0,072 | 0,498 | 0,283 | 0,049 |
| S9 ruído e leitura do próximo +100% | 1,000 | 1,000 | 0,717 | 1,000 | 1,000 | 0,041 |
| S11 deriva de frequência até 0,01 rad | 0,890 | 1,000 | 0,336 | 1,000 | 1,000 | 0,980 |

| Cenário | Shewhart | CUSUM | CUSUM móvel | Diferencial | Dif. calibrado | Multicomprimento |
|---|---|---|---|---|---|---|
| S3 constante θ = 0,01 | 0,992 | 0,992 | 0,990 | 0,991 | 0,992 | 0,997 |
| S7 rampa lenta até 0,015 | 0,989 | 0,992 | 0,992 | 0,993 | 0,994 | 0,999 |
| S10 adaptativo (atinge a referência) | 0,992 | 0,992 | 0,992 | 0,041 | 0,051 | 0,996 |

### Leitura

1. Detectores com base fixa não servem sob deriva: o CUSUM dispara em 94%
   das réplicas sem ataque com deriva comum de 50%.
2. A referência móvel tolera parte da deriva, mas uma rampa lenta a
   treina (S7: 15%).
3. Os diferenciais resolvem a deriva comum, mas caem quando os qubits
   derivam de forma diferente (S2, S8, S9) e ficam cegos a um atacante que
   atinge também a referência (S10: 5%).
4. A vantagem de poder do multicomprimento com canário de 10 janelas vem
   do circuito mais longo: com 30 janelas, toda a linha de base chega a
   0,99 em S3 e S7. O que só ele entrega é robustez. É o único com falso
   alarme de até 5% sob toda deriva incoerente (S1, S2, S8, S9) e, entre
   os que toleram deriva comum, o único que detecta o atacante adaptativo
   (S10). Shewhart e CUSUM também detectam S10, mas porque S10 não tem
   deriva.
5. Limite: uma deriva de frequência (S11) é um deslocamento coerente de
   fase, fisicamente igual ao ataque. O multicomprimento dispara em 90% das
   réplicas e a linha de base também se degrada, mais ainda com canário
   longo. Separar ataque de dessintonia e de ZZ legítimo de vizinhos é o
   problema aberto que sobra para o artigo.

### Detector proposto: canário multicomprimento

A coerência medida do canário de m janelas é c_m = 1 − 2P_m =
A·r^m·cos(mθ), com A dado pelo erro de leitura e r pela decoerência. Logo
log c_m = log A + m·log r + log cos(mθ): a parte de ruído é afim em m. Com
três comprimentos igualmente espaçados (0, L, 2L), a segunda diferença

    D = log c_0 − 2 log c_L + log c_2L = log cos(2Lθ) − 2 log cos(Lθ) ≈ −L²θ²

não depende de A nem de r, qualquer que seja a deriva dos dois, e não usa
qubit de referência. O detector é um CUSUM gaussiano sobre −D, com
variância e viés de segunda ordem estimados nas últimas 20 janelas (a
variância da própria janela é inflada por um ataque intermitente). A
identidade é conferida contra a matriz densidade do QSim, com erro de
leitura, a 1e-12 (`tests/test_tema1_canario.py`).

No benchmark empírico com histórico sintético (deriva real por qubit,
bases heterogêneas), o multicomprimento teve falso alarme entre 4,3% e
4,9% e detecção entre 0,91 e 0,95 em todas as distâncias no chip, contra
0,48 a 0,51 do diferencial calibrado com 14% a 18% de falso alarme.

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

**Conversão de T1/T2.** O canário mede na base X, então só a coerência
importa, e sob relaxamento térmico ela decai por e^(−τ/T2) por janela,
qualquer que seja T1. A conversão usada (`p_coerencia`) escolhe o
depolarizante com esse mesmo decaimento, e com ela a fórmula fechada do
canário é exata: o benchmark empírico confere isso, antes de rodar,
contra o canal de relaxamento térmico do próprio QSim (`t1`, `t2`,
`gate_time`), com diferença máxima de 1e-12. A conversão por fidelidade
média, usada numa versão anterior, erra P(1) em até 0,013 quando T1 e T2
diferem, mais que o sinal de um ataque com θ = 0,02. Consequência para o
artigo: a deriva que importa para este canário é a de T2.

A análise recupera o κ verdadeiro de um
histórico sintético em todas as distâncias
(`tests/test_tema1_calibracao.py`). Pelo que se encontrou na literatura,
as flutuações de T1 em processadores IBM são da ordem de 20% da média e
atribuídas a defeitos locais, mas a correlação entre qubits não foi
medida ([Carroll et al., npj QI 2022](https://link.springer.com/article/10.1038/s41534-022-00643-y)).

Limitação: a calibração da IBM é diária, então o pipeline mede deriva na
escala de dias. Deriva dentro do dia exige rodar canários no hardware.

## Limitações e próximos passos

- Rodar a coleta em pelo menos um processador de 127 ou 133 qubits por 90
  dias e repetir o benchmark empírico com os parâmetros medidos, incluindo
  a deriva do erro de leitura, que o CSV já registra.
- Deriva de frequência e ZZ legítimo de vizinhos (S11): medir em quadratura
  (base Y), onde o ataque tem sinal linear em θ com sinal definido, e
  canários com eco de Hahn, que cancelam dessintonia estática. Modelar a
  ocupação dos vizinhos por outros usuários.
- Atacante que conhece o protocolo: fase aleatória por shot (crosstalk
  incoerente, sem assinatura quadrática) e ataques sincronizados com os
  comprimentos do canário.
- Custo: o canário multicomprimento ocupa mais tempo de máquina por janela
  (circuitos de até 60 janelas ociosas).
- Validar o canário e o modelo de injeção em hardware real.
