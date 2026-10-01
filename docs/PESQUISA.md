# QSim — Guia de Uso para Pesquisa

Este documento descreve como utilizar o QSim como instrumento de pesquisa em
computação quântica. Todos os fluxos descritos aqui podem ser executados pela
interface web (console em `http://localhost:10000`), e os fluxos programáticos
contam adicionalmente com o SDK Python que fala diretamente com o NATS.

O QSim é um simulador teórico-empírico: o mesmo circuito pode ser executado no
regime exato (evolução unitária do vetor de estado, resultados determinísticos
dadas as amplitudes) e no regime empírico (trajetórias estocásticas com modelos
de ruído e amostragem finita), permitindo confrontar previsão teórica e
comportamento experimental no mesmo ambiente.

## 1. Acesso e formas de entrada

O console web oferece três formas equivalentes de descrever um circuito, em
abas no painel esquerdo. O **Editor** é uma grade clicável: selecione uma porta
na paleta e clique na célula desejada; portas de dois e três qubits pedem
cliques sucessivos (controles e alvo) na mesma coluna, e clicar numa célula
ocupada remove a operação. A aba **JSON** aceita a lista de operações no
formato nativo do QSim, útil para colar circuitos gerados por scripts. A aba
**QASM** aceita OpenQASM 2.0 e 3.0 (subconjunto: um registro quântico, portas
nativas, `measure`, `barrier`, `reset`, parâmetros com expressões de `pi`),
permitindo importar circuitos escritos para Qiskit ou outros frameworks.

Se o gateway estiver com `QSIM_ADMIN_TOKEN` definido (recomendado), informe o
token no campo do topo; ele é usado nas chamadas REST e no WebSocket.

Os controles abaixo das abas definem o experimento: número de qubits (até 31),
shots (até 1 milhão), precisão (`complex64` ou `complex128`), verificação
cruzada, seed, prioridade, observáveis, ruído e rótulo. O rótulo identifica o
job no histórico e nas exportações — use nomes significativos como
`ghz5-ruido-0.01`.

## 2. Linhas de pesquisa suportadas

### 2.1 Estudo de algoritmos quânticos

Qualquer circuito de até 31 qubits sobre o conjunto de portas nativo (h, x, y,
z, s, sdg, t, tdg, sx, rx, ry, rz, p, u, cx, cz, cp, swap, ccx, cswap) pode ser
executado com estatística completa. Exemplos prontos para colar na aba QASM:

Estado de Bell e GHZ (correlações não clássicas):

    OPENQASM 3;
    qubit[3] q;
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];

Transformada de Fourier quântica em 3 qubits (base da estimativa de fase):

    OPENQASM 3;
    qubit[3] q;
    h q[2];
    cp(pi/2) q[1], q[2];
    h q[1];
    cp(pi/4) q[0], q[2];
    cp(pi/2) q[0], q[1];
    h q[0];
    swap q[0], q[2];

Grover com 2 qubits marcando |11> (uma iteração é exata nesse tamanho):

    OPENQASM 3;
    qubit[2] q;
    h q[0]; h q[1];
    cz q[0], q[1];
    h q[0]; h q[1];
    x q[0]; x q[1];
    cz q[0], q[1];
    x q[0]; x q[1];
    h q[0]; h q[1];

O resultado esperado do Grover é a concentração de probabilidade em |11>
(100% no caso ideal). Teleporte quântico pode ser estudado com medições no
meio do circuito (`measure` + correções condicionais aproximadas por
pós-seleção nas contagens).

Para inspeção de amplitudes, marque `return_statevector` (via JSON) em
circuitos de até 12 qubits; acima disso o sistema retorna as probabilidades
dos 256 estados mais prováveis, suficiente para verificar padrões de
interferência.

### 2.2 Ruído, decoerência e mitigação de erros

O modelo de ruído opera por trajetórias estocásticas (método de Monte Carlo
quântico): cada shot evolui uma cópia independente do estado com erros
sorteados. Três canais estão disponíveis, combináveis em três níveis:

    "noise": {
      "depolarizing": 0.001,
      "amplitude_damping": 0.0005,
      "readout_error": 0.01,
      "per_gate":  { "cx": { "depolarizing": 0.01 } },
      "per_qubit": { "3":  { "depolarizing": 0.005 } }
    }

O nível global aplica-se a toda porta; `per_gate` modela o fato experimental
de portas de dois qubits serem uma ordem de grandeza piores que as de um
qubit; `per_qubit` modela qubits fisicamente defeituosos. As contribuições
somam-se (saturando em 1), permitindo compor perfis realistas de hardware.

O fluxo típico na interface é executar o mesmo circuito duas vezes — uma sem
ruído (teórico) e outra com ruído (empírico) — e abrir os dois no painel
**Comparador**, que sobrepõe os histogramas e calcula a distância de variação
total (TVD) entre as distribuições. A TVD em função da intensidade de ruído é
a curva de decoerência do circuito; a profundidade em que a TVD satura mede a
profundidade útil sob aquele modelo de erro.

Na CPU, as trajetórias são paralelizadas automaticamente entre os núcleos
(`parallel: true`, padrão) — nos dois Xeon de 28 núcleos isso significa 28
trajetórias simultâneas.

### 2.3 Precisão numérica (complex64 vs complex128)

Linha de pesquisa diretamente ligada ao uso de GPUs: `complex64` dobra o
throughput e o número de qubits que cabem na VRAM, mas acumula erro de
arredondamento. O campo **Verificação** executa o circuito nas duas precisões
e reporta a fidelidade do estado |<psi64|psi128>|^2 e a TVD entre as
distribuições, com aprovação automática nos limiares (fidelidade > 1-1e-6,
TVD < 1e-4). Medição de referência do projeto: 400 portas aleatórias em 20
qubits mantêm fidelidade 0,999999094 e TVD 4,7e-7 — base quantitativa para
justificar `complex64` em produção num artigo ou relatório.

Para estudar o crescimento do erro com a profundidade, varra a profundidade
do circuito com `verify: true` e exporte a fidelidade de cada ponto (campo
`metadata.verification` do JSON exportado).

### 2.4 Emaranhamento

Dois diagnósticos são calculados sob demanda para circuitos de até 24 qubits:

**Vetores de Bloch** (`return_bloch`): o vetor (x, y, z) de cada qubit,
exibido como esfera em miniatura no console. O comprimento r do vetor mede a
pureza do estado reduzido: r = 1 indica qubit puro (separável do resto),
r = 0 indica qubit maximamente misturado — assinatura de emaranhamento
máximo com o restante do registro.

**Entropia de emaranhamento** (`return_entanglement`): a entropia de von
Neumann S da partição qubit-versus-resto, em bits, exibida como barra por
qubit. S = 0 para estados produto; S = 1 para emaranhamento máximo.

Experimento clássico: compare GHZ e W de 3 qubits. Ambos exibem S = 1 por
qubit... mas respondem diferente à perda de um qubit — meça um qubit do GHZ
(`measure q[0]`) e observe as entropias dos remanescentes colapsarem a zero,
enquanto no estado W sobrevive emaranhamento. O estado W pode ser preparado
com rotações: ry(1.9106332362490186) em q0 seguida de ch/cx (ou cole a
preparação por amplitudes que preferir).

### 2.5 Observáveis e tomografia parcial

O campo **Observáveis** aceita duas sintaxes: `z<idx>` para <Z> de um qubit e
strings de Pauli completas (`ZZ`, `XX`, `ZIXY`...) do tamanho do registro,
com o caractere na posição j atuando no qubit j. Valores esperados são
calculados exatamente a partir do vetor de estado (até 26 qubits), sem erro
estatístico de amostragem.

Isso viabiliza, por exemplo, o teste de desigualdades de Bell/CHSH: no estado
de Bell, meça `ZZ` e `XX` (ambos +1) e componha as correlações nas bases
giradas aplicando as rotações correspondentes antes da medição. Também é a
base de estimadores de energia tipo VQE: a energia de um Hamiltoniano
H = soma de termos de Pauli é a soma ponderada dos valores esperados
retornados.

### 2.6 Estatística de medição e reprodutibilidade

Todo job aceita `seed`; com a mesma seed, mesmo circuito e mesma precisão, o
resultado é bit a bit reprodutível — requisito para publicação. Sem seed, o
gerador é semeado pelo sistema. Para estudar flutuação estatística, execute o
mesmo circuito com shots crescentes (100, 1k, 10k, 100k) e observe a TVD
contra a distribuição exata (aba Comparador) cair com 1/sqrt(shots).

### 2.7 Benchmarking de desempenho

O QSim expõe sua própria instrumentação como objeto de pesquisa: o campo
`metadata` de cada resultado traz modo de execução, backend efetivo, contagem
de operações antes e depois da fusão de portas e tempo total; o Grafana
(porta 3000) traz percentis de latência, jobs por classe e utilização de GPU
via dcgm-exporter. O script `tests/benchmark.py` mede portas/s por número de
qubits, ganho da fusão (8,5x em circuitos densos de 1 qubit na medição de
referência) e taxa de amostragem (0,73 milhão de shots/s em CPU). Comparações
c64 vs c128 e CPU vs GPU (3090 vs 4060 Ti vs P40) saem naturalmente
submetendo o mesmo circuito com `backend` e `precision` diferentes.

### 2.8 Escolha do motor de execução

O campo Motor do console (ou "backend" no JSON) escolhe onde o circuito
roda: "aer" usa o motor C++ do Qiskit Aer (padrão recomendado para
produção; "aer-gpu" com qiskit-aer-gpu instalado), "cupy" e "custatevec"
usam as GPUs via QSim, e "numpy" executa o núcleo Python de referência.
Rodar o mesmo circuito em dois motores e comparar no painel Comparador é em
si um experimento de validação cruzada; a suíte tests/test_backend_aer.py
automatiza isso e confirma fidelidade 1,0 entre Aer e núcleo.

## 3. SDK Python para varreduras de parâmetros

Experimentos que varrem um parâmetro (ângulo, intensidade de ruído,
profundidade) usam o SDK, que conversa diretamente com o NATS sem passar pelo
HTTP. Exemplo completo — curva de decoerência de um GHZ de 4 qubits:

    from sdk.qsim_client import sweep_sync

    def ghz_com_ruido(p):
        return {
            "num_qubits": 4, "shots": 4096, "seed": 42,
            "noise": {"depolarizing": p},
            "operations": [
                {"gate": "h", "targets": [0]},
                {"gate": "cx", "targets": [0, 1]},
                {"gate": "cx", "targets": [1, 2]},
                {"gate": "cx", "targets": [2, 3]},
            ],
        }

    pontos = sweep_sync(ghz_com_ruido, [0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    for ponto in pontos:
        counts = ponto["result"]["counts"]
        fiel = (counts.get("0000", 0) + counts.get("1111", 0)) / 4096
        print(f"p = {ponto['value']:.3f}  fracao GHZ = {fiel:.4f}")

E a curva <Z(theta)> = cos(theta) de uma rotação Ry, verificada no teste de
integração do projeto:

    from sdk.qsim_client import sweep_sync
    import numpy as np

    curva = sweep_sync(
        lambda th: {"num_qubits": 1, "shots": 1, "observables": ["z0"],
                    "operations": [{"gate": "ry", "targets": [0],
                                    "params": [th]}]},
        np.linspace(0, 2 * np.pi, 63).tolist(),
    )

O método `sweep` controla a concorrência (padrão 4 jobs simultâneos) e
preserva a ordem dos valores. Jobs individuais usam `run_sync(circuit)` e a
prioridade (0-9) permite que varreduras longas rodem em prioridade baixa sem
atrapalhar o uso interativo do console.

## 4. Exportação para publicação

Todo job concluído oferece três exportações no console (e em
`GET /api/jobs/{id}/export?fmt=...`): **JSON** com o resultado completo,
circuito e metadados (arquivo de reprodutibilidade para material
suplementar); **CSV** com estado, contagens e probabilidades (entrada direta
para pandas/matplotlib/R); **LaTeX** com a tabela de resultados formatada em
booktabs e notação de kets, pronta para inclusão em artigo. O histórico
completo fica em SQLite e sobrevive a reinícios do sistema
(`GET /api/history`).

## 5. Validade científica dos resultados

A confiabilidade do simulador é sustentada por sete suítes verificáveis:
oráculo físico (Bell, GHZ, QFT, unitariedade), testes metamórficos
(U·U† = I, identidades de circuito como HZH = X e Toffoli via
decomposição), testes diferenciais entre precisões e backends, validação
cruzada externa contra o Qiskit Aer (fidelidade 1,000000000000 em 20
circuitos aleatórios na medição de referência), suíte de segurança com
entradas hostis, fuzzer de circuitos e teste de integração ponta a ponta.
Execute tudo com:

    PYTHONPATH=. python3 tests/test_physics.py
    PYTHONPATH=. python3 tests/test_metamorphic.py
    PYTHONPATH=. python3 tests/test_differential.py
    PYTHONPATH=. python3 tests/test_research.py
    PYTHONPATH=. python3 tests/test_security.py
    PYTHONPATH=. python3 tests/test_aer_crosscheck.py   # requer qiskit-aer
    PYTHONPATH=. python3 tests/integration_e2e.py       # requer nats-server

## 6. Limites operacionais

Vetor de estado exato até 31 qubits (16 GiB em complex64); verificação
cruzada e observáveis Pauli até 26 qubits; Bloch e entropia até 24 qubits;
vetor de estado completo retornado até 12 qubits; 1 milhão de shots por job;
quota de 8 jobs ativos simultâneos por origem (configurável). Jobs de 21-29
qubits são roteados à classe `medium` e de 30-31 à classe `large` — em
instalação apenas-CPU essas classes não têm consumidor por padrão (defina
`QSIM_ENGINE_CLASSES=small,medium,noisy` num worker se a RAM permitir).
Circuitos com ruído são sempre roteados à classe `noisy` e custam uma
trajetória completa por shot: dimensione shots de acordo (milhares, não
milhões, em registros grandes).

Como todo simulador de vetor de estado, o QSim reproduz a mecânica quântica
ideal ou com ruído modelado — ele não substitui hardware real para efeitos
não modelados (crosstalk coerente, drift de calibração, leakage), mas
fornece a linha de base exata contra a qual esses efeitos são medidos.
