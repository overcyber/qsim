"""Detectores sequenciais sobre contagens de canarios.

Todas as funcoes operam sobre matrizes (R, W): R replicas de Monte Carlo,
W janelas de monitoramento. c[r, w] = numero de shots com resultado 1 no
canario proximo (exposto ao atacante) na janela w, de n shots; f[r, w] = o
mesmo para o canario de referencia distante (sujeito a mesma deriva, sem
acoplamento com o atacante).

Cada detector devolve um escore por janela; o alarme e o primeiro w >=
inicio com escore > h. O limiar h e calibrado por Monte Carlo para uma
probabilidade de falso alarme fixa no horizonte (calibrar_limiar).

  shewhart          z da contagem contra a base estimada na fase de
                    calibracao (janelas 0..inicio-1); sem memoria.
  cusum             CUSUM de Page com a razao de verossimilhanca binomial
                    entre p0 (base estimada) e p1 = p0 (1 + delta).
  cusum_movel       o mesmo CUSUM com p0 reestimado numa janela movel de L
                    janelas, defasada de g. Absorve deriva, mas tambem
                    absorve um ataque que cresce devagar.
  cusum_diferencial CUSUM condicional: dado t = c + f, c ~ Bin(t, q) com
                    q = p_prox / (p_prox + p_ref). Sem ataque q = 1/2 em
                    qualquer nivel de deriva comum; com ataque q > 1/2.
                    Nao usa base estimada; supoe bases iguais.
  cusum_diferencial_calibrado
                    o mesmo, com a razao de base entre os canarios
                    estimada na calibracao (qubits reais tem bases
                    diferentes; sem isso o diferencial dispara sem ataque).
  cusum_multicomprimento
                    proposto. Um so qubit, canarios de comprimentos
                    m0 < m1 < m2 igualmente espacados (ex.: 0, 30, 60
                    janelas) dividindo os mesmos n shots. Com coerencia
                    c_m = 1 - 2 P_m = A r^m cos(m theta), log c_m e afim em m
                    para qualquer A (leitura) e r (decoerencia); a segunda
                    diferenca D = log c0 - 2 log c1 + log c2 cancela os dois
                    e fica so com o termo coerente, ~ -(m1 - m0)^2 theta^2.
                    CUSUM gaussiano sobre -D, com variancia e vies de
                    segunda ordem estimados numa janela movel. Nao usa
                    referencia; nao distingue o ataque de outro
                    deslocamento coerente de fase (dessintonia, ZZ legitimo).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _p0_calibracao(c: np.ndarray, n: int, inicio: int) -> np.ndarray:
    p0 = c[:, :inicio].sum(axis=1) / (inicio * n)
    return np.clip(p0, _EPS, 1 - _EPS)[:, None]


def _llr_binomial(c: np.ndarray, n: int, p0: np.ndarray,
                  p1: np.ndarray) -> np.ndarray:
    p0 = np.clip(p0, _EPS, 1 - _EPS)
    p1 = np.clip(p1, _EPS, 1 - _EPS)
    return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))


def _cusum(llr: np.ndarray, inicio: int) -> np.ndarray:
    s = np.full(llr.shape, -np.inf)
    acc = np.zeros(llr.shape[0])
    for w in range(inicio, llr.shape[1]):
        acc = np.maximum(0.0, acc + llr[:, w])
        s[:, w] = acc
    return s


def shewhart(c: np.ndarray, n: int, inicio: int, **_) -> np.ndarray:
    p0 = _p0_calibracao(c, n, inicio)
    z = (c - n * p0) / np.sqrt(n * p0 * (1 - p0))
    z[:, :inicio] = -np.inf
    return z


def cusum(c: np.ndarray, n: int, inicio: int, delta: float, **_) -> np.ndarray:
    p0 = _p0_calibracao(c, n, inicio)
    return _cusum(_llr_binomial(c, n, p0, p0 * (1 + delta)), inicio)


def cusum_movel(c: np.ndarray, n: int, inicio: int, delta: float,
                janela_ref: int, defasagem: int, **_) -> np.ndarray:
    if janela_ref + defasagem > inicio:
        raise ValueError("janela_ref + defasagem deve caber antes de inicio")
    soma = np.concatenate([np.zeros((c.shape[0], 1)), np.cumsum(c, axis=1)],
                          axis=1)
    p_ref = np.full(c.shape, np.nan)
    for w in range(inicio, c.shape[1]):
        fim = w - defasagem
        p_ref[:, w] = (soma[:, fim] - soma[:, fim - janela_ref]) / (
            janela_ref * n)
    llr = np.zeros(c.shape)
    llr[:, inicio:] = _llr_binomial(c[:, inicio:], n, p_ref[:, inicio:],
                                    p_ref[:, inicio:] * (1 + delta))
    return _cusum(llr, inicio)


def cusum_diferencial(c: np.ndarray, n: int, inicio: int, delta: float,
                      f: np.ndarray, **_) -> np.ndarray:
    q1 = (1 + delta) / (2 + delta)
    llr = c * np.log(2 * q1) + f * np.log(2 * (1 - q1))
    return _cusum(llr, inicio)


def cusum_diferencial_calibrado(c: np.ndarray, n: int, inicio: int,
                                delta: float, f: np.ndarray,
                                **_) -> np.ndarray:
    """CUSUM diferencial com a razao de base estimada na calibracao.

    Qubits diferentes tem ruido de base diferente (T1 e T2 variam pelo
    chip), entao sem ataque q = p_prox / (p_prox + p_ref) nao e 1/2. q0 e
    estimado nas janelas 0..inicio-1; sob ataque, a razao de chances
    p_prox / p_ref cresce pelo fator (1 + delta).
    """
    soma_c = c[:, :inicio].sum(axis=1)
    soma_f = f[:, :inicio].sum(axis=1)
    q0 = np.clip((soma_c + 0.5) / (soma_c + soma_f + 1.0), _EPS, 1 - _EPS)
    r1 = q0 / (1 - q0) * (1 + delta)
    q1 = r1 / (1 + r1)
    q0, q1 = q0[:, None], q1[:, None]
    llr = c * np.log(q1 / q0) + f * np.log((1 - q1) / (1 - q0))
    return _cusum(llr, inicio)


def theta_projeto(p_fundo: float, janelas: int, delta: float) -> float:
    """Rotacao por janela que aumenta P(1) do canario de `janelas` em
    (1 + delta): o mesmo ponto de projeto dos CUSUM de base fixa."""
    r = (1 - 4 * p_fundo / 3) ** janelas
    p0 = (1 - r) / 2
    return float(np.arccos((1 - 2 * (1 + delta) * p0) / r) / janelas)


def _var_log_coerencia(p1: np.ndarray, n: np.ndarray | int) -> np.ndarray:
    c = 1 - 2 * p1
    return 4 * p1 * (1 - p1) / (n * c * c)


def alocacao_multicomprimento(comprimentos: list[int], p_fundo: float,
                              leitura: float, total: int) -> list[int]:
    """Shots por comprimento que minimizam Var(D) no ponto nominal:
    n_m proporcional a |w_m| sqrt(v_m), pesos (1, -2, 1)."""
    pesos = np.array([1.0, 2.0, 1.0])
    p1 = np.array([leitura + (1 - 2 * leitura)
                   * (1 - (1 - 4 * p_fundo / 3) ** m) / 2
                   for m in comprimentos])
    frac = pesos * np.sqrt(_var_log_coerencia(p1, 1))
    frac /= frac.sum()
    n = np.floor(frac * total).astype(int)
    n[np.argmax(frac)] += total - n.sum()
    return [int(x) for x in n]


def _media_movel(x: np.ndarray, janela: int) -> np.ndarray:
    soma = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)],
                          axis=1)
    fim = np.arange(1, x.shape[1] + 1)
    ini = np.maximum(0, fim - janela)
    return (soma[:, fim] - soma[:, ini]) / (fim - ini)


def estatistica_multicomprimento(multi: list[np.ndarray], n_multi: list[int],
                                 janela_var: int
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """(D, Var(D)) por janela. D com correcao de vies de segunda ordem
    (E[log c_hat] ~ log c - v/2); variancia pela taxa media das ultimas
    janela_var janelas, o que acompanha a deriva sem ser inflada por um
    ataque intermitente na propria janela."""
    D = 0.0
    var = 0.0
    for cnt, nm, w in zip(multi, n_multi, (1.0, -2.0, 1.0)):
        p_hat = (cnt + 0.5) / (nm + 1.0)
        p_mov = np.clip(_media_movel(cnt, janela_var) / nm, 0.5 / nm,
                        0.5 - 1e-3)
        v = _var_log_coerencia(p_mov, nm)
        c = np.clip(1 - 2 * p_hat, 1e-3, None)
        D = D + w * (np.log(c) + v / 2)
        var = var + w * w * v
    return D, var


def cusum_multicomprimento(c: np.ndarray, n: int, inicio: int, delta: float,
                           multi: list[np.ndarray] | None = None,
                           n_multi: list[int] | None = None,
                           comprimentos: list[int] | None = None,
                           theta_proj: float | None = None,
                           janela_var: int = 20, **_) -> np.ndarray:
    if multi is None or n_multi is None or comprimentos is None \
            or theta_proj is None:
        raise ValueError("cusum_multicomprimento exige multi, n_multi, "
                         "comprimentos e theta_proj")
    m = np.asarray(comprimentos)
    if len(m) != 3 or m[1] - m[0] != m[2] - m[1] or m[1] <= m[0]:
        raise ValueError("comprimentos devem ser 3, crescentes e igualmente "
                         "espacados")
    mu1 = -float(np.log(np.cos(m[0] * theta_proj))
                 - 2 * np.log(np.cos(m[1] * theta_proj))
                 + np.log(np.cos(m[2] * theta_proj)))
    D, var = estatistica_multicomprimento(multi, n_multi, janela_var)
    mu0 = D[:, :inicio].mean(axis=1, keepdims=True)
    x = -(D - mu0)
    llr = (mu1 / var) * (x - mu1 / 2)
    return _cusum(llr, inicio)


DETECTORES = {
    "shewhart": shewhart,
    "cusum": cusum,
    "cusum_movel": cusum_movel,
    "cusum_diferencial": cusum_diferencial,
    "cusum_diferencial_calibrado": cusum_diferencial_calibrado,
    "cusum_multicomprimento": cusum_multicomprimento,
}

# Detectores propostos neste trabalho (os demais sao a linha de base).
PROPOSTOS = ("cusum_multicomprimento",)


def calibrar_limiar(escore: np.ndarray, inicio: int, fap: float) -> float:
    """Limiar com P(algum alarme em [inicio, W)) = fap sob H0."""
    maximos = escore[:, inicio:].max(axis=1)
    return float(np.quantile(maximos, 1 - fap, method="higher"))


def primeiro_alarme(escore: np.ndarray, h: float, inicio: int) -> np.ndarray:
    """Indice da primeira janela >= inicio com escore > h; -1 se nenhuma.

    A comparacao e estrita porque o limiar vem de um quantil 'higher' dos
    maximos de calibracao: escores discretos iguais ao limiar nao disparam.
    """
    marca = escore[:, inicio:] > h
    tem = marca.any(axis=1)
    idx = np.where(tem, marca.argmax(axis=1) + inicio, -1)
    return idx
