"""Comparaciones pareadas entre configuraciones (ADR-000).

Todas las celdas ven los mismos 300 documentos, así que la comparación
correcta es **por documento**: la diferencia de ROUGE entre dos estrategias
sobre el mismo artículo. Eso elimina la varianza entre artículos, que en este
corpus es mucho mayor que la diferencia entre estrategias.

- Prueba: Wilcoxon de rangos con signo (no asume normalidad; los ROUGE por
  documento no lo son).
- Tamaño del efecto: diferencia media con IC 95 % por bootstrap pareado.
- Comparaciones múltiples: corrección de Holm sobre toda la familia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ESTRATEGIAS_FACTORIAL = ("truncation", "map_reduce", "extractive_abstractive")


def comparar_pareado(
    df: pd.DataFrame,
    a: str,
    b: str,
    metrica: str,
    *,
    n_bootstrap: int = 5000,
    seed: int = 20262,
) -> dict:
    """Compara `a` contra `b` (ids de configuración) en `metrica`, por documento.

    Diferencia positiva significa que `a` supera a `b`.
    """
    from scipy.stats import wilcoxon

    pivote = df[df["config_id"].isin([a, b])].pivot_table(
        index="doc_id", columns="config_id", values=metrica
    )
    pivote = pivote.dropna()
    diferencias = (pivote[a] - pivote[b]).to_numpy()
    n = len(diferencias)
    if n == 0:
        raise ValueError(f"{a} y {b} no comparten documentos.")

    rng = np.random.default_rng(seed)
    medias = rng.choice(diferencias, size=(n_bootstrap, n), replace=True).mean(axis=1)
    bajo, alto = np.quantile(medias, [0.025, 0.975])

    p = 1.0 if np.allclose(diferencias, 0) else float(wilcoxon(diferencias).pvalue)
    return {
        "a": a,
        "b": b,
        "metrica": metrica,
        "n": n,
        "media_a": float(pivote[a].mean()),
        "media_b": float(pivote[b].mean()),
        "diferencia": float(diferencias.mean()),
        "ic95_bajo": float(bajo),
        "ic95_alto": float(alto),
        "gana_a": float((diferencias > 0).mean()),
        "p_wilcoxon": p,
    }


def holm(p_valores: pd.Series) -> pd.Series:
    """p-valores ajustados por Holm-Bonferroni, alineados con la entrada."""
    orden = p_valores.sort_values()
    m = len(orden)
    ajustados = (orden * np.arange(m, 0, -1)).cummax().clip(upper=1.0)
    return ajustados.reindex(p_valores.index)


def pares_del_diseño(configs: pd.DataFrame) -> list[tuple[str, str]]:
    """Las comparaciones que responden la pregunta de investigación.

    `configs` tiene una fila por configuración con `config_id`, `modelo` y
    `estrategia`. Devuelve:

    1. Dentro de cada modelo de contexto corto: cada estrategia contra
       `truncation`, y `map_reduce` contra `extractive_abstractive`.
    2. Cada celda del factorial contra el piso `lead_k`.
    """
    pares: list[tuple[str, str]] = []
    factorial = configs[configs["estrategia"].isin(ESTRATEGIAS_FACTORIAL)]
    for _, grupo in factorial.groupby("modelo"):
        por_estrategia = dict(zip(grupo["estrategia"], grupo["config_id"], strict=True))
        base = por_estrategia.get("truncation")
        for estrategia in ("map_reduce", "extractive_abstractive"):
            if base and estrategia in por_estrategia:
                pares.append((por_estrategia[estrategia], base))
        if {"map_reduce", "extractive_abstractive"} <= por_estrategia.keys():
            pares.append(
                (por_estrategia["map_reduce"], por_estrategia["extractive_abstractive"])
            )

    piso = configs.loc[configs["estrategia"] == "lead_k", "config_id"]
    if len(piso):
        pares.extend((c, piso.iloc[0]) for c in factorial["config_id"])
    return pares


def pruebas_pareadas(
    df: pd.DataFrame, metricas: list[str], pares: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Tabla de comparaciones pareadas con p ajustado por Holm."""
    if pares is None:
        configs = df[["config_id", "modelo", "estrategia"]].drop_duplicates()
        pares = pares_del_diseño(configs)
    filas = [comparar_pareado(df, a, b, m) for a, b in pares for m in metricas]
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    tabla["p_holm"] = holm(tabla["p_wilcoxon"])
    tabla["significativo"] = tabla["p_holm"] < 0.05
    return tabla


def recuperacion_frente_a(
    resumen: pd.DataFrame, referencia: str, metrica: str = "rougeLsum"
) -> pd.DataFrame:
    """La afirmación central del ADR-003, en cifras.

    «La estrategia X recupera el N % del ROUGE-Lsum de LED a un M % de su
    latencia y un P % de su pico de memoria.» `resumen` es la salida de
    `resumir_por_configuracion`; `referencia`, el `config_id` del baseline.
    """
    tabla = resumen.reset_index().set_index("config_id")
    if referencia not in tabla.index:
        raise KeyError(f"No hay resultados de {referencia!r}.")
    ref = tabla.loc[referencia]
    return pd.DataFrame(
        {
            f"{metrica}_pct": 100 * tabla[metrica] / ref[metrica],
            "latencia_p50_pct": 100 * tabla["latencia_p50"] / ref["latencia_p50"],
            "memoria_pct": 100 * tabla["memoria_pico_mb"] / ref["memoria_pico_mb"],
        }
    ).sort_values(f"{metrica}_pct", ascending=False)
