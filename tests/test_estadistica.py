"""Tests de las comparaciones pareadas. Sin red."""

from __future__ import annotations

import pandas as pd
import pytest

from resumidor.eval.estadistica import holm, pares_del_diseño, pruebas_pareadas


def _resultados() -> pd.DataFrame:
    filas = []
    for i in range(40):
        base = 0.30 + (i % 7) * 0.01
        for config, estrategia, extra in [
            ("truncation_bart", "truncation", 0.0),
            ("map_reduce_bart", "map_reduce", 0.05),
            ("lead_k", "lead_k", -0.08),
        ]:
            filas.append(
                {
                    "config_id": config,
                    "modelo": "m" if estrategia != "lead_k" else "tok",
                    "estrategia": estrategia,
                    "doc_id": f"d{i}",
                    "rougeLsum": base + extra + (i % 3) * 0.001,
                }
            )
    return pd.DataFrame(filas)


def test_pares_del_diseño_compara_contra_truncation_y_el_piso() -> None:
    configs = _resultados()[["config_id", "modelo", "estrategia"]].drop_duplicates()
    pares = pares_del_diseño(configs)
    assert ("map_reduce_bart", "truncation_bart") in pares
    assert ("truncation_bart", "lead_k") in pares
    assert ("map_reduce_bart", "lead_k") in pares


def test_detecta_una_mejora_consistente() -> None:
    tabla = pruebas_pareadas(
        _resultados(), ["rougeLsum"], [("map_reduce_bart", "truncation_bart")]
    )
    fila = tabla.iloc[0]
    assert fila["diferencia"] == pytest.approx(0.05)
    assert fila["ic95_bajo"] > 0
    assert fila["gana_a"] == 1.0
    assert fila["significativo"]


def test_holm_es_monotono_y_acotado() -> None:
    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    ajustado = holm(p)
    assert (ajustado >= p).all()
    assert ajustado.max() <= 1.0
    assert ajustado[0] == pytest.approx(0.04)
