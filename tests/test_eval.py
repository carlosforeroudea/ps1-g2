"""Tests de las métricas de calidad. No descargan modelos (ROUGE es local)."""

from __future__ import annotations

import pandas as pd
import pytest

from resumidor.eval.aggregate import resumir_por_configuracion
from resumidor.eval.metrics import preparar_para_lsum, rouge


class TestPrepararParaLsum:
    """ROUGE-Lsum necesita una oración por línea o degenera en ROUGE-L."""

    def test_respeta_los_saltos_existentes(self) -> None:
        # Así vienen los abstracts del corpus.
        assert preparar_para_lsum("uno .\n dos .") == "uno .\ndos ."

    def test_traduce_el_separador_de_pegasus(self) -> None:
        # PEGASUS emite `<n>` en lugar de un salto de línea.
        assert preparar_para_lsum("uno . <n> dos .") == "uno .\ndos ."

    def test_segmenta_un_bloque_corrido(self) -> None:
        # BART devuelve un bloque sin saltos: hay que segmentarlo o
        # rougeLsum mediría sobre una sola "oración" gigante.
        assert preparar_para_lsum("Uno. Dos! Tres?") == "Uno.\nDos!\nTres?"

    def test_descarta_lineas_vacias(self) -> None:
        assert preparar_para_lsum("uno .\n\n\n dos .") == "uno .\ndos ."

    def test_texto_vacio(self) -> None:
        assert preparar_para_lsum("   ") == ""


class TestRouge:
    def test_identidad_da_uno(self) -> None:
        r = rouge("the cat sat on the mat .", "the cat sat on the mat .")
        assert r.rouge1 == pytest.approx(1.0)
        assert r.rouge2 == pytest.approx(1.0)
        assert r.rougeLsum == pytest.approx(1.0)

    def test_sin_solapamiento_da_cero(self) -> None:
        r = rouge("alpha beta gamma", "delta epsilon zeta")
        assert r.rouge1 == pytest.approx(0.0)

    def test_solapamiento_parcial_queda_en_medio(self) -> None:
        r = rouge("the cat sat on the mat", "the cat stood on the rug")
        assert 0.0 < r.rouge1 < 1.0
        # ROUGE-2 castiga más que ROUGE-1 porque exige bigramas completos.
        assert r.rouge2 <= r.rouge1


def _fila(config_id: str, modelo: str, **extra) -> dict:
    base = {
        "config_id": config_id,
        "modelo": modelo,
        "estrategia": "truncation",
        "doc_id": "test-00000",
        "latencia_s": 10.0,
        "pico_memoria_mb": 1000.0,
        "ratio_compresion": 0.02,
        "tokens_salida": 200,
        "invocaciones_modelo": 1,
        "rouge1": 0.4,
        "rouge2": 0.1,
        "rougeLsum": 0.35,
    }
    return {**base, **extra}


class TestAgregacion:
    def test_una_fila_por_configuracion(self) -> None:
        df = pd.DataFrame(
            [
                _fila("a", "bart", doc_id="d1"),
                _fila("a", "bart", doc_id="d2"),
                _fila("b", "pegasus", doc_id="d1", rouge1=0.5),
            ]
        )
        tabla = resumir_por_configuracion(df)
        assert len(tabla) == 2
        assert tabla["n"].tolist() == [1, 2]  # ordenada por rouge1 descendente

    def test_reporta_p50_y_p95_no_la_media(self) -> None:
        """La media engaña en una distribución con cola larga (ADR-005)."""
        df = pd.DataFrame(
            [
                _fila("a", "bart", doc_id=f"d{i}", latencia_s=lat)
                for i, lat in enumerate([1.0, 1.0, 1.0, 1.0, 100.0])
            ]
        )
        tabla = resumir_por_configuracion(df)
        assert tabla["latencia_p50"].iloc[0] == pytest.approx(1.0)
        assert tabla["latencia_p95"].iloc[0] > 50  # el atípico sí aparece aquí

    def test_la_memoria_se_agrega_como_maximo(self) -> None:
        # Es un *pico*: promediarlo no significaría nada.
        df = pd.DataFrame(
            [
                _fila("a", "bart", doc_id="d1", pico_memoria_mb=100.0),
                _fila("a", "bart", doc_id="d2", pico_memoria_mb=900.0),
            ]
        )
        tabla = resumir_por_configuracion(df)
        assert tabla["memoria_pico_mb"].iloc[0] == pytest.approx(900.0)

    def test_funciona_sin_bertscore(self) -> None:
        # BERTScore es opcional: carga un modelo de 1,4 GB.
        df = pd.DataFrame([_fila("a", "bart")])
        tabla = resumir_por_configuracion(df)
        assert "bertscore" not in tabla.columns
        assert "rouge1" in tabla.columns
