"""Tests del runner y del contador de invocaciones. No usan red."""

from __future__ import annotations

import json
import pathlib

from resumidor.domain import Document
from resumidor.instrumentation import medir, metodo_memoria
from resumidor.models.base import SummarizerModel
from resumidor.models.counting import ContadorDeInvocaciones
from resumidor.runner import ejecutar_documento
from resumidor.strategies.truncation import Truncation

from .conftest import FakeModel


class FakeModelConDispositivo(FakeModel):
    dispositivo: str = "cpu"


def test_el_contador_satisface_el_puerto(model: FakeModel) -> None:
    # Si el envoltorio no es un SummarizerModel, las estrategias no lo aceptan.
    assert isinstance(ContadorDeInvocaciones(model), SummarizerModel)


def test_el_contador_delega_sin_alterar(model: FakeModel) -> None:
    c = ContadorDeInvocaciones(model)
    assert c.name == model.name
    assert c.context_window == model.context_window
    assert c.count_tokens("a b c") == 3
    assert c.truncate("a b c d", 2) == "a b"


def test_el_contador_cuenta_las_invocaciones(model: FakeModel) -> None:
    c = ContadorDeInvocaciones(model)
    assert c.invocaciones == 0
    c.generate("hola mundo", max_new_tokens=2)
    c.generate("otra vez", max_new_tokens=2)
    assert c.invocaciones == 2
    c.reiniciar()
    assert c.invocaciones == 0


def test_truncation_registra_exactamente_una_invocacion(
    long_document: Document,
) -> None:
    """El número es observado, no declarado por la estrategia.

    Es lo que hará creíble el N+1 de map-reduce cuando se implemente.
    """
    resultado = ejecutar_documento(
        long_document, FakeModelConDispositivo(), Truncation(max_new_tokens=4)
    )
    assert resultado.cost.model_calls == 1


def test_el_resultado_trae_siempre_sus_medidas(long_document: Document) -> None:
    # RT-2: no existe una ruta que devuelva resumen sin costo.
    r = ejecutar_documento(
        long_document, FakeModelConDispositivo(), Truncation(max_new_tokens=4)
    )
    assert r.summary
    assert r.cost.latency_seconds >= 0
    assert r.cost.input_tokens == 30  # el documento de prueba
    assert r.cost.output_tokens > 0
    assert 0 < r.compression_ratio < 1
    assert r.model_name == "fake/model"
    assert r.strategy_name == "truncation"


def test_la_latencia_mide_la_estrategia_completa() -> None:
    """La instrumentación envuelve la estrategia, no el modelo (ADR-005 §3)."""
    import time

    with medir("cpu") as m:
        time.sleep(0.05)
    assert m.latencia_s >= 0.05


def test_el_metodo_de_memoria_queda_registrado() -> None:
    # Las cifras solo son comparables entre corridas del mismo método.
    assert metodo_memoria("cuda") == "cuda.max_memory_allocated"
    assert metodo_memoria("mps") == "mps.driver_allocated_memory"
    assert metodo_memoria("cpu") == "rss.ru_maxrss"


def test_la_fila_de_resultado_es_serializable(
    long_document: Document, tmp_path: pathlib.Path
) -> None:
    from resumidor.config import ExperimentoConfig
    from resumidor.runner import _fila

    config = ExperimentoConfig(
        id="prueba",
        model={"kind": "bart", "checkpoint": "facebook/bart-large-cnn"},
        strategy={"kind": "truncation"},
        sample={"dataset": "d", "split": "test", "n": 1, "seed": 7},
    )
    r = ejecutar_documento(
        long_document, FakeModelConDispositivo(), Truncation(max_new_tokens=4)
    )
    fila = _fila(config, long_document, r, "cpu", "rss.ru_maxrss")

    linea = json.dumps(fila, ensure_ascii=False)
    devuelta = json.loads(linea)

    # Lo que pide la tarjeta: modelo, artículo, resumen y tiempo.
    for campo in ("modelo", "doc_id", "resumen_generado", "latencia_s"):
        assert campo in devuelta
    # Y lo que necesitan los tests pareados del ADR-000.
    for campo in ("config_id", "semilla", "tokens_entrada", "invocaciones_modelo"):
        assert campo in devuelta
    # El dispositivo resuelto, no "auto": la latencia no es comparable entre
    # dispositivos distintos.
    assert devuelta["dispositivo"] == "cpu"
    assert devuelta["doc_id"] == long_document.doc_id
