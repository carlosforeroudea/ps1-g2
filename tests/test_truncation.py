"""Tests de `Truncation` y del contrato de los puertos (ADR-005)."""

from __future__ import annotations

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel
from resumidor.strategies.base import ContextStrategy
from resumidor.strategies.truncation import Truncation

from .conftest import FakeModel


def test_el_modelo_falso_satisface_el_puerto(model: FakeModel) -> None:
    # Si esto falla, el contrato pide algo propio de Hugging Face y las
    # estrategias dejaron de ser agnósticas del modelo.
    assert isinstance(model, SummarizerModel)


def test_truncation_satisface_el_puerto_de_estrategia() -> None:
    assert isinstance(Truncation(), ContextStrategy)


def test_truncation_invoca_al_modelo_exactamente_una_vez(
    model: FakeModel, long_document: Document
) -> None:
    # La aridad es la asimetría que motiva el ADR-005 §2 y §3:
    # 1 aquí, N+1 en MapReduce. La instrumentación depende de esto.
    Truncation().summarize(long_document, model)
    assert len(model.calls) == 1


def test_truncation_respeta_la_ventana_de_contexto(
    model: FakeModel, long_document: Document
) -> None:
    # El documento tiene 30 tokens y la ventana 10.
    assert model.count_tokens(long_document.text) == 30

    Truncation().summarize(long_document, model)

    entrada = model.calls[0]
    assert model.count_tokens(entrada) <= model.context_window


def test_truncation_descarta_el_final_del_documento(
    model: FakeModel, long_document: Document
) -> None:
    """La pérdida de información que el proyecto quiere cuantificar.

    Conclusions desaparece por completo: es exactamente lo que el
    planteamiento del problema señala como riesgo del truncamiento.
    """
    Truncation().summarize(long_document, model)

    entrada = model.calls[0]
    assert "i0" in entrada
    assert "m0" not in entrada
    assert "c0" not in entrada


def test_truncation_no_recorta_un_documento_que_ya_cabe(
    long_document: Document,
) -> None:
    modelo_amplio = FakeModel(context_window=1000)

    Truncation().summarize(long_document, modelo_amplio)

    assert modelo_amplio.calls[0] == long_document.text


def test_la_ventana_del_modelo_manda_no_la_estrategia(
    long_document: Document,
) -> None:
    """`Truncation` no codifica ningún límite propio.

    Es lo que permite añadir LED (16.384 tokens, ADR-003) sin tocar
    ninguna de las cuatro estrategias.
    """
    for ventana in (5, 10, 25):
        modelo = FakeModel(context_window=ventana)
        Truncation().summarize(long_document, modelo)
        assert modelo.count_tokens(modelo.calls[0]) == ventana


def test_max_new_tokens_acota_la_salida(
    model: FakeModel, long_document: Document
) -> None:
    resumen = Truncation(max_new_tokens=3).summarize(long_document, model)
    assert model.count_tokens(resumen) <= 3
