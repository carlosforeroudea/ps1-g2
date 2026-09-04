from __future__ import annotations

from resumidor.domain import CostMetrics, Document, Section, SummaryResult


def _result(input_tokens: int, output_tokens: int) -> SummaryResult:
    return SummaryResult(
        summary="resumen",
        cost=CostMetrics(
            latency_seconds=1.0,
            peak_memory_mb=100.0,
            model_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        model_name="fake/model",
        strategy_name="truncation",
    )


def test_document_text_une_las_secciones(long_document: Document) -> None:
    text = long_document.text
    assert text.startswith("i0")
    assert "m0" in text
    assert text.endswith("c9")


def test_document_sin_secciones_da_texto_vacio() -> None:
    assert Document(doc_id="vacio", sections=()).text == ""


def test_ratio_de_compresion() -> None:
    assert _result(input_tokens=1000, output_tokens=250).compression_ratio == 0.25


def test_ratio_de_compresion_con_entrada_vacia_no_divide_por_cero() -> None:
    # Un documento degenerado no debe abortar una corrida de 300.
    assert _result(input_tokens=0, output_tokens=0).compression_ratio == 0.0


def test_los_tipos_del_dominio_son_inmutables() -> None:
    # RT-1: un resultado que se puede mutar tras medirlo no es auditable.
    section = Section(title="Introduction", text="texto")
    try:
        section.text = "otro"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("Section debería ser inmutable")
