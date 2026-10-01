"""Tests de `MapReduce`, `ExtractiveAbstractive`, `LeadK` y del muestreo.

Sin red: todo corre sobre el `FakeModel` de `conftest.py`, que tokeniza por
espacios. Lo que se verifica es el contrato de cada estrategia —aridad frente
al modelo y respeto de la ventana—, que es lo que sostiene la comparación de
costo entre celdas (ADR-005 §2 y §3).
"""

from __future__ import annotations

import collections

import pytest

from resumidor.data.corpus import indices_estratificados
from resumidor.domain import Document, Section
from resumidor.runner import ejecutar_documento
from resumidor.strategies import (
    ESTRATEGIAS,
    ContextStrategy,
    ExtractiveAbstractive,
    LeadK,
    MapReduce,
)
from resumidor.strategies._texto import empaquetar, oraciones
from resumidor.strategies.extractive_abstractive import centralidad_textrank

from .conftest import FakeModel


class FakeModelConDispositivo(FakeModel):
    dispositivo: str = "cpu"


def _articulo(n_oraciones: int, palabras: int = 6) -> Document:
    """Artículo al estilo de la config `section`: una oración por línea."""
    lineas = [
        " ".join(f"s{i}w{j}" for j in range(palabras)) for i in range(n_oraciones)
    ]
    return Document(
        doc_id="fake-002",
        sections=(Section(title="", text="\n".join(lineas)),),
        reference_summary="ref",
    )


# --- registro y puertos -------------------------------------------------


@pytest.mark.parametrize("kind", sorted(ESTRATEGIAS))
def test_todas_satisfacen_el_puerto(kind: str) -> None:
    assert isinstance(ESTRATEGIAS[kind](), ContextStrategy)
    assert ESTRATEGIAS[kind]().name == kind


# --- utilidades de texto ------------------------------------------------


def test_oraciones_respeta_las_lineas_del_corpus() -> None:
    doc = _articulo(4)
    assert len(oraciones(doc)) == 4


def test_oraciones_segmenta_texto_corrido() -> None:
    doc = Document("x", (Section("", "Uno dos. Tres cuatro? Cinco."),))
    assert oraciones(doc) == ["Uno dos.", "Tres cuatro?", "Cinco."]


def test_empaquetar_no_pierde_ni_reordena(model: FakeModel) -> None:
    unidades = [f"a{i} b{i} c{i}" for i in range(7)]  # 3 tokens cada una
    fragmentos = empaquetar(unidades, model, limite=10)
    assert " ".join(fragmentos) == " ".join(unidades)
    assert all(model.count_tokens(f) <= 10 for f in fragmentos)


def test_empaquetar_recorta_una_unidad_gigante(model: FakeModel) -> None:
    fragmentos = empaquetar(["x " * 25, "corta"], model, limite=10)
    assert [model.count_tokens(f) for f in fragmentos] == [10, 1]


# --- LeadK ----------------------------------------------------------------


def test_lead_k_no_invoca_al_modelo(model: FakeModel) -> None:
    resumen = LeadK(k=3).summarize(_articulo(10), model)
    assert model.calls == []
    assert resumen.split("\n") == oraciones(_articulo(10))[:3]


def test_lead_k_registra_cero_invocaciones() -> None:
    r = ejecutar_documento(_articulo(10), FakeModelConDispositivo(), LeadK(k=2))
    assert r.cost.model_calls == 0


# --- MapReduce ------------------------------------------------------------


def test_map_reduce_hace_n_mas_una_invocaciones() -> None:
    # 20 oraciones × 6 tokens = 120 tokens; ventana 40 - margen 4 = 36 → 6 por
    # fragmento → 4 fragmentos (36+36+36+12). Resúmenes de 9 tokens (36 // 4)
    # suman 36: la reducción cabe en un nivel.
    modelo = FakeModelConDispositivo(context_window=40)
    estrategia = MapReduce(max_new_tokens=8, map_min_new_tokens=2, margen=4)
    r = ejecutar_documento(_articulo(20), modelo, estrategia)
    assert r.cost.model_calls == 4 + 1


def test_map_reduce_nunca_excede_la_ventana() -> None:
    modelo = FakeModel(context_window=40)
    MapReduce(max_new_tokens=8, map_min_new_tokens=2, margen=4).summarize(
        _articulo(50), modelo
    )
    assert all(modelo.count_tokens(t) <= 40 for t in modelo.calls)


def test_map_reduce_cubre_todo_el_articulo() -> None:
    """La diferencia con `Truncation`: el final del artículo llega al modelo."""
    modelo = FakeModel(context_window=40)
    MapReduce(max_new_tokens=8, map_min_new_tokens=2, margen=4).summarize(
        _articulo(20), modelo
    )
    vistos = " ".join(modelo.calls)
    assert "s0w0" in vistos
    assert "s19w0" in vistos


def test_map_reduce_solo_la_reduccion_usa_la_politica_comun() -> None:
    modelo = FakeModel(context_window=40)
    MapReduce(max_new_tokens=8, map_min_new_tokens=2, margen=4).summarize(
        _articulo(20), modelo
    )
    *fase_map, reduccion = modelo.min_tokens_pedidos
    assert all(m is not None for m in fase_map)
    assert reduccion is None


def test_map_reduce_con_documento_que_cabe_es_una_invocacion() -> None:
    modelo = FakeModel(context_window=1000)
    MapReduce().summarize(_articulo(5), modelo)
    assert len(modelo.calls) == 1


def test_map_reduce_recursa_si_la_reduccion_no_cabe() -> None:
    # 100 fragmentos: el presupuesto se queda en el mínimo (5) y 100 × 5 no
    # cabe en 36, así que hace falta un segundo nivel de map.
    modelo = FakeModel(context_window=40)
    MapReduce(max_new_tokens=8, map_min_new_tokens=5, margen=4).summarize(
        _articulo(600), modelo
    )
    assert all(modelo.count_tokens(t) <= 40 for t in modelo.calls)
    assert len(modelo.calls) > 100 + 1


def test_presupuesto_map_se_reparte_la_ventana() -> None:
    mr = MapReduce(map_min_new_tokens=48, map_max_new_tokens=160)
    assert mr.presupuesto_map(8, 1016) == 127
    assert mr.presupuesto_map(2, 1016) == 160  # techo
    assert mr.presupuesto_map(40, 1016) == 48  # piso


# --- ExtractiveAbstractive ------------------------------------------------


def test_extractive_abstractive_invoca_una_vez_y_respeta_la_ventana() -> None:
    modelo = FakeModel(context_window=40)
    ExtractiveAbstractive(max_new_tokens=8, margen=4).summarize(_articulo(30), modelo)
    assert len(modelo.calls) == 1
    assert modelo.count_tokens(modelo.calls[0]) <= 40


def test_extractive_abstractive_conserva_el_orden_original() -> None:
    modelo = FakeModel(context_window=40)
    seleccion = ExtractiveAbstractive(margen=4).seleccionar(_articulo(30), modelo)
    todas = oraciones(_articulo(30))
    posiciones = [todas.index(o) for o in seleccion]
    assert posiciones == sorted(posiciones)


def test_extractive_abstractive_prefiere_las_oraciones_centrales() -> None:
    """La oración que comparte vocabulario con el resto gana a la aislada."""
    lineas = [
        "neural summarization models compress long scientific articles",
        "long scientific articles exceed the context window of models",
        "summarization models need the context window of long articles",
        "the weather yesterday was sunny and pleasant outside",
    ]
    doc = Document("x", (Section("", "\n".join(lineas)),))
    modelo = FakeModel(context_window=22)  # cabe una línea de 8 y poco más
    seleccion = ExtractiveAbstractive(margen=4).seleccionar(doc, modelo)
    assert lineas[3] not in seleccion


def test_textrank_ignora_los_marcadores_del_corpus() -> None:
    # Sin el filtro, `@xmath` haría "similares" a todas las oraciones.
    p = centralidad_textrank(["@xmath0 @xmath1 alpha", "@xmath2 beta", "gamma"])
    assert p == pytest.approx([1 / 3] * 3)


# --- muestreo estratificado -----------------------------------------------


def test_estratificado_reparte_igual_y_es_reproducible() -> None:
    longitudes = list(range(1000))
    a = indices_estratificados(longitudes, n=300, estratos=4, seed=7)
    b = indices_estratificados(longitudes, n=300, estratos=4, seed=7)
    assert a == b
    assert len({i for i, _ in a}) == 300
    assert collections.Counter(e for _, e in a) == {
        "Q1": 75,
        "Q2": 75,
        "Q3": 75,
        "Q4": 75,
    }


def test_estratificado_asigna_el_estrato_por_longitud() -> None:
    longitudes = list(range(1000))
    for indice, estrato in indices_estratificados(longitudes, n=40, estratos=4, seed=1):
        assert estrato == f"Q{min(indice // 250, 3) + 1}"


def test_un_prefijo_de_la_muestra_cubre_todos_los_estratos() -> None:
    # Una corrida interrumpida o con --limite no debe quedar solo en Q1.
    muestra = indices_estratificados(list(range(1000)), n=300, estratos=4, seed=3)
    assert {e for _, e in muestra[:30]} == {"Q1", "Q2", "Q3", "Q4"}
