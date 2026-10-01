"""Extractivo-abstractivo: selecciona las oraciones centrales y genera una vez.

Primera etapa **sin modelo generativo**: TextRank sobre similitud TF-IDF entre
oraciones (Mihalcea y Tarau 2004; es también la idea de LexRank, Erkan y Radev
2004). Se eligen oraciones de mayor centralidad hasta llenar la ventana del
modelo y se devuelven **en su orden original**, para que el modelo abstractivo
reciba un texto con la secuencia argumental del artículo.

Segunda etapa: una sola invocación al modelo. El costo queda muy cerca del de
`Truncation` (1 invocación, entrada del mismo tamaño) más el de la selección,
que es aritmética sobre una matriz de ~220×220. Si esta estrategia recupera
parte de la calidad de `MapReduce` a costo de `Truncation`, ese es el hallazgo.

Implementado con numpy y sin dependencias nuevas: la selección tiene que ser
exactamente la misma en el portátil, en el job de GCP y en la plataforma web.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel
from resumidor.strategies._texto import ajustar_a_ventana, oraciones

_PALABRA = re.compile(r"[a-z]{3,}")

# Marcadores del preprocesado del corpus (`@xmath12` para fórmulas, `@xcite`
# para citas). Aparecen en casi todas las oraciones y harían que la
# centralidad premiara oraciones con muchas fórmulas.
_RUIDO_CORPUS = frozenset({"xmath", "xcite"})

_STOPWORDS = frozenset(
    """
    the and for are but not you all any can had her was one our out has him his
    how its may new now old see two way who did get let put say she too use
    that with have this will your from they know want been good much some time
    very when come here just like long make many more only over such take than
    them well were what also into most other their there these those which
    would could should about after again being below between both during each
    few further itself nor off once own same then through under until while
    where whom why does doing having here's we our ours ourselves thus hence
    therefore however where fig figs eq eqs section table ref refs using used
    """.split()
)


def _terminos(oracion: str) -> list[str]:
    return [
        t
        for t in _PALABRA.findall(oracion.lower())
        if t not in _STOPWORDS and t not in _RUIDO_CORPUS
    ]


def centralidad_textrank(
    textos: list[str], *, amortiguacion: float = 0.85, iteraciones: int = 100
) -> np.ndarray:
    """Puntaje TextRank de cada texto, sobre similitud coseno TF-IDF."""
    n = len(textos)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    conteos = [Counter(_terminos(t)) for t in textos]
    vocabulario = sorted({t for c in conteos for t in c})
    if not vocabulario:
        return np.full(n, 1.0 / n)
    indice = {t: i for i, t in enumerate(vocabulario)}

    tf = np.zeros((n, len(vocabulario)))
    for fila, conteo in enumerate(conteos):
        for termino, veces in conteo.items():
            tf[fila, indice[termino]] = 1.0 + np.log(veces)

    df = (tf > 0).sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    x = tf * idf
    normas = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, normas, out=np.zeros_like(x), where=normas > 0)

    similitud = x @ x.T
    np.fill_diagonal(similitud, 0.0)

    # Matriz de transición por filas. Una oración sin similitud con ninguna
    # otra reparte su peso uniformemente (nodo colgante de PageRank).
    sumas = similitud.sum(axis=1, keepdims=True)
    transicion = np.where(sumas > 0, similitud / np.where(sumas > 0, sumas, 1), 1 / n)

    puntaje = np.full(n, 1.0 / n)
    for _ in range(iteraciones):
        nuevo = (1 - amortiguacion) / n + amortiguacion * (transicion.T @ puntaje)
        if np.abs(nuevo - puntaje).sum() < 1e-8:
            puntaje = nuevo
            break
        puntaje = nuevo
    return puntaje


@dataclass(frozen=True)
class ExtractiveAbstractive:
    """Selecciona por TextRank hasta llenar la ventana y genera una vez."""

    max_new_tokens: int = 256
    min_palabras: int = 5
    """Oraciones más cortas no compiten: en este corpus son sobre todo restos
    de fórmulas y encabezados aplanados."""
    margen: int = 8

    @property
    def name(self) -> str:
        return "extractive_abstractive"

    def seleccionar(self, document: Document, model: SummarizerModel) -> list[str]:
        """Oraciones elegidas, en orden de aparición en el artículo."""
        todas = oraciones(document)
        candidatas = [
            (i, o) for i, o in enumerate(todas) if len(o.split()) >= self.min_palabras
        ] or list(enumerate(todas))
        if not candidatas:
            return []

        puntajes = centralidad_textrank([o for _, o in candidatas])
        # Orden estable: a igual puntaje gana la oración que aparece antes.
        orden = sorted(range(len(candidatas)), key=lambda j: (-puntajes[j], j))

        limite = model.context_window - self.margen
        usados = 0
        elegidas: list[int] = []
        for j in orden:
            n = model.count_tokens(candidatas[j][1])
            # Se sigue buscando aunque una no quepa: una oración más corta y
            # algo menos central puede aprovechar el hueco que queda.
            if usados + n <= limite:
                elegidas.append(j)
                usados += n

        return [
            candidatas[j][1] for j in sorted(elegidas, key=lambda j: candidatas[j][0])
        ]

    def summarize(self, document: Document, model: SummarizerModel) -> str:
        seleccion = self.seleccionar(document, model)
        texto = " ".join(seleccion) if seleccion else document.text
        return model.generate(
            ajustar_a_ventana(texto, model), max_new_tokens=self.max_new_tokens
        )
