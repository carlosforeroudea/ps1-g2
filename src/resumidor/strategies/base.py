"""Puerto `ContextStrategy`.

El otro eje de extensión (ADR-005 §1) y la variable independiente
principal del experimento: la pregunta de investigación del ADR-000 es,
literalmente, cuál de estas implementaciones gana.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel


@runtime_checkable
class ContextStrategy(Protocol):
    """Cómo se adapta un documento largo a la ventana de un modelo.

    **La estrategia orquesta; el modelo solo genera** (ADR-005 §2).

    Recibe el modelo por inyección y es responsable de la generación
    completa, incluido el número de invocaciones —que difiere entre
    implementaciones: 0 en `LeadK`, 1 en `Truncation`, N+1 en
    `MapReduce`—. Esa asimetría es la razón de que el contrato sea
    `Document -> str` y no `Document -> str` seguido de una llamada
    externa al modelo: map-reduce no se puede expresar así sin un caso
    especial que terminaría replicándose por todo el sistema.

    Es también la razón de que la instrumentación envuelva la estrategia
    y no el modelo (ADR-005 §3): medir el modelo haría que `MapReduce`
    reportara el costo de 1 invocación de N, invalidando exactamente la
    comparación que el proyecto quiere hacer.
    """

    @property
    def name(self) -> str:
        """Identificador de la estrategia, para trazabilidad."""
        ...

    def summarize(self, document: Document, model: SummarizerModel) -> str:
        """Produce un resumen de `document` usando `model`.

        Libre de invocar `model.generate` tantas veces como necesite.
        Debe garantizar que cada texto que le pasa al modelo cabe en
        `model.context_window`.
        """
        ...
