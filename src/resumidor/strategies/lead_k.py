"""Piso trivial: las primeras k oraciones del artículo (ADR-003).

No invoca ningún modelo generativo. Su papel es interpretativo: si una celda
del factorial no supera a `LeadK`, su ROUGE no dice nada sobre la estrategia;
y si todas las celdas quedan bajas, `LeadK` permite distinguir "el problema es
difícil" de "las estrategias no funcionan".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel
from resumidor.strategies._texto import oraciones


@dataclass(frozen=True)
class LeadK:
    """Devuelve las primeras `k` oraciones, una por línea."""

    k: int = 6

    usa_modelo_generativo: ClassVar[bool] = False

    @property
    def name(self) -> str:
        return "lead_k"

    def summarize(self, document: Document, model: SummarizerModel) -> str:
        # Una oración por línea: es el formato que ROUGE-Lsum necesita y el
        # que `preparar_para_lsum` respeta sin volver a segmentar.
        return "\n".join(oraciones(document)[: self.k])
