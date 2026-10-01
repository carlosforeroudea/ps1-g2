"""Resumidor inteligente de artículos científicos — Grupo 2.

Arquitectura: `docs/arquitectura.md`. Decisiones: `docs/adr/`.
"""

from resumidor.domain import CostMetrics, Document, Section, SummaryResult
from resumidor.models.base import SummarizerModel
from resumidor.strategies import (
    ContextStrategy,
    ExtractiveAbstractive,
    LeadK,
    MapReduce,
    Truncation,
)

__all__ = [
    "ContextStrategy",
    "CostMetrics",
    "Document",
    "ExtractiveAbstractive",
    "LeadK",
    "MapReduce",
    "Section",
    "SummarizerModel",
    "SummaryResult",
    "Truncation",
]
