"""Resumidor inteligente de artículos científicos — Grupo 2.

Arquitectura: `docs/arquitectura.md`. Decisiones: `docs/adr/`.
"""

from resumidor.domain import CostMetrics, Document, Section, SummaryResult
from resumidor.models.base import SummarizerModel
from resumidor.strategies.base import ContextStrategy
from resumidor.strategies.truncation import Truncation

__all__ = [
    "ContextStrategy",
    "CostMetrics",
    "Document",
    "Section",
    "SummarizerModel",
    "SummaryResult",
    "Truncation",
]
