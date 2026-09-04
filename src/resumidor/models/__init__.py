"""Adaptadores de modelos preentrenados.

Contexto corto (objeto de estudio): BART, PEGASUS.
Contexto largo (baseline superior, ADR-003): LED, LongT5.
Los adaptadores concretos son Fase 3.
"""

from resumidor.models.base import SummarizerModel

__all__ = ["SummarizerModel"]
