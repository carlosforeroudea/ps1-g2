"""Estrategias de manejo de contexto largo.

`Truncation` está implementada como baseline del factorial.
`MapReduce`, `ExtractiveAbstractive` y `LeadK` son Fase 3.
"""

from resumidor.strategies.base import ContextStrategy
from resumidor.strategies.truncation import Truncation

__all__ = ["ContextStrategy", "Truncation"]
