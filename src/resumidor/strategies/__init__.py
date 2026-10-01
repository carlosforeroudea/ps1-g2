"""Estrategias de manejo de contexto largo.

`ESTRATEGIAS` es el registro único: la validación de configuraciones (RT-6) y
el runner leen de aquí, así que una estrategia existe para el experimento en
cuanto aparece en este diccionario.
"""

from resumidor.strategies.base import ContextStrategy
from resumidor.strategies.extractive_abstractive import ExtractiveAbstractive
from resumidor.strategies.lead_k import LeadK
from resumidor.strategies.map_reduce import MapReduce
from resumidor.strategies.truncation import Truncation

ESTRATEGIAS: dict[str, type] = {
    "truncation": Truncation,
    "map_reduce": MapReduce,
    "extractive_abstractive": ExtractiveAbstractive,
    "lead_k": LeadK,
}

__all__ = [
    "ESTRATEGIAS",
    "ContextStrategy",
    "ExtractiveAbstractive",
    "LeadK",
    "MapReduce",
    "Truncation",
]
