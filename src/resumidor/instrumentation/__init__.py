"""Instrumentación de costo: latencia y pico de memoria.

Requisito de primera clase, no añadido posterior (ADR-005 §3, objetivo
específico 5). **Envuelve la estrategia completa, no cada invocación al
modelo**: medir el modelo haría que `MapReduce` reportara el costo de 1
invocación de N e invalidaría la comparación entre celdas.

Sobre el pico de memoria: cada backend expone un contador distinto y con
semántica distinta. Aquí se mide lo mejor que permite cada uno y se registra
**con qué método**, para que el informe no compare cifras incomparables. Era
el punto abierto que dejó `docs/arquitectura.md` §5.
"""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Medicion:
    """Resultado mutable que el gestor de contexto rellena al salir."""

    latencia_s: float = 0.0
    pico_memoria_mb: float = 0.0
    metodo_memoria: str = "desconocido"
    """Cómo se obtuvo `pico_memoria_mb`. Las cifras solo son comparables
    entre corridas que compartan método."""
    invocaciones_modelo: int = 0
    _extra: dict = field(default_factory=dict)


def metodo_memoria(dispositivo: str) -> str:
    """Qué contador se usará en este dispositivo.

    Se expone aparte para poder registrarlo junto a cada resultado sin tener
    que ejecutar una medición: las cifras de memoria solo son comparables
    entre corridas que compartan método.
    """
    return {
        "cuda": "cuda.max_memory_allocated",
        "mps": "mps.driver_allocated_memory",
    }.get(dispositivo, "rss.ru_maxrss")


def _rss_maximo_mb() -> float:
    """Marca de agua de memoria residente del proceso.

    `ru_maxrss` viene en bytes en macOS y en kilobytes en Linux.
    """
    bruto = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return bruto / (1024**2) if sys.platform == "darwin" else bruto / 1024


@contextmanager
def medir(dispositivo: str) -> Iterator[Medicion]:
    """Mide latencia y pico de memoria del bloque completo.

    Semántica de `metodo_memoria`:

    - `cuda.max_memory_allocated` — pico real de memoria de tensores durante
      el bloque. Es la única medida exacta y aislada de las tres.
    - `mps.driver_allocated_memory` — memoria asignada por el driver al
      *salir* del bloque, no el pico. Aproximación por defecto: PyTorch no
      expone un contador de pico para MPS.
    - `rss.ru_maxrss` — marca de agua del proceso entero, monótona y
      creciente. Incluye intérprete, pesos del modelo y cualquier otra cosa
      que el proceso haya reservado antes. Sobreestima.

    Solo la primera es comparable documento a documento. Las otras dos sirven
    para dimensionar, no para concluir.
    """
    medicion = Medicion()
    torch = None

    if dispositivo in ("cuda", "mps"):
        import torch as _torch

        torch = _torch

    if dispositivo == "cuda" and torch is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elif dispositivo == "cpu":
        rss_inicial = _rss_maximo_mb()

    inicio = time.perf_counter()
    try:
        yield medicion
    finally:
        if dispositivo == "cuda" and torch is not None:
            torch.cuda.synchronize()
        elif dispositivo == "mps" and torch is not None:
            torch.mps.synchronize()

        medicion.latencia_s = time.perf_counter() - inicio

        if dispositivo == "cuda" and torch is not None:
            medicion.pico_memoria_mb = torch.cuda.max_memory_allocated() / (1024**2)
            medicion.metodo_memoria = "cuda.max_memory_allocated"
        elif dispositivo == "mps" and torch is not None:
            medicion.pico_memoria_mb = torch.mps.driver_allocated_memory() / (1024**2)
            medicion.metodo_memoria = "mps.driver_allocated_memory"
        else:
            medicion.pico_memoria_mb = max(0.0, _rss_maximo_mb() - rss_inicial)
            medicion.metodo_memoria = "rss.ru_maxrss"


__all__ = ["Medicion", "medir", "metodo_memoria"]
