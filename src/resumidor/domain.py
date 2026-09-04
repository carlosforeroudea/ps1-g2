"""Tipos del dominio.

Este módulo no importa nada del resto del proyecto: es la base de la
regla de dependencias de `docs/arquitectura.md` §7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """Una sección del artículo.

    La config `section` del corpus (ADR-002) preserva estos límites.
    Son la unidad de fragmentación semántica de `MapReduce`: cortar por
    sección conserva unidades de sentido, cortar cada N tokens no.
    """

    title: str
    text: str


@dataclass(frozen=True)
class Document:
    """Un artículo científico con su estructura preservada.

    Es el punto donde convergen las dos rutas de ingesta: el corpus
    (experimentos) y la carga de PDF (plataforma web, Fase 4).
    """

    doc_id: str
    sections: tuple[Section, ...]
    reference_summary: str | None = None
    """El abstract, cuando el documento viene del corpus. La plataforma
    web no tiene referencia: por eso es opcional."""

    @property
    def text(self) -> str:
        """El artículo completo como texto plano."""
        return "\n\n".join(section.text for section in self.sections)


@dataclass(frozen=True)
class CostMetrics:
    """Medidas de costo computacional de una ejecución.

    Objetivo específico 5. Se producen siempre, junto con el resumen y
    no por separado (RT-2).
    """

    latency_seconds: float
    peak_memory_mb: float
    model_calls: int
    """Invocaciones al modelo generativo. Se registra explícitamente
    porque es lo que hace visible en los datos por qué map-reduce cuesta
    lo que cuesta: 0 para `LeadK`, 1 para `Truncation`, N+1 para
    `MapReduce` (ADR-005 §2)."""
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SummaryResult:
    """Un resumen junto con el costo de haberlo producido.

    No existe una ruta en el sistema que devuelva lo uno sin lo otro.
    """

    summary: str
    cost: CostMetrics
    model_name: str
    strategy_name: str

    @property
    def compression_ratio(self) -> float:
        """Tokens de salida sobre tokens de entrada.

        Devuelve 0.0 ante una entrada vacía en lugar de propagar una
        división por cero: un documento vacío es un caso degenerado del
        corpus, no un fallo del sistema, y no debe abortar una corrida
        de 300 documentos.
        """
        if self.cost.input_tokens == 0:
            return 0.0
        return self.cost.output_tokens / self.cost.input_tokens
