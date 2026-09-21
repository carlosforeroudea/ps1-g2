"""Envoltorio que cuenta las invocaciones al modelo.

`CostMetrics.model_calls` es lo que hace visible en los datos por qué
map-reduce cuesta lo que cuesta: 1 invocación en `Truncation`, N+1 en
`MapReduce` (ADR-005 §2). Contarlo aquí, y no pidiéndoselo a la estrategia,
tiene una ventaja concreta: el número es **observado**, no declarado. Una
estrategia que invoque el modelo más veces de las que cree no puede
subreportarlo.
"""

from __future__ import annotations

from resumidor.models.base import SummarizerModel


class ContadorDeInvocaciones:
    """Delega todo en el modelo envuelto y cuenta las llamadas a `generate`."""

    def __init__(self, modelo: SummarizerModel) -> None:
        self._modelo = modelo
        self.invocaciones = 0

    def reiniciar(self) -> None:
        self.invocaciones = 0

    @property
    def name(self) -> str:
        return self._modelo.name

    @property
    def context_window(self) -> int:
        return self._modelo.context_window

    def count_tokens(self, text: str) -> int:
        return self._modelo.count_tokens(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self._modelo.truncate(text, max_tokens)

    def generate(self, text: str, max_new_tokens: int) -> str:
        self.invocaciones += 1
        return self._modelo.generate(text, max_new_tokens=max_new_tokens)
