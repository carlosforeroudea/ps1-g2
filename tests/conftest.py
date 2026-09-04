"""Dobles de prueba.

Que un modelo falso de veinte líneas satisfaga `SummarizerModel` y
funcione con `Truncation` sin cambios es la prueba de que los puertos
del ADR-005 están bien cortados. Si el contrato exigiera algo propio de
Hugging Face, esto no sería posible y todos los tests dependerían de
descargar pesos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from resumidor.domain import Document, Section


@dataclass
class FakeModel:
    """Modelo determinista que tokeniza por espacios.

    Registra sus invocaciones para que los tests puedan verificar la
    aridad de cada estrategia (`model_calls` en `CostMetrics`).
    """

    context_window: int = 10
    name: str = "fake/model"
    calls: list[str] = field(default_factory=list)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def generate(self, text: str, max_new_tokens: int) -> str:
        self.calls.append(text)
        return " ".join(text.split()[:max_new_tokens]).upper()


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def long_document() -> Document:
    """Documento de 30 tokens: 3x la ventana del `FakeModel`.

    Reproduce a escala la brecha real del corpus (~5.905 tokens frente
    a 1.024, ADR-002).
    """
    return Document(
        doc_id="fake-001",
        sections=(
            Section(title="Introduction", text=" ".join(f"i{n}" for n in range(10))),
            Section(title="Methods", text=" ".join(f"m{n}" for n in range(10))),
            Section(title="Conclusions", text=" ".join(f"c{n}" for n in range(10))),
        ),
        reference_summary="resumen de referencia",
    )
