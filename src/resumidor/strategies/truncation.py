"""Estrategia de truncamiento: el baseline del factorial.

Es la estrategia más barata y la que más información pierde. Con
artículos de ~5.905 tokens (ADR-002) frente a una ventana de 1.024,
descarta del orden del 80 % del documento —incluidas, típicamente,
las secciones de resultados y conclusiones, que son las que determinan
la pertinencia del artículo—.

Esa pérdida es justamente lo que el proyecto quiere cuantificar: es el
punto de referencia contra el cual se mide si map-reduce y
extractivo-abstractivo justifican su costo adicional.
"""

from __future__ import annotations

from dataclasses import dataclass

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel


@dataclass(frozen=True)
class Truncation:
    """Recorta el documento a la ventana del modelo y genera una vez."""

    max_new_tokens: int = 256

    @property
    def name(self) -> str:
        return "truncation"

    def summarize(self, document: Document, model: SummarizerModel) -> str:
        text = document.text

        # Solo se recorta si hace falta. `truncate` implica un ciclo
        # encode/decode del tokenizer, que no reproduce el texto original:
        # normaliza espacios y puede alterar caracteres. Aplicarlo a un
        # documento que ya cabe degradaría la entrada sin ninguna ganancia
        # —relevante para artículos cortos en la plataforma web—.
        if model.count_tokens(text) > model.context_window:
            text = model.truncate(text, model.context_window)

        return model.generate(text, max_new_tokens=self.max_new_tokens)
