"""Puerto `SummarizerModel`.

Uno de los dos ejes de extensión del sistema (ADR-005 §1) y una de las
dos variables independientes del experimento.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SummarizerModel(Protocol):
    """Un modelo preentrenado de resumen abstractivo.

    La razón de ser de este protocolo es `context_window`: es lo que
    permite que las estrategias sean agnósticas del modelo. `Truncation`
    no sabe si recorta a 1.024 tokens (BART, PEGASUS) o a 16.384 (LED);
    pregunta. Sin esto, añadir el baseline del ADR-003 obligaría a tocar
    las cuatro estrategias.

    El modelo no conoce la estrategia. La dependencia va en un solo
    sentido.
    """

    @property
    def name(self) -> str:
        """Identificador del checkpoint, p. ej. `facebook/bart-large-cnn`.

        Va al resultado para trazabilidad: sin esto no se puede auditar
        qué produjo cada fila del experimento.
        """
        ...

    @property
    def context_window(self) -> int:
        """Máximo de tokens de entrada que el modelo acepta."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens según el tokenizer de *este* modelo.

        Es específico del modelo a propósito: contar por espacios da
        ~5.905 tokens por artículo (ADR-002), pero el conteo real en
        subpalabras es mayor. Solo el tokenizer del modelo da la cifra
        con la que se compara contra `context_window`.
        """
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Recorta `text` a lo sumo `max_tokens` tokens.

        Vive aquí y no en las estrategias porque solo el tokenizer sabe
        dónde caen los límites de token. `MapReduce` y
        `ExtractiveAbstractive` también lo necesitan, no solo
        `Truncation`.
        """
        ...

    def generate(self, text: str, max_new_tokens: int) -> str:
        """Genera un resumen de `text`.

        `text` DEBE caber en `context_window`; garantizarlo es
        responsabilidad de la estrategia que invoca. El modelo no
        recorta por su cuenta: hacerlo silenciosamente ocultaría
        pérdida de información justo en el fenómeno bajo estudio.
        """
        ...
