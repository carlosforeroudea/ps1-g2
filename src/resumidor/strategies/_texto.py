"""Utilidades de texto compartidas por las estrategias.

La unidad de trabajo es la **oración**, no el token ni la sección. La config
`section` del corpus entrega el artículo con una oración por línea (ADR-002,
corrección de la semana 6); no trae títulos de sección. Cortar por oraciones es
lo que el corpus sí ofrece, y evita que `MapReduce` parta una frase entre dos
fragmentos o que `ExtractiveAbstractive` seleccione media oración.
"""

from __future__ import annotations

import re

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel

# Solo para textos sin saltos de línea (p. ej. un PDF de la plataforma web).
_FIN_DE_ORACION = re.compile(r"(?<=[.!?])\s+")


def oraciones(document: Document) -> list[str]:
    """Oraciones del documento, en orden, sin líneas vacías."""
    resultado: list[str] = []
    for section in document.sections:
        texto = section.text.strip()
        if not texto:
            continue
        partes = texto.split("\n") if "\n" in texto else _FIN_DE_ORACION.split(texto)
        resultado.extend(p.strip() for p in partes if p.strip())
    return resultado


def empaquetar(unidades: list[str], model: SummarizerModel, limite: int) -> list[str]:
    """Agrupa unidades consecutivas en fragmentos de a lo sumo `limite` tokens.

    Empaquetado voraz y en orden: no reordena ni descarta nada. Una unidad que
    por sí sola excede el límite se recorta y ocupa su propio fragmento; en
    este corpus ocurre con "oraciones" que son en realidad tablas o fórmulas
    aplanadas.

    Cuenta los tokens de cada unidad por separado, lo que incluye los tokens
    especiales de inicio y fin en cada una: sobreestima un poco el tamaño del
    fragmento, y es a propósito. El error va del lado seguro.
    """
    fragmentos: list[str] = []
    actual: list[str] = []
    tokens_actual = 0

    for unidad in unidades:
        n = model.count_tokens(unidad)
        if n > limite:
            if actual:
                fragmentos.append(" ".join(actual))
                actual, tokens_actual = [], 0
            fragmentos.append(model.truncate(unidad, limite))
            continue
        if actual and tokens_actual + n > limite:
            fragmentos.append(" ".join(actual))
            actual, tokens_actual = [], 0
        actual.append(unidad)
        tokens_actual += n

    if actual:
        fragmentos.append(" ".join(actual))
    return fragmentos


def ajustar_a_ventana(text: str, model: SummarizerModel) -> str:
    """Recorta solo si hace falta (`truncate` no es idempotente)."""
    if model.count_tokens(text) > model.context_window:
        return model.truncate(text, model.context_window)
    return text
