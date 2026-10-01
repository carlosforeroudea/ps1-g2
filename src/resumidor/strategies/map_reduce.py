"""Map-reduce: resume cada fragmento y luego resume los resúmenes.

Es la estrategia que cubre el artículo completo con un modelo de contexto
corto, a cambio de N+1 invocaciones. Con ~7.800 tokens por artículo y una
ventana de 1.024, N ronda 8: el costo esperado es un orden de magnitud mayor
que `Truncation`, y justo esa es la cifra que el experimento tiene que poner
frente a la ganancia de calidad.

Decisiones que afectan a la comparación y deben constar en el informe:

- **Fragmentación por oraciones**, no por tokens fijos: ningún fragmento
  empieza o termina a mitad de frase (ADR-002).
- **Presupuesto de la fase map adaptativo**: cada fragmento recibe
  `ventana // N` tokens de salida, acotados entre `map_min_new_tokens` y
  `map_max_new_tokens`. Así la concatenación de los N resúmenes cabe en la
  ventana en una sola reducción para la inmensa mayoría de artículos, en vez
  de truncar la reducción —que reintroduciría la pérdida que la estrategia
  existe para evitar—.
- **Reducción recursiva** cuando aun así no cabe (artículos de la cola larga):
  se vuelve a aplicar el map sobre los resúmenes, hasta `max_niveles`. Cada
  nivel extra suma invocaciones, y quedan contadas en `model_calls`.
- La política de generación común (`min_new_tokens=150`) solo rige la
  **reducción final**, que es el resumen que se evalúa. En la fase map se
  sobrescribe: 150 tokens por fragmento × 8 fragmentos no cabe en 1.024.
"""

from __future__ import annotations

from dataclasses import dataclass

from resumidor.domain import Document
from resumidor.models.base import SummarizerModel
from resumidor.strategies._texto import ajustar_a_ventana, empaquetar, oraciones


@dataclass(frozen=True)
class MapReduce:
    """Fragmenta por oraciones, resume cada fragmento y reduce."""

    max_new_tokens: int = 256
    """Salida de la reducción final: el resumen que se evalúa."""
    map_max_new_tokens: int = 160
    map_min_new_tokens: int = 48
    margen: int = 8
    """Tokens de la ventana reservados a los especiales de inicio y fin."""
    max_niveles: int = 3

    @property
    def name(self) -> str:
        return "map_reduce"

    def presupuesto_map(self, n_fragmentos: int, limite: int) -> int:
        """Tokens de salida por fragmento para que la reducción quepa."""
        reparto = limite // max(n_fragmentos, 1)
        return max(self.map_min_new_tokens, min(self.map_max_new_tokens, reparto))

    def summarize(self, document: Document, model: SummarizerModel) -> str:
        limite = model.context_window - self.margen
        unidades = oraciones(document)

        for _ in range(self.max_niveles):
            fragmentos = empaquetar(unidades, model, limite)
            if len(fragmentos) <= 1:
                break
            presupuesto = self.presupuesto_map(len(fragmentos), limite)
            # Mínimo a la mitad del presupuesto: sin él, BART cierra algunos
            # fragmentos en una sola frase y la reducción pierde contenido.
            unidades = [
                model.generate(
                    fragmento,
                    max_new_tokens=presupuesto,
                    min_new_tokens=presupuesto // 2,
                )
                for fragmento in fragmentos
            ]

        # Si tras `max_niveles` todavía no cabe, se recorta la reducción. Es
        # el último recurso y solo alcanza a artículos extremos de la cola.
        texto = ajustar_a_ventana(" ".join(unidades), model)
        return model.generate(texto, max_new_tokens=self.max_new_tokens)
