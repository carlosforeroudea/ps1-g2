"""Métricas de calidad: ROUGE y BERTScore.

Son las implementaciones de referencia que usa la literatura citada en el
anteproyecto (Lin 2004; Zhang et al. 2020b). Reimplementar ROUGE es una fuente
clásica de cifras que luego no son comparables con las publicadas.

Vive en `src/` y no dentro del notebook a propósito: la misma función tiene
que poder ejecutarse en el análisis local y en el job de GCP sin duplicarse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# PEGASUS emite `<n>` como separador de oración en lugar de un salto de línea.
_SEPARADOR_PEGASUS = re.compile(r"\s*<n>\s*")
# Corte de oración conservador: punto/interrogación/cierre seguido de espacio.
_FIN_DE_ORACION = re.compile(r"(?<=[.!?])\s+")


def preparar_para_lsum(texto: str) -> str:
    """Normaliza el texto a una oración por línea.

    ROUGE-Lsum calcula la subsecuencia común **oración a oración**, así que
    necesita los saltos de línea. Si se le pasa un bloque sin dividir,
    degenera en ROUGE-L y la cifra deja de ser comparable con la literatura.

    Los tres formatos que aparecen en este proyecto se normalizan aquí:
    los abstracts del corpus ya vienen con saltos, PEGASUS emite `<n>`, y
    BART devuelve un bloque corrido que hay que segmentar.
    """
    texto = _SEPARADOR_PEGASUS.sub("\n", texto.strip())
    if "\n" in texto:
        lineas = texto.split("\n")
    else:
        lineas = _FIN_DE_ORACION.split(texto)
    return "\n".join(linea.strip() for linea in lineas if linea.strip())


@lru_cache(maxsize=1)
def _scorer():
    from rouge_score import rouge_scorer

    # `use_stemmer=True` es la convención de la literatura de resumen.
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


@dataclass(frozen=True)
class CalidadRouge:
    rouge1: float
    rouge2: float
    rougeLsum: float


def rouge(prediccion: str, referencia: str) -> CalidadRouge:
    """ROUGE-1/2/Lsum de un par (generado, referencia), como F-measure."""
    puntajes = _scorer().score(
        preparar_para_lsum(referencia), preparar_para_lsum(prediccion)
    )
    return CalidadRouge(
        rouge1=puntajes["rouge1"].fmeasure,
        rouge2=puntajes["rouge2"].fmeasure,
        rougeLsum=puntajes["rougeLsum"].fmeasure,
    )


def bertscore(
    predicciones: list[str],
    referencias: list[str],
    *,
    modelo: str = "roberta-large",
    dispositivo: str | None = None,
) -> list[float]:
    """BERTScore F1 por par.

    Se calcula **en lote y aparte de ROUGE** porque carga un modelo propio
    (~1,4 GB) y es caro: hacerlo documento a documento multiplicaría el costo
    por 300. Devuelve una lista alineada con la entrada.

    Es la métrica que complementa a ROUGE cuando el resumen es correcto pero
    usa otras palabras — el caso habitual en resumen abstractivo.
    """
    from bert_score import score as bert_score_fn

    from resumidor.device import resolver_dispositivo

    preds = [preparar_para_lsum(p) for p in predicciones]
    refs = [preparar_para_lsum(r) for r in referencias]
    # Un resumen vacío puntúa 0, como en ROUGE. Además hay que sacarlo del
    # lote: bert_score lo codifica con `build_inputs_with_special_tokens`,
    # que transformers 5 ya no expone, y la llamada aborta toda la corrida.
    validos = [i for i, (p, r) in enumerate(zip(preds, refs)) if p.strip() and r.strip()]
    f1_por_par = [0.0] * len(preds)
    if validos:
        _, _, f1 = bert_score_fn(
            [preds[i] for i in validos],
            [refs[i] for i in validos],
            lang="en",
            model_type=modelo,
            device=resolver_dispositivo(dispositivo),
            verbose=False,
        )
        for i, v in zip(validos, f1):
            f1_por_par[i] = float(v)
    return f1_por_par
