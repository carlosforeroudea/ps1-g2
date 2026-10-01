"""Evaluación de calidad y costo (objetivo específico 5)."""

from resumidor.eval.aggregate import (
    calificar,
    cargar_resultados,
    resumir_por_configuracion,
)
from resumidor.eval.estadistica import (
    comparar_pareado,
    pruebas_pareadas,
    recuperacion_frente_a,
)
from resumidor.eval.metrics import bertscore, preparar_para_lsum, rouge

__all__ = [
    "bertscore",
    "calificar",
    "cargar_resultados",
    "comparar_pareado",
    "preparar_para_lsum",
    "pruebas_pareadas",
    "recuperacion_frente_a",
    "resumir_por_configuracion",
    "rouge",
]
