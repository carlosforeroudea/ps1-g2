"""Runner de experimentos: ejecuta una configuración y registra el resultado.

Recibe toda la configuración desde el YAML (ADR-005 §4): añadir una celda del
factorial es un archivo nuevo, no código nuevo.

**Escribe una fila por documento, en cuanto termina cada uno.** No acumula en
memoria para volcar al final. Dos razones, ambas del diseño adoptado:

1. En GPU *spot*, una expropiación cuesta un documento, no la corrida entera.
2. Los tests pareados del ADR-000 necesitan el resultado por documento, no
   solo el agregado.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from datetime import UTC, datetime

from resumidor.config import ExperimentoConfig
from resumidor.data.corpus import cargar_muestra
from resumidor.domain import CostMetrics, Document, SummaryResult
from resumidor.instrumentation import medir, metodo_memoria
from resumidor.models.counting import ContadorDeInvocaciones
from resumidor.models.huggingface import HFSummarizer
from resumidor.strategies.base import ContextStrategy
from resumidor.strategies.truncation import Truncation

_ESTRATEGIAS = {"truncation": Truncation}


def construir_estrategia(config: ExperimentoConfig) -> ContextStrategy:
    parametros = config.strategy.model_dump(exclude={"kind"})
    clase = _ESTRATEGIAS[config.strategy.kind]
    admitidos = {c for c in clase.__dataclass_fields__}
    return clase(**{k: v for k, v in parametros.items() if k in admitidos})


def construir_modelo(config: ExperimentoConfig) -> HFSummarizer:
    return HFSummarizer(
        config.model.checkpoint,
        dispositivo=config.model.device,
        num_beams=config.model.num_beams,
        # `exclude_none`: solo se envían los parámetros declarados, para no
        # imponer valores por defecto nuestros sobre los del checkpoint
        # cuando la configuración no dice nada.
        generacion=config.model.generation.model_dump(exclude_none=True),
    )


def ejecutar_documento(
    documento: Document,
    modelo: HFSummarizer,
    estrategia: ContextStrategy,
) -> SummaryResult:
    """Resume un documento midiendo el costo de la estrategia completa.

    La instrumentación envuelve `estrategia.summarize`, no `modelo.generate`
    (ADR-005 §3): medir el modelo haría que map-reduce reportara el costo de
    una invocación de N.
    """
    tokens_entrada = modelo.count_tokens(documento.text)
    contador = ContadorDeInvocaciones(modelo)

    with medir(modelo.dispositivo) as medicion:
        resumen = estrategia.summarize(documento, contador)

    return SummaryResult(
        summary=resumen,
        cost=CostMetrics(
            latency_seconds=medicion.latencia_s,
            peak_memory_mb=medicion.pico_memoria_mb,
            model_calls=contador.invocaciones,
            input_tokens=tokens_entrada,
            output_tokens=modelo.count_tokens(resumen),
        ),
        model_name=modelo.name,
        strategy_name=estrategia.name,
    )


def _fila(
    config: ExperimentoConfig,
    documento: Document,
    resultado: SummaryResult,
    dispositivo: str,
    metodo: str,
) -> dict:
    """Una fila de resultado, autocontenida y auditable.

    Lleva la configuración que la produjo para que el archivo se pueda leer
    sin el YAML al lado.
    """
    return {
        "config_id": config.id,
        "doc_id": documento.doc_id,
        "modelo": resultado.model_name,
        "estrategia": resultado.strategy_name,
        # El dispositivo RESUELTO, no el configurado: con `device: auto` la
        # fila diría "auto" y la latencia no sería comparable entre corridas.
        "dispositivo": dispositivo,
        "resumen_generado": resultado.summary,
        "resumen_referencia": documento.reference_summary,
        "latencia_s": round(resultado.cost.latency_seconds, 3),
        "pico_memoria_mb": round(resultado.cost.peak_memory_mb, 1),
        "metodo_memoria": metodo,
        "invocaciones_modelo": resultado.cost.model_calls,
        "tokens_entrada": resultado.cost.input_tokens,
        "tokens_salida": resultado.cost.output_tokens,
        "ratio_compresion": round(resultado.compression_ratio, 5),
        "semilla": config.sample.seed,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def ejecutar(
    config: ExperimentoConfig,
    salida: pathlib.Path,
    *,
    limite: int | None = None,
    streaming: bool = False,
) -> Iterator[dict]:
    """Ejecuta la configuración y va escribiendo `salida` documento a documento.

    `limite` acota la muestra sin tocar el YAML: sirve para la prueba local
    pequeña sin inventar una configuración paralela que luego diverja.
    """
    n = min(limite, config.sample.n) if limite else config.sample.n

    modelo = construir_modelo(config)
    estrategia = construir_estrategia(config)

    # Antes de medir nada: descarga, carga y calentamiento del grafo. Sin
    # el calentamiento, el primer documento absorbe el costo de compilación
    # del dispositivo (448,8 s frente a 28,3 s medidos en la semana 6).
    modelo.precargar()

    documentos = cargar_muestra(
        dataset=config.sample.dataset,
        config=config.sample.config,
        split=config.sample.split,
        n=n,
        seed=config.sample.seed,
        streaming=streaming,
    )

    medida_memoria = metodo_memoria(modelo.dispositivo)

    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        for documento in documentos:
            resultado = ejecutar_documento(documento, modelo, estrategia)
            fila = _fila(
                config, documento, resultado, modelo.dispositivo, medida_memoria
            )
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()  # el punto de la escritura incremental
            yield fila
