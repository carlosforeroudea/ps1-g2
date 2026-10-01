"""Esquema y validación de las configuraciones de experimento.

RT-6 (`docs/arquitectura.md`): una configuración inválida debe fallar **antes**
de la corrida, no durante. Un checkpoint mal escrito tiene que reventar en
segundos, no doce horas después.

Cada celda del factorial es un YAML en `experiments/configs/` (ADR-005 §4).
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from resumidor.strategies import ESTRATEGIAS

# Aparecer en el registro de `resumidor.strategies` es lo que habilita una
# estrategia en el experimento.
ESTRATEGIAS_IMPLEMENTADAS = frozenset(ESTRATEGIAS)


class GeneracionConfig(BaseModel):
    """Política de generación común a todas las celdas.

    Sobrescribe la configuración propia de cada checkpoint. Es un **control
    experimental**: sin ella, cada modelo escribe con la longitud que aprendió
    de su dominio de afinado y el ROUGE mide eso en lugar de la capacidad de
    seleccionar contenido (ver `HFSummarizer.generate`).
    """

    model_config = ConfigDict(extra="forbid")

    min_new_tokens: int | None = Field(default=None, ge=0, le=1024)
    length_penalty: float | None = Field(default=None, ge=0.0, le=5.0)
    no_repeat_ngram_size: int | None = Field(default=None, ge=0, le=10)
    early_stopping: bool | None = None


class ModeloConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    checkpoint: str
    device: str = "auto"
    num_beams: int = Field(default=4, ge=1, le=16)
    ventana: int | None = Field(default=None, ge=16)
    """Ventana de contexto explícita. Solo para modelos que no la declaran en
    su config (LongT5, de atención relativa); en el resto se lee del modelo y
    declararla aquí sería una fuente de desincronización."""
    generation: GeneracionConfig = Field(default_factory=GeneracionConfig)


class EstrategiaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # map-reduce añade parámetros propios

    kind: str
    max_new_tokens: int = Field(default=256, ge=16, le=1024)

    @field_validator("kind")
    @classmethod
    def _implementada(cls, v: str) -> str:
        if v not in ESTRATEGIAS_IMPLEMENTADAS:
            raise ValueError(
                f"Estrategia {v!r} todavía no implementada. "
                f"Disponibles: {sorted(ESTRATEGIAS_IMPLEMENTADAS)}."
            )
        return v

    @model_validator(mode="after")
    def _parametros_conocidos(self) -> EstrategiaConfig:
        """Un parámetro mal escrito (`map_max_new_token`) no debe ignorarse.

        Sin esto, la celda correría con el valor por defecto y nadie lo
        notaría hasta comparar resultados.
        """
        clase = ESTRATEGIAS[self.kind]
        admitidos = {f.name for f in dataclasses.fields(clase)} | {"max_new_tokens"}
        desconocidos = set(self.model_extra or {}) - admitidos
        if desconocidos:
            raise ValueError(
                f"Parámetros desconocidos para {self.kind!r}: {sorted(desconocidos)}. "
                f"Admitidos: {sorted(admitidos)}."
            )
        return self


class MuestraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    config: str | None = None
    split: str = "test"
    n: int = Field(ge=1)
    seed: int
    muestreo: Literal["aleatorio", "estratificado"] = "aleatorio"
    """`estratificado`: cuartiles de longitud del artículo en palabras, el
    mismo criterio del notebook 03, con `n / estratos` artículos por estrato."""
    estratos: int = Field(default=4, ge=2, le=10)

    @model_validator(mode="after")
    def _estratos_iguales(self) -> MuestraConfig:
        if self.muestreo == "estratificado" and self.n % self.estratos:
            raise ValueError(
                f"n={self.n} no se reparte en {self.estratos} estratos iguales."
            )
        return self


class ExperimentoConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # `metrics` y anotaciones futuras

    id: str
    model: ModeloConfig
    strategy: EstrategiaConfig
    sample: MuestraConfig


def cargar_config(ruta: str | pathlib.Path) -> ExperimentoConfig:
    """Lee y valida un YAML de experimento."""
    ruta = pathlib.Path(ruta)
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ValueError(f"{ruta} no contiene un mapeo YAML válido.")
    return ExperimentoConfig(**datos)
