"""Selección de dispositivo de cómputo, en un solo lugar.

RT-4 (`docs/arquitectura.md`): el código debe correr en CPU, MPS y CUDA sin
cambios. Centralizarlo aquí es lo que evita que aparezcan `if torch.cuda...`
dispersos por los adaptadores.
"""

from __future__ import annotations

from typing import Literal

Dispositivo = Literal["cuda", "mps", "cpu"]


def detectar_dispositivo() -> Dispositivo:
    """Devuelve el mejor dispositivo disponible.

    El orden es deliberado: CUDA en la nube (Fase 3), MPS en los portátiles
    del equipo, CPU como opción siempre válida.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolver_dispositivo(preferido: str | None = None) -> Dispositivo:
    """Valida el dispositivo pedido por configuración, o detecta uno.

    Falla ruidosamente si se pide un dispositivo que no está: una corrida de
    300 documentos que cae silenciosamente a CPU tarda horas de más y sus
    medidas de costo no son comparables con las demás celdas.
    """
    if preferido in (None, "auto"):
        return detectar_dispositivo()

    if preferido not in ("cuda", "mps", "cpu"):
        raise ValueError(
            f"Dispositivo desconocido: {preferido!r}. "
            "Usa 'cuda', 'mps', 'cpu' o 'auto'."
        )

    disponible = detectar_dispositivo()
    if preferido != "cpu" and disponible != preferido:
        raise RuntimeError(
            f"Se pidió el dispositivo {preferido!r} pero no está disponible "
            f"(detectado: {disponible!r}). Cambiar a CPU en silencio haría que "
            "las medidas de latencia de esta celda no sean comparables con las "
            "demás; corrige la configuración de forma explícita."
        )
    return preferido  # type: ignore[return-value]
