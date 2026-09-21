"""Adaptador de un modelo seq2seq de Hugging Face al puerto `SummarizerModel`.

Cubre BART y PEGASUS (objeto de estudio) y sirve igual para LED y LongT5
(baseline del ADR-003): la ventana de contexto se descubre por introspección,
así que añadirlos no requiere tocar este archivo.
"""

from __future__ import annotations

from functools import cached_property

from resumidor.device import resolver_dispositivo

# Atributos donde los distintos modelos declaran su ventana de contexto.
# El orden importa: LED declara `max_encoder_position_embeddings` Y
# `max_position_embeddings` (este último es el del *decoder*, 1.024), así que
# el del encoder debe consultarse primero o se leería la ventana equivocada.
_ATRIBUTOS_VENTANA = (
    "max_encoder_position_embeddings",
    "max_position_embeddings",
    "n_positions",
)

_VENTANA_MAXIMA_RAZONABLE = 1_000_000


def ventana_de_contexto(checkpoint: str) -> int:
    """Ventana de contexto real del modelo, en tokens.

    La fuente correcta es la **config del modelo**, no el tokenizer.
    `tokenizer.model_max_length` devuelve un centinela de ~1e30 cuando el
    checkpoint no lo declara —que es el caso de `facebook/bart-large-cnn`—.
    Confiar en él haría que `Truncation` no truncara nunca y que todo el
    proyecto midiera un fenómeno que no ocurre. Lo detectó el `make smoke`
    de la semana 5; ver `docs/arquitectura.md` §4.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)

    for atributo in _ATRIBUTOS_VENTANA:
        valor = getattr(config, atributo, None)
        if isinstance(valor, int) and 0 < valor < _VENTANA_MAXIMA_RAZONABLE:
            return valor

    raise RuntimeError(
        f"No se pudo determinar la ventana de contexto de {checkpoint!r}. "
        "Los modelos de atención relativa (LongT5) no la declaran y necesitan "
        "un límite explícito en la configuración del experimento."
    )


class HFSummarizer:
    """Modelo preentrenado de Hugging Face que satisface `SummarizerModel`.

    Carga perezosa: el tokenizer y los pesos se descargan la primera vez que
    se usan, no al construir. Así validar una configuración (RT-6) no cuesta
    los 4 GB de descarga.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        dispositivo: str | None = None,
        num_beams: int = 4,
        ventana: int | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._dispositivo = resolver_dispositivo(dispositivo)
        self._num_beams = num_beams
        self._ventana_forzada = ventana

    # --- identidad -----------------------------------------------------
    @property
    def name(self) -> str:
        return self._checkpoint

    @property
    def dispositivo(self) -> str:
        return self._dispositivo

    @cached_property
    def context_window(self) -> int:
        if self._ventana_forzada is not None:
            return self._ventana_forzada
        return ventana_de_contexto(self._checkpoint)

    # --- carga perezosa ------------------------------------------------
    @cached_property
    def _tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self._checkpoint)

    @cached_property
    def _modelo(self):
        from transformers import AutoModelForSeq2SeqLM

        modelo = AutoModelForSeq2SeqLM.from_pretrained(self._checkpoint)
        modelo.to(self._dispositivo)
        modelo.eval()
        return modelo

    def precargar(self) -> None:
        """Fuerza la descarga y la carga en memoria.

        Se invoca antes de empezar a medir: si los pesos se cargaran dentro
        del primer documento, su latencia incluiría la descarga y esa celda
        quedaría inflada frente a las demás.
        """
        _ = self._tokenizer, self._modelo, self.context_window

    # --- puerto SummarizerModel ----------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, truncation=False)["input_ids"])

    def truncate(self, text: str, max_tokens: int) -> str:
        """Recorta a lo sumo `max_tokens` tokens.

        No es idempotente: el ciclo codificar/decodificar normaliza espacios y
        no reproduce el texto original. Por eso las estrategias solo lo
        invocan cuando `count_tokens` confirma que hace falta.
        """
        ids = self._tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"]
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def generate(self, text: str, max_new_tokens: int) -> str:
        import torch

        entradas = self._tokenizer(
            text,
            truncation=True,
            max_length=self.context_window,
            return_tensors="pt",
        ).to(self._dispositivo)

        with torch.no_grad():
            salida = self._modelo.generate(
                **entradas,
                max_new_tokens=max_new_tokens,
                num_beams=self._num_beams,
            )

        return self._tokenizer.decode(salida[0], skip_special_tokens=True)
