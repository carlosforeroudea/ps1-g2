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
        generacion: dict | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._dispositivo = resolver_dispositivo(dispositivo)
        self._num_beams = num_beams
        self._ventana_forzada = ventana
        self._generacion = dict(generacion or {})

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

    def precargar(self, *, calentar: bool = True) -> None:
        """Descarga los pesos, los carga en memoria y calienta el cómputo.

        El calentamiento no es opcional en la práctica. Cargar los pesos no
        basta: la **primera** invocación real compila el grafo y reserva los
        búferes del dispositivo, y ese costo aterriza entero sobre el primer
        documento medido. Verificado en la semana 6 con PEGASUS en MPS: el
        primer documento tardó 448,8 s y los siguientes 28,3 y 18,1 s. Sin
        calentar, la latencia del documento 1 no es comparable con la del
        resto ni con la de otras celdas.

        La generación de calentamiento se descarta.
        """
        _ = self._tokenizer, self._modelo, self.context_window
        if calentar:
            self.generate("warm up the compute graph", max_new_tokens=8)

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
        """Genera un resumen bajo la política de generación del experimento.

        Los parámetros de `generacion` **sobrescriben la configuración propia
        del checkpoint**, y eso es deliberado. Cada checkpoint trae la suya,
        heredada del corpus en que fue afinado: `facebook/bart-large-cnn`
        lleva `max_length=142`, `min_length=56` y `length_penalty=2.0`, que
        son longitudes de noticia. Medido en la semana 6, BART producía 96
        tokens de media frente a los 209 de los abstracts de referencia,
        mientras PEGASUS-arxiv producía 205.

        Sin igualar la política, la diferencia de ROUGE entre modelos
        reflejaría la longitud que cada uno aprendió de su dominio de
        afinado, no su capacidad de seleccionar contenido — que es lo que el
        proyecto quiere medir. Igualarla es un control experimental, y debe
        declararse como tal en el informe.
        """
        import torch

        entradas = self._tokenizer(
            text,
            truncation=True,
            max_length=self.context_window,
            return_tensors="pt",
        ).to(self._dispositivo)

        parametros = {
            "max_new_tokens": max_new_tokens,
            "num_beams": self._num_beams,
            # Neutraliza los topes en tokens absolutos del checkpoint, que
            # conviven con `max_new_tokens` y confunden a `generate`.
            "max_length": None,
            "min_length": None,
            **self._generacion,
        }

        with torch.no_grad():
            salida = self._modelo.generate(**entradas, **parametros)

        return self._tokenizer.decode(salida[0], skip_special_tokens=True)
