# Arquitectura del sistema

Resumidor inteligente de artículos científicos — Grupo 2, Los Predictores
Fase 2 (semanas 5–6) · Cubre el objetivo específico 3
Decisiones de respaldo: [ADR-001](adr/001-despliegue-en-alcance.md),
[ADR-002](adr/002-corpus-experimental.md),
[ADR-003](adr/003-rol-de-led-y-longt5.md),
[ADR-004](adr/004-stack-y-entorno.md),
[ADR-005](adr/005-arquitectura-de-la-canalizacion.md)

---

## 1. La restricción que define el diseño

Medido con el tokenizer real de BART sobre 30 artículos del split de
test (`make smoke`, semana 5):

| | Tokens |
|---|---|
| Artículo promedio | **7.839** |
| Mediana | 7.048 |
| Rango | 1.446 – 20.745 |
| Ventana de BART y PEGASUS | **1.024** |

**La brecha es de 7,7×. Ninguno de los 30 artículos cabe completo, y el
truncamiento descarta ~87 % del artículo promedio.**

Es peor de lo que estimaba el anteproyecto: contar por espacios da 5.905
tokens (ADR-002), un 33 % menos que el conteo real en subpalabras.

Esa brecha no es un detalle de implementación que se resuelve en una
función: es el objeto de estudio. Por eso la arquitectura no la esconde
—la expone como un punto de extensión explícito— y hace que el costo de
cerrarla sea siempre medible. Ese 87 % es lo que map-reduce y
extractivo-abstractivo deben recuperar para justificar su costo.

(Muestra de 30 artículos, suficiente para dimensionar el problema. El
EDA completo sobre los 6.440 del split es tarea de la semana 6.)

Todo lo demás en este documento se deriva de ahí.

---

## 2. Requisitos

### Funcionales
| ID | Requisito | Objetivo |
|----|-----------|----------|
| RF-1 | Resumir un artículo científico extenso en inglés | 4 |
| RF-2 | Ofrecer 3 estrategias de manejo de contexto intercambiables | 4 |
| RF-3 | Ofrecer múltiples modelos preentrenados intercambiables | 1, 4 |
| RF-4 | Evaluar calidad: ROUGE-1, ROUGE-2, ROUGE-Lsum, BERTScore | 5 |
| RF-5 | Medir costo: latencia p50/p95, pico de memoria, ratio de compresión | 5 |
| RF-6 | Ejecutar el factorial completo desde configuración declarativa | 7 |
| RF-7 | Plataforma web: cargar artículo, elegir opción, obtener resumen | 8 |

### Técnicos y de calidad
| ID | Requisito | Origen |
|----|-----------|--------|
| RT-1 | Reproducibilidad: misma muestra, misma semilla, mismas versiones | ADR-000, ADR-004 |
| RT-2 | Toda ejecución produce resumen **y** medidas de costo, inseparables | ADR-005 |
| RT-3 | Añadir estrategia o modelo = implementar un protocolo, sin tocar código existente | ADR-005 |
| RT-4 | Ejecutable en CPU, MPS y CUDA sin cambios de código | ADR-004 |
| RT-5 | La plataforma consume la canalización evaluada, no una reimplementación | ADR-001, ADR-005 |
| RT-6 | Configuración inválida falla antes de la corrida, no durante | ADR-005 |

---

## 3. Canalización de referencia

```
                         ┌─────────────────────────────────┐
   Documento             │   ContextStrategy (orquesta)    │
   (~7.800 tokens)       │                                 │
        │                │   fragmenta / selecciona /      │
        ▼                │   trunca, e invoca el modelo    │
   ┌─────────┐           │   1..N veces según su lógica    │
   │ Ingesta │──────────▶│                 │               │
   └─────────┘  texto    │                 ▼               │
   PDF · txt    normali- │        ┌──────────────────┐     │
   · corpus     zado     │        │  SummarizerModel │     │
                         │        │  (1.024 tokens)  │     │
                         │        └──────────────────┘     │
                         └────────────────┬────────────────┘
                                          │
                      ╔═══════════════════▼═══════════════════╗
                      ║  Instrumentación — envuelve TODA la    ║
                      ║  estrategia, no cada invocación        ║
                      ║  latencia · pico de memoria            ║
                      ╚═══════════════════╤═══════════════════╝
                                          │
                                          ▼
                            SummaryResult(texto + costo)
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
              ┌────────────────────┐            ┌────────────────────┐
              │ Evaluación (exp.)  │            │ Plataforma web     │
              │ ROUGE · BERTScore  │            │ (Fase 4)           │
              │ ratio compresión   │            │                    │
              └────────────────────┘            └────────────────────┘
```

Dos lecturas importantes del diagrama:

- **El modelo está *dentro* de la estrategia.** No es un paso posterior.
  Es lo que permite que map-reduce lo invoque N veces sin que nada más
  en el sistema se entere (ADR-005 §2).
- **La instrumentación rodea la estrategia completa.** Si rodeara el
  modelo, map-reduce reportaría el costo de 1 de N invocaciones y la
  comparación entre celdas sería inválida (ADR-005 §3).

---

## 4. Abstracciones centrales

Son dos. El resto del sistema son detalles alrededor de ellas.

### `SummarizerModel`

Envuelve un modelo preentrenado y —esto es lo que importa— **expone su
ventana de contexto como dato consultable**.

```python
class SummarizerModel(Protocol):
    @property
    def context_window(self) -> int:
        """Máximo de tokens de entrada que acepta el modelo."""

    @property
    def name(self) -> str:
        """Identificador del checkpoint, para trazabilidad."""

    def count_tokens(self, text: str) -> int:
        """Tokens reales según el tokenizer de este modelo."""

    def truncate(self, text: str, max_tokens: int) -> str:
        """Recorta a lo sumo `max_tokens`. Vive aquí y no en las
        estrategias porque solo el tokenizer sabe dónde caen los
        límites de token."""

    def generate(self, text: str, max_new_tokens: int) -> str:
        """Genera un resumen. `text` DEBE caber en context_window;
        garantizarlo es responsabilidad de la estrategia."""
```

`context_window` y `count_tokens` son la razón de ser del protocolo: sin
ellos cada estrategia tendría que codificar los límites de cada modelo, y
añadir LED (16.384 tokens) obligaría a tocar las cuatro estrategias.

`truncate` lo necesitan tres de las cuatro estrategias, no solo
`Truncation`: `MapReduce` acota cada fragmento y la reducción final,
y `ExtractiveAbstractive` acota lo que preselecciona. Nótese que **no
es idempotente**: un ciclo encode/decode normaliza espacios y no
reproduce el texto original, así que las estrategias solo lo invocan
cuando `count_tokens` confirma que hace falta.

> **Trampa verificada — de dónde sale `context_window`.**
> No del tokenizer. `tokenizer.model_max_length` devuelve un centinela
> de ~1e30 cuando el checkpoint no lo declara en su
> `tokenizer_config.json`, que es exactamente el caso de
> `facebook/bart-large-cnn`. El `make smoke` de la semana 5 lo detectó:
> con ese valor, `Truncation` no truncaría nunca y el proyecto entero
> mediría un fenómeno que no ocurre.
>
> La fuente correcta es la config del modelo:
> `AutoConfig.from_pretrained(ckpt).max_position_embeddings` → 1024.
> Los adaptadores de Fase 3 deben leerla de ahí. LongT5 usa atención
> relativa y no declara el atributo: su límite se fija por
> configuración explícita, no por introspección.

Implementaciones: BART, PEGASUS (contexto corto, objeto de estudio);
LED, LongT5 (baseline superior, ADR-003).

### `ContextStrategy`

Cómo se adapta un documento que no cabe. Recibe el modelo por inyección
y orquesta la generación completa.

```python
class ContextStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def summarize(self, document: Document, model: SummarizerModel) -> str:
        """Produce un resumen del documento usando el modelo.
        Libre de invocar `model.generate` tantas veces como necesite."""
```

| Implementación | Invocaciones al modelo | Idea |
|---|---|---|
| `LeadK` | 0 | Primeras k oraciones. Piso trivial (ADR-003) |
| `Truncation` | 1 | Recorta a `context_window` y genera |
| `MapReduce` | N + 1 | Resume cada fragmento, luego resume los resúmenes |
| `ExtractiveAbstractive` | 1 | Preselecciona extractivamente, luego genera |

La columna de invocaciones es exactamente la asimetría que motiva las
decisiones §2 y §3 del ADR-005.

### `Document`

El artículo con su estructura preservada. La config `section` del corpus
(ADR-002) da los límites de sección, que `MapReduce` usa como unidad de
fragmentación semántica en lugar de cortar a ciegas cada 1.024 tokens.

```python
@dataclass(frozen=True)
class Document:
    doc_id: str
    sections: tuple[Section, ...]
    reference_summary: str | None  # abstract, si viene del corpus
```

---

## 5. Instrumentación y resultado

RT-2 se cumple por construcción: no existe una ruta que devuelva un
resumen sin sus medidas.

```python
@dataclass(frozen=True)
class CostMetrics:
    latency_seconds: float
    peak_memory_mb: float
    model_calls: int          # delata la aridad de la estrategia
    input_tokens: int
    output_tokens: int

@dataclass(frozen=True)
class SummaryResult:
    summary: str
    cost: CostMetrics
    model_name: str
    strategy_name: str

    @property
    def compression_ratio(self) -> float:
        return self.output_tokens / self.input_tokens
```

**Punto abierto que hay que cerrar en Fase 3 (ADR-005):** medir pico de
memoria de forma comparable entre CPU, MPS y CUDA no es directo — cada
backend expone contadores distintos y con semánticas distintas. Debe
resolverse en un solo módulo (`instrumentation/`) y documentarse qué mide
exactamente en cada dispositivo. Si no, las cifras de memoria no serán
comparables ni siquiera entre corridas del propio equipo.

`model_calls` se registra a propósito: hace visible en los datos por qué
map-reduce cuesta lo que cuesta.

---

## 6. Experimentos declarativos

Cada celda del factorial es un YAML en `experiments/configs/`:

```yaml
id: mapreduce_bart
model:
  kind: bart
  checkpoint: facebook/bart-large-cnn
strategy:
  kind: map_reduce
  chunk_by: section
  reduce_max_tokens: 256
sample:
  dataset: ccdv/arxiv-summarization
  config: section
  split: test
  n: 300
  seed: 20262
```

El runner materializa el producto cartesiano y escribe resultados
crudos por documento (no solo agregados) — los tests pareados del
ADR-000 los necesitan.

`sample` idéntico en todas las celdas es lo que hace válido el pareado.
Y RT-6 obliga a validar el YAML completo antes de cargar el primer
modelo: un checkpoint mal escrito debe fallar en segundos, no doce horas
después.

> **Pendiente de Fase 3:** elegir los checkpoints exactos. Para PEGASUS
> existen variantes ya afinadas en arXiv (`google/pegasus-arxiv`). Usarlas
> es legítimo —no es fine-tuning nuestro, y el split de test sigue siendo
> ciego— pero **no es comparable** con un `bart-large-cnn` afinado en
> noticias. O ambos modelos usan checkpoints de dominio general, o ambos
> usan checkpoints de dominio científico. Mezclarlos confundiría el efecto
> de la estrategia con el efecto del dominio del preentrenamiento.

---

## 7. Estructura de paquetes

```
src/resumidor/
  domain.py            # Document, Section, SummaryResult, CostMetrics
  models/              # SummarizerModel + adaptadores HF
  strategies/          # ContextStrategy + las 4 implementaciones
  instrumentation/     # medición de latencia y memoria
  data/                # carga y muestreo del corpus (ADR-002)
  eval/                # ROUGE, BERTScore, agregación, tests pareados
  config.py            # esquema de configuración y validación (RT-6)

experiments/
  configs/             # una celda del factorial por archivo
  results/             # salidas crudas (ignorado por Git)

tests/
paper/                 # informe final
```

Regla de dependencias, en un solo sentido:

```
domain ← models ← strategies ← instrumentation ← runner
   ↑                                                ↑
   └──────────── eval ──────────┘        plataforma web
```

`domain.py` no importa nada del proyecto. La capa web importa el runner;
**nada en `src/resumidor` importa la capa web** (RT-5, ADR-005 §5).

---

## 8. Frontera con la plataforma web (Fase 4)

La plataforma es un consumidor más de la canalización, al mismo nivel que
el banco de experimentos:

- Las "opciones predefinidas de procesamiento" del objetivo 8 son las
  mismas configuraciones de la §6.
- La configuración por defecto en producción es **la que gane el criterio
  de compromiso calidad/costo de la Fase 3** (objetivo 6). No se elige por
  conveniencia de implementación: ese vínculo es lo que hace que las dos
  mitades del proyecto sean el mismo proyecto (ADR-001).
- La ingesta de PDF es propia de la plataforma; en experimentos el texto
  ya viene del corpus. Ambas rutas convergen en `Document`.

Requisitos que la plataforma añade y la investigación no tenía: latencia
tolerable para un usuario interactivo, límite de tamaño de carga, y
comportamiento definido cuando el documento excede incluso lo que la
estrategia puede manejar.

---

## 9. Estado de implementación (fin de semana 5)

| Componente | Estado |
|---|---|
| `domain.py`, protocolos, `Truncation` | Esqueleto, con tests |
| `MapReduce`, `ExtractiveAbstractive`, `LeadK` | Fase 3 |
| Adaptadores de modelos | Fase 3 |
| Instrumentación | Fase 3 (contrato ya fijado aquí) |
| Corpus y EDA | Semana 6 |
| Runner y evaluación | Fase 3 |
| Plataforma web | Fase 4 |

El esqueleto de la semana 5 existe para verificar que los contratos de
la §4 son usables, no para funcionar como sistema.
