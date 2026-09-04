# ADR-005 — Arquitectura de la canalización
Estado: Aceptado | Fecha: 2026-09-03 | Autores: Forero, Martelo, Mercado

## Contexto
El sistema tiene que servir a dos consumidores con exigencias
distintas y un requisito compartido:

1. El banco de experimentos (objetivos 4–7): ejecutar 6 celdas del
   factorial más los baselines (ADR-003) sobre la misma muestra, con
   instrumentación de costo y sin variación accidental entre celdas.
2. La plataforma web (objetivo 8, ADR-001): un artículo a la vez,
   interactivo, con opciones predefinidas.

El requisito compartido es el que fija la arquitectura: la plataforma
debe ejecutar **la misma** canalización que produjo la evidencia. Si
son dos implementaciones, la afirmación "el sistema desplegado usa la
configuración que ganó el experimento" deja de ser cierta y el
proyecto pierde su vínculo interno.

## Decisión

### 1. Las dos variables independientes son los dos ejes de extensión
Estrategia de manejo de contexto y modelo son ortogonales e
intercambiables, y se modelan como protocolos (puertos) con
adaptadores:

- `ContextStrategy` — cómo se adapta un documento largo al modelo.
  Implementaciones: `Truncation`, `MapReduce`, `ExtractiveAbstractive`
  y `LeadK` (piso trivial).
- `SummarizerModel` — envuelve un modelo preentrenado y **expone su
  ventana de contexto** como propiedad consultable.
  Implementaciones: BART, PEGASUS, LED, LongT5.

Añadir una celda al factorial debe ser una implementación nueva de un
protocolo, nunca un `if` dentro de código existente.

Que el modelo exponga su ventana es lo que permite que la estrategia
sea agnóstica del modelo: `Truncation` no sabe si recorta a 1.024 o a
16.384, solo pregunta.

### 2. La estrategia orquesta; el modelo solo genera
Decisión no obvia, y es la clave del diseño.

Las tres estrategias tienen aridad distinta frente al modelo:
`Truncation` lo invoca **una vez**; `MapReduce` lo invoca **una vez
por fragmento más una reducción final**; `ExtractiveAbstractive`
preselecciona sin modelo generativo y luego lo invoca **una vez**.

Por tanto: **la estrategia recibe el modelo por inyección y es
responsable de la orquestación completa**, incluido el número de
invocaciones. El modelo no conoce la estrategia.

La alternativa —que la estrategia devuelva texto y un orquestador
externo llame al modelo— se rechaza: no puede expresar map-reduce sin
un caso especial, y ese caso especial reaparecería en cada punto del
sistema.

### 3. La instrumentación envuelve la estrategia, no el modelo
Consecuencia directa de lo anterior. Si midiéramos el modelo,
`MapReduce` reportaría el costo de una invocación entre muchas y la
comparación entre celdas sería inválida — precisamente al revés de lo
que el proyecto quiere medir. El costo de una estrategia es el costo
de todo lo que hace para producir un resumen.

La instrumentación es un **requisito de primera clase, presente en el
contrato desde el diseño**, no un añadido posterior. El objetivo
específico 5 exige latencia p50/p95, pico de memoria y ratio de
compresión; retrofitear eso en la Fase 3 obliga a repetir corridas.

Toda ejecución devuelve resumen **y** medidas de costo, juntos. Es
deliberadamente imposible obtener un resumen sin sus medidas.

### 4. Los experimentos se declaran, no se programan
Cada celda es una entrada YAML en `experiments/configs/`: modelo,
estrategia, hiperparámetros, muestra y semilla. Un runner materializa
el producto cartesiano.

Así, añadir el baseline LED del ADR-003 es una línea de configuración.
Y la semilla y la muestra quedan versionadas junto al resultado, que
es lo que hace auditable el n≥300 pareado del ADR-000.

### 5. Frontera con la plataforma web
`src/resumidor` **no importa nada de la capa web**. La dependencia va
en un solo sentido: la plataforma consume la canalización. Las
"opciones predefinidas de procesamiento" del objetivo 8 son
exactamente las mismas configuraciones del punto 4.

## Consecuencias
(+) Cada celda del factorial es una configuración, no código nuevo:
    menos superficie donde introducir variación accidental entre
    condiciones.
(+) La plataforma sirve la canalización evaluada, por construcción.
(+) Las medidas de costo no se pueden omitir por olvido.
(−) Inyectar el modelo en la estrategia acopla ambos protocolos: un
    cambio en la firma de `SummarizerModel` toca las cuatro
    estrategias. Es aceptable — son cuatro, y el acoplamiento es
    explícito y probado.
(−) La configuración declarativa exige validación temprana: un YAML
    con un nombre de modelo mal escrito debe fallar antes de la
    corrida, no doce horas después.
(−) Medir pico de memoria de forma comparable entre CPU, MPS y CUDA
    no es trivial. Debe resolverse en un solo módulo y documentarse
    qué mide exactamente en cada dispositivo, o las cifras de la
    Fase 3 no serán comparables entre sí.
