# ADR-003 — Rol de LED y LongT5 en el diseño experimental
Estado: Aceptado | Fecha: 2026-09-03 | Autores: Forero, Martelo, Mercado
Precisa: ADR-000 (sección "Decisión", variable independiente)

## Contexto
Hay una ambigüedad entre los dos documentos de referencia.

El ADR-000 define la variable independiente como
`estrategia × modelo (BART | PEGASUS | LongT5/LED)`, es decir, los
modelos de contexto largo como un nivel más del factorial.

El anteproyecto dice lo opuesto: "el presente proyecto se enfoca en
estudiar estrategias que permitan utilizar modelos preentrenados de
contexto corto, **en lugar de** desarrollar o utilizar exclusivamente
modelos diseñados para ventanas de contexto extensas".

Tratarlos como nivel pleno del factorial contradice el enfoque
declarado y multiplica el costo de cómputo. Excluirlos por completo
deja la tesis sin punto de referencia: sin saber cuánta calidad
sacrifica el contexto corto, la afirmación "ofrece un compromiso
calidad-costo suficiente" no es verificable ni falsable.

## Decisión
LED y LongT5 entran como **baseline superior de referencia**, fuera
del factorial pero dentro del reporte.

Diseño resultante:

- **Factorial (objeto de estudio):**
  3 estrategias {truncamiento, map-reduce, extractivo-abstractivo}
  × 2 modelos de contexto corto {BART, PEGASUS} = 6 celdas.

- **Baselines de referencia (fuera del factorial):**
  - Techo/piso: LED y LongT5 sobre el documento completo, sin
    estrategia de manejo de contexto. Aportan el techo de calidad y
    el piso de costo contra el cual se lee todo lo demás.
  - Piso trivial: `lead-k` extractivo (primeras k oraciones), sin
    modelo generativo. Sin él, un ROUGE bajo en todas las celdas
    sería ininterpretable.

Todas las condiciones se miden con las mismas métricas, sobre la misma
muestra y con la misma instrumentación (ADR-002, ADR-005). La única
diferencia es el rol argumentativo, no el tratamiento experimental.

Esto vuelve la tesis falsable y cuantitativa. La conclusión del
proyecto toma la forma:

> "La estrategia X sobre <modelo de contexto corto> recupera el N %
> del ROUGE-Lsum de LED a un M % de su latencia y un P % de su pico
> de memoria."

Si N resulta ser bajo, el hallazgo sigue siendo válido y publicable:
la respuesta a la pregunta de investigación sería que el contexto
corto **no** basta, y el proyecto lo habría demostrado con evidencia
propia en lugar de asumirlo. Es coherente con la consecuencia (+) ya
registrada en el ADR-000.

## Corrección de la semana 6 — los checkpoints importan tanto como el papel

Este ADR asignaba a LED y LongT5 el papel de baseline superior, pero no
fijaba **qué checkpoint**. Al implementarlo se eligieron
`allenai/led-base-16384` y `google/long-t5-tglobal-base`, y ambos son
**modelos base sin afinar para resumen**: verificado contra el Hub, ninguno
declara dataset de la tarea.

La medición lo delató de inmediato (4 documentos, política de generación
idéntica):

| Modelo | ROUGE-1 | Latencia p50 |
|---|---|---|
| PEGASUS-arxiv | 0,457 | 113 s |
| LongT5 base | 0,450 | 79 s |
| BART-cnn | 0,434 | 25 s |
| **LED base** | **0,240** | **257 s** |

Un techo de calidad que queda muy por debajo del suelo no es un techo: es un
modelo que no sabe hacer la tarea. Con esos checkpoints, la afirmación
central de este ADR —«la estrategia X recupera el N % del ROUGE del
baseline»— no significaba nada.

Es el mismo error que este proyecto ya había evitado en el factorial al
descartar `google/pegasus-large` en favor de `google/pegasus-arxiv`. Se
advirtió para el objeto de estudio y se repitió en la referencia.

### Decisión corregida

**LED → `allenai/led-large-16384-arxiv`.** Afinado por los autores de LED
sobre `scientific_papers`, la misma familia que nuestro corpus, con el split
de test ciego. Es la referencia publicada para esta tarea exacta y el
baseline superior legítimo.

**LongT5 → deja de ser baseline y pasa a ser control.** No existe checkpoint
oficial afinado en arXiv; los de la comunidad son de otros dominios. Se
conserva como `control_longt5_sin_afinar` porque, puesto junto a LED-large
afinado, separa cuánto del rendimiento viene de la arquitectura de contexto
largo y cuánto del afinado. **No entra en la afirmación del techo de
calidad.**

### Regla que queda

Todo checkpoint que entre al experimento debe declarar su dataset de afinado
antes de usarse. Un modelo base y uno afinado no son comparables, y la
diferencia se confunde con el efecto que el proyecto quiere medir.

## Consecuencias
(+) Resuelve la contradicción entre ADR-000 y anteproyecto sin
    invalidar ninguno de los dos.
(+) La afirmación central del proyecto pasa de cualitativa
    ("compromiso suficiente") a cuantitativa y verificable.
(+) El `lead-k` cuesta prácticamente nada y protege contra la
    interpretación errónea de resultados bajos.
(−) LED y LongT5 sobre documentos de ~6.000 tokens son la corrida más
    cara del proyecto en memoria y latencia. Debe presupuestarse
    cómputo en nube específicamente para ella, y puede requerir una
    muestra menor que las celdas del factorial (lo cual es aceptable:
    es referencia, no objeto de comparación pareada).
(−) Añadir el baseline al reporte exige explicar por qué no es el
    objeto de estudio, o el lector asumirá que se eligió el peor
    camino a propósito.
