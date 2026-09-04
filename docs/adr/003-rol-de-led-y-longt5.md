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
