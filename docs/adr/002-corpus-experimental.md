# ADR-002 — Corpus experimental
Estado: Aceptado | Fecha: 2026-09-03 | Autores: Forero, Martelo, Mercado

## Contexto
El ADR-000 exige n≥300 con tests pareados, lo que obliga a un corpus
con resúmenes de referencia ya disponibles: producirlos manualmente
para 300+ artículos es inviable en 16 semanas.

El anteproyecto ancla el planteamiento del problema en Cohan et al.
(2018) —"los artículos científicos promedian 4.938 palabras"— y esa
cita ya está en la bibliografía. Usar el mismo corpus del que sale la
estadística que motiva el proyecto hace la evidencia autoconsistente.

## Decisión
Corpus: `ccdv/arxiv-summarization` (Hugging Face), que es la versión
publicada del dataset de arXiv de Cohan et al. (2018).

Estructura: `id`, `article` (cuerpo del artículo), `abstract` (resumen
de referencia). Splits: 203.037 train / 6.436 validation / 6.440 test.

Configuración: `section`, que preserva la separación por secciones del
artículo original. No es un detalle cosmético — es la unidad de
fragmentación semántica que necesitan map-reduce y extractivo-
abstractivo. Con la config plana habría que reinferir los límites de
sección con heurísticas frágiles.

Universo de muestreo: **solo el split `test`** (6.440 artículos). No
se usan train ni validation porque no hay fine-tuning (ADR-000): sin
entrenamiento, la distinción train/test no protege de nada, y el split
de test es el que la literatura reporta, lo que hace comparables
nuestras cifras con las publicadas.

Descarga: en streaming o por split. **No se descargan los 7,26 GB
completos**; el split de test es una fracción menor del total.

Muestreo: aleatorio simple sobre el split de test, semilla fija y
versionada en la configuración del experimento. La misma muestra para
todas las celdas del factorial — es lo que habilita los tests pareados
que exige el ADR-000.

## Estadísticas que justifican el problema
Medidas sobre el propio corpus (tokens por separación de espacios):

| Split      | Tokens/artículo | Tokens/abstract |
|------------|-----------------|-----------------|
| train      | 6.038           | 299             |
| validation | 5.894           | 172             |
| test       | 5.905           | 174             |

El promedio del split de test (5.905) es **~5,8×** la ventana de 1.024
tokens de BART y PEGASUS. La brecha es aún mayor que las 4.938
palabras citadas en el anteproyecto, y el conteo real en tokens de
subpalabra la ampliará todavía más (un token BPE es menor que una
palabra). El smoke test del entorno mide esa cifra exacta.

## Consecuencias
(+) Resúmenes de referencia disponibles sin trabajo de anotación.
(+) Comparabilidad directa con la literatura, que reporta ROUGE sobre
    este mismo split.
(+) El dominio del corpus (arXiv) coincide con la estadística de
    volumen citada en el planteamiento del problema.
(−) arXiv no es biomédico. El planteamiento del problema enfatiza el
    caso clínico (Bastian et al. 2010, Tang et al. 2023) y el corpus
    no lo cubre. Hay dos salidas: acotar las conclusiones al dominio
    de arXiv, o añadir `ccdv/pubmed-summarization` como réplica si el
    presupuesto de cómputo lo permite. **Debe declararse como
    limitación en el informe final.**
(−) El abstract como resumen de referencia tiene un sesgo conocido:
    es un resumen de autor, no un resumen extractivo-fiel, lo que
    penaliza sistemáticamente el ROUGE de cualquier sistema.
