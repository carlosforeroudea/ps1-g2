# Escenarios de arquitectura y despliegue

`escenarios-arquitectura.drawio` — cuatro páginas: tres escenarios evaluados y
la **arquitectura adoptada** (`★ FINAL`), que es la que hay que mostrar. Se abre
en [draw.io](https://app.diagrams.net) (*Archivo → Abrir desde → Dispositivo*)
o con la extensión *Draw.io Integration* de VS Code, y es editable a mano.

Cada página cubre el ciclo completo en cuatro bandas: **desarrollo local →
experimentación (Fase 3) → registro y selección (MLOps) → despliegue
(Fase 4)**.

---

## Qué restringe el diseño: los números del EDA

Los tres escenarios no son variaciones de estilo. Salen de lo que midieron
los notebooks, y cualquier alternativa que ignore estas cifras no es viable:

| Hallazgo | Fuente | Consecuencia arquitectónica |
|---|---|---|
| **99,1 %** de los artículos excede la ventana de 1.024 de BART (solo 57 de 6.440 caben) | `02_longitud_y_contexto` | La estrategia de contexto no es opcional: es la ruta normal, no el caso de borde |
| Map-reduce = **8,8 invocaciones de media**, 7 en la mediana, **19 en el p95**, **108 en el peor caso** | `02_longitud_y_contexto` | **La plataforma web no puede responder en la misma petición.** De ahí la cola de trabajos en los tres escenarios |
| El factorial completo ≈ **7.700 invocaciones** (~11 GPU-h) | derivado de 300 × 8 configuraciones | Cabría en la cuota gratuita de Colab/Kaggle. Se paga GPU igualmente, pero por reproducibilidad, no por capacidad |
| LED procesa completo el 91,5 %; LongT5 solo el 12,6 % | `02_longitud_y_contexto` | Ni el baseline superior se libra: el 8,5 % restante necesita estrategia también |
| Artículo mediano = 7.075 tokens; máximo = **110.531** (108× la ventana) | `02_longitud_y_contexto` | El worker necesita un tope de longitud y un camino definido para el caso patológico |

La consecuencia de diseño más importante es la segunda. Con 7 a 19 pasadas
del modelo por documento, una arquitectura petición-respuesta obligaría a
retirar map-reduce de la plataforma —justo la estrategia que el experimento
podría declarar ganadora—, lo que rompería el vínculo del
[ADR-001](../adr/001-despliegue-en-alcance.md): *la plataforma sirve la
configuración que ganó*. La cola asíncrona no es sobreingeniería, es lo que
mantiene ese vínculo en pie.

---

## Los tres escenarios

| | **1 · Costo cero estricto** | **2 · ZeroGPU** | **3 · GCP escala a cero** |
|---|---|---|---|
| **Desarrollo** | uv local + GitHub Actions | igual | igual + Docker |
| **Experimentación** | Colab / Kaggle (T4 gratis) | igual | Cloud Run Job con GPU L4 |
| **Registro** | HF Dataset privado | HF Dataset privado | Cloud Storage + BigQuery |
| **UI** | HF Space · Gradio (CPU) | HF Space · Gradio (CPU) | Cloud Run (`min-instances=0`) |
| **Cómputo de inferencia** | CPU del propio Space | **ZeroGPU bajo demanda** | Cloud Run Worker · GPU L4 |
| **Cola** | hilo dentro del Space | worker asíncrono | Cloud Tasks |
| **Observabilidad** | logs del Space | logs del Space | Cloud Logging + Monitoring |
| **Costo en reposo** | 0 | 0 | 0 |
| **Costo en demostración** | 0 | 0 | unos pocos USD/hora de GPU |
| **Latencia (artículo mediano, map-reduce)** | minutos | decenas de segundos | decenas de segundos |
| **Esfuerzo de montaje** | bajo | bajo–medio | alto |

### 1 · Costo cero estricto
Todo sobre capas gratuitas permanentes, sin tarjeta de crédito. Lo más simple
de sostener y el más rápido de montar. El precio lo paga la latencia: en CPU,
las 7 pasadas del artículo mediano son minutos, y el Space se duerme tras 48 h
de inactividad. Sirve para una demostración pactada; no para uso abierto.

### 2 · ZeroGPU — adoptado para el despliegue
Misma factura que el 1 (cero), pero la generación corre en GPU asignada por
invocación mediante `@spaces.GPU`. Resuelve el único problema real del
escenario 1 sin introducir costo ni complejidad de infraestructura. La cola
sigue siendo necesaria: el p95 son 19 pasadas y el peor caso 108.

Su riesgo es la dependencia de un solo proveedor y de una cuota gratuita cuyo
alcance conviene verificar antes de la sustentación.

### 3 · GCP escala a cero — del que se toma la Fase 3
Contenedores, IaC y CI/CD reales, con `min-instances=0` en todos los servicios
para que el reposo cueste cero. Es el más defendible en una sustentación de
ingeniería de sistemas y el plan B si HF cambia su plan gratuito.

**Aviso de costo:** la GPU L4 de Cloud Run **no está en capa gratuita** —se
paga por segundo de instancia activa—. Antes de la primera corrida hay que
fijar un **tope de gasto y una alerta de presupuesto**, o una demostración
que quede encendida consume el presupuesto entero.

---

## Decisión adoptada — página `★ FINAL`

No se eligió un escenario puro, sino un **híbrido** que separa dónde se paga y
dónde no:

| Fase | Dónde | Costo |
|---|---|---|
| **Fase 3 · Experimentación** | GCP · Vertex AI Custom Job con **GPU L4 spot** | **~3 USD** por pasada completa · ~10 USD con repeticiones |
| **Fase 4 · Despliegue** | **HF Spaces + ZeroGPU** | **0 USD** |
| Desarrollo y CI | local + GitHub Actions | 0 USD |
| Artifact Registry | imagen del contenedor | ~0,70 USD/mes |

**Total ≈ 10 USD**, y 0 en reposo.

### Por qué se paga la Fase 3 y no la Fase 4

El motivo no es la velocidad, es la **auditabilidad**.

El [ADR-000](../adr/000-alcance-y-pregunta-de-investigacion.md) exige n≥300 con
tests pareados. Eso solo vale si se puede demostrar que las 8 configuraciones
vieron la misma muestra con las mismas versiones. Un cuaderno de Colab no lo
da: el entorno lo fija el proveedor cada día. Un contenedor con *digest* fijo
sí. Esa es la **cadena de reproducibilidad** que dibuja la banda inferior de la
página final:

```
uv.lock → imagen @sha256 → config.yaml + semilla 42 → resultado por documento
                                                    → tests pareados auditables
```

A la plataforma web, en cambio, nadie le audita la reproducibilidad: solo tiene
que estar arriba el día de la sustentación. ZeroGPU da el mismo resultado
visible por 0 USD, y Spaces es una plataforma de despliegue legítima. Pagar ahí
no compraría rigor, solo comodidad.

### Dos decisiones que hacen viable el *spot*

- **Escritura incremental por documento.** Una instancia *spot* puede ser
  expropiada sin aviso. Si el runner persiste cada fila al terminarla, una
  expropiación cuesta un documento, no seis horas. Las filas por documento ya
  hacen falta para los tests pareados, así que no es trabajo extra.
- **Tareas en paralelo.** Cuatro tareas simultáneas cuestan las mismas
  GPU-horas totales, pero bajan el reloj de ~11 h a menos de 3.

## Antes de empezar

- **Las cuotas de capas gratuitas cambian.** Las de este documento (ZeroGPU,
  Colab, Cloud Run, Cloud Tasks, Firestore) hay que verificarlas contra la
  documentación vigente antes de comprometerse en la Fase 4.
- **Falta elegir el formato de ingesta.** Los diagramas asumen que la
  plataforma acepta PDF y texto; extraer texto de un PDF es trabajo propio de
  Fase 4 que aún no está dimensionado.
- **Falta definir el comportamiento en el caso patológico.** El artículo más
  largo del corpus son 110.531 tokens: 108 pasadas de map-reduce. La
  plataforma necesita un tope explícito y un mensaje al usuario, no un
  trabajo que nunca termina.
- **La cuota de GPU es el bloqueo número uno.** Las cuentas en periodo de
  prueba de GCP no traen cuota de GPU: hay que solicitarla y normalmente
  convertir la cuenta a una de pago (el crédito restante sigue aplicando).
  Verificar esto **antes** de contar con el bono de 300 USD, y antes de que la
  Fase 3 dependa de ello.
- **Alerta de presupuesto activa antes de la primera corrida.** Una VM con L4
  olvidada encendida cuesta ~540 USD al mes. El gasto planeado no es el riesgo;
  el olvido sí.

La decisión de la página `★ FINAL` todavía **no está registrada como ADR**.
Conviene fijarla como ADR-006 para que quede trazable junto a las demás.
