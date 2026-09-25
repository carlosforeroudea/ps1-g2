# Ejecución en Google Cloud — runbook

Implementa la banda 2 de la arquitectura adoptada
([diagramas](diagramas/README.md)): la Fase 3 corre en GCP dentro de un
contenedor con *digest* fijo, para que los resultados del experimento sean
auditables.

**Proyecto:** `udea-509713` · **Cuenta:** `carlosforero135923@gmail.com`

La configuración de `gcloud` está aislada en un perfil llamado `udea`, para
que no se mezcle con otras cuentas de trabajo:

```bash
gcloud config configurations activate udea      # universidad
gcloud config configurations activate default   # volver a lo demás
```

---

## Por qué en contenedor y no en Colab

El [ADR-000](adr/000-alcance-y-pregunta-de-investigacion.md) exige n≥300 con
tests pareados. Eso solo vale si se puede demostrar que las ocho
configuraciones vieron la misma muestra con las mismas versiones. Un cuaderno
de Colab no lo da: el entorno lo fija el proveedor cada día.

La cadena que sí lo da:

```
uv.lock → imagen @sha256 → config.yaml + semilla → resultado por documento
```

Ese es el único motivo para gastar aquí. No es velocidad.

---

## Paso 0 — Autenticación (una vez)

```bash
gcloud config configurations activate udea
gcloud auth login carlosforero135923@gmail.com
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud config set project udea-509713
gcloud config set compute/region us-central1
```

Verificar: `gcloud config list` debe mostrar la cuenta personal y
`udea-509713`.

## Paso 1 — Alerta de presupuesto, antes que nada

El riesgo de este proyecto no es el gasto planeado —unos pocos USD— sino el
olvido. Una instancia con L4 encendida un fin de semana cuesta más que todo
el experimento.

Crear en la consola, en *Facturación → Presupuestos y alertas*, un
presupuesto de **20 USD** con avisos al 50 %, 90 % y 100 %. Hacerlo **antes**
de la primera corrida, no después.

## Paso 2 — Habilitar las APIs

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com
```

## Paso 3 — Repositorio de imágenes y bucket

```bash
gcloud artifacts repositories create resumidor \
    --repository-format=docker \
    --location=us-central1 \
    --description="Imágenes del runner de experimentos"

gcloud storage buckets create gs://udea-509713-resumidor \
    --location=us-central1 \
    --uniform-bucket-level-access
```

## Paso 4 — Construir y publicar la imagen

Con Docker local (requiere Docker Desktop abierto):

```bash
export IMG=us-central1-docker.pkg.dev/udea-509713/resumidor/runner
docker build --platform linux/amd64 -t $IMG:latest .
docker push $IMG:latest
```

`--platform linux/amd64` no es opcional en un Mac con Apple Silicon: sin él
la imagen sale ARM y Vertex AI no la ejecuta.

Alternativa sin Docker local, construyendo en la nube. **Los dos flags
extra no son opcionales** (ver más abajo):

```bash
gcloud builds submit --tag "${IMG}:latest" --region=us-central1 \
    --service-account=projects/udea-509713/serviceAccounts/395466321539-compute@developer.gserviceaccount.com \
    --default-buckets-behavior=regional-user-owned-bucket .
```

> **Las comillas y las llaves NO son opcionales en zsh** —que es la shell por
> defecto de macOS—. Escrito como `$IMG:latest`, zsh interpreta `:l` como su
> modificador de expansión «pasar a minúsculas»: se come la `l` y deja `atest`
> como texto literal. El resultado es una imagen llamada
> `.../resumidor/runneratest`, el build termina en éxito, y el job de Vertex AI
> falla después con **`The image ... does not exist`** apuntando al nombre
> correcto. Cuesta mucho de diagnosticar porque nada falla donde se originó.
>
> Usa siempre `"${IMG}:latest"`.

> **Por qué esos flags.** En los proyectos nuevos de GCP ya no se crea
> automáticamente la cuenta de servicio de Cloud Build. Sin `--service-account`,
> `gcloud builds submit` falla con `PERMISSION_DENIED` **aunque seas `owner`
> del proyecto** — el error engaña, porque parece un problema de permisos del
> usuario. Se usa la cuenta de cómputo por defecto, a la que hay que concederle
> antes tres roles:
>
> ```bash
> SA=395466321539-compute@developer.gserviceaccount.com
> for R in roles/logging.logWriter roles/artifactregistry.writer roles/storage.objectAdmin; do
>     gcloud projects add-iam-policy-binding udea-509713 \
>         --member="serviceAccount:$SA" --role="$R" --condition=None
> done
> ```
>
> Y al usar una cuenta propia, Cloud Build necesita un bucket de logs propio:
> eso es `--default-buckets-behavior=regional-user-owned-bucket`.

El build tarda del orden de 10 minutos, casi todo descargando PyTorch con
las bibliotecas de CUDA (varios GB). La imagen resultante pesa unos 6-8 GB.

**Anotar el digest**, que es lo que hace auditable la corrida:

```bash
gcloud artifacts docker images describe $IMG:latest --format="value(image_summary.digest)"
```

### Tres trampas que costaron un build cada una

Las tres comparten el mismo patrón: **el error se manifiesta lejos de donde
se origina**. Están resueltas en el repositorio; quedan anotadas porque cada
una cuesta ~18 minutos de descubrir.

**1 · `README.md` debe copiarse a la imagen.** `pyproject.toml` lo declara en
`readme`, así que sin él hatchling falla con `OSError: Readme file does not
exist` — al final de un build de 10 minutos.

**2 · Los patrones de ignorado deben ir anclados con `/`.** Este costó tres
builds y merece detalle.

`gcloud builds submit` decide qué sube según `.gcloudignore`; si no existe, lo
deriva de `.gitignore`. Ambos usan **semántica gitignore**, en la que un
patrón sin barra interna coincide con un directorio de ese nombre a
**cualquier profundidad**. Así, tanto `data/*` en `.gitignore` como `data/` en
`.gcloudignore` excluyen `src/resumidor/data/` — el paquete llega mutilado.

El síntoma es `ModuleNotFoundError: No module named 'resumidor.data'`, y lo
desconcertante es que la rueda construida en local sí contiene el módulo:
el problema no es el empaquetado sino el contexto que se sube.

Escribe siempre `/data/`, nunca `data/`. Y **compruébalo antes de gastar 18
minutos de build**:

```bash
gcloud meta list-files-for-upload | grep "^src/"
```

Ese comando es local e instantáneo, y responde con certeza qué llega a Cloud
Build. Es la herramienta que debí usar desde el principio.

`.dockerignore` es un mecanismo **distinto**: gobierna qué ve el demonio de
Docker, no qué se sube a Cloud Build. Arreglar solo ese no tiene efecto aquí.

**3 · El paso de verificación del `Dockerfile` no es decorativo.** Importa
todos los módulos y valida las configuraciones durante la construcción, de
modo que un archivo ausente rompa el build en vez de un job remoto. Fue lo
que finalmente hizo visible el problema 2.

## Paso 5 — Probar la plomería en CPU

Antes de pedir GPU. Demuestra que el contenedor, el registro, el job y la
escritura a Cloud Storage funcionan, y **no necesita cuota de GPU**.

```bash
sed -e "s/PROYECTO/udea-509713/g" \
    -e "s/REGION/us-central1/g" \
    -e "s/BUCKET/udea-509713-resumidor/g" \
    deploy/gcp/job-cpu.yaml > /tmp/job-cpu.yaml

gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=resumidor-cpu-prueba \
    --config=/tmp/job-cpu.yaml
```

Seguimiento y resultado:

```bash
gcloud ai custom-jobs list --region=us-central1 --limit=3
gcloud storage cat gs://udea-509713-resumidor/results/truncation_bart.jsonl
```

Si esto produce dos filas de resultado, el entregable «avance verificable de
la ejecución en GCP» está cumplido.

## Paso 6 — La corrida real con GPU

Requiere cuota de GPU, que **las cuentas nuevas no traen**.

> **La cuota de Compute Engine NO es la que aplica.** Vertex AI tiene su
> propio conjunto de cuotas, separado del de Compute Engine. Mirar
> `gcloud compute regions describe` lleva a una conclusión equivocada.
>
> Las métricas correctas para un Custom Job con L4 son:
>
> | Métrica | Para qué |
> |---|---|
> | `aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_l4_gpus` | **Spot** — la que usa este proyecto |
> | `aiplatform.googleapis.com/custom_model_training_nvidia_l4_gpus` | Bajo demanda |
>
> (`custom_model_serving_*` es para endpoints de inferencia, no para jobs.)

Consultar y solicitar en la consola, filtrando por servicio *Vertex AI API*:

<https://console.cloud.google.com/iam-admin/quotas?project=udea-509713&service=aiplatform.googleapis.com>

Pedir **1 GPU** en `us-central1`, sobre la métrica *preemptible*. Con una
basta: el factorial se paraleliza por configuraciones, no dentro de una
corrida, y pedir más alarga la revisión sin acelerar nada.

**Estado comprobado (semana 6): la cuota está en 0.** Verificado enviando un
job de prueba, que fue rechazado con HTTP 429 y sin costo:

```
The following quota metrics exceed quota limits:
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_l4_gpus
```

Enviar un job es, de hecho, la forma más rápida y fiable de comprobar la
cuota: el CLI no expone `effectiveLimit` para estas métricas, y si la cuota
falta el rechazo es inmediato y gratuito.

La solicitud puede tardar entre horas y varios días, y a veces exige
convertir la cuenta de prueba en una de pago —el crédito restante sigue
aplicando—. **Conviene lanzarla cuanto antes**, aunque la Fase 3 todavía no
la necesite: es el único elemento del proyecto cuyo plazo no controlamos.

Con cuota disponible:

```bash
sed -e "s/PROYECTO/udea-509713/g" \
    -e "s/REGION/us-central1/g" \
    -e "s/BUCKET/udea-509713-resumidor/g" \
    deploy/gcp/job-gpu-l4.yaml > /tmp/job-gpu.yaml

gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=resumidor-truncation-bart \
    --config=/tmp/job-gpu.yaml
```

---

## Deuda de diseño conocida

**Las configuraciones van dentro de la imagen.** El `Dockerfile` copia
`experiments/configs/`, así que cambiar un checkpoint o un parámetro obliga a
reconstruir: 15-18 minutos por cambio. Se notó al corregir el checkpoint de
LED, que dejó la imagen desactualizada respecto al repositorio.

Alternativas para la Fase 3, cuando los cambios de configuración sean
frecuentes:

- Montar las configuraciones desde el bucket (`/gcs/BUCKET/configs/`) y pasar
  la ruta como argumento. La imagen deja de depender de ellas.
- Pasar la configuración completa como argumento serializado.

La primera es la más simple y mantiene la trazabilidad: el digest de la
imagen fija el *código*, y el objeto de Cloud Storage fija la *configuración*.
Ambos quedan registrados en el resultado.

---

## Costo estimado

| Concepto | Estimación |
|---|---|
| Prueba en CPU (paso 5) | céntimos |
| Una celda de 300 documentos en L4 | ~1–2 h · ~1–2 USD |
| Las 8 configuraciones, con repeticiones | ~10 USD |
| Artifact Registry (imagen de ~6 GB) | ~0,70 USD/mes |
| Cloud Storage (resultados) | despreciable |

Son estimaciones de orden de magnitud: confírmalas en la calculadora de GCP
antes de comprometerte.

## Limpieza al terminar el semestre

```bash
gcloud artifacts repositories delete resumidor --location=us-central1
gcloud storage rm -r gs://udea-509713-resumidor
```
