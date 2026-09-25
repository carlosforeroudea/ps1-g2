#!/usr/bin/env bash
# Estado de un vistazo: experimentos locales, build en la nube y jobs de GCP.
#
#   make estado            una foto
#   make estado-seguir     refresca cada 20 s
#
# No arranca nada ni cambia nada: solo consulta.

set -uo pipefail

PROYECTO="${PROYECTO:-udea-509713}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-udea-509713-resumidor}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

neg=$'\033[1m'; verde=$'\033[32m'; ama=$'\033[33m'; rojo=$'\033[31m'; fin=$'\033[0m'

titulo() { printf '\n%s── %s %s\n' "$neg" "$1" "$fin"; }

# ---------------------------------------------------------------- local
titulo "EXPERIMENTOS LOCALES"
resultados="$REPO/experiments/results"
if compgen -G "$resultados/*.jsonl" > /dev/null; then
    for f in "$resultados"/*.jsonl; do
        n=$(wc -l < "$f" | tr -d ' ')
        nombre=$(basename "$f" .jsonl)
        # La meta viene del propio YAML, no de un número escrito a mano.
        # awk toma el segundo campo: `tr -dc '0-9'` se tragaba también los
        # dígitos del comentario de la misma línea (ADR-000, n>=300).
        meta=$(awk '/^[[:space:]]+n:[[:space:]]/ {print $2; exit}' \
               "$REPO/experiments/configs/$nombre.yaml" 2>/dev/null)
        printf '  %-24s %3s filas  (configuración: n=%s)\n' "$nombre" "$n" "${meta:-?}"
    done
else
    printf '  %sSin resultados todavía%s\n' "$ama" "$fin"
fi

if pgrep -f "scripts/run_experiment.py" > /dev/null 2>&1; then
    transcurrido=$(ps -o etime= -p "$(pgrep -f 'scripts/run_experiment.py' | tail -1)" | tr -d ' ')
    printf '  %scorriendo%s  (%s)\n' "$verde" "$fin" "$transcurrido"
else
    printf '  %ssin procesos activos%s\n' "$ama" "$fin"
fi

# ------------------------------------------------------------------ gcp
if ! command -v gcloud > /dev/null 2>&1; then
    printf '\n  gcloud no está instalado; se omite la parte de GCP.\n'
    exit 0
fi

cuenta=$(gcloud config get-value account 2>/dev/null)
proyecto=$(gcloud config get-value project 2>/dev/null)
printf '\n  cuenta: %s · proyecto: %s\n' "$cuenta" "$proyecto"

if [[ "$proyecto" != "$PROYECTO" ]]; then
    printf '  %sAVISO: el proyecto activo no es %s.%s\n' "$rojo" "$PROYECTO" "$fin"
    printf '  Ejecuta: gcloud config configurations activate udea\n'
fi

titulo "CLOUD BUILD (últimos 3)"
gcloud builds list --region="$REGION" --limit=3 \
    --format="value(id,status,duration,createTime.date('%H:%M'))" 2>/dev/null \
    | awk -F'\t' '{printf "  %-10s %-10s %-8s %s\n", substr($1,1,8), $2, $3, $4}' \
    || printf '  (sin builds)\n'

titulo "IMAGEN PUBLICADA"
# `slice()` no aplica a cadenas en gcloud: el digest se recorta en shell.
img=$(gcloud artifacts docker images list "$REGION-docker.pkg.dev/$PROYECTO/resumidor" \
      --format="value(version,createTime.date('%Y-%m-%d %H:%M'))" 2>/dev/null \
      | tail -1 | awk -F'\t' '{printf "%s  (%s)", substr($1,1,19), $2}')
if [[ -n "$img" ]]; then
    printf '  %s%s%s\n' "$verde" "$img" "$fin"
else
    printf '  %saún no hay imagen publicada%s\n' "$ama" "$fin"
fi

titulo "JOBS DE VERTEX AI (últimos 3)"
gcloud ai custom-jobs list --region="$REGION" --limit=3 \
    --format="table[no-heading](displayName,state,createTime.date('%H:%M'))" \
    2>/dev/null | sed 's/^/  /' || printf '  (sin jobs)\n'

titulo "RESULTADOS EN CLOUD STORAGE"
gcloud storage ls -l "gs://$BUCKET/results/**" 2>/dev/null | sed 's/^/  /' \
    || printf '  %ssin resultados en el bucket todavía%s\n' "$ama" "$fin"

cat <<EOF

${neg}Consolas${fin}
  Builds   https://console.cloud.google.com/cloud-build/builds;region=$REGION?project=$PROYECTO
  Jobs     https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROYECTO
  Costo    https://console.cloud.google.com/billing
EOF
