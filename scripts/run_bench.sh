#!/usr/bin/env bash
# Ejecuta baterías y guarda CSV en results/raw + last.csv
set -euo pipefail

BACKEND="both"   # openmp|threads|both
N=5000000        # pequeño para VM (valida flujo)
MIN=0; MAX=255
REPS=1           # repeticiones por punto
THREADS=""       # si vacío: 1,2,4,... hasta nproc
SEED=42
OUTDIR="results/raw"; mkdir -p "$OUTDIR"
ts=$(date +%Y%m%d_%H%M%S)
CSV="$OUTDIR/run_${ts}.csv"

# Parseo simple
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --min) MIN="$2"; shift 2;;
    --max) MAX="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    *) echo "Arg desconocido: $1"; exit 1;;
  esac
done

echo "backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist" > "$CSV"

# Conjunto de hilos
if [[ -z "${THREADS}" ]]; then
  MAXT=$(nproc --all || echo 8)
  SET="1"
  t=2; while [[ $t -le $MAXT ]]; do SET="$SET $t"; t=$((t*2)); done
else
  SET="${THREADS}"
fi

run_openmp () {
  for variant in private atomic mutex; do
    for t in $SET; do
      for r in $(seq 1 $REPS); do
        echo "[OpenMP] var=$variant t=$t r=$r N=$N"
        OMP_NUM_THREADS=$t ./bin/hist_openmp --n $N --min $MIN --max $MAX \
          --variant $variant --seed $SEED --rep 1 >> "$CSV"
      done
    done
  done
}

run_threads () {
  for variant in private atomic mutex; do
    for t in $SET; do
      for r in $(seq 1 $REPS); do
        echo "[threads] var=$variant t=$t r=$r N=$N"
        ./bin/hist_threads --n $N --threads $t --min $MIN --max $MAX \
          --variant $variant --seed $SEED --rep 1 >> "$CSV"
      done
    done
  done
}

case "$BACKEND" in
  openmp)  run_openmp ;;
  threads) run_threads ;;
  both)    run_openmp; run_threads ;;
  *) echo "backend inválido: $BACKEND"; exit 2;;
esac

cp "$CSV" "$OUTDIR/last.csv"
echo "[✓] Resultados: $CSV  y  $OUTDIR/last.csv"
