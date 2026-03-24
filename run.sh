#!/bin/bash
set -e

if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install -q -r requirements.txt --disable-pip-version-check

FASTUTIL_VER="8.2.2"
FASTUTIL_JAR="fastutil-${FASTUTIL_VER}.jar"
if [ ! -f "$FASTUTIL_JAR" ]; then
    echo "[*] Downloading ${FASTUTIL_JAR}..."
    curl -sO "https://repo1.maven.org/maven2/it/unimi/dsi/fastutil/${FASTUTIL_VER}/${FASTUTIL_JAR}"
fi

TYPE=""
for arg in "$@"; do
  case $arg in
    --type=*)
      TYPE="${arg#*=}"
      shift
      ;;
    --type)
      TYPE="$2"
      shift 2
      ;;
  esac
done

case $TYPE in
    compare)
        python3 -m scripts.benchmarks.compare "$@"
        ;;
    sweep)
        python3 -m scripts.benchmarks.parameter_sweep "$@"
        ;;
    lhs)
        python3 -m scripts.benchmarks.latin_hypercube "$@"
        ;;
    bayesian)
      python3 -m scripts.benchmarks.bayesian_opt "$@"
        ;;
    metadata)
          python3 -m scripts.dataset_metadata "$@"
            ;;
    *)
        echo "Usage: ./run.sh --type {compare|sweep|latin} [options]"
        echo "Example: ./run.sh --type compare --methods mags --group one"
        exit 1
        ;;
esac