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

python3 -m scripts "$@"