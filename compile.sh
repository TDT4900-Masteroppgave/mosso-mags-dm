#!/bin/bash
set -e

echo "[*] Compiling Java sources..."

rm -rf class mosso-1.0.jar
mkdir -p class

FASTUTIL_JAR=$(ls fastutil-*.jar 2>/dev/null | head -n 1)

if [ -z "$FASTUTIL_JAR" ]; then
    echo "[!] Error: fastutil JAR not found in the current directory."
    exit 1
fi

javac -cp "./$FASTUTIL_JAR" -d class $(find ./src -name "*.java")

echo "[*] Creating JAR archive..."
cd class
jar cf mosso-1.0.jar ./
mv mosso-1.0.jar ../
cd ..

rm -rf class

echo "[*] Build complete: mosso-1.0.jar"