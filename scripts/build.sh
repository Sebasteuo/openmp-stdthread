#!/usr/bin/env bash
# Compila: hist_seq, hist_openmp, hist_threads
set -euo pipefail
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:-"-O3 -std=c++17 -march=native -pipe"}

mkdir -p bin
echo "[*] Compilando seq..."
$CXX $CXXFLAGS src/hist_seq.cpp -o bin/hist_seq

echo "[*] Compilando OpenMP..."
$CXX $CXXFLAGS -fopenmp src/hist_openmp.cpp -o bin/hist_openmp

echo "[*] Compilando std::thread..."
$CXX $CXXFLAGS -pthread src/hist_threads.cpp -o bin/hist_threads

echo "[✓] Listo: bin/hist_seq, bin/hist_openmp, bin/hist_threads"
