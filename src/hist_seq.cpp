/**
 * CE-4302 — Taller 01
 * Secuencial (“baseline”) con CLI
 *
 * Qué hace
 * - Lee parámetros por línea de comandos: --n, --min, --max, --seed, --rep, --csv-header
 * - Genera N enteros uniformes en [min, max] (mismo RNG siempre que la semilla sea la misma)
 * - Construye el histograma (cuántas veces aparece cada valor del rango)
 * - Imprime una línea CSV por repetición con los tiempos: gen_ms, hist_ms y total_ms
 *
 * Formato de salida (una línea por repetición):
 *   backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist
 *   - backend = "seq"   (es la versión secuencial)
 *   - variant = "baseline"
 *   - threads = 1       (sin paralelismo)
 *   - sum_hist debe ser igual a N (sanity check de que el histograma está bien)
 *
 * Ejemplo:
 *   ./bin/hist_seq --csv-header
 *   ./bin/hist_seq --n 5000000 --min 0 --max 255 --seed 42
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <cstdint>

using namespace std;

/* --------------------------- Parámetros de entrada --------------------------- */
struct Args {
    uint64_t N = 1'000'000;  // cuántos datos genero
    int minv = 0;            // valor mínimo (incl.)
    int maxv = 255;          // valor máximo (incl.)
    unsigned seed = 12345u;  // semilla del RNG (para reproducir resultados)
    int rep = 1;             // cuántas veces repito la corrida (una línea CSV por vez)
    bool csv_header = false; // si viene, solo imprime el encabezado CSV y termina
};

/* ------------------------------ Helpers de CLI ------------------------------ */
// El flag 'name' aparece tal cual en argv?
static bool has_flag(int argc, char** argv, const string& name){
    for (int i = 0; i < argc; i++) if (name == argv[i]) return true;
    return false;
}
// Obtiene string posterior a 'name', o 'def' si no está
static string get_str(int argc, char** argv, const string& name, const string& def){
    for (int i = 0; i < argc - 1; i++) if (name == argv[i]) return string(argv[i + 1]);
    return def;
}
// Obtiene long long posterior a 'name', o 'def' si no está
static long long get_ll(int argc, char** argv, const string& name, long long def){
    for (int i = 0; i < argc - 1; i++) if (name == argv[i]) return stoll(argv[i + 1]);
    return def;
}
// Obtiene int posterior a 'name', o 'def' si no está
static int get_int(int argc, char** argv, const string& name, int def){
    for (int i = 0; i < argc - 1; i++) if (name == argv[i]) return stoi(argv[i + 1]);
    return def;
}

/* ---------------------------------- Main ----------------------------------- */
int main(int argc, char** argv){
    ios::sync_with_stdio(false); // pequeño speed-up: evita sync con stdio de C

    // 1) Parseo básico de argumentos
    Args a;
    a.csv_header = has_flag(argc, argv, "--csv-header");
    if (a.csv_header) {
        // Solo imprimo el encabezado del CSV y salgo, para concatenar luego.
        cout << "backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist\n";
        return 0;
    }
    a.N    = (uint64_t)get_ll(argc, argv, "--n",   (long long)a.N);
    a.minv =             get_int(argc, argv, "--min", a.minv);
    a.maxv =             get_int(argc, argv, "--max", a.maxv);
    a.seed = (unsigned)  get_int(argc, argv, "--seed", (int)a.seed);
    a.rep  =             get_int(argc, argv, "--rep",  a.rep);

    // Si se pasó min y max al revés, lo arreglamos para no fallar
    if (a.maxv < a.minv) swap(a.maxv, a.minv);
    const int bins = a.maxv - a.minv + 1; // cantidad de celdas del histograma

    // 2) Podemos repetir la corrida 'rep' veces (sirve para tomar medianas luego)
    for (int r = 0; r < a.rep; ++r) {

        // 2.1) Generación de datos (secuencial, sin hilos)
        vector<int> data(a.N);
        auto t0g = chrono::steady_clock::now();

        // RNG de 64 bits. Truco: desfaso la semilla con el índice de repetición
        // para que cada repetición tenga su propia secuencia, pero siga siendo reproducible.
        mt19937_64 rng(a.seed + (unsigned)r * 17u);
        uniform_int_distribution<int> dist(a.minv, a.maxv);

        // Lleno el vector con valores uniformes [min, max]
        for (uint64_t i = 0; i < a.N; ++i) data[i] = dist(rng);

        auto t1g = chrono::steady_clock::now();

        // 2.2) Construcción del histograma (conteo por valor)
        auto t0h = chrono::steady_clock::now();

        // hist[b] cuenta cuántas veces aparece (minv + b)
        vector<uint64_t> hist(bins, 0);
        for (uint64_t i = 0; i < a.N; ++i) {
            const int idx = data[i] - a.minv; // mapeo a [0, bins-1]
            ++hist[idx];
        }

        auto t1h = chrono::steady_clock::now();

        // 2.3) Tiempos y chequeo final
        const uint64_t sum_hist = accumulate(hist.begin(), hist.end(), uint64_t{0});
        const auto gen_ms  = chrono::duration_cast<chrono::milliseconds>(t1g - t0g).count();
        const auto hist_ms = chrono::duration_cast<chrono::milliseconds>(t1h - t0h).count();
        const auto tot_ms  = gen_ms + hist_ms;

        // sum_hist debe ser exactamente N. Si no, algo se cromo
        cout << "seq" << "," << "baseline" << "," << 1 << ","
             << a.N << "," << bins << "," << a.minv << "," << a.maxv << "," << a.seed << ","
             << gen_ms << "," << hist_ms << "," << tot_ms << "," << sum_hist << "\n";
    }
    return 0;
}