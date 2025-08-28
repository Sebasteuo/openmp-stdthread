/**
 * CE-4302 — Taller 01
 * OpenMP: 3 formas de hacer el histograma en paralelo
 *
 * Que hace este programa:
 * - Genera N números enteros uniformes en [min, max] usando un RNG por hilo.
 * - Calcula un histograma con 3 variantes:
 *   1) private: cada hilo cuenta en su arreglo local y al final se suman todos.
 *      (casi no hay contencións entre hilos, va a ser la más rápida)
 *   2) atomic : todos escriben en el mismo arreglo global, pero con ++ atómico.
 *      (hay algo de contención, rinde menos si muchos caen en el mismo bin)
 *   3) mutex  : igual global, pero con un lock por cada bin.
 *      (la más lenta cuando hay choque, se incluye para fines de contraste)
 *
 * Salida: una línea CSV por corrida
 * Formato:
 *   backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist
 *
 * Ejemplos:
 *   OMP_NUM_THREADS=4 ./bin/hist_openmp --n 5000000 --variant private --seed 42
 *   OMP_NUM_THREADS=4 ./bin/hist_openmp --n 5000000 --variant atomic  --seed 42
 *   OMP_NUM_THREADS=4 ./bin/hist_openmp --n 5000000 --variant mutex   --seed 42
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <cstdint>
#include <omp.h>

using namespace std;

/* ---------------------- Parámetros y helpers de CLI ---------------------- */
struct Args {
    uint64_t N = 10'000'000;   // cuántos datos generamos
    int minv = 0, maxv = 255;  // rango de valores
    unsigned seed = 12345u;    // semilla base (para reproducir)
    int rep = 1;               // cuántas veces repetimos (líneas CSV)
    string variant = "private";// private | atomic | mutex
    int threads = -1;          // si >0, fija ese número de hilos
};

static bool has_flag(int argc, char** argv, const string& name){
    for (int i=0;i<argc;i++) if (name == argv[i]) return true;
    return false;
}
static long long get_ll(int argc, char** argv, const string& name, long long def){
    for (int i=0;i<argc-1;i++) if (name == argv[i]) return stoll(argv[i+1]);
    return def;
}
static int get_int(int argc, char** argv, const string& name, int def){
    for (int i=0;i<argc-1;i++) if (name == argv[i]) return stoi(argv[i+1]);
    return def;
}
static string get_str(int argc, char** argv, const string& name, string def){
    for (int i=0;i<argc-1;i++) if (name == argv[i]) return string(argv[i+1]);
    return def;
}

/* --------------------------------- Main ---------------------------------- */
int main(int argc, char** argv){
    ios::sync_with_stdio(false);

    // Si me piden el encabezado del CSV, lo imprimo y salgo
    if (has_flag(argc, argv, "--csv-header")) {
        cout << "backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist\n";
        return 0;
    }

    // Tomamos los argumentos (con valores por defecto sensatos xD).
    Args a;
    a.N       = (uint64_t)get_ll(argc, argv, "--n",        (long long)a.N);
    a.minv    =            get_int(argc, argv, "--min",     a.minv);
    a.maxv    =            get_int(argc, argv, "--max",     a.maxv);
    a.seed    = (unsigned) get_int(argc, argv, "--seed",   (int)a.seed);
    a.rep     =            get_int(argc, argv, "--rep",     a.rep);
    a.variant =            get_str(argc, argv, "--variant", a.variant);
    a.threads =            get_int(argc, argv, "--threads", a.threads);

    // Si alguien pasó min y max al revés, lo arreglamos.
    if (a.maxv < a.minv) swap(a.maxv, a.minv);
    const int bins = a.maxv - a.minv + 1;

    // Si nos dieron un #hilos fijo, lo aplicamos al runtime de OpenMP.
    if (a.threads > 0) omp_set_num_threads(a.threads);

    // Podemos repetir varias veces para tener más de una muestra.
    for (int r = 0; r < a.rep; ++r) {

        /* --------------------- (1) Generar los datos --------------------- *
         * Hacemos esto en paralelo. Cada hilo usa su propio RNG para evitar
         * contencións y que la generación sea thread-safe.
         */
        vector<int> data(a.N);
        int T = 1; // aquí guardamos cuántos hilos realmente se usaron

        auto t0g = chrono::steady_clock::now();
        #pragma omp parallel
        {
            // Guardamos T una sola vez.
            #pragma omp single
            { T = omp_get_num_threads(); }

            int tid = omp_get_thread_num();
            mt19937_64 rng(a.seed + (unsigned)tid * 1337u + (unsigned)r * 17u);
            uniform_int_distribution<int> dist(a.minv, a.maxv);

            // Repartimos las posiciones [0..N) entre los hilos (bloques estáticos).
            #pragma omp for schedule(static)
            for (long long i = 0; i < (long long)a.N; ++i) {
                data[i] = dist(rng);
            }
        }
        auto t1g = chrono::steady_clock::now();

        /* ---------------- (2) Calcular el histograma ----------------
         * Elegimos la variante según --variant:
         *  - private: cuenta local por hilo y luego sumamos todo.
         *  - atomic : todos suman en el mismo arreglo pero con operación atómica.
         *  - mutex  : igual que atomic pero usando locks por bin.
         */
        auto t0h = chrono::steady_clock::now();
        uint64_t sum_hist = 0;

        if (a.variant == "private") {
            // Arreglo global donde quedará el resultado final
            vector<long long> hist_global(bins, 0);

            #pragma omp parallel
            {
                // Cada hilo trabaja con su copia local (sin contención).
                vector<long long> hist_local(bins, 0);

                #pragma omp for schedule(static)
                for (long long i = 0; i < (long long)a.N; ++i) {
                    const int idx = data[i] - a.minv; // mapeamos al rango [0, bins-1]
                    ++hist_local[idx];
                }

                // Al final, cada hilo “deposita” su conteo local en el global.
                #pragma omp critical
                {
                    for (int b = 0; b < bins; ++b) hist_global[b] += hist_local[b];
                }
            }

            sum_hist = accumulate(hist_global.begin(), hist_global.end(), uint64_t{0});

        } else if (a.variant == "atomic") {
            // Un único histograma global. Aquí las escrituras son atómicas.
            vector<long long> hist(bins, 0);

            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)a.N; ++i) {
                const int idx = data[i] - a.minv;
                #pragma omp atomic update
                hist[idx] += 1;
            }

            sum_hist = accumulate(hist.begin(), hist.end(), uint64_t{0});

        } else if (a.variant == "mutex") {
            // Un único histograma global con un lock por cada bin.
            vector<long long> hist(bins, 0);
            vector<omp_lock_t> locks(bins);
            for (int b = 0; b < bins; ++b) omp_init_lock(&locks[b]);

            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)a.N; ++i) {
                const int idx = data[i] - a.minv;
                omp_set_lock(&locks[idx]);
                ++hist[idx];
                omp_unset_lock(&locks[idx]);
            }

            for (int b = 0; b < bins; ++b) omp_destroy_lock(&locks[b]);
            sum_hist = accumulate(hist.begin(), hist.end(), uint64_t{0});

        } else {
            cerr << "Variant no soportada: " << a.variant
                 << " (use: private | atomic | mutex)\n";
            return 2;
        }

        auto t1h = chrono::steady_clock::now();

        /* ---------------- (3) Medir tiempos y hacer el CSV ----------------
         * gen_ms  : milisegundos generando los datos
         * hist_ms : milisegundos haciendo el histograma
         * total_ms: suma de ambos
         * sum_hist: la suma de todos los bins (debe ser igual a N; sanity check)
         */
        const auto gen_ms  = chrono::duration_cast<chrono::milliseconds>(t1g - t0g).count();
        const auto hist_ms = chrono::duration_cast<chrono::milliseconds>(t1h - t0h).count();
        const auto tot_ms  = gen_ms + hist_ms;

        cout << "openmp" << "," << a.variant << "," << T << ","
             << a.N << "," << bins << "," << a.minv << "," << a.maxv << "," << a.seed << ","
             << gen_ms << "," << hist_ms << "," << tot_ms << "," << sum_hist << "\n";
    }

    return 0;
}