/**
 * CE-4302 — Taller 01
 * std::thread: 3 formas de hacer el histograma en paralelo
 *
 * Qué hace:
 * - Genera N enteros en [min, max] en paralelo usando hilos de C++.
 * - Calcula el histograma con tres variantes para comparar contención:
 *   1) private: cada hilo cuenta en su propio arreglo local y al final se suman todos.
 *      (casi no hay contencións,  suele escalar mejor)
 *   2) atomic : todos actualizan un arreglo global, pero con incrementos atómicos.
 *      (hay contención moderada si muchos caen en el mismo bin)
 *   3) mutex  : arreglo global protegido con un mutex por bin.
 *      (la más pesada en choque, útil para ver el peor caso)
 *
 * Salida (una línea por corrida)
 *   backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist
 *   - backend = "threads"
 *   - variant = "private" | "atomic" | "mutex"
 *   - threads = # de hilos usados (--threads)
 */

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <cstdint>
#include <atomic>
#include <mutex>

using namespace std;

/* ---------------------- Parámetros de ejecución y CLI ---------------------- */
struct Args {
    uint64_t N = 10'000'000;      // cuántos datos generamos
    int minv = 0, maxv = 255;     // rango inclusivo [minv, maxv]
    unsigned seed = 12345u;       // semilla base (para reproducir)
    int rep = 1;                  // cuántas veces repetimos (líneas CSV)
    string variant = "private";   // private | atomic | mutex
    int threads = (int)thread::hardware_concurrency(); // si 0, lo corregimos abajo
};

// Helpers de CLI sencillitos (sin libs externas)
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

/* ---------------------------------- Main ---------------------------------- */
int main(int argc, char** argv){
    ios::sync_with_stdio(false);

    // Si piden el encabezado CSV, lo imprimimos y jalamos
    if (has_flag(argc, argv, "--csv-header")){
        cout << "backend,variant,threads,N,bins,min,max,seed,gen_ms,hist_ms,total_ms,sum_hist\n";
        return 0;
    }

    // Parseo de argumentos con defaults
    Args a;
    a.N       = (uint64_t)get_ll(argc, argv, "--n",        (long long)a.N);
    a.minv    =            get_int(argc, argv, "--min",     a.minv);
    a.maxv    =            get_int(argc, argv, "--max",     a.maxv);
    a.seed    = (unsigned) get_int(argc, argv, "--seed",   (int)a.seed);
    a.rep     =            get_int(argc, argv, "--rep",     a.rep);
    a.variant =            get_str(argc, argv, "--variant", a.variant);
    a.threads =            get_int(argc, argv, "--threads", a.threads);

    // Arreglamos rango si vino invertido
    if (a.maxv < a.minv) swap(a.maxv, a.minv);
    const int bins = a.maxv - a.minv + 1;

    // Si hardware_concurrency() devolvió 0 o el user pasó <1, usamos un valor razonable
    if (a.threads <= 0) a.threads = 4;
    const int T = a.threads;

    // Chequeo básico de la variable
    if (a.variant != "private" && a.variant != "atomic" && a.variant != "mutex"){
        cerr << "Variant no soportada: " << a.variant << " (use: private | atomic | mutex)\n";
        return 2;
    }

    // Podemos repetir para tener más muestras (luego se puede calcular mediana)
    for (int r = 0; r < a.rep; ++r){
        /* ------------------------- 1) Generar datos ------------------------- *
         * Vamos a dividir el trabajo en bloques por hilo:
         *   - chunk = ceil(N / T)
         *   - hilo t procesa [t*chunk, min(N,(t+1)*chunk))
         * Cada hilo tiene su RNG propio (semilla base + offset por hilo + repetición).
         */
        vector<int> data(a.N);
        auto t0g = chrono::steady_clock::now();

        const uint64_t chunk = (a.N + (uint64_t)T - 1) / (uint64_t)T;
        vector<thread> ths;
        ths.reserve(T);

        for (int t = 0; t < T; ++t){
            ths.emplace_back([&, t](){
                const uint64_t start = (uint64_t)t * chunk;
                const uint64_t end   = min<uint64_t>(a.N, (uint64_t)(t+1) * chunk);

                mt19937_64 rng(a.seed + (unsigned)t * 1337u + (unsigned)r * 17u);
                uniform_int_distribution<int> dist(a.minv, a.maxv);

                for (uint64_t i = start; i < end; ++i){
                    data[i] = dist(rng);
                }
            });
        }
        for (auto& th : ths) th.join();
        auto t1g = chrono::steady_clock::now();

        /* --------------------- 2) Histograma por variante -------------------- *
         * - private: cada hilo usa un vector local y luego hacemos una reducción.
         * - atomic : un solo vector global de atomics<long long>.
         * - mutex  : un vector global con un mutex por bin (región crítica cortita).
         */
        auto t0h = chrono::steady_clock::now();
        uint64_t sum_hist = 0;

        if (a.variant == "private"){
            // locals[t] es el histograma del hilo t
            vector<vector<uint64_t>> locals(T, vector<uint64_t>(bins, 0));
            ths.clear();

            for (int t = 0; t < T; ++t){
                ths.emplace_back([&, t](){
                    const uint64_t start = (uint64_t)t * chunk;
                    const uint64_t end   = min<uint64_t>(a.N, (uint64_t)(t+1) * chunk);
                    auto& local = locals[t];
                    for (uint64_t i = start; i < end; ++i){
                        const int idx = data[i] - a.minv; // [0, bins-1]
                        ++local[idx];
                    }
                });
            }
            for (auto& th : ths) th.join();

            // Reducción final: sumamos todos los locales al global (O(bins))
            vector<uint64_t> hist_global(bins, 0);
            for (int t = 0; t < T; ++t)
                for (int b = 0; b < bins; ++b)
                    hist_global[b] += locals[t][b];

            sum_hist = accumulate(hist_global.begin(), hist_global.end(), uint64_t{0});

        } else if (a.variant == "atomic"){
            // Un único histograma global; incrementos atómicos (relaxed = rápido y suficiente aquí)
            vector<atomic<long long>> hist(bins);
            for (int b = 0; b < bins; ++b) hist[b].store(0LL, memory_order_relaxed);

            ths.clear();
            for (int t = 0; t < T; ++t){
                ths.emplace_back([&, t](){
                    const uint64_t start = (uint64_t)t * chunk;
                    const uint64_t end   = min<uint64_t>(a.N, (uint64_t)(t+1) * chunk);
                    for (uint64_t i = start; i < end; ++i){
                        const int idx = data[i] - a.minv;
                        hist[idx].fetch_add(1LL, memory_order_relaxed);
                    }
                });
            }
            for (auto& th : ths) th.join();

            for (int b = 0; b < bins; ++b)
                sum_hist += (uint64_t)hist[b].load(memory_order_relaxed);

        } else { // mutex
            // Un único histograma global; protegemos cada bin con su propio mutex
            vector<long long> hist(bins, 0);
            vector<mutex> bin_mtx(bins); // un mutex por celda

            ths.clear();
            for (int t = 0; t < T; ++t){
                ths.emplace_back([&, t](){
                    const uint64_t start = (uint64_t)t * chunk;
                    const uint64_t end   = min<uint64_t>(a.N, (uint64_t)(t+1) * chunk);
                    for (uint64_t i = start; i < end; ++i){
                        const int idx = data[i] - a.minv;
                        lock_guard<mutex> lk(bin_mtx[idx]); // sección crítica bien pequena
                        ++hist[idx];
                    }
                });
            }
            for (auto& th : ths) th.join();

            sum_hist = accumulate(hist.begin(), hist.end(), uint64_t{0});
        }

        auto t1h = chrono::steady_clock::now();

        /* ------------------- 3) Tiempos y salida CSV ------------------- *
         * gen_ms  : milisegundos generando los datos
         * hist_ms : milisegundos construyendo el histograma
         * total_ms: suma de ambos
         * sum_hist: suma de todos los bins (debería ser exactamente N)
         */
        const auto gen_ms  = chrono::duration_cast<chrono::milliseconds>(t1g - t0g).count();
        const auto hist_ms = chrono::duration_cast<chrono::milliseconds>(t1h - t0h).count();
        const auto tot_ms  = gen_ms + hist_ms;

        cout << "threads" << "," << a.variant << "," << T << ","
             << a.N << "," << bins << "," << a.minv << "," << a.maxv << "," << a.seed << ","
             << gen_ms << "," << hist_ms << "," << tot_ms << "," << sum_hist << "\n";
    }

    return 0;
}
