#include <benchmark/benchmark.h>
#include <chrono>

#include "gof.hpp"

#if CUDA_DEVICE!=0
#pragma message("Cuda Device != 0")
typedef Kokkos::CudaSpace MemSpace;
typedef Kokkos::LayoutLeft Layout;
#else 	
#pragma message("Cuda Device == 0")
typedef Kokkos::HostSpace MemSpace;
typedef Kokkos::LayoutRight Layout;
#endif

#define TO_DEAD_LOW     2
#define TO_DEAD_HIGH    3
#define TO_LIVE         3
#define NREPEATS        1
#if CUDA_DEVICE!=0
#define RANGE_MIN       256
#define RANGE_MAX       16384
#else
#define RANGE_MIN       16384
#define RANGE_MAX       16384
#endif

// static void BM_naive(benchmark::State& state) {
//     Cell_Arrays_Naive<MemSpace, Layout> cell_arrays(state.range(0)+1, state.range(0)+1, TO_DEAD_LOW, TO_DEAD_HIGH, TO_LIVE);

//     for (auto _ : state) {
//         auto start = std::chrono::high_resolution_clock::now();
//         cell_arrays.update_steps(NREPEATS);
//         Kokkos::fence();
//         auto end = std::chrono::high_resolution_clock::now();
//         auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//         state.SetIterationTime(elapsed_seconds.count());
//     }
// }
// BENCHMARK(BM_naive)->RangeMultiplier(2)->Range(RANGE_MIN, RANGE_MAX)->UseManualTime();

static void BM_portable(benchmark::State& state) {
    Cell_Arrays_Portable<MemSpace, Layout> cell_arrays(state.range(0)+1, state.range(0)+1, TO_DEAD_LOW, TO_DEAD_HIGH, TO_LIVE);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        cell_arrays.update_steps(NREPEATS);
        Kokkos::fence();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_portable)->RangeMultiplier(2)->Range(RANGE_MIN, RANGE_MAX)->UseManualTime();

static void BM_no_ghost(benchmark::State& state) {
    Cell_Arrays_No_Ghost<MemSpace, Layout> cell_arrays(state.range(0)+1, state.range(0)+1, TO_DEAD_LOW, TO_DEAD_HIGH, TO_LIVE);
 
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        cell_arrays.update_steps(NREPEATS);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_no_ghost)->RangeMultiplier(2)->Range(RANGE_MIN, RANGE_MAX)->UseManualTime();

static void BM_final(benchmark::State& state) {
    Cell_Arrays_Final<MemSpace, Layout> cell_arrays(state.range(0)+1, state.range(0)+1, TO_DEAD_LOW, TO_DEAD_HIGH, TO_LIVE);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        cell_arrays.update_steps(NREPEATS);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_final)->RangeMultiplier(2)->Range(RANGE_MIN, RANGE_MAX)->UseManualTime();


int main(int argc, char** argv) {                                     
::benchmark::Initialize(&argc, argv);                               
if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; 
Kokkos::initialize();
::benchmark::RunSpecifiedBenchmarks();               
Kokkos::finalize();               
::benchmark::Shutdown();                                            
return 0;                                                           
}          