#include "gof.hpp"

#define TO_DEAD_LOW     2
#define TO_DEAD_HIGH    3
#define TO_LIVE         3
#define NREPEAT 10
#define NCHECKS 5

#ifdef VISUAL
#define BOARD_SIZE 21
#else
#define BOARD_SIZE 255
#endif


/* First template parameters have to correspond to Host devices, and second to Cuda devices*/
template <typename MS1, typename L1, typename MS2, typename L2> 
int compare_boards(Kokkos::View<bool**, L1, MS1>& host_view, typename Kokkos::View<bool**, L1, MS1>::HostMirror& host_mirror, Kokkos::View<bool**, L2, MS2>& device_view, typename Kokkos::View<bool**, L2, MS2>::HostMirror& device_mirror) {
    Kokkos::deep_copy(host_mirror, host_view);
    Kokkos::deep_copy(device_mirror, device_view);
    int differential = 0;
    Kokkos::parallel_reduce("Comparison for", Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename Kokkos::HostSpace::execution_space>({0,0}, {BOARD_SIZE, BOARD_SIZE}), 
    [=](int64_t i, int64_t j, int& update) {
        update += (host_mirror(i,j) != device_mirror(i,j));
    }, differential);
    return differential;
}

int main(int argc, char** argv) {
    Kokkos::initialize(); {
    Cell_Arrays_Portable<Kokkos::CudaSpace, Kokkos::LayoutLeft> portable_cuda(BOARD_SIZE, BOARD_SIZE, TO_DEAD_LOW, TO_DEAD_HIGH, TO_LIVE);
    Cell_Arrays_Portable<Kokkos::HostSpace, Kokkos::LayoutRight> portable_host(BOARD_SIZE, BOARD_SIZE, TO_DEAD_LOW, TO_DEAD_HIGH, TO_LIVE);
    
    // Make sure both instances have the same starting board.
    Kokkos::deep_copy(portable_cuda.h_status, portable_host.status);
    Kokkos::deep_copy(portable_cuda.status, portable_cuda.h_status);
    Kokkos::deep_copy(portable_cuda.h_nb_alive, portable_host.h_nb_alive);
    Kokkos::deep_copy(portable_cuda.nb_alive, portable_cuda.h_nb_alive);    

    #ifdef VISUAL
    portable_cuda.sync_host();
    portable_host.sync_host();
    portable_cuda.print_status();
    portable_host.print_status();
    #endif
    for (int i = 0; i < NCHECKS; ++i) {
        portable_cuda.update_steps(NREPEAT);
        portable_host.update_steps(NREPEAT);
        int diff = compare_boards<Kokkos::HostSpace, Kokkos::LayoutRight, Kokkos::CudaSpace, Kokkos::LayoutLeft>(portable_host.status, portable_host.h_status, portable_cuda.status, portable_cuda.h_status);
        std::cout << "diff : " << diff << "\n";
        #ifdef VISUAL
        portable_cuda.sync_host();
        portable_host.sync_host();
        portable_cuda.print_status();
        portable_host.print_status();
        #endif
        assert (diff == 0);
    }
    }
    Kokkos::finalize(); 
    std::cout << "Test ended with no issue\n";
}