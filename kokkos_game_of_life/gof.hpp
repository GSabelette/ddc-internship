#include <Kokkos_Core.hpp>
#include <string>
#include <cstdio>
#include <pthread.h>
#include <cstdlib>

#ifndef CUDA_DEVICE
#define CUDA_DEVICE 0
#endif

template <typename MemSpace, typename Layout>
struct Cell_Arrays_Naive {
    using ExecSpace = typename MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

    Kokkos::View<bool**, Layout, MemSpace> status;
    typename Kokkos::View<bool**, Layout, MemSpace>::HostMirror h_status;
    Kokkos::View<int**,  Layout, MemSpace> alive_neighbors;

    int* nb_alive;
    int* dev_nb_alive;

    Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace> gamerules; 
    typename Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace>::HostMirror h_gamerules;

    int N;
    int M;

    Cell_Arrays_Naive(int n, int m, int to_dead_low, int to_dead_high, int to_live) : N(n), M(m), status("Status", n, m), alive_neighbors("Neighbors", n, m), gamerules("Game rules")
    {
        h_status = Kokkos::create_mirror_view(status);

        nb_alive = (int*)malloc(sizeof(int));
        *nb_alive = 0;
        init_random();

        #if defined (__CUDA_ARCH__)
        cudaMalloc((void**)&dev_nb_alive, sizeof(int));
        cudaMemcpy(dev_nb_alive, nb_alive, sizeof(int), cudaMemcpyHostToDevice);
        #else
        dev_nb_alive = (int*)malloc(sizeof(int));
        *dev_nb_alive = *nb_alive;
        #endif

        init_gamerules(to_dead_low, to_dead_high, to_live); 
    }

    void init_gamerules(int to_dead_low, int to_dead_high, int to_live) {
        h_gamerules = Kokkos::create_mirror_view(gamerules);
        h_gamerules(0) = to_dead_low;
        h_gamerules(1) = to_dead_high;
        h_gamerules(2) = to_live;
        Kokkos::deep_copy(gamerules, h_gamerules);
    }

    void update_neigh() {
        // Borders
        // Update corners
        Kokkos::parallel_for("Corners", range_policy(0,1), KOKKOS_CLASS_LAMBDA(int i){
            alive_neighbors(0, 0) = status(0, 1) + status(1, 0) + status(1, 1);
            alive_neighbors(0, M-1) = status(0, M-2) + status(1, M-1) + status(1, M-2);
            alive_neighbors(N-1, 0) = status(N-2, 0) + status(N-1, 1) + status(N-2, 1);
            alive_neighbors(N-1, M-1) = status(N-2, M-1) + status(N-1, M-2) + status(N-2, M-2);
        });
        // First and last Lines excluding corners
        Kokkos::parallel_for("First and last lines", range_policy(1,M-1), KOKKOS_CLASS_LAMBDA(int j) {
            alive_neighbors(0, j) = status(0, j-1) + status(0, j+1) + status (1, j-1) + status(1, j) + status (1, j+1);
            alive_neighbors(N-1, j) = status(N-1, j-1) + status(N-1, j+1) + status (N-2, j-1) + status(N-2, j) + status (N-2, j+1);
        });
        // Remaining borders
        Kokkos::parallel_for("Remaining borders", range_policy(1,N-1), KOKKOS_CLASS_LAMBDA(int i) {
            alive_neighbors(i, 0) = status(i-1, 0) + status(i+1, 0) + status(i-1, 1) + status(i, 1) + status(i+1, 1);
            alive_neighbors(i, M-1) = status(i-1, M-1) + status(i+1, M-1) + status(i-1, M-2) + status(i, M-2) + status(i+1, M-2);
        });
        // Inner
        Kokkos::parallel_for("Inner for", mdrange_policy({1,1},{N-1, M-1}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j) {
            int cur_neighbors = 0;
            for (int k = i-1; k < i+2; ++k) {
                for (int t = j-1; t < j+2; ++t) {
                    cur_neighbors += status(k,t);
                }
            }
            if (status(i, j)) cur_neighbors--;
            alive_neighbors(i, j) = cur_neighbors;
        });
    } 
    
    void update_status() {
        Kokkos::parallel_for("Status", mdrange_policy({0,0}, {N, M}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j) {
            if (status(i, j) && (alive_neighbors(i, j) < gamerules(0) || alive_neighbors(i, j) > gamerules(1))) {
                status(i, j) = false;
                #if defined(__CUDA_ARCH__) 
                atomicAdd(dev_nb_alive, -1);
                #else 
                __atomic_fetch_sub(dev_nb_alive, 1, __ATOMIC_SEQ_CST);
                #endif
            }
            else if (!status(i, j) && alive_neighbors(i, j) == gamerules(2)) {
                status(i, j) = true;
                #if defined(__CUDA_ARCH__)
                atomicAdd(dev_nb_alive, 1);
                #else 
                __atomic_fetch_add(dev_nb_alive, 1, __ATOMIC_SEQ_CST);
                #endif
            }
        }); 
    }

    void update_steps(int nrepeat) {
        for (int repeat = 0; repeat < nrepeat; repeat++) {
            update_neigh();
            update_status();
        }
    }

    void init_blinker() {
        if (N % 2 != 1 || M % 2 != 1) {
            std::cout << "Could not init a blinker : Dimensions are not odd\n";
            return;
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (i == (N/2) && (j > 0 && j < M-1)) {h_status(i, j) = true; *nb_alive = *nb_alive + 1;}
                else h_status(i, j) = false; 
            }
        }       
        Kokkos::deep_copy(status, h_status); 
    }

    void init_random() {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (rand() % 2 == 0) {h_status(i, j) = true; *nb_alive = *nb_alive + 1;}
                else h_status(i, j) = false;
            }
        }
        Kokkos::deep_copy(status, h_status); 
    }

    void sync_host() {
        Kokkos::deep_copy(h_status, status);
        #if defined(__CUDA_ARCH__)
        cudaMemcpy(nb_alive, dev_nb_alive, sizeof(int), cudaMemcpyDeviceToHost);
        #else
        *nb_alive = *dev_nb_alive;
        #endif
    }

    void print_status() {
        std::cout << "Number of alive cells : " << *nb_alive << "\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) std::cout << h_status(i, j);
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

template <typename MemSpace, typename Layout>
struct Cell_Arrays_Portable {
    using ExecSpace = typename MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

    Kokkos::View<bool**, Layout, MemSpace> status;
    typename Kokkos::View<bool**, Layout, MemSpace>::HostMirror h_status;
    Kokkos::View<int**,  Layout, MemSpace> alive_neighbors;

    Kokkos::View<int, MemSpace>  nb_alive;
    typename Kokkos::View<int, MemSpace>::HostMirror h_nb_alive; 

    Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace> gamerules; 
    typename Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace>::HostMirror h_gamerules;

    int N;
    int M;

    Cell_Arrays_Portable(int n, int m, int to_dead_low, int to_dead_high, int to_live) : N(n), M(m), status("Status", n, m), alive_neighbors("Neighbors", n, m), nb_alive("Nb Alive"), gamerules("Game rules")
    {
        h_status = Kokkos::create_mirror_view(status);
        h_nb_alive = Kokkos::create_mirror_view(nb_alive);
        h_nb_alive() = 0;

        init_random();

        init_gamerules(to_dead_low, to_dead_high, to_live); 
    }

    void init_gamerules(int to_dead_low, int to_dead_high, int to_live) {
        h_gamerules = Kokkos::create_mirror_view(gamerules);
        h_gamerules(0) = to_dead_low;
        h_gamerules(1) = to_dead_high;
        h_gamerules(2) = to_live;
        Kokkos::deep_copy(gamerules, h_gamerules);
    }

    KOKKOS_INLINE_FUNCTION
    void single_update_neigh(int i, int j) {
        int cur_neighbors = 0;
        for (int k = i-1; k < i+2; ++k) {
            for (int t = j-1; t < j+2; ++t) {
                cur_neighbors += status(k,t);
            }
        }
        if (status(i, j)) cur_neighbors--;
        alive_neighbors(i, j) = cur_neighbors;
    }

    void update_neigh() {
        // Borders
        // Update corners
        Kokkos::parallel_for("Corners", range_policy(0,1), KOKKOS_CLASS_LAMBDA(int i){
            alive_neighbors(0, 0) = status(0, 1) + status(1, 0) + status(1, 1);
            alive_neighbors(0, M-1) = status(0, M-2) + status(1, M-1) + status(1, M-2);
            alive_neighbors(N-1, 0) = status(N-2, 0) + status(N-1, 1) + status(N-2, 1);
            alive_neighbors(N-1, M-1) = status(N-2, M-1) + status(N-1, M-2) + status(N-2, M-2);
        });
        // First and last Lines excluding corners
        Kokkos::parallel_for("First and last lines", range_policy(1,M-1), KOKKOS_CLASS_LAMBDA(int j) {
            alive_neighbors(0, j) = status(0, j-1) + status(0, j+1) + status (1, j-1) + status(1, j) + status (1, j+1);
            alive_neighbors(N-1, j) = status(N-1, j-1) + status(N-1, j+1) + status (N-2, j-1) + status(N-2, j) + status (N-2, j+1);
        });
        // Remaining borders
        Kokkos::parallel_for("Remaining borders", range_policy(1,N-1), KOKKOS_CLASS_LAMBDA(int i) {
            alive_neighbors(i, 0) = status(i-1, 0) + status(i+1, 0) + status(i-1, 1) + status(i, 1) + status(i+1, 1);
            alive_neighbors(i, M-1) = status(i-1, M-1) + status(i+1, M-1) + status(i-1, M-2) + status(i, M-2) + status(i+1, M-2);
        });
        // Inner
        Kokkos::parallel_for("Inner for", mdrange_policy({1,1},{N-1, M-1}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j) {
            int cur_neighbors = 0;
            for (int k = i-1; k < i+2; ++k) {
                for (int t = j-1; t < j+2; ++t) {
                    cur_neighbors += status(k,t);
                }
            }
            if (status(i, j)) cur_neighbors--;
            alive_neighbors(i, j) = cur_neighbors;
        });
    } 

    void update_status() {
        Kokkos::parallel_for("Status", mdrange_policy({0,0}, {N, M}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j) {
            if (status(i, j) && (alive_neighbors(i, j) < gamerules(0) || alive_neighbors(i, j) > gamerules(1))) {
                status(i, j) = false;
                Kokkos::atomic_add(&nb_alive(), -1);
            }
            else if (!status(i, j) && alive_neighbors(i, j) == gamerules(2)) {
                status(i, j) = true;
                Kokkos::atomic_add(&nb_alive(), 1);
            }
        }); 
    }

    void update_steps(int nrepeat) {
        for (int repeat = 0; repeat < nrepeat; repeat++) {
            update_neigh();
            update_status();
        }
    }

    void init_blinker() {
        if (N % 2 != 1 || M % 2 != 1) {
            std::cout << "Could not init a blinker : Dimensions are not odd\n";
            return;
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (i == (N/2) && (j > 0 && j < M-1)) {h_status(i, j) = true; h_nb_alive() += 1;}
                else h_status(i, j) = false; 
            }
        }       
        Kokkos::deep_copy(status, h_status); 
        Kokkos::deep_copy(nb_alive, h_nb_alive);
    }

    void init_random() {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (rand() % 2 == 0) {h_status(i, j) = true; h_nb_alive() += 1;}
                else h_status(i, j) = false;
            }
        }
        Kokkos::deep_copy(status, h_status); 
        Kokkos::deep_copy(nb_alive, h_nb_alive);
    }

    void sync_host() {
        Kokkos::deep_copy(h_status, status);
        Kokkos::deep_copy(h_nb_alive, nb_alive);
    }

    void print_status() {
        std::cout << "Number of alive cells : " << h_nb_alive() << "\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) std::cout << h_status(i, j);
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    double get_nb_alive() {
        double tot_alive = 0;
        Kokkos::parallel_reduce("Nb Alive", mdrange_policy({0,0}, {N,M}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j, double& update) {
            update += status(i, j);
        }, tot_alive);
        return tot_alive;
    }
};

template <typename MemSpace, typename Layout>
struct Cell_Arrays_No_Ghost {
    using ExecSpace = typename MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

    Kokkos::View<bool**, Layout, MemSpace> status;
    typename Kokkos::View<bool**, Layout, MemSpace>::HostMirror h_status;
    Kokkos::View<int**,  Layout, MemSpace> alive_neighbors;

    double nb_alive = 0;

    Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace> gamerules; 
    typename Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace>::HostMirror h_gamerules;

    int N;
    int M;

    Cell_Arrays_No_Ghost(int n, int m, int to_dead_low, int to_dead_high, int to_live) : N(n), M(m), status("Status", n, m), alive_neighbors("Neighbors", n, m), gamerules("Game rules")
    {
        h_status = Kokkos::create_mirror_view(status);
        init_random();

        init_gamerules(to_dead_low, to_dead_high, to_live); 
    }

    void init_gamerules(int to_dead_low, int to_dead_high, int to_live) {
        h_gamerules = Kokkos::create_mirror_view(gamerules);
        h_gamerules(0) = to_dead_low;
        h_gamerules(1) = to_dead_high;
        h_gamerules(2) = to_live;
        Kokkos::deep_copy(gamerules, h_gamerules);
    }

    void update_neigh() {
        // Borders
        // Update corners
        Kokkos::parallel_for("Corners", range_policy(0,1), KOKKOS_CLASS_LAMBDA(int i){
            alive_neighbors(0, 0) = status(0, 1) + status(1, 0) + status(1, 1);
            alive_neighbors(0, M-1) = status(0, M-2) + status(1, M-1) + status(1, M-2);
            alive_neighbors(N-1, 0) = status(N-2, 0) + status(N-1, 1) + status(N-2, 1);
            alive_neighbors(N-1, M-1) = status(N-2, M-1) + status(N-1, M-2) + status(N-2, M-2);
        });
        // First and last Lines excluding corners
        Kokkos::parallel_for("First and last lines", range_policy(1,M-1), KOKKOS_CLASS_LAMBDA(int j) {
            alive_neighbors(0, j) = status(0, j-1) + status(0, j+1) + status (1, j-1) + status(1, j) + status (1, j+1);
            alive_neighbors(N-1, j) = status(N-1, j-1) + status(N-1, j+1) + status (N-2, j-1) + status(N-2, j) + status (N-2, j+1);
        });
        // Remaining borders
        Kokkos::parallel_for("Remaining borders", range_policy(1,N-1), KOKKOS_CLASS_LAMBDA(int i) {
            alive_neighbors(i, 0) = status(i-1, 0) + status(i+1, 0) + status(i-1, 1) + status(i, 1) + status(i+1, 1);
            alive_neighbors(i, M-1) = status(i-1, M-1) + status(i+1, M-1) + status(i-1, M-2) + status(i, M-2) + status(i+1, M-2);
        });
        // Inner
        Kokkos::parallel_for("Inner for", mdrange_policy({1,1},{N-1, M-1}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j) {
            int cur_neighbors = 0;
            for (int k = i-1; k < i+2; ++k) {
                for (int t = j-1; t < j+2; ++t) {
                    cur_neighbors += status(k,t);
                }
            }
            if (status(i, j)) cur_neighbors--;
            alive_neighbors(i, j) = cur_neighbors;
        });
    }  

    void update_status() {
        nb_alive = 0;
        Kokkos::parallel_reduce("Nb Alive", mdrange_policy({0,0}, {N,M}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j, double& update) {
            if (status(i, j) && (alive_neighbors(i, j) < gamerules(0) || alive_neighbors(i, j) > gamerules(1))) 
                status(i, j) = false;
                     
            else if (!status(i, j) && alive_neighbors(i, j) == gamerules(2)) 
                status(i, j) = true;
            
            update += status(i, j);
        }, nb_alive);
    }

    void update_steps(int nrepeat) {
        for (int repeat = 0; repeat < nrepeat; repeat++) {
            update_neigh();
            update_status(); 
            // // Every 10% of total steps, sync host and print current nb of alive cells
            // if ((repeat+1)%(nrepeat/10) == 0) {
            //     sync_host();
            //     std::cout << "Step " << nrepeat + 1 << " : " << h_nb_alive() << " alive cells\n";
            // }
        }
    }

    void init_blinker() {
        if (N % 2 != 1 || M % 2 != 1) {
            std::cout << "Could not init a blinker : Dimensions are not odd\n";
            return;
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (i == (N/2) && (j > 0 && j < M-1)) {h_status(i, j) = true; nb_alive += 1;}
                else h_status(i, j) = false; 
            }
        }       
        Kokkos::deep_copy(status, h_status); 
    }

    void init_random() {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (rand() % 2 == 0) h_status(i, j) = true;
                else h_status(i, j) = false;
            }
        }
        Kokkos::deep_copy(status, h_status); 
    }

    void sync_host() {
        Kokkos::deep_copy(h_status, status);
    }

    void print_status() {
        std::cout << "Number of alive cells : " << nb_alive << "\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) std::cout << h_status(i, j);
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

template <typename MemSpace, typename Layout>
struct Cell_Arrays_Final {
    using ExecSpace = typename MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

    Kokkos::View<bool**, Layout, MemSpace> status;
    typename Kokkos::View<bool**, Layout, MemSpace>::HostMirror h_status;
    Kokkos::View<int**,  Layout, MemSpace> alive_neighbors;

    double nb_alive = 0;

    Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace> gamerules; 
    typename Kokkos::View<int[3], Kokkos::LayoutRight, MemSpace>::HostMirror h_gamerules;

    int N;
    int M;

    Cell_Arrays_Final(int n, int m, int to_dead_low, int to_dead_high, int to_live) : N(n+2), M(m+2), status("Status", n+2, m+2), alive_neighbors("Neighbors", n+2, m+2), gamerules("Game rules")
    {
        h_status = Kokkos::create_mirror_view(status);
        init_blinker();
        init_gamerules(to_dead_low, to_dead_high, to_live); 
    }

    void init_gamerules(int to_dead_low, int to_dead_high, int to_live) {
        h_gamerules = Kokkos::create_mirror_view(gamerules);
        h_gamerules(0) = to_dead_low;
        h_gamerules(1) = to_dead_high;
        h_gamerules(2) = to_live;
        Kokkos::deep_copy(gamerules, h_gamerules);
    }

    void update_neigh() {
        // Inner
        Kokkos::parallel_for("Inner for", mdrange_policy({1,1},{N-1, M-1}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j) {
            int cur_neighbors = 0;
            for (int k = i-1; k < i+2; ++k) {
                for (int t = j-1; t < j+2; ++t) {
                    cur_neighbors += status(k,t);
                }
            }
            if (status(i, j)) cur_neighbors--;
            alive_neighbors(i, j) = cur_neighbors;
        });
    } 

    void update_status() {
        nb_alive = 0;
        Kokkos::parallel_reduce("Nb Alive", mdrange_policy({1,1}, {N-1,M-1}), KOKKOS_CLASS_LAMBDA(int64_t i, int64_t j, double& update) {
            if (status(i, j) && (alive_neighbors(i, j) < gamerules(0) || alive_neighbors(i, j) > gamerules(1))) 
                status(i, j) = false;
                     
            else if (!status(i, j) && alive_neighbors(i, j) == gamerules(2)) 
                status(i, j) = true;
            
            update += status(i, j);
        }, nb_alive);
    }

    void update_steps(int nrepeat) {
        for (int repeat = 0; repeat < nrepeat; repeat++) {
            update_neigh();
            update_status(); 
            // // Every 10% of total steps, sync host and print current nb of alive cells
            // if ((repeat+1)%(nrepeat/10) == 0) {
            //     sync_host();
            //     std::cout << "Step " << nrepeat + 1 << " : " << h_nb_alive() << " alive cells\n";
            // }
        }
    }

    void init_blinker() {
        if (N % 2 != 1 || M % 2 != 1) {
            std::cout << "Could not init a blinker : Dimensions are not odd\n";
            return;
        }

        // Initialize ghost zone
        // Corners
        h_status(0, 0) = false;
        h_status(0, M-1) = false;
        h_status(N-1, 0) = false;
        h_status(N-1, M-1) = false;
        // First and last Lines excluding corners
        for (int j = 1; j < M-1; ++j) {
            h_status(0, j) = false;
            h_status(N-1, j) = false;
        }
        // Remaining borders
        for (int i  = 1; i < N-1; ++i) {
            h_status(i, 0) = false;
            h_status(i, M-1) = false;
        }

        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                if (i == (N/2) && (j > 1 && j < M-2)) {h_status(i, j) = true; nb_alive += 1;}
                else h_status(i, j) = false; 
            }
        }   
        Kokkos::deep_copy(status, h_status); 
    }

    void init_random() {
        // Initialize ghost zone
        // Corners
        h_status(0, 0) = false;
        h_status(0, M-1) = false;
        h_status(N-1, 0) = false;
        h_status(N-1, M-1) = false;
        // First and last Lines excluding corners
        for (int j = 1; j < M-1; ++j) {
            h_status(0, j) = false;
            h_status(N-1, j) = false;
        }
        // Remaining borders
        for (int i  = 1; i < N-1; ++i) {
            h_status(i, 0) = false;
            h_status(i, M-1) = false;
        }

        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                if (rand() % 2 == 0) h_status(i, j) = true;
                else h_status(i, j) = false;
            }
        }
        Kokkos::deep_copy(status, h_status); 
    }

    void sync_host() {
        Kokkos::deep_copy(h_status, status);
    }

    void print_status() {
        std::cout << "Number of alive cells : " << nb_alive << "\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) std::cout << h_status(i, j);
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};
