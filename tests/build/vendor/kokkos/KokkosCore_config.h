
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error \
    "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

// KOKKOS_VERSION % 100 is the patch level
// KOKKOS_VERSION / 100 % 100 is the minor version
// KOKKOS_VERSION / 10000 is the major version
#define KOKKOS_VERSION 30500

/* Execution Spaces */
#define KOKKOS_ENABLE_SERIAL
#define KOKKOS_ENABLE_OPENMP
/* #undef KOKKOS_ENABLE_OPENMPTARGET */
/* #undef KOKKOS_ENABLE_THREADS */
#define KOKKOS_ENABLE_CUDA
/* #undef KOKKOS_ENABLE_HIP */
/* #undef KOKKOS_ENABLE_HPX */
/* #undef KOKKOS_ENABLE_MEMKIND */
/* #undef KOKKOS_ENABLE_LIBRT */
/* #undef KOKKOS_ENABLE_SYCL */

#ifndef __CUDA_ARCH__
#define KOKKOS_ENABLE_TM
#define KOKKOS_USE_ISA_X86_64
/* #undef KOKKOS_USE_ISA_KNC */
/* #undef KOKKOS_USE_ISA_POWERPCLE */
/* #undef KOKKOS_USE_ISA_POWERPCBE */
#endif

/* General Settings */
/* #undef KOKKOS_ENABLE_CXX14 */
#define KOKKOS_ENABLE_CXX17
/* #undef KOKKOS_ENABLE_CXX20 */

/* #undef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE */
/* #undef KOKKOS_ENABLE_CUDA_UVM */
#define KOKKOS_ENABLE_CUDA_LAMBDA
/* #undef KOKKOS_ENABLE_CUDA_CONSTEXPR */
/* #undef KOKKOS_ENABLE_CUDA_LDG_INTRINSIC */
/* #undef KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC */
/* #undef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE */
/* #undef KOKKOS_ENABLE_HPX_ASYNC_DISPATCH */
/* #undef KOKKOS_ENABLE_DEBUG */
/* #undef KOKKOS_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK */
/* #undef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK */
/* #undef KOKKOS_ENABLE_COMPILER_WARNINGS */
/* #undef KOKKOS_ENABLE_PROFILING_LOAD_PRINT */
/* #undef KOKKOS_ENABLE_TUNING */
#define KOKKOS_ENABLE_DEPRECATED_CODE_3
#define KOKKOS_ENABLE_DEPRECATION_WARNINGS
/* #undef KOKKOS_ENABLE_LARGE_MEM_TESTS */
/* #undef KOKKOS_ENABLE_DUALVIEW_MODIFY_CHECK */
#define KOKKOS_ENABLE_COMPLEX_ALIGN
#define KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
/* #undef KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION */
/* #undef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION */

/* TPL Settings */
/* #undef KOKKOS_ENABLE_HWLOC */
/* #undef KOKKOS_USE_LIBRT */
/* #undef KOKKOS_ENABLE_HBWSPACE */
#define KOKKOS_ENABLE_LIBDL
/* #undef KOKKOS_ENABLE_LIBQUADMATH */
/* #undef KOKKOS_IMPL_CUDA_CLANG_WORKAROUND */

#define KOKKOS_COMPILER_CUDA_VERSION 115

/* #undef KOKKOS_ARCH_SSE42 */
/* #undef KOKKOS_ARCH_ARMV80 */
/* #undef KOKKOS_ARCH_ARMV8_THUNDERX */
/* #undef KOKKOS_ARCH_ARMV81 */
/* #undef KOKKOS_ARCH_ARMV8_THUNDERX2 */
/* #undef KOKKOS_ARCH_AMD_AVX2 */
/* #undef KOKKOS_ARCH_AVX */
/* #undef KOKKOS_ARCH_AVX2 */
#define KOKKOS_ARCH_AVX512XEON
/* #undef KOKKOS_ARCH_KNC */
/* #undef KOKKOS_ARCH_AVX512MIC */
/* #undef KOKKOS_ARCH_POWER7 */
/* #undef KOKKOS_ARCH_POWER8 */
/* #undef KOKKOS_ARCH_POWER9 */
/* #undef KOKKOS_ARCH_INTEL_GEN */
/* #undef KOKKOS_ARCH_INTEL_DG1 */
/* #undef KOKKOS_ARCH_INTEL_GEN9 */
/* #undef KOKKOS_ARCH_INTEL_GEN11 */
/* #undef KOKKOS_ARCH_INTEL_GEN12LP */
/* #undef KOKKOS_ARCH_INTEL_XEHP */
/* #undef KOKKOS_ARCH_INTEL_GPU */
/* #undef KOKKOS_ARCH_KEPLER */
/* #undef KOKKOS_ARCH_KEPLER30 */
/* #undef KOKKOS_ARCH_KEPLER32 */
/* #undef KOKKOS_ARCH_KEPLER35 */
/* #undef KOKKOS_ARCH_KEPLER37 */
/* #undef KOKKOS_ARCH_MAXWELL */
/* #undef KOKKOS_ARCH_MAXWELL50 */
/* #undef KOKKOS_ARCH_MAXWELL52 */
/* #undef KOKKOS_ARCH_MAXWELL53 */
/* #undef KOKKOS_ARCH_PASCAL */
/* #undef KOKKOS_ARCH_PASCAL60 */
/* #undef KOKKOS_ARCH_PASCAL61 */
#define KOKKOS_ARCH_VOLTA
#define KOKKOS_ARCH_VOLTA70
/* #undef KOKKOS_ARCH_VOLTA72 */
/* #undef KOKKOS_ARCH_TURING75 */
/* #undef KOKKOS_ARCH_AMPERE */
/* #undef KOKKOS_ARCH_AMPERE80 */
/* #undef KOKKOS_ARCH_AMPERE86 */
/* #undef KOKKOS_ARCH_AMD_ZEN */
/* #undef KOKKOS_ARCH_AMD_ZEN2 */
/* #undef KOKKOS_ARCH_AMD_ZEN3 */

/* #undef KOKKOS_IMPL_DISABLE_SYCL_DEVICE_PRINTF */
