namespace detail {
    namespace transparent_global_static {
        template<class Tag, typename cst_type>
        __constant__ __device__ cst_type device_value;

        template<class Tag, typename cst_type>
        static cst_type host_value;
    }
}

template <class Tag, typename cst_type>
class transparent_global {
    private :

    static __device__ __host__ cst_type& device_constant() {
        return detail::transparent_global_static::device_value<Tag, cst_type>;
    }

    static __host__ cst_type& host_constant() {
        return detail::transparent_global_static::host_value<Tag, cst_type>;
    }

    public :
    __host__ __device__ transparent_global() = delete;

    static void init(cst_type& val) {
        host_constant() = val;
        cudaMemcpyToSymbol(device_constant(), &host_constant(), sizeof(cst_type));
    }

    static __host__ __device__ cst_type& get() {
        #ifdef __CUDA_ARCH__
            return device_constant();
        #else
            return host_constant();
        #endif
    }
};