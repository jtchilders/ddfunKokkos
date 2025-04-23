#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "ddouble.hpp"

namespace ddfun {

class DDRandom {
private:
    // The random pool needs to be a class member since it must persist
    Kokkos::Random_XorShift64_Pool<> rand_pool;

public:
    // Constructor that takes a seed
    DDRandom(uint64_t seed = 332211) : rand_pool(seed) {}

    // Device-callable function to generate a single uniform random ddouble
    KOKKOS_INLINE_FUNCTION
    ddouble get_uniform_ddouble(typename Kokkos::Random_XorShift64_Pool<>::generator_type& gen) const {
        // Generate the high and low parts
        double hi = gen.drand();
        double lo = gen.drand() * ldexp(1.0, -52);
        
        // Create and return the ddouble
        return ddouble(hi, lo);
    }
    
    // Device-callable function to generate a single normal random ddouble
    KOKKOS_INLINE_FUNCTION
    ddouble get_normal_ddouble(typename Kokkos::Random_XorShift64_Pool<>::generator_type& gen) const {
        // Use standard Box-Muller transform with regular doubles first
        double u1 = gen.drand();
        double u2 = gen.drand();
        
        // Ensure u1 is never exactly zero to avoid log(0)
        if (u1 < 1.0e-300) u1 = 1.0e-300;
        
        double radius = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        
        // Generate standard normal in double precision
        double z = radius * Kokkos::cos(theta);
        
        // Convert to ddouble with extra precision
        ddouble result = ddouble(z);
        
        // Add some controlled noise for the lower bits
        double lo_noise = gen.drand() * ldexp(1.0, -52);
        // Keep the same sign as z
        if (z < 0.0) lo_noise = -lo_noise;
        result.lo = lo_noise;
        
        return result;
    }
    
    // Helper functions to get and free a generator state
    KOKKOS_INLINE_FUNCTION
    typename Kokkos::Random_XorShift64_Pool<>::generator_type get_state() const {
        return rand_pool.get_state();
    }
    
    KOKKOS_INLINE_FUNCTION
    void free_state(typename Kokkos::Random_XorShift64_Pool<>::generator_type& gen) const {
        rand_pool.free_state(gen);
    }

    // Generate uniform random numbers on device
    void generate_uniform(Kokkos::View<ddouble*> results) {
        const int N = results.extent(0);
        
        // Create a local copy of the random pool to use in the lambda
        auto rand_pool_local = rand_pool;
        
        // Capture a pointer to this object to access the methods from within the lambda
        const DDRandom* self_ptr = this;
        
        Kokkos::parallel_for("generate_uniform", N, 
        KOKKOS_LAMBDA(const int i) {
            // Get a generator instance for this thread
            auto gen = rand_pool_local.get_state();
            
            // Use the single-value function via the captured pointer
            results(i) = self_ptr->get_uniform_ddouble(gen);
            
            // Return the generator
            rand_pool_local.free_state(gen);
        });
    }

    // Generate normal random numbers on device using safer approach
    void generate_normal(Kokkos::View<ddouble*> results) {
        const int N = results.extent(0);
        
        // Create a local copy of the random pool to use in the lambda
        auto rand_pool_local = rand_pool;
        
        // Capture a pointer to this object to access the methods from within the lambda
        const DDRandom* self_ptr = this;
        
        Kokkos::parallel_for("generate_normal", N, 
        KOKKOS_LAMBDA(const int i) {
            // Get a generator instance for this thread
            auto gen = rand_pool_local.get_state();
            
            // Use the single-value function via the captured pointer
            results(i) = self_ptr->get_normal_ddouble(gen);
            
            // Return the generator
            rand_pool_local.free_state(gen);
        });
    }
};

} // namespace ddfun 