#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
// nvprof format profiler struct
struct KernelProfiler {
    cudaEvent_t start, stop;
    float total_time;
    float min_time;
    float max_time;
    int count;
    int threshold_count;
    const char* name;
    
    static std::vector<KernelProfiler*>& profilers() {
        static std::vector<KernelProfiler*> all_profilers;
        return all_profilers;
    }
    
    static float& get_time() {
        static float total = 0.0f;
        return total;
    }
    
    KernelProfiler(const char* kernel_name) :
        total_time(0.0f), 
        min_time(10000.0f), 
        max_time(-1.0f), 
        count(0), 
        threshold_count(938 * 3), 
        name(kernel_name) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        profilers().push_back(this);
    }
    
    ~KernelProfiler() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void start_timing() {
        count++;
        // if (++count <= threshold_count || count >= 938 * 29) return;
        cudaEventRecord(start);
    }
    
    void stop_timing() {
        // if (count <= threshold_count || count >= 938 * 29) return;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        min_time = fminf(min_time, ms);
        max_time = fmaxf(max_time, ms);
        get_time() += ms;
    }
    
    static void print() {
        std::sort(profilers().begin(), profilers().end(), 
            [](KernelProfiler* a, KernelProfiler* b) { return a->total_time > b->total_time; });
        
        // print header
        printf("\n==== CUDA Kernel Profiling Results ====\n\n");
        printf("%-30s  %12s  %12s  %8s  %12s  %12s\n", 
              "Kernel Name", "Time(%)", "Time", "Calls", "Avg", "Min");
        printf("------------------------------------------------------------------------------------------------\n");
        
        float total_time = get_time();
        for (auto profiler : profilers()) {
            if (profiler->count > 0) {
                float time_percent = (profiler->total_time / total_time) * 100.0f;
                printf(
                    "%-30s  %11.2f%%  %12.2f  %8d  %12.6f  %12.6f\n", 
                    profiler->name, 
                    time_percent,
                    profiler->total_time, 
                    profiler->count, 
                    profiler->total_time / profiler->count, 
                    profiler->min_time
                );
            }
        }
    }
};