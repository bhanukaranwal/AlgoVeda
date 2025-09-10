/*!
 * AlgoVeda Core C++ Interface
 * High-performance C++ components for critical path operations
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <thread>

namespace algoveda {
namespace core {

// Forward declarations
class Application;
class ThreadPool;
class MemoryManager;
class PerformanceMonitor;

// Type aliases for convenience
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;
using ThreadId = std::thread::id;

// Error handling
class AlgoVedaException : public std::exception {
public:
    explicit AlgoVedaException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

// Configuration structure
struct CoreConfig {
    size_t thread_pool_size = std::thread::hardware_concurrency();
    size_t memory_pool_size_mb = 1024;  // 1GB default
    bool enable_numa_awareness = true;
    bool enable_cpu_affinity = true;
    bool enable_huge_pages = false;
    std::vector<int> cpu_affinity_mask;
    
    // Performance monitoring
    bool enable_profiling = true;
    std::chrono::milliseconds metrics_interval{1000};
    
    // Memory management
    size_t small_object_threshold = 1024;  // 1KB
    size_t large_object_threshold = 1024 * 1024;  // 1MB
    double memory_growth_factor = 1.5;
};

// Main application class
class Application {
public:
    explicit Application(const CoreConfig& config = CoreConfig{});
    ~Application();
    
    // Non-copyable
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    
    // Initialization and lifecycle
    void initialize();
    void start();
    void stop();
    void shutdown();
    
    // Component access
    ThreadPool& get_thread_pool() { return *thread_pool_; }
    MemoryManager& get_memory_manager() { return *memory_manager_; }
    PerformanceMonitor& get_performance_monitor() { return *performance_monitor_; }
    
    // Status
    bool is_running() const { return running_.load(); }
    bool is_initialized() const { return initialized_.load(); }
    
    // Configuration
    const CoreConfig& get_config() const { return config_; }
    
private:
    CoreConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};
    
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<PerformanceMonitor> performance_monitor_;
    
    void setup_numa_policy();
    void setup_cpu_affinity();
    void setup_signal_handlers();
};

// High-performance thread pool
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    // Task submission
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;
    
    template<typename F>
    void submit_detached(F&& f);
    
    // Bulk operations
    template<typename Iterator, typename Function>
    void parallel_for(Iterator first, Iterator last, Function func);
    
    template<typename T, typename Function>
    void parallel_for_each(std::vector<T>& container, Function func);
    
    // Pool management
    void resize(size_t new_size);
    void pause();
    void resume();
    
    // Statistics
    size_t get_thread_count() const { return threads_.size(); }
    size_t get_pending_tasks() const;
    size_t get_completed_tasks() const { return completed_tasks_.load(); }
    
private:
    struct Task {
        std::function<void()> function;
        std::chrono::high_resolution_clock::time_point submit_time;
    };
    
    std::vector<std::thread> threads_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
    std::atomic<size_t> completed_tasks_{0};
    
    void worker_thread(size_t thread_id);
    void set_thread_affinity(std::thread& thread, int cpu_id);
};

// NUMA-aware memory manager
class MemoryManager {
public:
    explicit MemoryManager(const CoreConfig& config);
    ~MemoryManager();
    
    // Non-copyable
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    // Memory allocation
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    void* allocate_aligned(size_t size, size_t alignment);
    void* allocate_huge(size_t size);  // Huge page allocation
    
    void deallocate(void* ptr);
    void deallocate_aligned(void* ptr);
    void deallocate_huge(void* ptr);
    
    // Pool management
    template<typename T>
    class ObjectPool {
    public:
        explicit ObjectPool(size_t initial_size = 1000);
        ~ObjectPool();
        
        template<typename... Args>
        T* construct(Args&&... args);
        
        void destroy(T* obj);
        
        size_t size() const { return pool_.size(); }
        size_t capacity() const { return capacity_; }
        
    private:
        std::vector<std::unique_ptr<T>> pool_;
        std::stack<T*> free_objects_;
        std::mutex mutex_;
        size_t capacity_;
    };
    
    // Memory statistics
    struct MemoryStats {
        size_t total_allocated = 0;
        size_t total_deallocated = 0;
        size_t current_usage = 0;
        size_t peak_usage = 0;
        size_t allocation_count = 0;
        size_t deallocation_count = 0;
        double fragmentation_ratio = 0.0;
    };
    
    MemoryStats get_stats() const;
    void reset_stats();
    
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool is_free;
        std::chrono::high_resolution_clock::time_point allocation_time;
    };
    
    CoreConfig config_;
    std::vector<MemoryBlock> blocks_;
    std::mutex allocation_mutex_;
    mutable std::atomic<size_t> total_allocated_{0};
    mutable std::atomic<size_t> current_usage_{0};
    mutable std::atomic<size_t> peak_usage_{0};
    
    void* allocate_from_pool(size_t size);
    void* allocate_from_system(size_t size);
    bool try_coalesce_blocks();
};

// Performance monitoring and profiling
class PerformanceMonitor {
public:
    explicit PerformanceMonitor(const CoreConfig& config);
    ~PerformanceMonitor();
    
    // Timing utilities
    class Timer {
    public:
        Timer(const std::string& name, PerformanceMonitor& monitor);
        ~Timer();
        
        void stop();
        Duration elapsed() const;
        
    private:
        std::string name_;
        PerformanceMonitor& monitor_;
        TimePoint start_time_;
        bool stopped_;
    };
    
    // Profiling
    void start_profiling();
    void stop_profiling();
    bool is_profiling() const { return profiling_.load(); }
    
    // Metrics collection
    void record_latency(const std::string& operation, Duration latency);
    void record_throughput(const std::string& operation, size_t count);
    void record_counter(const std::string& counter, size_t value = 1);
    void record_gauge(const std::string& gauge, double value);
    
    // Statistics
    struct PerformanceStats {
        struct Metric {
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::lowest();
            double mean = 0.0;
            double stddev = 0.0;
            size_t count = 0;
            double p50 = 0.0;
            double p95 = 0.0;
            double p99 = 0.0;
        };
        
        std::unordered_map<std::string, Metric> latencies;
        std::unordered_map<std::string, size_t> counters;
        std::unordered_map<std::string, double> gauges;
        std::unordered_map<std::string, double> throughput;
    };
    
    PerformanceStats get_stats() const;
    void reset_stats();
    
    // Reporting
    void export_metrics(const std::string& format = "json") const;
    void start_metrics_server(int port = 9090);
    void stop_metrics_server();
    
private:
    struct TimingData {
        std::vector<Duration> samples;
        std::mutex mutex;
    };
    
    CoreConfig config_;
    std::atomic<bool> profiling_{false};
    std::unordered_map<std::string, TimingData> timing_data_;
    std::unordered_map<std::string, std::atomic<size_t>> counters_;
    std::unordered_map<std::string, std::atomic<double>> gauges_;
    
    std::thread metrics_thread_;
    std::atomic<bool> metrics_running_{false};
    
    void metrics_collection_loop();
    void calculate_statistics(const std::vector<Duration>& samples, 
                            PerformanceStats::Metric& metric) const;
};

// RAII timer for automatic performance measurement
#define ALGOVEDA_TIMER(name) \
    auto timer = algoveda::core::PerformanceMonitor::Timer(name, performance_monitor)

// Lock-free data structures
template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity = 1024);
    ~LockFreeQueue();
    
    bool push(const T& item);
    bool push(T&& item);
    bool pop(T& item);
    
    bool empty() const;
    size_t size() const;
    size_t capacity() const { return capacity_; }
    
private:
    struct Node {
        std::atomic<T*> data{nullptr};
        std::atomic<Node*> next{nullptr};
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    std::atomic<size_t> size_{0};
    size_t capacity_;
    
    Node* allocate_node();
    void deallocate_node(Node* node);
};

// Cache-friendly containers
template<typename T>
class CacheAlignedVector {
public:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    explicit CacheAlignedVector(size_t capacity = 0);
    ~CacheAlignedVector();
    
    void push_back(const T& value);
    void push_back(T&& value);
    void pop_back();
    
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    T& at(size_t index);
    const T& at(size_t index) const;
    
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    
    void reserve(size_t new_capacity);
    void resize(size_t new_size);
    void clear();
    
    // Iterator support
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
    void grow(size_t min_capacity);
    void* allocate_aligned(size_t size, size_t alignment);
};

// Utility functions
namespace utils {
    // CPU topology
    int get_numa_node_count();
    int get_cpu_count();
    int get_current_numa_node();
    std::vector<int> get_cpus_for_numa_node(int node);
    
    // Timing utilities
    inline TimePoint now() {
        return std::chrono::high_resolution_clock::now();
    }
    
    inline Duration duration_since(TimePoint start) {
        return std::chrono::duration_cast<Duration>(now() - start);
    }
    
    // Memory utilities
    void prefetch_read(const void* addr);
    void prefetch_write(void* addr);
    void memory_barrier();
    void cpu_relax();
    
    // Bit manipulation
    inline bool is_power_of_two(size_t n) {
        return n && !(n & (n - 1));
    }
    
    inline size_t next_power_of_two(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return ++n;
    }
}

} // namespace core
} // namespace algoveda

// Template implementations
#include "core_impl.hpp"
