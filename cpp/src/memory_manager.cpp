/*!
 * NUMA-Aware Memory Manager
 * High-performance memory allocation with pool management
 */

#include "algoveda/core.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
    #include <sys/mman.h>
    #include <numa.h>
    #include <numaif.h>
#endif

namespace algoveda {
namespace core {

MemoryManager::MemoryManager(const CoreConfig& config) 
    : config_(config) {
    
    // Reserve initial memory pools
    size_t initial_pool_size = config_.memory_pool_size_mb * 1024 * 1024;
    
    // Initialize memory blocks
    blocks_.reserve(1000); // Reserve space for block metadata
    
    // Allocate initial memory pool
    void* initial_pool = allocate_from_system(initial_pool_size);
    if (initial_pool) {
        blocks_.push_back({
            initial_pool,
            initial_pool_size,
            true, // Initially free
            std::chrono::high_resolution_clock::now()
        });
    }
}

MemoryManager::~MemoryManager() {
    // Free all allocated blocks
    for (const auto& block : blocks_) {
        if (block.ptr) {
            std::free(block.ptr);
        }
    }
    blocks_.clear();
}

void* MemoryManager::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    
    // Align size to prevent fragmentation
    size = (size + alignment - 1) & ~(alignment - 1);
    
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Try to allocate from existing pools first
    void* ptr = allocate_from_pool(size);
    if (ptr) {
        update_allocation_stats(size);
        return ptr;
    }
    
    // Allocate new memory from system
    ptr = allocate_from_system(size);
    if (ptr) {
        blocks_.push_back({
            ptr,
            size,
            false, // Not free
            std::chrono::high_resolution_clock::now()
        });
        update_allocation_stats(size);
    }
    
    return ptr;
}

void* MemoryManager::allocate_aligned(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    
    #ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            ptr = nullptr;
        }
    #endif
    
    if (ptr) {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        blocks_.push_back({
            ptr,
            size,
            false,
            std::chrono::high_resolution_clock::now()
        });
        update_allocation_stats(size);
    }
    
    return ptr;
}

void* MemoryManager::allocate_huge(size_t size) {
    #ifdef __linux__
        void* ptr = mmap(nullptr, size, 
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                        -1, 0);
        
        if (ptr == MAP_FAILED) {
            // Fallback to regular allocation
            return allocate(size);
        }
        
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        blocks_.push_back({
            ptr,
            size,
            false,
            std::chrono::high_resolution_clock::now()
        });
        update_allocation_stats(size);
        
        return ptr;
    #else
        return allocate(size);
    #endif
}

void MemoryManager::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Find the block containing this pointer
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [ptr](const MemoryBlock& block) {
            return block.ptr == ptr;
        });
    
    if (it != blocks_.end()) {
        it->is_free = true;
        update_deallocation_stats(it->size);
        
        // Try to coalesce adjacent free blocks
        try_coalesce_blocks();
    }
}

void MemoryManager::deallocate_aligned(void* ptr) {
    if (!ptr) return;
    
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        std::free(ptr);
    #endif
    
    deallocate(ptr);
}

void MemoryManager::deallocate_huge(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [ptr](const MemoryBlock& block) {
            return block.ptr == ptr;
        });
    
    if (it != blocks_.end()) {
        #ifdef __linux__
            munmap(ptr, it->size);
        #endif
        
        update_deallocation_stats(it->size);
        blocks_.erase(it);
    }
}

void* MemoryManager::allocate_from_pool(size_t size) {
    // Find a suitable free block
    for (auto& block : blocks_) {
        if (block.is_free && block.size >= size) {
            // Split the block if it's much larger than needed
            if (block.size > size + config_.small_object_threshold) {
                size_t remaining_size = block.size - size;
                
                // Create new block for the remaining memory
                blocks_.push_back({
                    static_cast<char*>(block.ptr) + size,
                    remaining_size,
                    true,
                    std::chrono::high_resolution_clock::now()
                });
                
                block.size = size;
            }
            
            block.is_free = false;
            return block.ptr;
        }
    }
    
    return nullptr;
}

void* MemoryManager::allocate_from_system(size_t size) {
    void* ptr = nullptr;
    
    #ifdef __linux__
        if (config_.enable_numa_awareness && numa_available() != -1) {
            // Allocate on current NUMA node
            int numa_node = numa_node_of_cpu(sched_getcpu());
            if (numa_node >= 0) {
                ptr = numa_alloc_onnode(size, numa_node);
            }
        }
    #endif
    
    if (!ptr) {
        ptr = std::aligned_alloc(64, size); // 64-byte aligned for cache efficiency
        if (!ptr) {
            ptr = std::malloc(size);
        }
    }
    
    if (ptr) {
        // Initialize memory to zero for security
        std::memset(ptr, 0, size);
    }
    
    return ptr;
}

bool MemoryManager::try_coalesce_blocks() {
    bool coalesced = false;
    
    // Sort blocks by address for efficient coalescing
    std::sort(blocks_.begin(), blocks_.end(),
        [](const MemoryBlock& a, const MemoryBlock& b) {
            return a.ptr < b.ptr;
        });
    
    for (size_t i = 0; i < blocks_.size() - 1; ++i) {
        auto& current = blocks_[i];
        auto& next = blocks_[i + 1];
        
        // Check if blocks are adjacent and both free
        if (current.is_free && next.is_free &&
            static_cast<char*>(current.ptr) + current.size == next.ptr) {
            
            // Merge blocks
            current.size += next.size;
            blocks_.erase(blocks_.begin() + i + 1);
            coalesced = true;
            --i; // Check the same position again
        }
    }
    
    return coalesced;
}

void MemoryManager::update_allocation_stats(size_t size) {
    total_allocated_.fetch_add(size, std::memory_order_relaxed);
    current_usage_.fetch_add(size, std::memory_order_relaxed);
    
    // Update peak usage
    size_t current = current_usage_.load(std::memory_order_relaxed);
    size_t peak = peak_usage_.load(std::memory_order_relaxed);
    while (current > peak && 
           !peak_usage_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
        peak = peak_usage_.load(std::memory_order_relaxed);
    }
}

void MemoryManager::update_deallocation_stats(size_t size) {
    current_usage_.fetch_sub(size, std::memory_order_relaxed);
}

MemoryManager::MemoryStats MemoryManager::get_stats() const {
    MemoryStats stats;
    
    stats.total_allocated = total_allocated_.load(std::memory_order_relaxed);
    stats.current_usage = current_usage_.load(std::memory_order_relaxed);
    stats.peak_usage = peak_usage_.load(std::memory_order_relaxed);
    
    // Calculate fragmentation
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    size_t free_blocks = 0;
    size_t total_free_memory = 0;
    
    for (const auto& block : blocks_) {
        if (block.is_free) {
            free_blocks++;
            total_free_memory += block.size;
        }
    }
    
    if (total_free_memory > 0) {
        stats.fragmentation_ratio = static_cast<double>(free_blocks) / 
                                   (total_free_memory / 1024.0); // Blocks per KB
    }
    
    stats.allocation_count = blocks_.size();
    
    return stats;
}

void MemoryManager::reset_stats() {
    total_allocated_.store(0, std::memory_order_relaxed);
    peak_usage_.store(current_usage_.load(std::memory_order_relaxed), 
                     std::memory_order_relaxed);
}

// ObjectPool implementation
template<typename T>
MemoryManager::ObjectPool<T>::ObjectPool(size_t initial_size) : capacity_(initial_size) {
    pool_.reserve(initial_size);
    
    for (size_t i = 0; i < initial_size; ++i) {
        auto obj = std::make_unique<T>();
        T* ptr = obj.get();
        pool_.push_back(std::move(obj));
        free_objects_.push(ptr);
    }
}

template<typename T>
MemoryManager::ObjectPool<T>::~ObjectPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.clear();
    // free_objects_ will be cleared automatically
}

template<typename T>
template<typename... Args>
T* MemoryManager::ObjectPool<T>::construct(Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    T* obj = nullptr;
    
    if (!free_objects_.empty()) {
        obj = free_objects_.top();
        free_objects_.pop();
        
        // Reconstruct the object in place
        obj->~T();
        new(obj) T(std::forward<Args>(args)...);
    } else {
        // Expand the pool
        auto new_obj = std::make_unique<T>(std::forward<Args>(args)...);
        obj = new_obj.get();
        pool_.push_back(std::move(new_obj));
        capacity_++;
    }
    
    return obj;
}

template<typename T>
void MemoryManager::ObjectPool<T>::destroy(T* obj) {
    if (!obj) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Verify the object belongs to this pool
    bool found = false;
    for (const auto& pool_obj : pool_) {
        if (pool_obj.get() == obj) {
            found = true;
            break;
        }
    }
    
    if (found) {
        free_objects_.push(obj);
    }
}

} // namespace core
} // namespace algoveda
