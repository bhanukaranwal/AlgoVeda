cmake_minimum_required(VERSION 3.18)
project(AlgoVeda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Compiler flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Source files
file(GLOB_RECURSE CPP_SOURCES "cpp/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "cuda/src/*.cu")

# Create library
add_library(algoveda_core SHARED ${CPP_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(algoveda_core ${CUDA_LIBRARIES} Threads::Threads)

# Set properties
set_target_properties(algoveda_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS algoveda_core DESTINATION lib)
install(DIRECTORY cpp/include/ DESTINATION include)
