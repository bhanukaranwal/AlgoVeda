use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=assembly/");

    // Build C++ components
    let mut cpp_build = cc::Build::new();
    cpp_build
        .cpp(true)
        .std("c++20")
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-funroll-loops")
        .include("cpp/include")
        .files(vec![
            "cpp/src/core/application.cpp",
            "cpp/src/core/thread_pool.cpp",
            "cpp/src/core/memory_manager.cpp",
            "cpp/src/trading/order_book_engine.cpp",
            "cpp/src/trading/matching_engine.cpp",
            "cpp/src/calculations/options_pricing.cpp",
            "cpp/src/performance/simd_operations.cpp",
        ])
        .compile("algoveda_cpp");

    // Build CUDA kernels if feature enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=curand");
        
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        
        // Compile CUDA kernels
        let nvcc_output = std::process::Command::new("nvcc")
            .args(&[
                "-O3",
                "--gpu-architecture=sm_75",
                "--ptx",
                "cuda/src/kernels/options_pricing.cu",
                "-o", "cuda/kernels/options_pricing.ptx"
            ])
            .output()
            .expect("Failed to compile CUDA kernels");

        if !nvcc_output.status.success() {
            panic!("CUDA compilation failed: {}", String::from_utf8_lossy(&nvcc_output.stderr));
        }
    }

    // Generate bindings for C++ headers
    let bindings = bindgen::Builder::default()
        .header("cpp/include/algoveda/core.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
