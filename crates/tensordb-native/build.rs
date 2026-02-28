fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("cpp/native.cc")
        .include("include")
        .flag_if_supported("-std=c++17")
        .compile("tensordb_native");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/tensordb_native.h");
    println!("cargo:rerun-if-changed=cpp/native.cc");
    println!("cargo:rerun-if-changed=cpp/native.h");
}
