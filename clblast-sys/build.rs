extern crate cmake;
use cmake::Config;

use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let dst = Config::new("CLBlast")
        // .define("BUILD_SHARED_LIBS", "OFF")
        // .define("_GLIBCXX_USE_CXX11_ABI", "0")
        // .define("COMPILE_TARGET", "DESKTOP_x86_64")
        // .define("FLAVOR", "DESKTOP")
        // .define("LIBOPENVPN3_NOT_BUILD_EXAMPLES", "TRUE")
        .build();
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    // println!("cargo:rustc-link-lib=OpenCL");
    println!("cargo:rustc-link-lib=clblast");
    // println!("cargo:rustc-link-lib=dylib=OpenCL");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "windows" {
        println!("cargo:rustc-link-lib=shell32");
    }

    println!("cargo:rerun-if-changed=CLBlast/include/clblast_c.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("CLBlast/include/clblast_c.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // .allowlist_function("^CLBlast*")
        // .allowlist_var("^CLBLAST*")
        // .allowlist_var("^CLBlast*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from("./src");
    bindings
       
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // pkg_config::Config::new()
    //     .atleast_version("1.0")
    //     .probe("OpenCL")
    //     .unwrap();
    //   pkg_config::Config::new()
    //     .atleast_version("1.0")
    //     .probe("clblast")
    //     .unwrap();
    // let src = ["src/file1.c", "src/otherfile.c"];
    // let mut builder = cc::Build::new();
    // let build = builder
    //     .files(src.iter())
    //     .include("include")
    //     .flag("-Wno-unused-parameter")
    //     .define("USE_ZLIB", None);
    // build.compile("foo");
}
