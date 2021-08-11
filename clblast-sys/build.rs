extern crate cmake;
use cmake::Config;

use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let dst = Config::new("CLBlast")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=clblast");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "windows" {
        println!("cargo:rustc-link-lib=shell32");
    }

    println!("cargo:rerun-if-changed=CLBlast/include/clblast_c.h");

    let bindings = bindgen::Builder::default()
        .header("CLBlast/include/clblast_c.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("clblast.CLBlast.*")
        .allowlist_function("CLBlast.*")
        .allowlist_function("clblast.CLBlast.*")
        .allowlist_var("clblast.CLBLAST.*")
        .allowlist_var("clblast.CLBlast.*")
        .allowlist_type("cl_double2")
        .blocklist_type("_?cl_event")
        .blocklist_type("_?cl_command_queue")
        .blocklist_type("_?cl_mem")
        .blocklist_type("_?cl_device_id")
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // // for debugging:
    // bindings.write_to_file("bindings.debug.rs");

}
