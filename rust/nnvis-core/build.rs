// build.rs for nnvis-core
//
// Regenerates src/generated/model_bundle_generated.rs from the .fbs schema
// if `flatc` is available on PATH. If `flatc` is not present the pre-generated
// file that is already checked in will be used as-is, so the build does not
// require flatc on every machine.

use std::path::Path;
use std::process::Command;

fn main() {
    let schema = "schema/model_bundle.fbs";
    let out_dir = "src/generated";

    // Tell Cargo to re-run this script if the schema changes.
    println!("cargo:rerun-if-changed={}", schema);
    println!("cargo:rerun-if-changed=build.rs");

    // Only attempt regeneration if the schema exists and flatc is available.
    if !Path::new(schema).exists() {
        return;
    }

    let flatc_available = Command::new("flatc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !flatc_available {
        eprintln!(
            "cargo:warning=flatc not found on PATH; using pre-generated FlatBuffers code. \
             Install flatbuffers (e.g. `brew install flatbuffers`) to regenerate."
        );
        return;
    }

    std::fs::create_dir_all(out_dir).expect("failed to create generated dir");

    let status = Command::new("flatc")
        .args(["--rust", "-o", out_dir, schema])
        .status()
        .expect("failed to spawn flatc");

    if !status.success() {
        panic!("flatc exited with non-zero status: {}", status);
    }
}
