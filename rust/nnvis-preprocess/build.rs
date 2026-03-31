// build.rs for nnvis-preprocess
//
// Uses prost-build + protox to generate Rust types from the ONNX proto schema.
// protox is a pure-Rust protobuf compiler — no external `protoc` binary needed.
// Generated code lands in OUT_DIR and is included via include! in src/onnx_proto.rs.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = "proto/onnx.proto";

    println!("cargo:rerun-if-changed={}", proto_path);
    println!("cargo:rerun-if-changed=build.rs");

    let file_descriptor_set = protox::compile([proto_path], ["proto"])?;

    prost_build::Config::new()
        .compile_fds(file_descriptor_set)?;

    Ok(())
}
