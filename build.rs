use std::{error::Error, path::PathBuf};

use shaderc::{self, ShaderKind};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src/shaders");
    let outdir = PathBuf::from("target").join("shaders"); // target MUST be hardcoded, otherwise I cannot use include! macro
    std::fs::create_dir_all(outdir.clone())?;
    let mut compiler = shaderc::Compiler::new().expect("Failed to find a SPIR-V compiler");
    let options = shaderc::CompileOptions::new();
    for entry in std::fs::read_dir("src/shaders")? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let in_path = entry.path();
            let shader_kind = in_path
                .extension()
                .and_then(|ext| match ext.to_string_lossy().as_ref() {
                    "vert" => Some(ShaderKind::Vertex),
                    "comp" => Some(ShaderKind::Compute),
                    "frag" => Some(ShaderKind::Fragment),
                    _ => None,
                })
                .expect("Unsupported shader type");
            let source_text = std::fs::read_to_string(&in_path)?;
            let compiled_bytes = compiler.compile_into_spirv(
                &source_text,
                shader_kind,
                in_path.as_path().to_str().unwrap(),
                "main",
                options.as_ref(),
            )?;
            let outfile = outdir.clone().join(format!(
                "{}.spv",
                in_path.as_path().file_name().unwrap().to_str().unwrap()
            ));

            std::fs::write(&outfile, &compiled_bytes.as_binary_u8())?;
        }
    }
    Ok(())
}
