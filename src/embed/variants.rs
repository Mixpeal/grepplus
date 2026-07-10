use crate::core::error::{GpError, Result};
use crate::embed::download::is_installed;
use crate::embed::hf::install_local_id;
use crate::embed::hf::HfFile;
use inquire::error::InquireError;
use inquire::Select;
use std::io::{self, IsTerminal};

pub const MAX_ONNX_PICKER: usize = 10;
pub const MAX_MIRROR_PICKER: usize = 10;
/// Unquantized `model.onnx` exports ≥ this size are skipped for non-interactive installs.
const FULL_PRECISION_MIN_BYTES: u64 = 1_500 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct OnnxVariant {
    pub label: String,
    pub model_file: String,
    pub extra_files: Vec<String>,
    pub total_bytes: u64,
    pub rank: i32,
}

pub fn discover_onnx_variants(files: &[HfFile]) -> Vec<OnnxVariant> {
    let paths: Vec<&HfFile> = files
        .iter()
        .filter(|f| f.path.to_lowercase().ends_with(".onnx"))
        .collect();
    let mut out = Vec::new();
    for file in &paths {
        let stem = file.path.trim_end_matches(".onnx");
        let mut extra = Vec::new();
        let mut total = file.size;
        for other in files {
            let op = &other.path;
            if op.starts_with(stem) && (op.ends_with(".onnx_data") || op.contains(".onnx_data_")) {
                extra.push(op.clone());
                total += other.size;
            }
        }
        if file.size < 5 * 1024 * 1024 && extra.is_empty() {
            let name = file.path.to_lowercase();
            if name.contains("model.onnx") && paths.len() > 1 {
                continue;
            }
        }
        let label = onnx_label(&file.path);
        out.push(OnnxVariant {
            label,
            model_file: file.path.clone(),
            extra_files: extra,
            total_bytes: total,
            rank: onnx_rank(&file.path),
        });
    }
    out.sort_by_key(|v| v.rank);
    out.dedup_by(|a, b| a.label == b.label);
    out.truncate(MAX_ONNX_PICKER);
    out
}

fn onnx_rank(path: &str) -> i32 {
    let lower = path.to_lowercase();
    if lower.contains("q4f16") {
        return 0;
    }
    if lower.contains("quantized") {
        return 1;
    }
    if lower.contains("q4") {
        return 2;
    }
    if lower.contains("int8") {
        return 3;
    }
    if lower.contains("fp16") || lower.contains("f16") {
        return 4;
    }
    if lower.ends_with("model.onnx") {
        return 10;
    }
    12
}

fn onnx_label(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.trim_end_matches(".onnx").to_string()
}

pub fn quant_label_from_path(path: &str) -> String {
    onnx_label(path)
}

fn is_generic_model_onnx(variant: &OnnxVariant) -> bool {
    variant.label == "model" || variant.model_file.to_lowercase().ends_with("model.onnx")
}

/// Huge unquantized `model.onnx` — excluded from non-interactive default installs.
pub fn is_full_precision(variant: &OnnxVariant, _all: &[OnnxVariant]) -> bool {
    is_generic_model_onnx(variant) && variant.total_bytes >= FULL_PRECISION_MIN_BYTES
}

/// Generic `model.onnx` shown under the "Full precision" group when quants also exist.
fn is_full_precision_group(variant: &OnnxVariant, all: &[OnnxVariant]) -> bool {
    if !is_generic_model_onnx(variant) {
        return false;
    }
    all.iter()
        .any(|v| v.model_file != variant.model_file && !is_generic_model_onnx(v))
}

fn visible_variants(variants: &[OnnxVariant], include_full: bool) -> Vec<&OnnxVariant> {
    variants
        .iter()
        .filter(|v| include_full || !is_full_precision(v, variants))
        .collect()
}

const BACK_TO_EXPORT_LABEL: &str = "← Change ONNX export";

/// Maps ONNX quant labels to local install ids for the quant picker.
#[derive(Debug, Clone, Copy)]
pub struct VariantPickInstallCtx<'a> {
    pub base_id: &'a str,
    pub default_quant: Option<&'a str>,
}

impl VariantPickInstallCtx<'_> {
    fn variant_installed(&self, quant_label: &str) -> bool {
        let id = install_local_id(self.base_id, quant_label, self.default_quant);
        is_installed(&id)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OnnxVariantPick<'a> {
    Selected(&'a OnnxVariant),
    ChangeExport,
}

#[derive(Debug, Clone, Copy)]
enum QuantPickEntry<'a> {
    Back,
    Group(&'static str),
    Variant(&'a OnnxVariant),
}

impl QuantPickEntry<'_> {
    fn label(self, install_ctx: Option<&VariantPickInstallCtx<'_>>) -> String {
        match self {
            Self::Back => BACK_TO_EXPORT_LABEL.to_string(),
            Self::Group(name) => name.to_string(),
            Self::Variant(v) => {
                let installed = install_ctx.is_some_and(|ctx| ctx.variant_installed(&v.label));
                let tag = if installed { " (installed)" } else { "" };
                format!(
                    "  {:<16} ~{:>4} MB{tag}",
                    v.label,
                    (v.total_bytes / 1024 / 1024).max(1)
                )
            }
        }
    }
}

/// Pick an ONNX variant. On a TTY with multiple variants, shows a picker unless `quant` is set
/// or `non_interactive` is true. When `allow_back` is true, the picker includes an option to
/// return to the ONNX export step.
pub fn pick_onnx_variant<'a>(
    variants: &'a [OnnxVariant],
    quant: Option<&str>,
    non_interactive: bool,
    include_full: bool,
    allow_back: bool,
    install_ctx: Option<&VariantPickInstallCtx<'_>>,
) -> Result<OnnxVariantPick<'a>> {
    if variants.is_empty() {
        return Err(GpError::Model("no ONNX variants found in repo".into()));
    }
    if let Some(q) = quant {
        let qn = q.to_lowercase();
        if let Some(v) = variants.iter().find(|v| v.label.to_lowercase() == qn) {
            return Ok(OnnxVariantPick::Selected(v));
        }
        if let Some(v) = variants
            .iter()
            .find(|v| v.model_file.to_lowercase().contains(&qn))
        {
            return Ok(OnnxVariantPick::Selected(v));
        }
        return Err(GpError::Model(format!(
            "ONNX variant {q} not found (available: {})",
            variants
                .iter()
                .map(|v| v.label.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if !non_interactive && io::stdin().is_terminal() && (variants.len() > 1 || allow_back) {
        return pick_interactive_quant(variants, allow_back, install_ctx);
    }
    let choices = visible_variants(variants, include_full);
    if choices.is_empty() {
        return Err(GpError::Model(
            "only full-precision ONNX found — pass --include-full or --quant model".into(),
        ));
    }
    Ok(OnnxVariantPick::Selected(choices[0]))
}

fn build_quant_picker_entries<'a>(
    variants: &'a [OnnxVariant],
    allow_back: bool,
) -> Vec<QuantPickEntry<'a>> {
    let mut quant = Vec::new();
    let mut full = Vec::new();
    for v in variants {
        if is_full_precision_group(v, variants) {
            full.push(v);
        } else {
            quant.push(v);
        }
    }

    let mut entries = Vec::new();
    if allow_back {
        entries.push(QuantPickEntry::Back);
    }
    let grouped = !quant.is_empty() && !full.is_empty();
    if grouped && !full.is_empty() {
        entries.push(QuantPickEntry::Group("Full precision"));
    }
    entries.extend(full.into_iter().map(QuantPickEntry::Variant));
    if grouped && !quant.is_empty() {
        entries.push(QuantPickEntry::Group("Quantized"));
    }
    entries.extend(quant.into_iter().map(QuantPickEntry::Variant));
    entries
}

fn quant_picker_start_cursor(entries: &[QuantPickEntry<'_>], variants: &[OnnxVariant]) -> usize {
    entries
        .iter()
        .position(|e| {
            matches!(
                e,
                QuantPickEntry::Variant(v)
                    if !is_full_precision_group(v, variants)
            )
        })
        .or_else(|| {
            entries
                .iter()
                .position(|e| matches!(e, QuantPickEntry::Variant(_)))
        })
        .unwrap_or(0)
}

fn pick_interactive_quant<'a>(
    variants: &'a [OnnxVariant],
    allow_back: bool,
    install_ctx: Option<&VariantPickInstallCtx<'_>>,
) -> Result<OnnxVariantPick<'a>> {
    let entries = build_quant_picker_entries(variants, allow_back);
    let labels: Vec<String> = entries.iter().map(|e| e.label(install_ctx)).collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let page_size = labels.len().max(7);
    let starting_cursor = quant_picker_start_cursor(&entries, variants);
    let help = if allow_back {
        "↑/↓ navigate, Enter confirm — Full precision first, then quantized"
    } else {
        "↑/↓ navigate, Enter confirm — recommended quant is highlighted"
    };

    loop {
        let selection = Select::new("Choose a quantization", label_refs.clone())
            .with_page_size(page_size)
            .with_starting_cursor(starting_cursor)
            .with_help_message(help)
            .prompt();
        match selection {
            Ok(label) => {
                let idx = labels.iter().position(|l| l == label).unwrap_or(0);
                match entries[idx] {
                    QuantPickEntry::Back => return Ok(OnnxVariantPick::ChangeExport),
                    QuantPickEntry::Group(_) => continue,
                    QuantPickEntry::Variant(v) => return Ok(OnnxVariantPick::Selected(v)),
                }
            }
            Err(InquireError::OperationCanceled) => {
                return Err(GpError::Config("install cancelled".into()));
            }
            Err(e) => return Err(GpError::Config(format!("variant picker: {e}"))),
        }
    }
}

pub fn pick_mirror_repo(repos: &[String], non_interactive: bool) -> Result<String> {
    if repos.is_empty() {
        return Err(GpError::Model("no ONNX mirror repos found".into()));
    }
    if non_interactive || !io::stdin().is_terminal() || repos.len() == 1 {
        return Ok(repos[0].clone());
    }
    let shown = repos.len().min(MAX_MIRROR_PICKER);
    let choices: Vec<String> = repos[..shown].to_vec();
    let labels: Vec<&str> = choices.iter().map(String::as_str).collect();
    let selection = Select::new("Choose an ONNX export", labels)
        .with_help_message("Hugging Face repo with ONNX weights — ↑/↓ navigate, Enter confirm")
        .prompt();
    match selection {
        Ok(repo) => Ok(repo.to_string()),
        Err(InquireError::OperationCanceled) => Err(GpError::Config("pull cancelled".into())),
        Err(e) => Err(GpError::Config(format!("mirror picker: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_small_only_model_onnx() {
        let files = vec![HfFile {
            path: "onnx/model.onnx".into(),
            size: 130_000_000,
        }];
        let v = discover_onnx_variants(&files);
        let picked = pick_onnx_variant(&v, None, true, false, false, None).unwrap();
        match picked {
            OnnxVariantPick::Selected(v) => assert_eq!(v.label, "model"),
            OnnxVariantPick::ChangeExport => panic!("unexpected back"),
        }
    }

    #[test]
    fn hides_full_precision_from_picker() {
        let files = vec![
            HfFile {
                path: "onnx/model.onnx".into(),
                size: 5_000_000_000,
            },
            HfFile {
                path: "onnx/model_q4f16.onnx".into(),
                size: 500_000,
            },
            HfFile {
                path: "onnx/model_q4f16.onnx_data".into(),
                size: 300_000_000,
            },
        ];
        let v = discover_onnx_variants(&files);
        let picked = pick_onnx_variant(&v, None, true, false, false, None).unwrap();
        match picked {
            OnnxVariantPick::Selected(v) => assert_eq!(v.label, "model_q4f16"),
            OnnxVariantPick::ChangeExport => panic!("unexpected back"),
        }
    }

    #[test]
    fn groups_quant_and_full_precision() {
        let files = vec![
            HfFile {
                path: "model.onnx".into(),
                size: 126_000_000,
            },
            HfFile {
                path: "onnx/model_qint8_avx512_vnni.onnx".into(),
                size: 32_000_000,
            },
            HfFile {
                path: "onnx/model_O4.onnx".into(),
                size: 63_000_000,
            },
        ];
        let v = discover_onnx_variants(&files);
        let entries = build_quant_picker_entries(&v, false);
        let labels: Vec<String> = entries.iter().map(|e| e.label(None)).collect();
        assert!(labels.contains(&"Quantized".to_string()));
        assert!(labels.contains(&"Full precision".to_string()));
        assert!(labels.iter().any(|l| l.contains("model_qint8_avx512_vnni")));
        assert!(labels.iter().any(|l| l.contains("model_O4")));
        assert!(labels
            .iter()
            .any(|l| l.contains("model") && l.contains("MB")));
        let quant_idx = labels.iter().position(|l| l == "Quantized").unwrap();
        let full_idx = labels.iter().position(|l| l == "Full precision").unwrap();
        assert!(full_idx < quant_idx);
    }

    #[test]
    fn discovers_onnx_variants_ranked() {
        let files = vec![
            HfFile {
                path: "onnx/model_fp16.onnx".into(),
                size: 1_000_000,
            },
            HfFile {
                path: "onnx/model_q4f16.onnx".into(),
                size: 500_000,
            },
            HfFile {
                path: "onnx/model_q4f16.onnx_data".into(),
                size: 300_000_000,
            },
        ];
        let v = discover_onnx_variants(&files);
        assert!(!v.is_empty());
        assert_eq!(v[0].label, "model_q4f16");
        assert!(!v[0].extra_files.is_empty());
    }
}
