mod router_traces;

pub use router_traces::generate_router_traces;

#[cfg(test)]
mod tests {
    #[test]
    fn research_module_exports_generate_router_traces() {
        let _f: fn(
            &crate::eval::AgentCodeHarness,
            Option<&std::path::Path>,
        ) -> Result<std::path::PathBuf, crate::core::error::GpError> =
            crate::research::generate_router_traces;
        let _ = _f;
    }
}
