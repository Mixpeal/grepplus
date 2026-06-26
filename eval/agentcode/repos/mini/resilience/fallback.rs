// graceful degradation when upstream fails
pub fn fallback_handler() {
    circuit_breaker::CircuitBreaker.open();
}

mod circuit_breaker {
    pub use super::super::breaker::CircuitBreaker;
}
