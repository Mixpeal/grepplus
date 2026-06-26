use std::cell::RefCell;
use std::rc::Rc;

use crate::traits::EmbedAccounting;

/// Thread-local embed counter passed through search → JIT reheat.
#[derive(Clone, Default)]
pub struct EmbedStatsCell {
    inner: Rc<RefCell<EmbedAccounting>>,
}

impl EmbedStatsCell {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_chunks(&self, n: usize, dim: usize) {
        let mut acc = self.inner.borrow_mut();
        acc.chunks_embedded += n;
        acc.bytes_embedded += n * dim * 4;
    }

    pub fn snapshot(&self) -> EmbedAccounting {
        self.inner.borrow().clone()
    }

    pub fn reset(&self) {
        *self.inner.borrow_mut() = EmbedAccounting::default();
    }
}
