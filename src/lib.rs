//! grep+ — hybrid code search (lexical + semantic + JIT indexing).

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

pub mod agent_eval;
pub mod chunk;
pub mod cli;
pub mod core;
pub mod embed;
pub mod eval;
pub mod fusion;
pub mod grep;
pub mod index;
pub mod laser;
pub mod research;
pub mod router;
pub mod search;
pub mod serve;
pub mod sketch;
