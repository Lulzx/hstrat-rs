#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod differentia;
pub mod errors;
pub mod policies;
pub mod column;
pub mod reconstruction;
pub mod serialization;

#[cfg(feature = "pyo3")]
pub mod pyo3_bindings;

pub use differentia::Differentia;
pub use errors::HstratError;
pub use column::HereditaryStratigraphicColumn;
pub use column::Stratum;

/// Largest power of 2 less than or equal to `x`. Returns 0 for input 0.
/// Must match Python hstrat's `_bit_floor` exactly.
#[inline]
pub fn bit_floor(x: u64) -> u64 {
    if x == 0 {
        0
    } else {
        1u64 << (63 - x.leading_zeros() as u64)
    }
}
