#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod column;
pub mod differentia;
pub mod errors;
pub mod policies;
pub mod reconstruction;
pub mod serialization;

#[cfg(feature = "pyo3")]
pub mod pyo3_bindings;

pub use column::HereditaryStratigraphicColumn;
pub use column::Stratum;
pub use differentia::Differentia;
pub use errors::HstratError;

// Reconstruction re-exports
pub use reconstruction::{
    AlifeDataFrame, TreeAlgorithm,
    // Juxtaposition
    calc_min_implausible_spurious_collisions,
    calc_probability_differentia_collision,
    calc_rank_of_first_retained_disparity_between,
    calc_rank_of_last_retained_commonality_between,
    calc_ranks_since_first_retained_disparity_with,
    calc_ranks_since_last_retained_commonality_with,
    does_definitively_share_no_common_ancestor,
    get_nth_common_rank_between,
    // MRCA
    calc_rank_of_mrca_bounds_among,
    calc_rank_of_mrca_bounds_between,
    calc_rank_of_mrca_uncertainty_among,
    calc_ranks_since_mrca_bounds_between,
    does_have_any_common_ancestor,
    does_share_any_common_ancestor_among,
    // Estimation
    Estimator,
    ballpark_rank_of_mrca_between,
    ballpark_ranks_since_mrca_with,
    calc_rank_of_mrca_bounds_provided_confidence_level,
    calc_rank_of_mrca_uncertainty_between,
    calc_ranks_since_mrca_bounds_with,
    calc_ranks_since_mrca_uncertainty_with,
    estimate_rank_of_mrca_between,
    estimate_rank_of_mrca_naive,
    estimate_ranks_since_mrca_with,
    // Priors
    ArbitraryPrior, ExponentialPrior, Prior, UniformPrior,
    // Postprocessors
    AssignOriginTimeNodeRankPostprocessor,
    AssignOriginTimeNaivePostprocessor,
    AssignOriginTimeExpectedValuePostprocessor,
    CompoundPostprocessor,
    TriePostprocessor,
    // Tree
    build_tree,
};

// Serialization re-exports
pub use serialization::{
    col_from_int, col_from_int_with_options,
    col_to_int, col_to_int_with_options,
    col_from_packet, col_to_packet,
};

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
