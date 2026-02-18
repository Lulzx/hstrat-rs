use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::Debug;

/// Trait for stratum retention policies.
///
/// A retention policy determines which strata (historical checkpoints) are
/// kept or discarded as new generations are deposited. Different policies
/// offer different trade-offs between memory usage and MRCA inference accuracy.
pub trait StratumRetentionPolicy: Clone + Debug + PartialEq + Send + Sync {
    /// Return the ranks that should be DROPPED after `num_strata_deposited`
    /// strata have been deposited, given the current set of retained ranks.
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64>;

    /// Iterate over retained ranks in ascending order for a column
    /// with `num_strata_deposited` strata deposited.
    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_>;

    /// Exact number of strata retained after `num_strata_deposited` depositions.
    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64;

    /// Whether the policy guarantees that when exactly one stratum must be
    /// dropped on deposition, it is always the second-most-recent stratum.
    ///
    /// This enables a constant-time pruning fast path in column updates.
    fn fast_drop_singleton_is_second_most_recent(&self) -> bool {
        false
    }

    /// The rank at a given column index (0-based) when `num_strata_deposited`
    /// strata have been deposited.
    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64;

    /// The absolute MRCA uncertainty for two identical columns at this depth.
    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64;

    /// Algorithm identifier string for serialization compatibility.
    fn algo_identifier(&self) -> &'static str;
}
